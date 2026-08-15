"""Explicit geometry-aware neural CP/Tucker models for Paper B."""
from __future__ import annotations
import math
import torch
from torch import nn
from .neural_geometry import WaveTask, source_geodesic_distance


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                         nn.GELU(), nn.Linear(hidden, out_dim))


def paired_phase_carriers(distance: torch.Tensor, time: torch.Tensor,
                          bands: torch.Tensor, speeds: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the canonical four-term CP carriers for ``k(d-c t)``.

    Component order is ``(band, speed, trig-term)`` with trig terms
    ``cos(d)cos(t)``, ``sin(d)sin(t)``, ``cos(d)sin(t)``, and
    ``sin(d)cos(t)``.  Keeping this construction in one pure function makes
    the angle-addition claim directly testable and prevents a silent reshape
    change from invalidating the claimed phase pairing.
    """
    if distance.ndim != 2 or distance.shape[1] != 1:
        raise ValueError("distance must have shape (N, 1)")
    if time.ndim != 2 or time.shape[1] != 1:
        raise ValueError("time must have shape (N, 1)")
    if len(distance) != len(time):
        raise ValueError("distance and time must have the same batch size")
    kd = distance[:, :, None] * bands[None, :, None]
    kd = kd.expand(-1, -1, len(speeds)).reshape(len(distance), -1)
    kct = time[:, :, None] * bands[None, :, None] * speeds[None, None, :]
    kct = kct.reshape(len(time), -1)
    sd, cd = torch.sin(kd), torch.cos(kd)
    st, ct = torch.sin(kct), torch.cos(kct)
    spatial = torch.stack([cd, sd, cd, sd], -1).reshape(len(distance), -1)
    temporal = torch.stack([ct, st, st, ct], -1).reshape(len(time), -1)
    return spatial, temporal


def tensor_mode_features(task: WaveTask, wrong_geometry: bool = False,
                         use_phase: bool = True):
    """Return separate `(geometry, time, spatial)` mode features."""
    g = task.domain.descriptor.float()
    t = torch.tensor([task.time], dtype=torch.float32)
    xy = task.domain.coords.float()
    sdf = task.domain.signed_distance[:, None].float()
    radius = torch.linalg.vector_norm(xy - torch.tensor([-.72, -.38]), dim=1, keepdim=True)
    distance = radius[:, 0] if wrong_geometry else source_geodesic_distance(task.domain)
    spatial = torch.cat([xy, sdf, distance[:, None]], 1)
    if use_phase:
        bands = torch.tensor([7., 13., 19., 27., 37.])
        spatial = torch.cat([spatial, torch.sin(distance[:, None]*bands),
                             torch.cos(distance[:, None]*bands)], 1)
        t = torch.cat([t, torch.sin(task.time*bands), torch.cos(task.time*bands)])
    return g, t, spatial


class GeometryNeuralCP(nn.Module):
    """Mode-wise neural factors contracted through an explicit CP core."""
    def __init__(self, rank=32, hidden=64, use_phase=True):
        super().__init__(); self.rank=rank; self.use_phase=use_phase
        self.geometry_factor=_mlp(7,rank,hidden)
        self.time_factor=_mlp(11 if use_phase else 1,rank,hidden)
        self.space_factor=_mlp(14 if use_phase else 4,rank,hidden)
        self.weight=nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def feature_task(self, task, wrong_geometry=False):
        return tensor_mode_features(task,wrong_geometry,self.use_phase)

    def forward_features(self, features, node_ids=None):
        g,t,x=features
        dev=self.weight.device;g,t,x=g.to(dev),t.to(dev),x.to(dev)
        if node_ids is not None:x=x[node_ids]
        gf=self.geometry_factor(g[None]).squeeze(0)
        tf=self.time_factor(t[None]).squeeze(0)
        xf=self.space_factor(x)
        return (xf*gf*tf*self.weight).sum(1)

    def forward_points(self,g,t,x):
        dev=self.weight.device;g,t,x=g.to(dev),t.to(dev),x.to(dev)
        return (self.geometry_factor(g)*self.time_factor(t)*self.space_factor(x)*self.weight).sum(1)

    def forward_task(self,task,node_ids=None,wrong_geometry=False):
        return self.forward_features(self.feature_task(task,wrong_geometry),node_ids)


class GeometryNeuralTucker(nn.Module):
    """Small nonlinear mode factors with a learned rank-(Rg,Rt,Rs) core."""
    def __init__(self, ranks=(6,10,16), hidden=64, use_phase=True, diagonal=False,
                 band_gates=False):
        super().__init__(); self.ranks=ranks;self.use_phase=use_phase;self.diagonal=diagonal
        self.band_gates=band_gates
        rg,rt,rs=ranks
        self.geometry_factor=_mlp(7,rg,hidden)
        self.time_factor=_mlp(11 if use_phase else 1,rt,hidden)
        self.space_factor=_mlp(14 if use_phase else 4,rs,hidden)
        self.core=nn.Parameter(torch.randn(rg,rt,rs)/math.sqrt(rg*rt*rs))
        if band_gates:
            self.raw_band_gate=nn.Parameter(torch.full((5,), 1.5))

    def _gate_features(self, t, x):
        if not self.band_gates: return t, x
        gate=torch.sigmoid(self.raw_band_gate)
        # time=[raw,sin5,cos5], space=[raw4,sin5,cos5]
        t=torch.cat([t[:1],t[1:6]*gate,t[6:11]*gate])
        x=torch.cat([x[:,:4],x[:,4:9]*gate,x[:,9:14]*gate],1)
        return t,x

    def feature_task(self,task,wrong_geometry=False):
        return tensor_mode_features(task,wrong_geometry,self.use_phase)

    def forward_features(self,features,node_ids=None):
        g,t,x=features;dev=self.core.device;g,t,x=g.to(dev),t.to(dev),x.to(dev)
        if node_ids is not None:x=x[node_ids]
        t,x=self._gate_features(t,x)
        gf=self.geometry_factor(g[None]).squeeze(0);tf=self.time_factor(t[None]).squeeze(0);xf=self.space_factor(x)
        core=self.core
        if self.diagonal:
            mask=torch.zeros_like(core);r=min(self.ranks)
            idx=torch.arange(r,device=dev);mask[idx,idx,idx]=1;core=core*mask
        return torch.einsum('g,t,x,gtx,nx->n',gf,tf,torch.ones(self.ranks[2],device=dev),core,xf)

    def forward_points(self,g,t,x):
        dev=self.core.device;g,t,x=g.to(dev),t.to(dev),x.to(dev)
        if self.band_gates:
            gate=torch.sigmoid(self.raw_band_gate)
            t=torch.cat([t[:,:1],t[:,1:6]*gate,t[:,6:11]*gate],1)
            x=torch.cat([x[:,:4],x[:,4:9]*gate,x[:,9:14]*gate],1)
        gf,tf,xf=self.geometry_factor(g),self.time_factor(t),self.space_factor(x)
        core=self.core
        if self.diagonal:
            mask=torch.zeros_like(core);r=min(self.ranks);idx=torch.arange(r,device=dev);mask[idx,idx,idx]=1;core=core*mask
        return torch.einsum('ng,nt,ns,gts->n',gf,tf,xf,core)

    def regularization(self):
        if not self.band_gates:return self.core.new_zeros(())
        return torch.sigmoid(self.raw_band_gate).mean()

    def band_summary(self):
        return torch.sigmoid(self.raw_band_gate).detach().cpu().tolist() if self.band_gates else None

    def forward_task(self,task,node_ids=None,wrong_geometry=False):
        return self.forward_features(self.feature_task(task,wrong_geometry),node_ids)


class SpeedAlignedPhaseCP(nn.Module):
    """CP whose mode factors implement paired traveling-phase identities.

    No network sees joint ``(distance,time)`` inputs.  Spatial and temporal
    carriers are computed separately and combined only by the CP product.
    Four components per (speed, band) span all sin/cos phase offsets.
    """
    def __init__(self, hidden=48):
        super().__init__()
        self.register_buffer('bands',torch.tensor([7.,13.,19.,27.,37.]))
        self.register_buffer('speeds',torch.tensor([.35,.65,1.0]))
        self.components=4*len(self.bands)*len(self.speeds)
        self.geometry_factor=_mlp(7,self.components,hidden)
        self.time_amplitude=_mlp(1,self.components,hidden)
        self.space_amplitude=_mlp(4,self.components,hidden)
        self.weight=nn.Parameter(torch.ones(self.components)/math.sqrt(self.components))

    def feature_task(self,task,wrong_geometry=False):
        return tensor_mode_features(task,wrong_geometry,False)

    def _carriers(self,t,x):
        # x[:,3] is intrinsic/Euclidean source distance; t[:,0] is time.
        return paired_phase_carriers(x[:, 3:4], t[:, 0:1], self.bands,
                                     self.speeds)

    def forward_points(self,g,t,x):
        dev=self.weight.device;g,t,x=g.to(dev),t.to(dev),x.to(dev)
        spatial,temporal=self._carriers(t,x)
        gf=self.geometry_factor(g);tf=self.time_amplitude(t)*temporal
        xf=self.space_amplitude(x)*spatial
        return (gf*tf*xf*self.weight).sum(1)

    def forward_features(self,features,node_ids=None):
        g,t,x=features
        if node_ids is not None:x=x[node_ids]
        n=len(x);return self.forward_points(g[None].expand(n,-1),t[None].expand(n,-1),x)

    def forward_task(self,task,node_ids=None,wrong_geometry=False):
        return self.forward_features(self.feature_task(task,wrong_geometry),node_ids)


class PhaseEnvelopeCP(SpeedAlignedPhaseCP):
    """Traveling-phase CP with a small explicit separable envelope rank.

    For every phase component the amplitude envelope is

    ``sum_q weight[b,q] * E_distance[q](d) * E_time[q](t)``.

    Distance and time are processed by separate networks and interact only by
    multiplication, so the model remains an explicit CP expansion with
    ``components * envelope_rank`` effective terms.  There is no joint
    ``(distance, time)`` residual path.
    """

    def __init__(self, envelope_rank: int = 4, hidden: int = 48):
        super().__init__(hidden=hidden)
        if envelope_rank < 1:
            raise ValueError("envelope_rank must be positive")
        self.envelope_rank = int(envelope_rank)
        envelope_hidden = max(16, hidden // 2)
        self.distance_envelope = _mlp(1, self.envelope_rank, envelope_hidden)
        self.time_envelope = _mlp(1, self.envelope_rank, envelope_hidden)
        self.envelope_weight = nn.Parameter(
            torch.zeros(self.components, self.envelope_rank))

    def envelope_factors(self, t, x):
        distance = x[:, 3:4]
        ed = self.distance_envelope(distance)
        et = self.time_envelope(t[:, :1])
        return ed, et

    def forward_points(self, g, t, x):
        dev = self.weight.device
        g, t, x = g.to(dev), t.to(dev), x.to(dev)
        spatial, temporal = self._carriers(t, x)
        ed, et = self.envelope_factors(t, x)
        # A residual envelope makes every Q-model start exactly from the paired
        # carrier.  Increasing Q therefore adds capacity without discarding the
        # already validated phase-aligned solution.
        envelope = 1 + torch.einsum("nq,nq,bq->nb", ed, et, self.envelope_weight)
        gf = self.geometry_factor(g)
        tf = self.time_amplitude(t) * temporal
        xf = self.space_amplitude(x) * spatial
        return (gf * tf * xf * envelope * self.weight).sum(1)


class PhaseEnvelopeTucker(SpeedAlignedPhaseCP):
    """Paired phase carriers modulated by a small fixed-basis envelope core.

    RBF bases are evaluated separately in intrinsic distance and time.  Their
    only interaction is the explicit ``component x distance x time`` Tucker
    core.  Compared with learned CP envelope factors, this removes a difficult
    multiplicative optimization while retaining a finite, inspectable tensor
    representation of a moving envelope.
    """

    def __init__(self, distance_rank: int = 10, time_rank: int = 6,
                 hidden: int = 48):
        super().__init__(hidden=hidden)
        if distance_rank < 2 or time_rank < 2:
            raise ValueError("envelope Tucker ranks must be at least two")
        self.distance_rank = int(distance_rank)
        self.time_rank = int(time_rank)
        dc = torch.linspace(0., 3.8, self.distance_rank)
        tc = torch.linspace(.12, .44, self.time_rank)
        self.register_buffer("distance_centers", dc)
        self.register_buffer("time_centers", tc)
        self.distance_width = float((dc[1] - dc[0]) * 1.35)
        self.time_width = float((tc[1] - tc[0]) * 1.35)
        self.envelope_core = nn.Parameter(
            torch.zeros(self.components, self.distance_rank, self.time_rank))

    def envelope_factors(self, t, x):
        distance = x[:, 3:4]
        ed = torch.exp(-.5 * ((distance - self.distance_centers) /
                              self.distance_width).square())
        et = torch.exp(-.5 * ((t[:, :1] - self.time_centers) /
                              self.time_width).square())
        return ed, et

    def forward_points(self, g, t, x):
        dev = self.weight.device
        g, t, x = g.to(dev), t.to(dev), x.to(dev)
        spatial, temporal = self._carriers(t, x)
        ed, et = self.envelope_factors(t, x)
        envelope = 1 + torch.einsum("nd,nt,bdt->nb", ed, et, self.envelope_core)
        gf = self.geometry_factor(g)
        tf = self.time_amplitude(t) * temporal
        xf = self.space_amplitude(x) * spatial
        return (gf * tf * xf * envelope * self.weight).sum(1)
