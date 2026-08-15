import torch

from geoaware.operator_spectral_funbat import (
    ModeAdaptiveVariationalCP,
    all_grid_indices,
    fourier_features,
    generic_spectral_dictionary,
    nonnegative_cp_spectrum,
    operator_joint_spectrum,
)


def test_fourier_features_define_psd_kernel():
    _, spectra = generic_spectral_dictionary(4)
    x = torch.linspace(0, 1, 17)[:-1]
    phi = fourier_features(x, spectra)
    kernel = phi[2] @ phi[2].T
    assert torch.linalg.eigvalsh(kernel).min() > -1e-5
    assert torch.allclose(kernel.diag(), torch.ones(len(x)), atol=1e-5)


def test_nonnegative_operator_spectrum_separation():
    frequency = torch.arange(6, dtype=torch.float32)
    spectrum = operator_joint_spectrum("diffusion", frequency)
    result = nonnegative_cp_spectrum(spectrum, rank=3, steps=100, seed=3)
    assert result.reconstruction.shape == spectrum.shape
    assert all(torch.all(factor >= 0) for factor in result.factors)
    assert result.relative_error < 0.5


def test_elbo_routes_have_gradients_and_shapes():
    _, base = generic_spectral_dictionary(3)
    spectra = base[None].expand(3, -1, -1).clone()
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    model = ModeAdaptiveVariationalCP(coords, spectra, rank=2)
    indices = all_grid_indices((7, 7, 7))[:31]
    targets = torch.randn(len(indices))
    loss, _ = model.negative_elbo(indices, targets, total_count=len(indices), samples=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.routing_logits.grad is not None
    assert torch.isfinite(model.routing_logits.grad).all()
    assert model.posterior_mean(indices).shape == targets.shape


def test_hierarchical_route_shares_rank_and_shrinks_deviation():
    _, base = generic_spectral_dictionary(3)
    spectra = base[None].expand(3, -1, -1).clone()
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    model = ModeAdaptiveVariationalCP(coords, spectra, rank=2, routing="hierarchical")
    weights = model.routing_weights()
    assert torch.allclose(weights[:, 0], weights[:, 1])
    base_kl = model.kl_to_prior()
    with torch.no_grad():
        model.mode_deviation[0, 0] = 1.0
    assert model.kl_to_prior() > base_kl
