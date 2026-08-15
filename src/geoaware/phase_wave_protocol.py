"""Small, auditable gates for the wave-specific phase-factor track."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AbsoluteWaveGate:
    """Outcome of the external absolute-effectiveness gate.

    Pairwise significance is deliberately absent: it is meaningful only after
    this gate passes and is handled by the shared statistics protocol.
    """

    model_nrmse: float
    strongest_trivial_nrmse: float
    mse_skill: float
    max_nrmse: float
    min_mse_skill: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def absolute_wave_gate(model_nrmse: float, strongest_trivial_nrmse: float,
                       max_nrmse: float = 0.8,
                       min_mse_skill: float = 0.2) -> AbsoluteWaveGate:
    """Require useful absolute recovery before comparing two learned models.

    NRMSE is an RMSE ratio, hence MSE skill relative to the strongest trivial
    predictor is ``1 - (model/trivial)**2``.
    """
    if model_nrmse < 0 or strongest_trivial_nrmse <= 0:
        raise ValueError("NRMSE values must be nonnegative and baseline positive")
    if not 0 <= min_mse_skill < 1:
        raise ValueError("min_mse_skill must lie in [0, 1)")
    skill = 1.0 - (model_nrmse / strongest_trivial_nrmse) ** 2
    return AbsoluteWaveGate(
        model_nrmse=float(model_nrmse),
        strongest_trivial_nrmse=float(strongest_trivial_nrmse),
        mse_skill=float(skill),
        max_nrmse=float(max_nrmse),
        min_mse_skill=float(min_mse_skill),
        passed=bool(model_nrmse <= max_nrmse and skill >= min_mse_skill),
    )
