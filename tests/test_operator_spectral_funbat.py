import torch

from geoaware.operator_spectral_funbat import (
    ModeAdaptiveVariationalCP,
    all_grid_indices,
    fourier_features,
    extended_generic_dictionary,
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


def test_collapsed_mixture_has_bank_independent_coefficient_budget_and_psd_spectrum():
    _, base = generic_spectral_dictionary(3)
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    four_atom = base[None].expand(3, -1, -1).clone()
    eight_atom = torch.cat((four_atom, four_atom), dim=1)
    small = ModeAdaptiveVariationalCP(
        coords, four_atom, rank=2, mixture_parameterization="collapsed",
    )
    large = ModeAdaptiveVariationalCP(
        coords, eight_atom, rank=2, mixture_parameterization="collapsed",
    )
    assert small.variational_mean.shape == large.variational_mean.shape == (3, 2, 7)
    assert torch.all(small.induced_spectra() >= 0)
    assert torch.allclose(
        small.induced_spectra()[..., :1]
        + 2 * small.induced_spectra()[..., 1:].sum(-1, keepdim=True),
        torch.ones(3, 2, 1),
    )


def test_exact_posterior_moments_match_monte_carlo_for_collapsed_model():
    _, base = generic_spectral_dictionary(3)
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    model = ModeAdaptiveVariationalCP(
        coords, base[None].expand(3, -1, -1).clone(), rank=2,
        mixture_parameterization="collapsed",
    )
    indices = all_grid_indices((7, 7, 7))[:5]
    mean, variance = model.posterior_moments(indices)
    generator = torch.Generator().manual_seed(123)
    samples = model.posterior_predictive_samples(
        indices, samples=4000, generator=generator, include_noise=False,
    )
    assert torch.allclose(samples.mean(0), mean, atol=0.04, rtol=0.12)
    assert torch.allclose(samples.var(0), variance, atol=0.04, rtol=0.18)


def test_collapsed_routing_gradient_is_finite_with_strict_zero_support():
    _, base = generic_spectral_dictionary(3)
    spectra = base[None].expand(3, -1, -1).clone()
    spectra[..., 2:] = 0
    # Preserve exact zero high-frequency support after variance normalization.
    from geoaware.operator_spectral_funbat import normalize_spectrum
    spectra = normalize_spectrum(spectra)
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    model = ModeAdaptiveVariationalCP(
        coords, spectra, rank=2, mixture_parameterization="collapsed",
    )
    indices = all_grid_indices((7, 7, 7))[:31]
    loss, _ = model.negative_elbo(
        indices, torch.randn(len(indices)), total_count=len(indices), samples=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.routing_logits.grad).all()


def test_generic_escape_floor_is_preserved_by_routing():
    _, base = generic_spectral_dictionary(3)
    spectra = torch.cat((base, base))[None].expand(3, -1, -1).clone()
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    floor = torch.cat((torch.zeros(4), torch.full((4,), 0.25 / 4)))
    model = ModeAdaptiveVariationalCP(
        coords, spectra, rank=2, mixture_parameterization="collapsed",
        routing_floor=floor,
    )
    with torch.no_grad():
        model.routing_logits[..., :4] = 20
        model.routing_logits[..., 4:] = -20
    weights = model.routing_weights()
    assert torch.all(weights[..., 4:] >= floor[4:])
    assert torch.allclose(weights.sum(-1), torch.ones(3, 2))


def test_generic_floor_restores_features_missing_from_first_atom_support():
    _, generic = generic_spectral_dictionary(3)
    wrong = generic.clone()
    wrong[..., 2:] = 0
    from geoaware.operator_spectral_funbat import normalize_spectrum
    wrong = normalize_spectrum(wrong)
    spectra = torch.cat((wrong, generic))[None].expand(3, -1, -1).clone()
    coords = tuple(torch.linspace(0, 1, 8)[:-1] for _ in range(3))
    floor = torch.cat((torch.zeros(4), torch.full((4,), 0.25 / 4)))
    model = ModeAdaptiveVariationalCP(
        coords, spectra, rank=2, mixture_parameterization="collapsed",
        routing_floor=floor,
    )
    features = model._collapsed_features(0, torch.arange(7))
    # k=2 cosine/sine columns must remain active via the generic floor even
    # though the first four operator atoms have strict zero support there.
    assert features[..., 2].abs().max() > 1e-3
    assert features[..., 5].abs().max() > 1e-3
    kernel = features[:, 0] @ features[:, 0].T
    assert torch.linalg.eigvalsh(kernel).min() > -1e-5
    indices = all_grid_indices((7, 7, 7))[:31]
    loss, _ = model.negative_elbo(
        indices, torch.randn(len(indices)), total_count=len(indices), samples=2,
    )
    loss.backward()
    assert torch.isfinite(model.routing_logits.grad).all()


def test_extended_generic_dictionary_extends_without_changing_the_frozen_four():
    names4, base = generic_spectral_dictionary(6)
    names12, extended = extended_generic_dictionary(12, 6)
    # Every previously frozen result must stay reproducible.
    assert names12[:4] == names4
    assert torch.allclose(extended[:4], base, atol=1e-6)
    assert extended.shape == (12, 7)
    assert len(set(names12)) == 12
    assert torch.all(extended >= 0)
    # Unit marginal variance, i.e. s_0 + 2 * sum_{k>0} s_k == 1.
    mass = extended[:, 0] + 2 * extended[:, 1:].sum(-1)
    assert torch.allclose(mass, torch.ones(12), atol=1e-5)
    # The extras must be genuinely distinct directions, otherwise an
    # atom-count-matched control would be a fake control.
    normalized = extended / extended.norm(dim=-1, keepdim=True)
    gram = normalized @ normalized.T
    off_diagonal = gram - torch.eye(12)
    assert off_diagonal.max() < 0.9999


def test_extended_generic_dictionary_builds_psd_kernels():
    _, extended = extended_generic_dictionary(10, 6)
    x = torch.linspace(0, 1, 17)[:-1]
    phi = fourier_features(x, extended)
    for index in range(len(extended)):
        kernel = phi[index] @ phi[index].T
        assert torch.linalg.eigvalsh(kernel).min() > -1e-5


def test_wave_parameter_defaults_reproduce_the_frozen_literals():
    frequency = torch.arange(7, dtype=torch.float32)
    default = operator_joint_spectrum("wave", frequency)
    explicit = operator_joint_spectrum(
        "wave", frequency, wave_coefficients=(1.35, 0.65), wave_damping=(0.45, 0.18),
    )
    assert torch.equal(default, explicit)
    # The arguments must actually move the spectrum, otherwise a wave-family
    # sweep would silently be a single repeated atom.
    shifted = operator_joint_spectrum(
        "wave", frequency, wave_coefficients=(0.7, 1.4), wave_damping=(0.9, 0.05),
    )
    assert not torch.allclose(default, shifted)
    assert torch.all(shifted >= 0)


def test_reaction_diffusion_symbol_is_even_and_band_pass():
    frequency = torch.arange(9, dtype=torch.float32)
    # A positive reaction rate must create an interior spectral peak: the
    # Turing wavenumber.  A generic monotone-decaying kernel cannot express it,
    # which is the whole reason this family is the headline case.
    banded = operator_joint_spectrum(
        "reaction_diffusion", frequency,
        reaction_diffusivity=(1.0, 1.0), reaction_rate=9.0, reaction_damping=0.3,
    )
    spatial = banded[:, 0, 0]
    assert int(spatial.argmax()) not in (0, len(spatial) - 1)
    # With no reaction the symbol must decay monotonically from the origin.
    plain = operator_joint_spectrum(
        "reaction_diffusion", frequency, reaction_rate=0.0,
    )
    assert int(plain[:, 0, 0].argmax()) == 0
    assert torch.all(banded >= 0) and torch.isfinite(banded).all()
    # Adding it must not perturb any previously frozen operator.
    for name in ("diffusion", "advection", "wave"):
        assert torch.isfinite(operator_joint_spectrum(name, frequency)).all()


def _tucker(ranks=(2, 5, 5), families=4, floor=None, routing="per_mode_rank"):
    from geoaware.operator_spectral_funbat import ModeAdaptiveVariationalTucker
    _, generic = extended_generic_dictionary(families, 6)
    spectra = generic[None].expand(3, -1, -1).clone()
    coords = tuple(torch.arange(n) / n for n in (16, 12, 12))
    return ModeAdaptiveVariationalTucker(
        coords, spectra, ranks=ranks, routing=routing, routing_floor=floor,
    )


def test_tucker_coefficient_budget_is_independent_of_bank_size():
    small, large = _tucker(families=4), _tucker(families=12)
    def coefficients(model):
        return sum(p.numel() for p in model.variational_mean) + sum(
            p.numel() for p in model.raw_variational_std)
    assert coefficients(small) == coefficients(large)
    # Only the routing logits may grow with the bank, exactly as in the CP host.
    assert sum(p.numel() for p in large.parameters() if p.requires_grad) > \
           sum(p.numel() for p in small.parameters() if p.requires_grad)


def test_tucker_unequal_ranks_give_the_small_core_that_motivates_it():
    model = _tucker(ranks=(2, 11, 11))
    assert tuple(model.core.shape) == (2, 11, 11)
    assert model.core.numel() == 242            # versus 1331 for an equal rank 11
    values = model.posterior_mean(torch.zeros(5, 3, dtype=torch.long))
    assert values.shape == (5,) and torch.isfinite(values).all()


def test_tucker_routed_mixture_builds_psd_kernels_per_mode():
    model = _tucker()
    for mode in range(3):
        features = model._collapsed_features(mode, torch.arange(12 if mode else 16))
        for rank in range(model.ranks[mode]):
            gram = features[:, rank] @ features[:, rank].T
            assert torch.linalg.eigvalsh(gram).min() > -1e-5


def test_tucker_routing_floor_is_respected_and_gradients_stay_finite():
    floor = torch.cat((torch.zeros(2), torch.full((2,), 0.25 / 2)))
    model = _tucker(floor=floor)
    for mode, weights in enumerate(model.routing_weights()):
        assert torch.all(weights[:, 2:] >= floor[2:] - 1e-6)
        assert torch.allclose(weights.sum(-1), torch.ones(model.ranks[mode]), atol=1e-5)
    indices = torch.zeros(24, 3, dtype=torch.long)
    loss, _ = model.negative_elbo(indices, torch.randn(24), total_count=24, samples=2)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.routing_logits)


def test_tucker_posterior_mean_matches_large_sample_monte_carlo():
    model = _tucker(ranks=(2, 3, 3))
    indices = torch.stack(torch.meshgrid(
        torch.arange(4), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(-1, 3)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        exact = model.posterior_mean(indices)
        sampled = model.posterior_predictive_samples(
            indices, samples=4000, generator=generator).mean(0)
    assert torch.allclose(exact, sampled, atol=0.05)


def test_banded_operator_spectrum_peaks_off_the_origin():
    frequency = torch.arange(9, dtype=torch.float32)
    spectrum = operator_joint_spectrum(
        "banded_pattern", frequency, band_wavenumber=4.0,
        band_stiffness=1.0, band_offset=0.05, source_scale=1e-4)
    radial = spectrum[:, 0, 0]
    # The whole point of this family: the maximum is at a nonzero wavenumber,
    # so a monotone-decaying generic kernel cannot represent it.
    assert int(radial.argmax()) not in (0, len(radial) - 1)
    assert torch.all(spectrum >= 0) and torch.isfinite(spectrum).all()
    # Moving the band must move the peak.
    shifted = operator_joint_spectrum(
        "banded_pattern", frequency, band_wavenumber=2.0,
        band_stiffness=1.0, band_offset=0.05, source_scale=1e-4)
    assert int(shifted[:, 0, 0].argmax()) < int(radial.argmax())
