#!/usr/bin/env python3
"""Quadrupolar statistical anisotropy g_* from residual shear on the CRBC bounce.

CRBC_EFT_선정과_계수계약_kr.md §10 concluded that an isotropic FLRW background cannot
generate the angular factor Y_LM of the coherent-relic template, and that the only
dynamical source of a preferred direction inside CRBC is residual shear. Shear forces
L = 2 and produces a multiplicative modulation rather than an additive term,

    P(k, k_hat) = P_iso(k) [ 1 + g_*(k) (k_hat . n_hat)^2 ],

which Planck already constrains: Kim & Komatsu (2013) give Delta g_* = 0.016 (1 sigma).
This script computes g_* for the gated CRBC realization, so the relic becomes a prediction
that existing data can reject.

Setup. Axisymmetric Bianchi I on top of the §7.3 background, treated as a test anisotropy:

    a_1 = a_2 = a e^{beta},  a_3 = a e^{-2 beta},   sigma^2 = 3 beta_dot^2,  beta_dot = C/a^3

so sigma ~ a^-3 as required for Bianchi I without anisotropic stress. The physical
wavenumber becomes direction dependent, and to first order in beta

    a^2 k_phys^2 = k^2 [ 1 - 2 beta + 6 beta mu ],     mu = (k_hat . n_hat)^2,

which is the only place the anisotropy enters the mode equation for the entropy field.

beta is normalized to vanish at the read-out time. A constant beta is a rescaling of
comoving coordinates and is unobservable, so only the change in beta between horizon exit
and read-out can produce a physical quadrupole; anchoring beta at read-out enforces that.

Method. Rather than expanding P in beta analytically, the modes are integrated at several
mu and the modulation is measured. That tests the quadrupolar *form* as well as its
amplitude: if P(k, mu) is not linear in mu, the template
P_iso [1 + g_* (k_hat . n_hat)^2] is itself wrong.

Two shear amplitudes are run to confirm that g_* is linear in C in the test regime, and
the self-consistency of treating shear as a test field is reported as sigma^2/rho.

Not claimed: the beyond-Horndeski stability gate of §7 is not re-derived on a
non-perturbative Bianchi I background. Treating shear perturbatively is justified only
while sigma^2/rho stays small, which the run reports; a shear large enough to violate that
would require redoing the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.crbc_beyond_horndeski_realization import reconstruct_equation_of_state
    from quantum_gravity.crbc_gpu.crbc_primordial_spectrum import build_background, post_bounce_horizon_peak
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from crbc_beyond_horndeski_realization import reconstruct_equation_of_state  # type: ignore[no-redef]
    from crbc_primordial_spectrum import build_background, post_bounce_horizon_peak  # type: ignore[no-redef]

PLANCK_G_STAR_SIGMA = 0.016  # Kim & Komatsu (2013), 1 sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-initial", type=float, default=8.0)
    parser.add_argument("--p-final", type=float, default=2.4)
    parser.add_argument("--eta-p", type=float, default=0.7)
    parser.add_argument("--tau-p", type=float, default=1.0)
    parser.add_argument("--k1", type=float, default=12.0)
    parser.add_argument("--k2", type=float, default=5.0)
    parser.add_argument("--tau1", type=float, default=1.7)
    parser.add_argument("--tau2-sq", type=float, default=0.59)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--entropic-lambda", type=float, default=104.0)

    parser.add_argument("--extent", type=float, default=2000.0)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--k-min", type=float, default=3e-4)
    parser.add_argument("--k-max", type=float, default=1e-1)
    parser.add_argument("--mu-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--shear-amplitudes", type=float, nargs="+", default=[1e-4, 2e-4])
    parser.add_argument("--wkb-ratio-threshold", type=float, default=100.0)
    parser.add_argument("--super-horizon-margin", type=float, default=3.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_bianchi_quadrupole.json"))
    return parser.parse_args()


def shear_profile(fields: dict[str, torch.Tensor], amplitude: float) -> dict[str, torch.Tensor]:
    """beta with beta_dot = C/a^3, anchored so beta(t_eval) = 0."""
    step = float(fields["step"].item())
    scale = fields["a"]
    beta_dot = amplitude / scale.pow(3)
    beta = torch.zeros_like(beta_dot)
    beta[1:] = torch.cumsum(0.5 * (beta_dot[1:] + beta_dot[:-1]) * step, dim=0)
    beta = beta - beta[-1]  # unobservable constant removed by anchoring at read-out
    return {"beta": beta, "beta_dot": beta_dot, "shear_sq": 3.0 * beta_dot.square()}


def integrate(
    fields: dict[str, torch.Tensor],
    k: torch.Tensor,
    steps: int,
    lam: float,
    beta: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Entropy modes with the direction-dependent physical wavenumber."""
    step = float(fields["step"].item()) * 2.0
    fine_step = float(fields["step"].item())
    scale, scale_dot, hubble = fields["a"], fields["a_dot"], fields["H"]

    scale_ddot = torch.empty_like(scale)
    scale_ddot[1:-1] = (scale[2:] - 2.0 * scale[1:-1] + scale[:-2]) / fine_step**2
    scale_ddot[0], scale_ddot[-1] = scale_ddot[1], scale_ddot[-2]
    a_second_conformal = scale_dot.square() + scale * scale_ddot
    comoving_hubble_sq = (scale * hubble).square()

    anisotropy = 1.0 - 2.0 * beta + 6.0 * beta * mu  # a^2 k_phys^2 / k^2
    friction = scale_dot / scale
    inverse_scale_sq = 1.0 / scale.square()

    def acceleration(index: int, u: torch.Tensor, u_dot: torch.Tensor) -> torch.Tensor:
        potential = k.square() * anisotropy[index] - lam * comoving_hubble_sq[index] - a_second_conformal[index]
        return -friction[index] * u_dot - potential * inverse_scale_sq[index] * u

    effective_k = k * anisotropy[0].clamp(min=1e-12).sqrt()
    u = (1.0 / torch.sqrt(2.0 * effective_k)).to(torch.complex128)
    u_dot = (-1j * effective_k / scale[0]) * u

    for i in range(steps):
        left, middle, right = 2 * i, 2 * i + 1, 2 * i + 2
        k1u, k1a = u_dot, acceleration(left, u, u_dot)
        k2u, k2a = u_dot + 0.5 * step * k1a, acceleration(middle, u + 0.5 * step * k1u, u_dot + 0.5 * step * k1a)
        k3u, k3a = u_dot + 0.5 * step * k2a, acceleration(middle, u + 0.5 * step * k2u, u_dot + 0.5 * step * k2a)
        k4u, k4a = u_dot + step * k3a, acceleration(right, u + step * k3u, u_dot + step * k3a)
        u = u + step * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        u_dot = u_dot + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0

    entropy = u / scale[-1]
    return k.pow(3) * entropy.abs().square() / (2.0 * torch.pi**2)


def linear_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    """Least-squares slope, intercept, and maximum absolute residual."""
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).square()).sum()
    intercept = y_mean - slope * x_mean
    residual = (y - (intercept + slope * x)).abs().max()
    return float(slope.item()), float(intercept.item()), float(residual.item())


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_end = post_bounce_horizon_peak(args, device)
    fields = build_background(args, 2 * args.steps + 1, device, t_end)
    step = float(fields["step"].item())
    k = torch.logspace(
        float(torch.log10(torch.tensor(args.k_min))),
        float(torch.log10(torch.tensor(args.k_max))),
        args.modes,
        dtype=torch.float64,
        device=device,
    )

    comoving_initial = (fields["a"][0] * fields["H"][0]).abs()
    comoving_final = (fields["a"][-1] * fields["H"][-1]).abs()
    effective_potential = (
        fields["a_dot"][0].square()
        + fields["a"][0] * (fields["a"][2] - 2.0 * fields["a"][1] + fields["a"][0]) / step**2
        + args.entropic_lambda * (fields["a"][0] * fields["H"][0]).square()
    ).abs()
    valid = (
        (k.square() / effective_potential > args.wkb_ratio_threshold)
        & (k / comoving_initial > args.wkb_ratio_threshold**0.5)
        & (comoving_final / k > args.super_horizon_margin)
    )

    equation_of_state = reconstruct_equation_of_state(fields["time"], {"H": fields["H"], "a": fields["a"]}, step, args.w)
    density_over_critical = max(equation_of_state["max_rho_over_rhoc"], 1e-30)
    critical_density = 4.0 / (3.0 * (1.0 + args.w) ** 2)  # 8 pi G rho_c at alpha = 1

    runs = []
    mu_grid = torch.tensor(args.mu_values, dtype=torch.float64, device=device)
    for amplitude in args.shear_amplitudes:
        profile = shear_profile(fields, amplitude)
        spectra = torch.stack([integrate(fields, k, args.steps, args.entropic_lambda, profile["beta"], mu)
                               for mu in args.mu_values])
        isotropic = spectra[0]
        ratio = spectra / isotropic  # (mu, k)

        # Quadrupolar form test: is P(mu)/P(0) linear in mu at each k?
        slopes, residuals = [], []
        for index in range(k.numel()):
            slope, _, residual = linear_fit(mu_grid, ratio[:, index])
            slopes.append(slope)
            residuals.append(residual)
        slope_tensor = torch.tensor(slopes, dtype=torch.float64, device=device)
        residual_tensor = torch.tensor(residuals, dtype=torch.float64, device=device)

        runs.append(
            {
                "shear_amplitude_C": amplitude,
                "g_star_from_endpoints": (ratio[-1] - 1.0)[valid].tolist(),
                "g_star_from_linear_fit_mean": float(slope_tensor[valid].mean().item()),
                "g_star_min": float(slope_tensor[valid].min().item()),
                "g_star_max": float(slope_tensor[valid].max().item()),
                "max_nonlinearity_in_mu": float(residual_tensor[valid].max().item()),
                "beta_at_start": float(profile["beta"][0].item()),
                "max_shear_sq_over_rho": float(
                    (profile["shear_sq"].max() / (critical_density * density_over_critical)).item()
                ),
            }
        )

    # Linearity of g_* in the shear amplitude.
    linear_in_amplitude = None
    if len(runs) >= 2:
        first, second = runs[0], runs[1]
        expected = first["g_star_from_linear_fit_mean"] * (
            second["shear_amplitude_C"] / first["shear_amplitude_C"]
        )
        linear_in_amplitude = {
            "predicted_from_first_run": expected,
            "measured": second["g_star_from_linear_fit_mean"],
            "relative_difference": abs(second["g_star_from_linear_fit_mean"] - expected) / abs(expected),
        }

    reference = runs[0]
    response = reference["g_star_from_linear_fit_mean"] / reference["shear_amplitude_C"]
    shear_bound = PLANCK_G_STAR_SIGMA / abs(response) if response != 0.0 else None

    report: dict[str, object] = {
        "purpose": "compute the quadrupolar statistical anisotropy g_* generated by residual "
        "shear on the gated CRBC bounce, and confront it with Planck",
        "background": "CRBC_EFT_선정과_계수계약_kr.md §7.3 realization with the §9 entropic mass, "
        "plus an axisymmetric Bianchi I test shear",
        "device": str(device),
        "grid": {"extent": args.extent, "rk4_steps": args.steps, "modes": args.modes, "t_eval": t_end},
        "valid_modes": int(valid.sum().item()),
        "valid_k_range": [float(k[valid].min().item()), float(k[valid].max().item())]
        if bool(valid.any().item())
        else None,
        "mu_values": args.mu_values,
        "runs": runs,
        "linearity_in_shear_amplitude": linear_in_amplitude,
        "planck_confrontation": {
            "g_star_per_unit_shear_amplitude": response,
            "planck_1sigma_delta_g_star": PLANCK_G_STAR_SIGMA,
            "reference": "Kim & Komatsu (2013)",
            "shear_amplitude_bound_C": shear_bound,
        },
        "not_claimed": [
            "The §7 beyond-Horndeski stability gate is not re-derived on a non-perturbative "
            "Bianchi I background; shear is treated as a test anisotropy.",
            "The initial shear amplitude is not predicted by CRBC, so g_* is bounded, not predicted.",
        ],
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "runs"}
    summary["runs"] = [{key: value for key, value in entry.items() if key != "g_star_from_endpoints"} for entry in runs]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
