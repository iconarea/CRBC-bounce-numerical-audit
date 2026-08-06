#!/usr/bin/env python3
"""Entropic mechanism on the gated CRBC background: what it requires, and how tuned it is.

The single-field adiabatic spectrum of the gated CRBC realization is strongly blue
(n_s ~ 3.35, see crbc_primordial_spectrum.py and CRBC_EFT_선정과_계수계약_kr.md §8), so the
bounce cannot by itself produce the observed curvature perturbation. The standard remedy
in the ekpyrotic literature is the entropic mechanism: a second field acquires a nearly
scale-invariant isocurvature spectrum during contraction, which is later converted to
curvature.

This script does not assume the mechanism works. It asks what the CRBC background demands
of the entropy sector, and how precisely that demand must be met.

Setup. Add an entropy direction s with a minimally coupled quadratic perturbation of
effective mass m_eff on the fixed CRBC background:

    delta_s'' + 2 H_conf delta_s' + (k^2 + a^2 m_eff^2) delta_s = 0,   ' = d/dtau
    u = a delta_s   ->   u'' + (k^2 + a^2 m_eff^2 - a''/a) u = 0.

Parametrization. During ekpyrotic contraction a ~ |tau|^beta with beta = 1/(p_i - 1), and
H_conf = a H = beta/tau, so 1/tau^2 = (a H)^2/beta^2. Writing a^2 m_eff^2 = -mu/tau^2 (the
form that yields a power-law spectrum) is therefore equivalent to the regular, bounce-safe
ansatz

    m_eff^2 = -lambda H^2,      lambda = mu / beta^2,

which is the standard ekpyrotic scaling form: a tachyonic entropic mass tracking H.

Analytic expectation. With a''/a = beta(beta-1)/tau^2,

    u'' + (k^2 - [beta(beta-1) + mu]/tau^2) u = 0,
    nu^2 = 1/4 + beta(beta-1) + lambda beta^2,
    n_s - 1 = 3 - 2 nu.

Exact scale invariance (nu = 3/2) needs mu = 2 - beta(beta-1), and matching Planck's
n_s = 0.9649 needs a slightly larger mu. The script measures n_s(lambda) by integration,
checks it against this formula, solves for the lambda that reproduces Planck, and reports
the tolerance dlambda corresponding to Planck's 1-sigma error. That tolerance is the
quantitative price of the mechanism.

What is not claimed: the entropy field is added by hand, not derived from CRBC
microphysics; its sound speed is taken to be 1; and the conversion efficiency
zeta = kappa * delta_s is a free parameter, so the amplitude of the curvature spectrum is
not predicted here, only its tilt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.crbc_primordial_spectrum import (
        build_background,
        post_bounce_horizon_peak,
        tilt,
    )
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from crbc_primordial_spectrum import (  # type: ignore[no-redef]
        build_background,
        post_bounce_horizon_peak,
        tilt,
    )

PLANCK_NS = 0.9649
PLANCK_NS_SIGMA = 0.0042


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
    parser.add_argument("--adiabatic-n-s-minus-one", type=float, default=2.3542,
                        help="single-field tilt of the SAME background, for comparison only; "
                             "the default is the tanh p(eta) value and is wrong for --derived-background")
    parser.add_argument("--derived-background", action="store_true",
                        help="use the derived background of section 13 instead of the tanh p(eta)")
    parser.add_argument("--w-ekpyrotic", type=float, default=4.3333333333333333,
                        help="ekpyrotic w_1 of the derived background (distinct from --w)")
    parser.add_argument("--gamma", type=float, default=26.0)
    parser.add_argument("--rho-h-over-rho-c", type=float, default=1e-2)
    parser.add_argument("--rho-initial", type=float, default=1.8e-10)
    parser.add_argument("--rho-stop", type=float, default=1e-5)
    parser.add_argument("--t-max", type=float, default=4e6)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)

    parser.add_argument("--extent", type=float, default=2000.0)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--k-min", type=float, default=3e-4)
    parser.add_argument("--k-max", type=float, default=1e-1)
    parser.add_argument("--wkb-ratio-threshold", type=float, default=100.0)
    parser.add_argument("--super-horizon-margin", type=float, default=3.0)
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=None,
        help="entropic mass coefficients m_eff^2 = -lambda H^2; default brackets the "
        "scale-invariant value implied by p_initial",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_entropic_mechanism.json"))
    return parser.parse_args()


def analytic_tilt(beta: float, lam: float) -> float:
    """n_s - 1 = 3 - 2 nu with nu^2 = 1/4 + beta(beta-1) + lambda beta^2."""
    nu_sq = 0.25 + beta * (beta - 1.0) + lam * beta**2
    return 3.0 - 2.0 * nu_sq**0.5 if nu_sq > 0.0 else float("nan")


def lambda_for_tilt(beta: float, target_n_s: float) -> float:
    """Invert the analytic relation for the lambda that gives a target n_s."""
    nu = (3.0 - (target_n_s - 1.0)) / 2.0
    return (nu**2 - 0.25 - beta * (beta - 1.0)) / beta**2


def integrate_entropy_modes(
    fields: dict[str, torch.Tensor], k: torch.Tensor, steps: int, lam: float
) -> dict[str, torch.Tensor]:
    """RK4 for u = a delta_s with a^2 m_eff^2 = -lambda (aH)^2, in cosmic time."""
    step = float(fields["step"].item()) * 2.0
    scale, scale_dot, hubble = fields["a"], fields["a_dot"], fields["H"]

    # a''/a = a_dot^2 + a a_ddot, since d/dtau = a d/dt.
    fine_step = float(fields["step"].item())
    scale_ddot = torch.empty_like(scale)
    scale_ddot[1:-1] = (scale[2:] - 2.0 * scale[1:-1] + scale[:-2]) / fine_step**2
    scale_ddot[0], scale_ddot[-1] = scale_ddot[1], scale_ddot[-2]
    a_second_conformal = scale_dot.square() + scale * scale_ddot

    comoving_hubble_sq = (scale * hubble).square()
    friction = scale_dot / scale
    inverse_scale_sq = 1.0 / scale.square()

    def acceleration(index: int, u: torch.Tensor, u_dot: torch.Tensor) -> torch.Tensor:
        potential = k.square() - lam * comoving_hubble_sq[index] - a_second_conformal[index]
        return -friction[index] * u_dot - potential * inverse_scale_sq[index] * u

    # Minkowski (Bunch-Davies) vacuum for the canonical variable u, sound speed 1.
    u = (1.0 / torch.sqrt(2.0 * k)).to(torch.complex128)
    u_dot = (-1j * k / scale[0]) * u

    for i in range(steps):
        left, middle, right = 2 * i, 2 * i + 1, 2 * i + 2
        k1u, k1a = u_dot, acceleration(left, u, u_dot)
        k2u, k2a = u_dot + 0.5 * step * k1a, acceleration(middle, u + 0.5 * step * k1u, u_dot + 0.5 * step * k1a)
        k3u, k3a = u_dot + 0.5 * step * k2a, acceleration(middle, u + 0.5 * step * k2u, u_dot + 0.5 * step * k2a)
        k4u, k4a = u_dot + step * k3a, acceleration(right, u + step * k3u, u_dot + step * k3a)
        u = u + step * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        u_dot = u_dot + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0

    return {"u": u, "u_dot": u_dot, "a_second_conformal": a_second_conformal}


def measure(args: argparse.Namespace, device: torch.device, steps: int, t_end: float, lam: float) -> dict[str, object]:
    fields = build_background(args, 2 * steps + 1, device, t_end)
    k = torch.logspace(
        float(torch.log10(torch.tensor(args.k_min))),
        float(torch.log10(torch.tensor(args.k_max))),
        args.modes,
        dtype=torch.float64,
        device=device,
    )
    result = integrate_entropy_modes(fields, k, steps, lam)

    entropy = result["u"] / fields["a"][-1]
    spectrum = k.pow(3) * entropy.abs().square() / (2.0 * torch.pi**2)

    comoving_initial = (fields["a"][0] * fields["H"][0]).abs()
    comoving_final = (fields["a"][-1] * fields["H"][-1]).abs()
    effective_potential = (result["a_second_conformal"][0] + lam * (fields["a"][0] * fields["H"][0]).square()).abs()
    valid = (
        (k.square() / effective_potential > args.wkb_ratio_threshold)
        & (k / comoving_initial > args.wkb_ratio_threshold**0.5)
        & (comoving_final / k > args.super_horizon_margin)
    )
    return {"k": k, "spectrum": spectrum, "valid": valid, "fields": fields}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    beta = 1.0 / (args.p_initial - 1.0)
    lambda_scale_invariant = lambda_for_tilt(beta, 1.0)
    lambda_planck = lambda_for_tilt(beta, PLANCK_NS)
    # Tolerance from d(n_s)/d(lambda) = -beta^2/nu.
    nu_planck = (3.0 - (PLANCK_NS - 1.0)) / 2.0
    lambda_tolerance = PLANCK_NS_SIGMA * nu_planck / beta**2

    if args.lambda_values is not None:
        lambdas = list(args.lambda_values)
    else:
        span = 0.25 * lambda_planck
        lambdas = [lambda_planck - span, lambda_planck - 0.5 * span, lambda_planck,
                   lambda_planck + 0.5 * span, lambda_planck + span]

    t_end = post_bounce_horizon_peak(args, device)
    runs = []
    for lam in lambdas:
        fine = measure(args, device, args.steps, t_end, lam)
        fitted = tilt(fine["k"], fine["spectrum"], fine["valid"])
        predicted = analytic_tilt(beta, lam)
        runs.append(
            {
                "lambda": lam,
                "measured_n_s_minus_one": fitted["n_s_minus_one"],
                "analytic_n_s_minus_one": predicted,
                "difference": None
                if fitted["n_s_minus_one"] is None
                else fitted["n_s_minus_one"] - predicted,
                "fitted_modes": fitted["fitted_modes"],
                "max_abs_log_residual": fitted.get("max_abs_log_residual"),
            }
        )

    coarse = measure(args, device, args.steps // 2, t_end, lambda_planck)
    fine = measure(args, device, args.steps, t_end, lambda_planck)
    shared = coarse["valid"] & fine["valid"]
    convergence = (
        float(((fine["spectrum"][shared] - coarse["spectrum"][shared]) / fine["spectrum"][shared]).abs().max().item())
        if bool(shared.any().item())
        else None
    )

    consistent = [entry for entry in runs if entry["difference"] is not None]
    max_difference = max((abs(entry["difference"]) for entry in consistent), default=None)

    report: dict[str, object] = {
        "purpose": "quantify what the entropic mechanism requires of the gated CRBC background",
        "background": "CRBC_EFT_선정과_계수계약_kr.md §7.3 realization, held fixed",
        "ansatz": "m_eff^2 = -lambda H^2 for the entropy direction; equivalent to a^2 m^2 = -mu/tau^2 "
        "with mu = lambda beta^2 during ekpyrotic contraction",
        "beta": beta,
        "device": str(device),
        "grid": {"extent": args.extent, "rk4_steps": args.steps, "modes": args.modes, "t_eval": t_end},
        "requirement": {
            "lambda_for_exact_scale_invariance": lambda_scale_invariant,
            "lambda_for_planck_n_s": lambda_planck,
            "lambda_tolerance_at_planck_1sigma": lambda_tolerance,
            "fractional_tuning": lambda_tolerance / lambda_planck,
            "planck_n_s": PLANCK_NS,
            "planck_n_s_sigma": PLANCK_NS_SIGMA,
        },
        "lambda_scan": runs,
        "validation": {
            "max_abs_difference_measured_vs_analytic": max_difference,
            "grid_convergence_at_planck_lambda": convergence,
        },
        "single_field_comparison": {
            "adiabatic_n_s_minus_one": args.adiabatic_n_s_minus_one,
            "note": "the adiabatic mode of the same background, from crbc_primordial_spectrum.py; "
        "background-dependent, so it must be supplied with --adiabatic-n-s-minus-one",
        },
        "not_claimed": [
            "The entropy field is added by hand, not derived from CRBC microphysics.",
            "Its sound speed is set to 1.",
            "The conversion zeta = kappa * delta_s has a free efficiency, so the amplitude is not predicted.",
            "A scale-invariant Gaussian component is not the coherent relic; that remains separate.",
        ],
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
