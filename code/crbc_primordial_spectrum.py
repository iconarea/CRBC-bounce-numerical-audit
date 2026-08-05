#!/usr/bin/env python3
"""Primordial curvature spectrum of the gated CRBC beyond-Horndeski realization.

This replaces the surrogate Gaussian potential of crbc_perturbation_scan.py with the
z''/z that follows from the coefficient trajectory which passed
crbc_eft_coefficient_gate.py (see CRBC_EFT_선정과_계수계약_kr.md §7.3).

Derivation of the quantities used, for the N = 1 (unit lapse) CRBC background:

    Ye-Piao Eq. (11) with N = 1:  L_2 = a^3 (M_pl^2/2) [ U zeta_dot^2 - V (grad zeta)^2/a^2 ]
    so with A_S = (M_pl^2/2) U = q_s and c_s^2 = V/U, in conformal time dtau = dt/a,

        S_2 = int dtau d^3x a^2 [ A_S zeta_tau^2 - A_S c_s^2 (grad zeta)^2 ],
        z^2 = 2 a^2 A_S      ->   z = a sqrt(2 q_s),
        v = z zeta,          ->   v'' + (c_s^2 k^2 - z''/z) v = 0,  ' = d/dtau.

No interpolation onto a conformal-time grid is needed, because d/dtau = a d/dt gives

        z''/z = (a^2 z_ddot + a a_dot z_dot) / z,
        v_ddot = -(a_dot/a) v_dot - (c_s^2 k^2 - z''/z) v / a^2,

both evaluated on the uniform cosmic-time grid the background is defined on.

Initial condition: the adiabatic (WKB) vacuum in the far contracting past,

        v_k = 1/sqrt(2 c_s k),   v_k' = (-i c_s k - c_s'/(2 c_s)) v_k,

which is only legitimate for modes satisfying c_s^2 k^2 >> |z''/z| at the start. The run
reports that ratio per mode and refuses to report a spectrum for modes that fail it.

Output: P_R(k) = k^3 |v_k/z|^2 / (2 pi^2), its tilt, and the convergence and freezing
diagnostics. Units are those of the realization (alpha = 1), so the amplitude is measured
in units of the bounce curvature scale; converting it to the observed value requires
rho_c/M_pl^4, which this construction does not fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.crbc_beyond_horndeski_realization import coefficients, crbc_background
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from crbc_beyond_horndeski_realization import coefficients, crbc_background  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Gated realization of CRBC_EFT_선정과_계수계약_kr.md §7.3.
    parser.add_argument("--p-initial", type=float, default=8.0)
    parser.add_argument("--p-final", type=float, default=2.4)
    parser.add_argument("--eta-p", type=float, default=0.7)
    parser.add_argument("--tau-p", type=float, default=1.0)
    parser.add_argument("--k1", type=float, default=12.0)
    parser.add_argument("--k2", type=float, default=5.0)
    parser.add_argument("--tau1", type=float, default=1.7)
    parser.add_argument("--tau2-sq", type=float, default=0.59)
    parser.add_argument("--w", type=float, default=1.0, help="only used if --p-initial is unset")

    parser.add_argument("--extent", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=100000, help="RK4 steps; background uses 2*steps+1 points")
    parser.add_argument("--modes", type=int, default=96)
    parser.add_argument("--k-min", type=float, default=1e-3)
    parser.add_argument("--k-max", type=float, default=2e-1)
    parser.add_argument(
        "--t-eval",
        type=float,
        default=None,
        help="time at which P_R is read; default is the post-bounce maximum of |aH|, "
        "where the largest range of modes is super-horizon",
    )
    parser.add_argument("--wkb-ratio-threshold", type=float, default=100.0)
    parser.add_argument(
        "--super-horizon-margin",
        type=float,
        default=3.0,
        help="required |aH|/k at t_eval; larger means the mode is more firmly frozen",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_primordial_spectrum.json"))
    return parser.parse_args()


def first_derivative(values: torch.Tensor, step: float) -> torch.Tensor:
    result = torch.empty_like(values)
    result[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * step)
    result[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * step)
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * step)
    return result


def second_derivative(values: torch.Tensor, step: float) -> torch.Tensor:
    result = torch.empty_like(values)
    result[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / step**2
    result[0] = result[1]
    result[-1] = result[-2]
    return result


def post_bounce_horizon_peak(args: argparse.Namespace, device: torch.device) -> float:
    """Time of maximal |aH| after the bounce; the widest super-horizon window."""
    probe = torch.linspace(1e-3, 20.0, 200001, dtype=torch.float64, device=device)
    background = crbc_background(probe, args.w, args)
    comoving = (background["a"] * background["H"]).abs()
    return float(probe[torch.argmax(comoving)].item())


def build_background(
    args: argparse.Namespace, points: int, device: torch.device, t_end: float
) -> dict[str, torch.Tensor]:
    """Background, coefficients, and the Mukhanov-Sasaki potential on a uniform t grid."""
    time = torch.linspace(-args.extent, t_end, points, dtype=torch.float64, device=device)
    step = float((time[1] - time[0]).item())
    background = crbc_background(time, args.w, args)

    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64, device=device)

    values = coefficients(
        time, step, background, scalar(args.k1), scalar(args.k2), scalar(args.tau1), scalar(args.tau2_sq)
    )

    scale = background["a"]
    q_s = values["q_t"] * values["U"]  # A_S = (M_pl^2/2) U
    cs_sq = values["cs_sq"]
    if bool((q_s <= 0.0).any().item()):
        raise ValueError("q_s <= 0 somewhere; the trajectory is not the gated one.")

    z = scale * torch.sqrt(2.0 * q_s)
    z_dot = first_derivative(z, step)
    z_ddot = second_derivative(z, step)
    scale_dot = first_derivative(scale, step)
    # z''/z with ' = d/dtau and d/dtau = a d/dt.
    z_potential = (scale.square() * z_ddot + scale * scale_dot * z_dot) / z

    cs = torch.sqrt(torch.clamp(cs_sq, min=1e-300))
    return {
        "time": time,
        "step": torch.tensor(step, dtype=torch.float64, device=device),
        "a": scale,
        "a_dot": scale_dot,
        "H": background["H"],
        "q_s": q_s,
        "cs_sq": cs_sq,
        "cs": cs,
        "cs_dot": first_derivative(cs, step),
        "z": z,
        "z_potential": z_potential,
    }


def integrate_modes(fields: dict[str, torch.Tensor], k: torch.Tensor, steps: int) -> dict[str, torch.Tensor]:
    """RK4 on v_ddot = -(a_dot/a) v_dot - (c_s^2 k^2 - z''/z) v / a^2, in cosmic time."""
    step = float(fields["step"].item()) * 2.0  # background grid is twice as fine as the RK4 grid
    scale, scale_dot = fields["a"], fields["a_dot"]
    cs_sq, z_potential = fields["cs_sq"], fields["z_potential"]

    friction = scale_dot / scale
    stiffness = 1.0 / scale.square()
    k_sq = k.square().unsqueeze(1)  # (modes, 1)

    def acceleration(index: int, v: torch.Tensor, v_dot: torch.Tensor) -> torch.Tensor:
        return -friction[index] * v_dot - (cs_sq[index] * k_sq[:, 0] - z_potential[index]) * stiffness[index] * v

    # Adiabatic vacuum in the far contracting past.
    cs0, cs_dot0, a0 = fields["cs"][0], fields["cs_dot"][0], scale[0]
    v = (1.0 / torch.sqrt(2.0 * cs0 * k)).to(torch.complex128)
    v_tau = (-1j * cs0 * k - cs_dot0 * a0 / (2.0 * cs0)) * v  # c_s' = a * c_s_dot
    v_dot = v_tau / a0

    for i in range(steps):
        left, middle, right = 2 * i, 2 * i + 1, 2 * i + 2
        k1v, k1a = v_dot, acceleration(left, v, v_dot)
        k2v, k2a = v_dot + 0.5 * step * k1a, acceleration(middle, v + 0.5 * step * k1v, v_dot + 0.5 * step * k1a)
        k3v, k3a = v_dot + 0.5 * step * k2a, acceleration(middle, v + 0.5 * step * k2v, v_dot + 0.5 * step * k2a)
        k4v, k4a = v_dot + step * k3a, acceleration(right, v + step * k3v, v_dot + step * k3a)
        v = v + step * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        v_dot = v_dot + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0

    return {"v": v, "v_dot": v_dot}


def analyse(args: argparse.Namespace, device: torch.device, steps: int, t_end: float) -> dict[str, object]:
    fields = build_background(args, 2 * steps + 1, device, t_end)
    k = torch.logspace(
        float(torch.log10(torch.tensor(args.k_min))),
        float(torch.log10(torch.tensor(args.k_max))),
        args.modes,
        dtype=torch.float64,
        device=device,
    )
    result = integrate_modes(fields, k, steps)

    z_end, a_end = fields["z"][-1], fields["a"][-1]
    curvature = result["v"] / z_end
    spectrum = k.pow(3) * curvature.abs().square() / (2.0 * torch.pi**2)

    # Initial-condition validity: the WKB vacuum needs the mode deep inside the sound
    # horizon, i.e. c_s^2 k^2 >> |z''/z| and c_s k >> |aH|, at the start.
    comoving_initial = (fields["a"][0] * fields["H"][0]).abs()
    wkb_ratio = (fields["cs_sq"][0] * k.square()) / fields["z_potential"][0].abs()
    sub_horizon_initial = fields["cs"][0] * k / comoving_initial

    # The spectrum is only meaningful for modes that are super-horizon at the read-out time.
    comoving_final = (a_end * fields["H"][-1]).abs()
    super_horizon_final = comoving_final / k

    valid = (
        (wkb_ratio > args.wkb_ratio_threshold)
        & (sub_horizon_initial > args.wkb_ratio_threshold**0.5)
        & (super_horizon_final > args.super_horizon_margin)
    )

    # Freezing: |R'| / (aH |R|) with R = v/z and ' = d/dtau = a d/dt.
    z_dot_end = (fields["z"][-1] - fields["z"][-2]) / float(fields["step"].item())
    curvature_rate = (result["v_dot"] * a_end / z_end - curvature * (a_end * z_dot_end / z_end)).abs()
    freezing = curvature_rate / (comoving_final * curvature.abs())

    return {
        "k": k,
        "spectrum": spectrum,
        "wkb_ratio": wkb_ratio,
        "sub_horizon_initial": sub_horizon_initial,
        "super_horizon_final": super_horizon_final,
        "valid": valid,
        "freezing": freezing,
        "z_potential_initial": float(fields["z_potential"][0].item()),
        "comoving_horizon_initial": float((fields["a"][0] * fields["H"][0]).abs().item()),
        "comoving_horizon_final": float((fields["a"][-1] * fields["H"][-1]).abs().item()),
        "min_cs_sq": float(fields["cs_sq"].min().item()),
        "max_cs_sq": float(fields["cs_sq"].max().item()),
    }


def tilt(k: torch.Tensor, spectrum: torch.Tensor, valid: torch.Tensor) -> dict[str, object]:
    """Least-squares slope of ln P_R against ln k; n_s - 1 is that slope."""
    if int(valid.sum().item()) < 3:
        return {"fitted_modes": int(valid.sum().item()), "n_s_minus_one": None, "n_s": None}
    x = torch.log(k[valid])
    y = torch.log(spectrum[valid])
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).square()).sum()
    residual = y - (y_mean + slope * (x - x_mean))
    return {
        "fitted_modes": int(valid.sum().item()),
        "n_s_minus_one": float(slope.item()),
        "n_s": float(slope.item()) + 1.0,
        "max_abs_log_residual": float(residual.abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_end = args.t_eval if args.t_eval is not None else post_bounce_horizon_peak(args, device)
    fine = analyse(args, device, args.steps, t_end)
    coarse = analyse(args, device, args.steps // 2, t_end)

    valid = fine["valid"] & coarse["valid"]
    convergence = (
        float(((fine["spectrum"][valid] - coarse["spectrum"][valid]) / fine["spectrum"][valid]).abs().max().item())
        if bool(valid.any().item())
        else None
    )

    fitted = tilt(fine["k"], fine["spectrum"], fine["valid"])
    k_list = fine["k"].cpu().tolist()
    spectrum_list = fine["spectrum"].cpu().tolist()

    report: dict[str, object] = {
        "source": "gated CRBC beyond-Horndeski realization, CRBC_EFT_선정과_계수계약_kr.md §7.3",
        "replaces": "the surrogate Gaussian potential of crbc_perturbation_scan.py",
        "parameters": {
            "p_initial": args.p_initial,
            "p_final": args.p_final,
            "k1": args.k1,
            "k2": args.k2,
            "tau1": args.tau1,
            "tau2_sq": args.tau2_sq,
        },
        "grid": {"extent": args.extent, "rk4_steps": args.steps, "modes": args.modes, "t_eval": t_end},
        "device": str(device),
        "background": {
            "min_c_s_sq": fine["min_cs_sq"],
            "max_c_s_sq": fine["max_cs_sq"],
            "comoving_horizon_initial_aH": fine["comoving_horizon_initial"],
            "comoving_horizon_final_aH": fine["comoving_horizon_final"],
            "z_potential_initial": fine["z_potential_initial"],
        },
        "mode_window": {
            "threshold_c_s_sq_k_sq_over_z_potential": args.wkb_ratio_threshold,
            "requires": "sub-horizon WKB vacuum at t_start and super-horizon at t_eval",
            "min_sub_horizon_ratio_at_start": float(fine["sub_horizon_initial"][fine["valid"]].min().item())
            if bool(fine["valid"].any().item())
            else None,
            "min_super_horizon_ratio_at_eval": float(fine["super_horizon_final"][fine["valid"]].min().item())
            if bool(fine["valid"].any().item())
            else None,
            "valid_modes": int(fine["valid"].sum().item()),
            "total_modes": args.modes,
            "valid_k_range": [
                float(fine["k"][fine["valid"]].min().item()),
                float(fine["k"][fine["valid"]].max().item()),
            ]
            if bool(fine["valid"].any().item())
            else None,
            "min_wkb_ratio_among_valid": float(fine["wkb_ratio"][fine["valid"]].min().item())
            if bool(fine["valid"].any().item())
            else None,
        },
        "convergence": {
            "steps_compared": [args.steps // 2, args.steps],
            "max_relative_spectrum_difference": convergence,
        },
        "freezing_diagnostic": {
            "max_R_rate_over_aH_among_valid": float(fine["freezing"][fine["valid"]].max().item())
            if bool(fine["valid"].any().item())
            else None,
            "note": "large values mean the mode has not settled by the end of the integration",
        },
        "spectrum": {
            "k": k_list,
            "P_R": spectrum_list,
            "valid": fine["valid"].cpu().tolist(),
        },
        "tilt": fitted,
        "analytic_expectation": {
            "formula": "n_s - 1 = 3 - 2|beta - 1/2| with beta = 1/(p_i - 1), from a ~ |tau|^beta "
            "and z ~ a in the asymptotic contracting phase",
            "beta": 1.0 / (args.p_initial - 1.0) if args.p_initial else None,
            "n_s_minus_one": 3.0 - 2.0 * abs(1.0 / (args.p_initial - 1.0) - 0.5) if args.p_initial else None,
            "ekpyrotic_limit": "n_s - 1 -> 2 as p_i -> infinity",
        },
        "units": "alpha = 1, a_B = 1. The amplitude is in units of the bounce curvature scale; "
        "converting to the observed 2.1e-9 requires rho_c/M_pl^4, which this construction does not fix.",
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "spectrum"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
