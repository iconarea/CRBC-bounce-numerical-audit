#!/usr/bin/env python3
"""Diagnose the negative sound-speed region in the Ye & Piao (2019) transcription.

The direct unitary-gauge transcription reports min(c_s^2) ~ -120 near the zero of
the gamma-crossing quantity M. Two hypotheses must be separated before that number
can be called a physical gradient instability:

  A. physical: c_s^2 is genuinely negative on a finite interval.
  B. artifact: c_s^2 has a simple pole at M=0, so its sampled minimum is set by how
     close the grid lands to the pole and is not a converged quantity.

Hypothesis B is decided without reference to any particular transcription by grid
refinement: a simple pole makes min(c_s^2) diverge as the grid resolves M=0, while a
physical negative region converges. The residue test (eta - eta_0) * c_s^2 -> const
confirms the pole order.

The script additionally evaluates the standard Horndeski gradient coefficient
F_S = (1/a) d(a M)/dt - F_T, which is regular through M=0, against the transcribed
form that divides by (a M) instead of a. This is a transcription check, not a claim
about the published paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.ye_piao_2019_reproduction import (
        cumulative_trapezoid,
        derivative,
    )
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from ye_piao_2019_reproduction import (  # type: ignore[no-redef]
        cumulative_trapezoid,
        derivative,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=[3001, 6001, 12001, 24001, 48001, 96001],
        help="odd grid sizes used for the pole-versus-physical refinement test",
    )
    parser.add_argument("--extent", type=float, default=15.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ye_piao_cs_sq_pole_diagnostic.json"),
    )
    return parser.parse_args()


def coefficients(points: int, extent: float, device: torch.device) -> dict[str, torch.Tensor]:
    """Background and scalar quadratic coefficients on a uniform conformal-time grid."""
    if points < 5 or points % 2 == 0:
        raise ValueError("points must be odd and at least 5 so eta=0 is represented.")
    eta = torch.linspace(-extent, extent, points, dtype=torch.float64, device=device)
    step = float((eta[1] - eta[0]).item())
    pi, pf, eta_p, tau_p = 8.0, 3.0, 0.7, 1.0
    xi, xf, eta_x, tau_x = pi / 3.0, pf / 3.0, -3.0, 3.0
    k1, k2, tau1, tau2_sq = 0.06, 2.0, 2.0, 0.6

    p = pi + 0.5 * (1.0 + torch.tanh((eta - eta_p) / tau_p)) * (pf - pi)
    x = xi + 0.5 * (1.0 + torch.tanh((eta - eta_x) / tau_x)) * (xf - xi)
    lapse = 1.0 / x
    physical_h = eta / (p * (1.0 + eta.square()))
    conformal_h = lapse * physical_h
    scale_factor = torch.exp(cumulative_trapezoid(conformal_h, step))

    c1 = k1 * (1.0 - torch.tanh(eta / tau1))
    c2 = torch.exp(-eta.square() / tau2_sq)
    f1 = c2 * (k2 * conformal_h + 1.0) / (2.0 * x.square())

    numerator = 1.0 - 4.0 * f1.square() * x.pow(4)
    denominator = 2.0 * conformal_h * (1.0 + 6.0 * f1 * x.square())
    m = numerator / denominator
    middle = points // 2
    m[middle] = 0.5 * (m[middle - 1] + m[middle + 1])

    u = c1 + 6.0 / lapse
    a_m = scale_factor * m
    d_a_m = derivative(a_m, step)

    # Transcribed form: divides the derivative by (a M); singular wherever M = 0.
    v_transcribed = 2.0 * (lapse * d_a_m / a_m - 1.0)
    # Horndeski gradient coefficient F_S = (1/a) d(a M)/dt - F_T with F_T = 1, dt = N d(eta).
    v_regular = 2.0 * (lapse * d_a_m / scale_factor - 1.0)

    return {
        "eta": eta,
        "step": torch.tensor(step, dtype=torch.float64, device=device),
        "a": scale_factor,
        "h": conformal_h,
        "u": u,
        "m": m,
        "a_m": a_m,
        "cs_sq_transcribed": v_transcribed / u,
        "cs_sq_regular": v_regular / u,
    }


def zero_crossings(values: torch.Tensor, eta: torch.Tensor) -> list[float]:
    """Linearly interpolated locations where `values` changes sign."""
    left, right = values[:-1], values[1:]
    indices = torch.where(left * right < 0.0)[0]
    crossings: list[float] = []
    for index in indices.tolist():
        y0, y1 = float(left[index].item()), float(right[index].item())
        e0, e1 = float(eta[index].item()), float(eta[index + 1].item())
        crossings.append(e0 + (e1 - e0) * y0 / (y0 - y1))
    return crossings


def diagnose(points: int, extent: float, device: torch.device) -> dict[str, object]:
    fields = coefficients(points, extent, device)
    eta = fields["eta"]
    cs_transcribed = fields["cs_sq_transcribed"]
    cs_regular = fields["cs_sq_regular"]
    m = fields["m"]

    m_zeros = zero_crossings(m, eta)
    min_index = int(torch.argmin(cs_transcribed).item())
    eta_min = float(eta[min_index].item())
    min_value = float(cs_transcribed[min_index].item())

    # Residue test: for a simple pole, (eta - eta_0) * c_s^2 approaches a constant.
    residue = None
    distance_to_pole = None
    if m_zeros:
        nearest_zero = min(m_zeros, key=lambda value: abs(value - eta_min))
        distance_to_pole = abs(eta_min - nearest_zero)
        residue = (eta_min - nearest_zero) * min_value

    return {
        "points": points,
        "grid_step": float(fields["step"].item()),
        "m_zero_crossings": m_zeros,
        "min_abs_m": float(m.abs().min().item()),
        "transcribed": {
            "min_c_s_sq": min_value,
            "max_c_s_sq": float(cs_transcribed.max().item()),
            "eta_at_min": eta_min,
            "distance_from_min_to_nearest_m_zero": distance_to_pole,
            "simple_pole_residue_estimate": residue,
            "negative_fraction": float((cs_transcribed < 0.0).double().mean().item()),
        },
        "regular": {
            "min_c_s_sq": float(cs_regular.min().item()),
            "max_c_s_sq": float(cs_regular.max().item()),
            "eta_at_min": float(eta[torch.argmin(cs_regular)].item()),
            "all_finite": bool(torch.isfinite(cs_regular).all().item()),
            "positive_everywhere": bool((cs_regular > 0.0).all().item()),
            "negative_fraction": float((cs_regular < 0.0).double().mean().item()),
        },
        "min_u": float(fields["u"].min().item()),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    refinements = [diagnose(points, args.extent, device) for points in sorted(args.refinements)]

    transcribed_minima = [entry["transcribed"]["min_c_s_sq"] for entry in refinements]
    regular_minima = [entry["regular"]["min_c_s_sq"] for entry in refinements]
    residues = [
        entry["transcribed"]["simple_pole_residue_estimate"]
        for entry in refinements
        if entry["transcribed"]["simple_pole_residue_estimate"] is not None
    ]

    finest = refinements[-1]
    coarsest = refinements[0]
    transcribed_diverges = abs(transcribed_minima[-1]) > 4.0 * abs(transcribed_minima[0])
    regular_converged = (
        abs(regular_minima[-1] - regular_minima[-2]) <= 1e-3 * max(1.0, abs(regular_minima[-1]))
        if len(regular_minima) > 1
        else False
    )

    verdict = {
        "transcribed_min_c_s_sq_by_resolution": dict(
            zip((entry["points"] for entry in refinements), transcribed_minima)
        ),
        "regular_min_c_s_sq_by_resolution": dict(
            zip((entry["points"] for entry in refinements), regular_minima)
        ),
        "transcribed_minimum_diverges_under_refinement": transcribed_diverges,
        "residue_estimates": residues,
        "residue_spread_relative": (
            (max(residues) - min(residues)) / abs(sum(residues) / len(residues)) if residues else None
        ),
        "regular_form_converged": regular_converged,
        "regular_form_positive_everywhere": finest["regular"]["positive_everywhere"],
        "conclusion": (
            "The negative sound speed of the direct transcription is a simple pole at the "
            "gamma-crossing zero of M, not a converged physical gradient instability."
            if transcribed_diverges
            else "The negative region survives refinement and must be treated as physical."
        ),
    }

    report: dict[str, object] = {
        "reference": "Ye & Piao, Commun. Theor. Phys. 71 (2019) 427-434",
        "purpose": "separate a gauge/transcription pole from a physical gradient instability",
        "device": str(device),
        "coarsest_grid": coarsest,
        "finest_grid": finest,
        "refinements": refinements,
        "verdict": verdict,
        "caveat": (
            "A regular F_S = (1/a) d(a M)/dt - F_T is used as the comparison form. Confirming "
            "that it is the published coefficient requires the paper's own equation, so this "
            "run establishes the pole diagnosis, not the reproduction verdict."
        ),
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
