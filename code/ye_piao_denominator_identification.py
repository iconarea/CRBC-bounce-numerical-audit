#!/usr/bin/env python3
"""Identify the denominator of Eq. (13) in Ye & Piao (2019) from the paper's own identity.

Text extraction of arXiv:1901.02202 cannot distinguish the two symbols that both render
as "M": the calligraphic coefficient of Eq. (16) and the effective Planck mass squared of
Eq. (11). Eq. (13) therefore has several consistent readings:

    V = 2 [ N / (a * D) * d(a * M_cal)/d(eta) - 1 ]

with D one of {M_cal, M_cal^2, M_pl_sq, M_pl_sq/2, A, 1}. Guessing is not acceptable, so
the reading is fixed by two published statements that do not depend on it:

  1. Eq. (26): c_s^2(eta -> +inf) = (-x H' + H x') / (3 H^2 x^2) -> 1 for x_f = p_f/3.
     With c1(+inf) = 0 this gives U(+inf) = 6/N^2 = 6 and therefore V(+inf) = 6.
  2. Fig. 3: the model is ghost-free and gradient-stable, i.e. U > 0 and c_s^2 > 0
     on the whole displayed range.

A reading that fails either test is rejected. If no reading passes, the transcription is
reported as unresolved rather than as a verdict on the published model.

Background and auxiliary definitions used (Eqs. 17-20, 24, 25):
    H = eta / (p(eta) (1 + eta^2))          physical Hubble rate
    x = 1/N,  A(N, eta) = 1/2 + f1 / N^2,   M_pl_sq = 2 N A
    f1 = c2 (k2 * H + 1) / (2 x^2)
    M_cal = (1 - 4 f1^2 x^4) / (2 H (1 + 6 f1 x^2))
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

PARAMETERS = {
    "p_i": 8.0,
    "p_f": 3.0,
    "eta_p": 0.7,
    "tau_p": 1.0,
    "eta_x": -3.0,
    "tau_x": 3.0,
    "k1": 0.06,
    "k2": 2.0,
    "tau1": 2.0,
    "tau2_sq": 0.6,
}

DENOMINATORS = ("M_cal", "M_cal_sq", "M_pl_sq", "half_M_pl_sq", "A", "unity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=48001)
    parser.add_argument("--extent", type=float, default=15.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ye_piao_denominator_identification.json"),
    )
    return parser.parse_args()


def background(points: int, extent: float, device: torch.device, hubble: str) -> dict[str, torch.Tensor]:
    par = PARAMETERS
    eta = torch.linspace(-extent, extent, points, dtype=torch.float64, device=device)
    step = float((eta[1] - eta[0]).item())
    x_i, x_f = par["p_i"] / 3.0, par["p_f"] / 3.0  # Eq. (26) fixes x_f = p_f/3

    p = par["p_i"] + 0.5 * (1.0 + torch.tanh((eta - par["eta_p"]) / par["tau_p"])) * (par["p_f"] - par["p_i"])
    x = x_i + 0.5 * (1.0 + torch.tanh((eta - par["eta_x"]) / par["tau_x"])) * (x_f - x_i)
    lapse = 1.0 / x
    physical_h = eta / (p * (1.0 + eta.square()))
    conformal_h = lapse * physical_h
    scale_factor = torch.exp(cumulative_trapezoid(conformal_h, step))

    h_used = physical_h if hubble == "physical" else conformal_h
    c1 = par["k1"] * (1.0 - torch.tanh(eta / par["tau1"]))
    c2 = torch.exp(-eta.square() / par["tau2_sq"])
    f1 = c2 * (par["k2"] * h_used + 1.0) / (2.0 * x.square())

    m_cal = (1.0 - 4.0 * f1.square() * x.pow(4)) / (2.0 * h_used * (1.0 + 6.0 * f1 * x.square()))
    middle = points // 2
    m_cal[middle] = 0.5 * (m_cal[middle - 1] + m_cal[middle + 1])

    a_coefficient = 0.5 + f1 * x.square()  # A = 1/2 + f1/N^2
    m_pl_sq = 2.0 * lapse * a_coefficient  # Eq. (7): M^2/2 = N A

    return {
        "eta": eta,
        "step": torch.tensor(step, dtype=torch.float64, device=device),
        "a": scale_factor,
        "x": x,
        "N": lapse,
        "H": physical_h,
        "c1": c1,
        "f1": f1,
        "M_cal": m_cal,
        "A": a_coefficient,
        "M_pl_sq": m_pl_sq,
    }


def evaluate_reading(fields: dict[str, torch.Tensor], denominator: str, extent: float) -> dict[str, object]:
    eta = fields["eta"]
    step = float(fields["step"].item())
    a, lapse, m_cal = fields["a"], fields["N"], fields["M_cal"]

    choices = {
        "M_cal": m_cal,
        "M_cal_sq": m_cal.square(),
        "M_pl_sq": fields["M_pl_sq"],
        "half_M_pl_sq": 0.5 * fields["M_pl_sq"],
        "A": fields["A"],
        "unity": torch.ones_like(m_cal),
    }
    d = choices[denominator]

    u = fields["c1"] + 6.0 / lapse.square()  # Eq. (12) with Eq. (22)
    v = 2.0 * (lapse * derivative(a * m_cal, step) / (a * d) - 1.0)  # Eq. (13) reading
    cs_sq = v / u

    interior = slice(1, -1)
    eta_i, cs_i, u_i, v_i = eta[interior], cs_sq[interior], u[interior], v[interior]
    finite = torch.isfinite(cs_i) & torch.isfinite(u_i)

    tail = (eta_i > 0.8 * extent) & finite
    asymptotic_v = float(v_i[tail].mean().item()) if bool(tail.any().item()) else float("nan")
    asymptotic_cs = float(cs_i[tail].mean().item()) if bool(tail.any().item()) else float("nan")

    negative = (cs_i < 0.0) & finite
    return {
        "denominator": denominator,
        "asymptotic_c_s_sq": asymptotic_cs,
        "asymptotic_V": asymptotic_v,
        "asymptotic_c_s_sq_error": abs(asymptotic_cs - 1.0),
        "ghost_free": bool((u_i[finite] > 0.0).all().item()),
        "gradient_stable": int(negative.sum().item()) == 0,
        "negative_fraction": float(negative.double().mean().item()),
        "min_c_s_sq": float(cs_i[finite].min().item()),
        "max_c_s_sq": float(cs_i[finite].max().item()),
        "passes_eq26": abs(asymptotic_cs - 1.0) < 0.02,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: dict[str, list[dict[str, object]]] = {}
    for hubble in ("physical", "conformal"):
        fields = background(args.points, args.extent, device, hubble)
        results[hubble] = [evaluate_reading(fields, name, args.extent) for name in DENOMINATORS]

    accepted = [
        {"hubble": hubble, **entry}
        for hubble, entries in results.items()
        for entry in entries
        if entry["passes_eq26"] and entry["ghost_free"] and entry["gradient_stable"]
    ]
    eq26_only = [
        {"hubble": hubble, **entry}
        for hubble, entries in results.items()
        for entry in entries
        if entry["passes_eq26"]
    ]

    if accepted:
        conclusion = (
            "Reading identified: {} with the {} Hubble convention reproduces both the "
            "asymptotic identity and the published stability claim.".format(
                accepted[0]["denominator"], accepted[0]["hubble"]
            )
        )
    elif eq26_only:
        conclusion = (
            "A reading satisfies the asymptotic identity but not the published stability "
            "claim. The transcription is not yet established; no stability verdict is issued."
        )
    else:
        conclusion = (
            "No candidate reading reproduces the paper's own asymptotic identity. The "
            "transcription remains unresolved and no stability verdict on the published "
            "model may be drawn from these runs."
        )

    report: dict[str, object] = {
        "reference": "G. Ye and Y.-S. Piao, Commun. Theor. Phys. 71 (2019) 427, arXiv:1901.02202",
        "problem": "Eq. (13) denominator is ambiguous under text extraction; fixed by published identities.",
        "acceptance_tests": {
            "eq26_asymptotic": "c_s^2(+inf) -> 1, equivalently V(+inf) -> 6 since U(+inf) = 6",
            "fig3_stability": "U > 0 and c_s^2 > 0 on the whole range",
        },
        "grid": {"points": args.points, "extent": args.extent},
        "device": str(device),
        "readings": results,
        "accepted_readings": accepted,
        "readings_passing_eq26_only": eq26_only,
        "transcription_resolved": bool(accepted),
        "conclusion": conclusion,
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("readings", "accepted_readings", "transcription_resolved", "conclusion")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
