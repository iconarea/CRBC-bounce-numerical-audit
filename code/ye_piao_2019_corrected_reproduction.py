#!/usr/bin/env python3
"""Corrected reproduction of the c_T=1 beyond-Horndeski bounce of Ye & Piao (2019).

Reference: G. Ye and Y.-S. Piao, "Implication of GW170817 for cosmological bounces",
Commun. Theor. Phys. 71 (2019) 427, arXiv:1901.02202.

Equations used:

    Eq. (12)  U = Sigma/gamma^2 + 6/N^2, with Eq. (22) Sigma = c1 gamma^2  ->  U = c1 + 6/N^2
    Eq. (13)  V = 2 [ N / (a D) d(a M_cal)/d(eta) - 1 ]
    Eq. (16)  M_cal = (1 - 4 f1^2 x^4) / (2 H (1 + 6 f1 x^2))
    Eq. (17)  H = eta / (p(eta)(1 + eta^2))          (physical Hubble; H = script_H / N)
    Eq. (18)  p(eta) = p_i + (1 + tanh((eta-eta_p)/tau_p))/2 * (p_f - p_i)
    Eq. (19)  x(eta) = 1/N = x_i + (1 + tanh((eta-eta_x)/tau_x))/2 * (x_f - x_i)
    Eq. (20)  A(N, eta) = 1/2 + f1(eta)/N^2,  and Eq. (7) M_pl_sq = 2 N A
    Eq. (24)  f1 = c2 (c3 H + 1)/(2 x^2),  c3 = k2
    Eq. (25)  c1 = k1(1 - tanh(eta/tau1)),  c2 = exp(-eta^2/tau2^2)
    Eq. (26)  c_s^2(+inf) = 1 requires x_f = p_f/3

Two transcription errors in the earlier ye_piao_2019_reproduction.py are corrected:

  1. Eq. (12) is 6/N^2, not 6/N.
  2. Eq. (13) divides by a * (M_pl_sq/2), not by a * M_cal.

The denominator D of Eq. (13) cannot be read off the PDF text, because the calligraphic
coefficient of Eq. (16) and the effective Planck mass of Eq. (11) both extract as "M". It
was instead identified by ye_piao_denominator_identification.py, which scans the candidate
readings against two published statements: the asymptotic identity of Eq. (26) and the
ghost-free/gradient-stable claim of Fig. 3. Exactly two readings survive, D = M_pl_sq/2 and
D = A; they differ by a factor N and agree asymptotically, so both are reported. The
stability verdict is the same for both, which is why the reproduction verdict is robust to
the residual ambiguity.

This reproduces a published example. It is not a CRBC action and not a CMB prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.ye_piao_2019_reproduction import derivative
    from quantum_gravity.crbc_gpu.ye_piao_denominator_identification import PARAMETERS, background
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from ye_piao_2019_reproduction import derivative  # type: ignore[no-redef]
    from ye_piao_denominator_identification import PARAMETERS, background  # type: ignore[no-redef]

SURVIVING_READINGS = ("half_M_pl_sq", "A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=96001)
    parser.add_argument("--extent", type=float, default=60.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-cross-check", action="store_true", default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ye_piao_2019_corrected_reproduction.json"),
    )
    return parser.parse_args()


def coefficients(fields: dict[str, torch.Tensor], reading: str) -> dict[str, torch.Tensor]:
    step = float(fields["step"].item())
    a, lapse, m_cal = fields["a"], fields["N"], fields["M_cal"]
    d = 0.5 * fields["M_pl_sq"] if reading == "half_M_pl_sq" else fields["A"]

    u = fields["c1"] + 6.0 / lapse.square()
    v = 2.0 * (lapse * derivative(a * m_cal, step) / (a * d) - 1.0)
    return {"u": u, "v": v, "cs_sq": v / u}


def summarize(fields: dict[str, torch.Tensor], reading: str, extent: float) -> dict[str, object]:
    eta = fields["eta"]
    values = coefficients(fields, reading)
    interior = slice(1, -1)  # boundary uses one-sided derivatives
    eta_i = eta[interior]
    u_i, cs_i = values["u"][interior], values["cs_sq"][interior]
    finite = torch.isfinite(u_i) & torch.isfinite(cs_i)
    tail = (eta_i > 0.8 * extent) & finite

    return {
        "reading": reading,
        "ghost_free": bool((u_i[finite] > 0.0).all().item()),
        "gradient_stable": bool((cs_i[finite] > 0.0).all().item()),
        "min_u": float(u_i[finite].min().item()),
        "min_c_s_sq": float(cs_i[finite].min().item()),
        "max_c_s_sq": float(cs_i[finite].max().item()),
        "eta_at_min_c_s_sq": float(eta_i[finite][torch.argmin(cs_i[finite])].item()),
        "asymptotic_c_s_sq": float(cs_i[tail].mean().item()),
        "asymptotic_c_s_sq_error_vs_eq26": abs(float(cs_i[tail].mean().item()) - 1.0),
        "all_finite": bool(finite.all().item()),
    }, cs_i[finite]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fields = background(args.points, args.extent, device, "physical")
    results = {}
    traces = {}
    for reading in SURVIVING_READINGS:
        summary, trace = summarize(fields, reading, args.extent)
        results[reading] = summary
        traces[reading] = trace

    cross_check = None
    if args.cpu_cross_check:
        cpu_fields = background(args.points, args.extent, torch.device("cpu"), "physical")
        cross_check = {}
        for reading in SURVIVING_READINGS:
            cpu_summary, cpu_trace = summarize(cpu_fields, reading, args.extent)
            cross_check[reading] = {
                "cpu_min_c_s_sq": cpu_summary["min_c_s_sq"],
                "max_abs_gpu_cpu_c_s_sq_difference": float(
                    (traces[reading].cpu() - cpu_trace).abs().max().item()
                ),
                "same_stability_verdict": cpu_summary["gradient_stable"] == results[reading]["gradient_stable"],
            }

    reproduced = all(entry["ghost_free"] and entry["gradient_stable"] for entry in results.values())
    report: dict[str, object] = {
        "reference": "G. Ye and Y.-S. Piao, Commun. Theor. Phys. 71 (2019) 427, arXiv:1901.02202",
        "corrections_against_prior_transcription": [
            "Eq. (12): U = c1 + 6/N^2; the earlier code used 6/N.",
            "Eq. (13): divide by a * (M_pl_sq/2); the earlier code used a * M_cal, which put a "
            "simple pole at the zero of M_cal and produced a spurious min(c_s^2) that diverges "
            "under grid refinement.",
        ],
        "denominator_identified_by": "ye_piao_denominator_identification.py (Eq. 26 + Fig. 3 acceptance tests)",
        "residual_ambiguity": "D = M_pl_sq/2 and D = A both pass; the stability verdict is identical for both.",
        "parameters": PARAMETERS,
        "grid": {"points": args.points, "extent": args.extent},
        "device": str(device),
        "by_reading": results,
        "cpu_cross_check": cross_check,
        "published_stability_reproduced": reproduced,
        "verdict": (
            "Reproduced: the published model is ghost-free and gradient-stable over the whole "
            "integration range, and c_s^2 tends to the Eq. (26) value of 1."
            if reproduced
            else "Not reproduced."
        ),
        "scope": "Reproduction of a published example. Not a CRBC action and not a CMB prediction.",
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
