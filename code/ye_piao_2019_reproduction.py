#!/usr/bin/env python3
"""SUPERSEDED first transcription of the Ye & Piao (2019) bounce. Do not cite its output.

Kept only so the correction is auditable. Use ye_piao_2019_corrected_reproduction.py.

Two errors against arXiv:1901.02202:
  * Eq. (12) is U = Sigma/gamma^2 + 6/N^2; this file uses 6/N.
  * Eq. (13) divides by a * (M_pl_sq/2); this file divides by a * M_cal, which places a
    simple pole at the zero of M_cal.

The second error is what produced this file's reported min(c_s^2) = -119.7. That number is
not a physical sound speed: ye_piao_cs_sq_pole_diagnostic.py shows it diverges from -53 to
-2250 as the grid is refined from 3001 to 96001 points, with a constant pole residue. With
Eqs. (12) and (13) transcribed correctly the published model is ghost-free and
gradient-stable everywhere, with c_s^2 -> 1 as required by the paper's Eq. (26).

The helper functions derivative() and cumulative_trapezoid() below are correct and are
imported by the replacement scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
except ModuleNotFoundError:
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]


def derivative(values: torch.Tensor, step: float) -> torch.Tensor:
    result = torch.empty_like(values)
    result[0] = (values[1] - values[0]) / step
    result[-1] = (values[-1] - values[-2]) / step
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * step)
    return result


def cumulative_trapezoid(values: torch.Tensor, step: float) -> torch.Tensor:
    integral = torch.zeros_like(values)
    integral[1:] = torch.cumsum(0.5 * (values[1:] + values[:-1]) * step, dim=0)
    return integral


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=12001)
    parser.add_argument("--extent", type=float, default=15.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/ye_piao_2019_beyond_horndeski_reproduction.json"))
    return parser.parse_args()


def evaluate(points: int, extent: float, device: torch.device) -> dict[str, torch.Tensor]:
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

    # Eq. (16), with the removable eta=0 singularity interpolated from neighbours.
    numerator = 1.0 - 4.0 * f1.square() * x.pow(4)
    denominator = 2.0 * conformal_h * (1.0 + 6.0 * f1 * x.square())
    m = numerator / denominator
    middle = points // 2
    m[middle] = 0.5 * (m[middle - 1] + m[middle + 1])

    # Eqs. (12), (13), with the construction Sigma=c1*gamma^2.
    u = c1 + 6.0 / lapse
    a_m = scale_factor * m
    v = 2.0 * (lapse * derivative(a_m, step) / a_m - 1.0)
    cs_sq = v / u
    return {"eta": eta, "u": u, "v": v, "cs_sq": cs_sq, "m": m}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    result = evaluate(args.points, args.extent, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    finite = torch.isfinite(result["u"]) & torch.isfinite(result["v"]) & torch.isfinite(result["cs_sq"])
    u, cs_sq, eta = result["u"][finite], result["cs_sq"][finite], result["eta"][finite]
    m = result["m"]
    crossing_indices = torch.where(m[:-1] * m[1:] <= 0.0)[0]
    crossing_detected = bool(crossing_indices.numel() > 0)
    report: dict[str, object] = {
        "reference": "Ye & Piao, Implication of GW170817 for Cosmological Bounces, Commun. Theor. Phys. 71 (2019) 427-434",
        "parameters": {"p_i": 8, "p_f": 3, "tau_p": 1, "eta_p": 0.7, "eta_x": -3, "tau_x": 3, "k1": 0.06, "k2": 2, "tau1": 2, "tau2_sq": 0.6},
        "summary": {
            "device": str(device), "finite_points": int(finite.sum().item()), "total_points": int(finite.numel()),
            "min_u": float(u.min().item()), "min_c_s_sq": float(cs_sq.min().item()), "max_c_s_sq": float(cs_sq.max().item()),
            "direct_unitary_gauge_scalar_stable_on_grid": bool((u > 0.0).all().item() and (cs_sq > 0.0).all().item()),
            "c_t_sq": 1.0, "eta_at_min_c_s_sq": float(eta[torch.argmin(cs_sq)].item()),
            "gamma_crossing_proxy_detected": crossing_detected,
            "min_abs_m": float(m.abs().min().item()),
            "requires_gauge_regular_first_order_evolution": crossing_detected,
        },
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
