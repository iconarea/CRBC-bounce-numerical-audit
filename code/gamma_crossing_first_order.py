#!/usr/bin/env python3
"""GPU reproduction of gauge-invariant first-order gamma-crossing evolution.

Implements Eqs. (16a,c), (21), and (25) of R. N. Raveendran, arXiv:1902.06639,
for its Galileon matter-bounce demonstration. This validates the numerical
integrator across gamma=0; it is not a beyond-Horndeski or CRBC prediction.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=12001)
    parser.add_argument("--extent", type=float, default=6.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/gamma_crossing_first_order.json"))
    return parser.parse_args()


def backgrounds(eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a0 = k0 = 1.0
    b0 = 100.0
    a = a0 * (1.0 + k0**2 * eta.square())
    h = 2.0 * k0**2 * eta / a.square()
    gamma = a0 * k0 / (2.0 * a) * (2.0 * k0 * eta - b0 / (a.pow(1.5) * a0**1.5))
    theta = 3.0 * a0**2 * k0**2 / (4.0 * a.pow(4)) * ((3.0 * k0 * eta - b0 / (a.pow(2.5) * a0**1.5)).square() + 8.0 + 31.0 * k0**2 * eta.square())
    return a, h, gamma, theta


def rhs(r: torch.Tensor, sigma: torch.Tensor, k_sq: torch.Tensor, a: torch.Tensor, h: torch.Tensor, gamma: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    difference = r - gamma * sigma
    return -k_sq * gamma / (a * theta) * difference, -a * h * sigma - k_sq / (a * theta) * difference + a * r


def integrate(modes: int, time_steps: int, extent: float, device: torch.device) -> dict[str, object]:
    if time_steps < 5 or time_steps % 2 == 0:
        raise ValueError("time_steps must be odd and at least 5.")
    eta = torch.linspace(-extent, extent, time_steps, dtype=torch.float64, device=device)
    dt = float((eta[1] - eta[0]).item())
    k_sq = torch.logspace(-5, -1, modes, dtype=torch.float64, device=device).square()
    r = torch.ones(modes, dtype=torch.complex128, device=device)
    sigma = torch.zeros_like(r)
    gamma_values = []
    for eta_value in eta[:-1]:
        a, h, gamma, theta = backgrounds(eta_value)
        gamma_values.append(gamma)
        k1r, k1s = rhs(r, sigma, k_sq, a, h, gamma, theta)
        k2r, k2s = rhs(r + 0.5 * dt * k1r, sigma + 0.5 * dt * k1s, k_sq, a, h, gamma, theta)
        k3r, k3s = rhs(r + 0.5 * dt * k2r, sigma + 0.5 * dt * k2s, k_sq, a, h, gamma, theta)
        k4r, k4s = rhs(r + dt * k3r, sigma + dt * k3s, k_sq, a, h, gamma, theta)
        r = r + dt * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
        sigma = sigma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0
    gamma_series = torch.stack(gamma_values)
    return {
        "r": r,
        "max_abs_r": float(r.abs().max().item()),
        "all_finite": bool(torch.isfinite(r.real).all().item() and torch.isfinite(r.imag).all().item()),
        "gamma_crossing_detected": bool((gamma_series[:-1] * gamma_series[1:] <= 0.0).any().item()),
        "min_abs_gamma": float(gamma_series.abs().min().item()),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu = integrate(args.modes, args.time_steps, args.extent, device)
    cpu = integrate(args.modes, args.time_steps, args.extent, torch.device("cpu"))
    report: dict[str, object] = {
        "reference": "R. N. Raveendran, A gauge invariant prescription to avoid gamma-crossing instability in Galileon bounce, arXiv:1902.06639",
        "gpu": {key: value for key, value in gpu.items() if key != "r"},
        "cpu": {key: value for key, value in cpu.items() if key != "r"},
        "max_abs_cpu_gpu_r_difference": float((gpu["r"].cpu() - cpu["r"]).abs().max().item()),
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
