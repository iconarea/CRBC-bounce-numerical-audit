#!/usr/bin/env python3
"""Integrate gauge-invariant (R, Sigma) modes from an EFT profile .npz.

The profile must contain one-dimensional conformal-time arrays `eta`, `a`,
`H`, `gamma`, and `theta`. It is deliberately agnostic about the action: the
upstream EFT derivation owns those coefficients, while this GPU stage owns the
crossing-safe first-order evolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
except ModuleNotFoundError:
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/gamma_crossing_profile.json"))
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, np.ndarray]:
    required = ("eta", "a", "H", "gamma", "theta")
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Missing profile arrays: {', '.join(missing)}")
        profile = {name: np.asarray(data[name], dtype=np.float64) for name in required}
    size = profile["eta"].size
    if any(value.ndim != 1 or value.size != size for value in profile.values()) or size < 5:
        raise ValueError("All profile arrays must be one-dimensional with equal length >= 5.")
    if not np.all(np.diff(profile["eta"]) > 0):
        raise ValueError("eta must be strictly increasing.")
    return profile


def integrate(profile: dict[str, torch.Tensor], modes: int, device: torch.device) -> dict[str, object]:
    eta, a, h, gamma, theta = (profile[name].to(device) for name in ("eta", "a", "H", "gamma", "theta"))
    if torch.any(a <= 0) or torch.any(theta <= 0):
        raise ValueError("Profile requires a>0 and theta>0 everywhere.")
    dt = eta[1:] - eta[:-1]
    if not torch.allclose(dt, dt[0].expand_as(dt), rtol=1e-8, atol=1e-12):
        raise ValueError("The current integrator requires a uniform eta grid.")
    step = float(dt[0].item())
    k_sq = torch.logspace(-5, -1, modes, dtype=torch.float64, device=device).square()
    r = torch.ones(modes, dtype=torch.complex128, device=device)
    sigma = torch.zeros_like(r)

    def rhs(r_now: torch.Tensor, s_now: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        difference = r_now - gamma[index] * s_now
        return -k_sq * gamma[index] / (a[index] * theta[index]) * difference, -a[index] * h[index] * s_now - k_sq / (a[index] * theta[index]) * difference + a[index] * r_now

    for index in range(eta.numel() - 1):
        k1r, k1s = rhs(r, sigma, index)
        k2r, k2s = rhs(r + 0.5 * step * k1r, sigma + 0.5 * step * k1s, index)
        k3r, k3s = rhs(r + 0.5 * step * k2r, sigma + 0.5 * step * k2s, index)
        k4r, k4s = rhs(r + step * k3r, sigma + step * k3s, index)
        r += step * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
        sigma += step * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0
    crossing = bool((gamma[:-1] * gamma[1:] <= 0).any().item())
    return {"max_abs_r": float(r.abs().max().item()), "all_finite": bool(torch.isfinite(r.real).all().item() and torch.isfinite(r.imag).all().item()), "gamma_crossing_detected": crossing, "min_abs_gamma": float(gamma.abs().min().item())}


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    device = resolve_device(args.device)
    tensors = {name: torch.from_numpy(value) for name, value in profile.items()}
    result = integrate(tensors, args.modes, device)
    report = {"profile": str(args.profile), "device": str(device), "modes": args.modes, "result": result}
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
