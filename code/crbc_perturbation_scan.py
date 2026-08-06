#!/usr/bin/env python3
"""GPU-batched perturbation sensitivity scan for the CRBC effective bounce.

The bounce potential used here is an explicitly phenomenological surrogate.
It is not derived from a stable covariant action and therefore cannot be used
as evidence for CRBC.  Its purpose is to identify which effective backgrounds
would be worth deriving from a physical action before any CMB data fit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import (
        ScanConfig,
        evaluate_background,
        resolve_device,
        sample_parameters,
        torch_dtype,
    )
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from crbc_background_scan import (  # type: ignore[no-redef]
        ScanConfig,
        evaluate_background,
        resolve_device,
        sample_parameters,
        torch_dtype,
    )


@dataclass(frozen=True)
class PerturbationConfig:
    points: int
    modes: int
    time_steps: int
    time_extent: float
    seed: int
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=1025)
    parser.add_argument("--time-extent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cpu-reference-points", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_perturbation_scan_gpu.json"))
    return parser.parse_args()


def stable_parameters(config: PerturbationConfig, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Draw the declared background proxy parameters and retain stable points."""
    background_config = ScanConfig(config.points, 257, config.time_extent, config.seed, config.dtype)
    params = sample_parameters(config.points, config.seed, device, dtype)
    background = evaluate_background(params, background_config, device, dtype)
    mask = background["valid"]
    if not bool(mask.any().item()):
        raise RuntimeError("No stable proxy points survived the background filter.")
    return {name: value[mask] for name, value in params.items()}


def bounce_potential(params: dict[str, torch.Tensor], time_value: torch.Tensor) -> torch.Tensor:
    """Return the surrogate z''/z potential for sensitivity testing only.

    The amplitude and width are deterministic functions of the declared
    background parameters so they cannot be tuned against a CMB map here.
    """
    alpha = 2.25 * params["rho_c"] * (1.0 + params["w"]).square()
    amplitude = 0.15 * alpha
    width = 1.0 / torch.sqrt(alpha)
    return amplitude[:, None] * torch.exp(-0.5 * (time_value / width[:, None]).square())


def acceleration(v_real: torch.Tensor, v_imag: torch.Tensor, omega_sq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return -omega_sq * v_real, -omega_sq * v_imag


def rk4_step(
    v_real: torch.Tensor,
    v_imag: torch.Tensor,
    p_real: torch.Tensor,
    p_imag: torch.Tensor,
    omega_sq: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One vectorized RK4 step for v'' + omega_sq*v = 0."""
    k1_vr, k1_vi = p_real, p_imag
    k1_pr, k1_pi = acceleration(v_real, v_imag, omega_sq)
    k2_vr, k2_vi = p_real + 0.5 * dt * k1_pr, p_imag + 0.5 * dt * k1_pi
    k2_pr, k2_pi = acceleration(v_real + 0.5 * dt * k1_vr, v_imag + 0.5 * dt * k1_vi, omega_sq)
    k3_vr, k3_vi = p_real + 0.5 * dt * k2_pr, p_imag + 0.5 * dt * k2_pi
    k3_pr, k3_pi = acceleration(v_real + 0.5 * dt * k2_vr, v_imag + 0.5 * dt * k2_vi, omega_sq)
    k4_vr, k4_vi = p_real + dt * k3_pr, p_imag + dt * k3_pi
    k4_pr, k4_pi = acceleration(v_real + dt * k3_vr, v_imag + dt * k3_vi, omega_sq)
    factor = dt / 6.0
    return (
        v_real + factor * (k1_vr + 2.0 * k2_vr + 2.0 * k3_vr + k4_vr),
        v_imag + factor * (k1_vi + 2.0 * k2_vi + 2.0 * k3_vi + k4_vi),
        p_real + factor * (k1_pr + 2.0 * k2_pr + 2.0 * k3_pr + k4_pr),
        p_imag + factor * (k1_pi + 2.0 * k2_pi + 2.0 * k3_pi + k4_pi),
    )


def integrate_transfer(params: dict[str, torch.Tensor], config: PerturbationConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a mode batch and return k plus a dimensionless transfer proxy."""
    k = torch.logspace(-3, 0, config.modes, device=device, dtype=dtype)
    time_grid = torch.linspace(-config.time_extent, config.time_extent, config.time_steps, device=device, dtype=dtype)
    dt = float((time_grid[1] - time_grid[0]).item())
    c_s = torch.sqrt(params["c_s_sq"])[:, None]
    omega_initial = c_s * k[None, :]
    v_real = 1.0 / torch.sqrt(2.0 * omega_initial)
    v_imag = torch.zeros_like(v_real)
    p_real = torch.zeros_like(v_real)
    p_imag = -omega_initial * v_real

    for time_value in time_grid[:-1]:
        potential = bounce_potential(params, time_value)
        omega_sq = params["c_s_sq"][:, None] * k.square()[None, :] - potential
        v_real, v_imag, p_real, p_imag = rk4_step(v_real, v_imag, p_real, p_imag, omega_sq, dt)

    transfer = 2.0 * omega_initial * (v_real.square() + v_imag.square())
    return k, transfer


def run(config: PerturbationConfig, device: torch.device, params: dict[str, torch.Tensor] | None = None) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    dtype = torch_dtype(config.dtype)
    if params is None:
        params = stable_parameters(config, device, dtype)
    started = time.perf_counter()
    k, transfer = integrate_transfer(params, config, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    low_k_change = (transfer[:, 0] - 1.0).abs()
    summary: dict[str, object] = {
        "device": str(device),
        "dtype": config.dtype,
        "stable_proxy_points": int(transfer.shape[0]),
        "modes": config.modes,
        "time_steps": config.time_steps,
        "elapsed_seconds": elapsed,
        "max_abs_transfer_change": float((transfer - 1.0).abs().max().item()),
        "median_low_k_abs_transfer_change": float(low_k_change.median().item()),
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return summary, k, transfer


def agreement(config: PerturbationConfig, points: int, device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"performed": False, "reason": "CUDA unavailable"}
    reference = PerturbationConfig(points, config.modes, config.time_steps, config.time_extent, config.seed, "float64")
    cpu_params = stable_parameters(reference, torch.device("cpu"), torch.float64)
    gpu_params = {name: value.to(device) for name, value in cpu_params.items()}
    cpu_summary, k_cpu, transfer_cpu = run(reference, torch.device("cpu"), cpu_params)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_summary, k_gpu, transfer_gpu = run(reference, device, gpu_params)
    k_difference = (k_cpu - k_gpu.cpu()).abs()
    return {
        "performed": True,
        "cpu_reference": cpu_summary,
        "gpu_reference": gpu_summary,
        "k_grid_max_abs_difference": float(k_difference.max().item()),
        "same_k_grid_within_tolerance": bool(torch.allclose(k_cpu, k_gpu.cpu(), rtol=1.0e-12, atol=1.0e-14)),
        "max_abs_transfer_difference": float((transfer_cpu - transfer_gpu.cpu()).abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.points < 1 or args.modes < 2 or args.time_steps < 3 or args.time_extent <= 0:
        raise ValueError("points >= 1, modes >= 2, time_steps >= 3, and time_extent > 0 are required.")
    device = resolve_device(args.device)
    config = PerturbationConfig(args.points, args.modes, args.time_steps, args.time_extent, args.seed, args.dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    summary, _, _ = run(config, device)
    report = {"config": asdict(config), "scan": summary, "cpu_gpu_agreement": agreement(config, args.cpu_reference_points, device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
