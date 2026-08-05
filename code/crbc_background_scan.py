#!/usr/bin/env python3
"""GPU-batched phenomenological prefilter for CRBC bounce backgrounds.

This program does not fit CMB data and does not establish a bounce theory.  It
screens a declared grid of effective background parameters, rejects candidates
that fail the explicitly stated EFT/stability proxies, and compares a GPU batch
calculation against the same CPU calculation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ScanConfig:
    points: int
    time_steps: int
    time_extent: float
    seed: int
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=16384)
    parser.add_argument("--time-steps", type=int, default=2049)
    parser.add_argument("--time-extent", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cpu-reference-points", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_background_scan.json"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def sample_parameters(points: int, seed: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Sample declared phenomenological parameters before any CMB calculation."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    uniform = lambda low, high: low + (high - low) * torch.rand(points, generator=generator, device=device, dtype=dtype)
    return {
        "rho_c": uniform(0.2, 1.0),
        "w": uniform(-0.25, 1.0),
        "q_s": uniform(-0.25, 1.5),
        "c_s_sq": uniform(-0.25, 1.5),
        "cutoff_ratio": uniform(0.25, 1.5),
    }


def evaluate_background(params: dict[str, torch.Tensor], config: ScanConfig, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Evaluate the closed effective bounce solution in dimensionless units.

    rho(t)=rho_c/(1+alpha*t^2), alpha=(9/4)*rho_c*(1+w)^2.  This normalization sets
    8*pi*G/3=1 and is used only for a reproducible screening baseline.
    """
    time_grid = torch.linspace(-config.time_extent, config.time_extent, config.time_steps, device=device, dtype=dtype)
    alpha = 2.25 * params["rho_c"] * (1.0 + params["w"]).square()
    denominator = 1.0 + alpha[:, None] * time_grid.square()[None, :]
    density = params["rho_c"][:, None] / denominator
    expansion = 2.0 * alpha[:, None] * time_grid[None, :] / (3.0 * (1.0 + params["w"])[:, None] * denominator)

    friedmann_rhs = density * (1.0 - density / params["rho_c"][:, None])
    friedmann_residual = (expansion.square() - friedmann_rhs).abs().amax(dim=1)
    maximum_density_fraction = (density / params["rho_c"][:, None]).amax(dim=1)
    stable = (params["q_s"] > 0.0) & (params["c_s_sq"] > 0.0) & (params["cutoff_ratio"] < 1.0)
    valid = stable & torch.isfinite(friedmann_residual) & (friedmann_residual < 2.0e-5)
    return {
        "friedmann_residual": friedmann_residual,
        "maximum_density_fraction": maximum_density_fraction,
        "valid": valid,
    }


def run_scan(config: ScanConfig, device: torch.device) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    dtype = torch_dtype(config.dtype)
    started = time.perf_counter()
    params = sample_parameters(config.points, config.seed, device, dtype)
    result = evaluate_background(params, config, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    valid_count = int(result["valid"].sum().item())
    summary: dict[str, object] = {
        "device": str(device),
        "dtype": config.dtype,
        "points": config.points,
        "time_steps": config.time_steps,
        "elapsed_seconds": elapsed,
        "valid_count": valid_count,
        "valid_fraction": valid_count / config.points,
        "max_friedmann_residual": float(result["friedmann_residual"].max().item()),
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return summary, result


def cpu_gpu_agreement(config: ScanConfig, cpu_reference_points: int, device: torch.device) -> dict[str, object]:
    reference_config = ScanConfig(
        points=min(config.points, cpu_reference_points),
        time_steps=config.time_steps,
        time_extent=config.time_extent,
        seed=config.seed,
        dtype="float64",
    )
    dtype = torch_dtype(reference_config.dtype)
    cpu_device = torch.device("cpu")
    cpu_params = sample_parameters(reference_config.points, reference_config.seed, cpu_device, dtype)
    cpu_started = time.perf_counter()
    cpu_result = evaluate_background(cpu_params, reference_config, cpu_device, dtype)
    cpu_elapsed = time.perf_counter() - cpu_started
    cpu_summary = {
        "device": "cpu",
        "dtype": reference_config.dtype,
        "points": reference_config.points,
        "time_steps": reference_config.time_steps,
        "elapsed_seconds": cpu_elapsed,
        "valid_count": int(cpu_result["valid"].sum().item()),
        "max_friedmann_residual": float(cpu_result["friedmann_residual"].max().item()),
    }
    if device.type != "cuda":
        return {"performed": False, "reason": "CUDA unavailable", "cpu_reference": cpu_summary}
    gpu_params = {name: value.to(device) for name, value in cpu_params.items()}
    torch.cuda.reset_peak_memory_stats(device)
    gpu_started = time.perf_counter()
    gpu_result = evaluate_background(gpu_params, reference_config, device, dtype)
    torch.cuda.synchronize(device)
    gpu_elapsed = time.perf_counter() - gpu_started
    gpu_summary = {
        "device": str(device),
        "dtype": reference_config.dtype,
        "points": reference_config.points,
        "time_steps": reference_config.time_steps,
        "elapsed_seconds": gpu_elapsed,
        "valid_count": int(gpu_result["valid"].sum().item()),
        "max_friedmann_residual": float(gpu_result["friedmann_residual"].max().item()),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    difference = (cpu_result["friedmann_residual"].cpu() - gpu_result["friedmann_residual"].cpu()).abs()
    return {
        "performed": True,
        "cpu_reference": cpu_summary,
        "gpu_reference": gpu_summary,
        "max_abs_residual_difference": float(difference.max().item()),
        "same_validity_mask": bool(torch.equal(cpu_result["valid"], gpu_result["valid"].cpu())),
    }


def main() -> None:
    args = parse_args()
    if args.points < 1 or args.time_steps < 3 or args.time_extent <= 0:
        raise ValueError("points >= 1, time_steps >= 3, and time_extent > 0 are required.")
    device = resolve_device(args.device)
    config = ScanConfig(args.points, args.time_steps, args.time_extent, args.seed, args.dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    summary, _ = run_scan(config, device)
    agreement = cpu_gpu_agreement(config, args.cpu_reference_points, device)
    report = {"config": asdict(config), "scan": summary, "cpu_gpu_agreement": agreement}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
