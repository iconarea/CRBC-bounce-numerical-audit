#!/usr/bin/env python3
"""GPU consistency gate for a minimal P(X, phi)=KX+LX^2-V bounce candidate.

For a single minimally coupled k-essence field in flat GR, this code checks the
simultaneous requirements of NEC violation, a positive scalar kinetic
coefficient, and positive sound-speed squared.  The expected result is no
surviving point: the inequalities are mutually inconsistent in this minimal
model.  That failure is a useful result, not a numerical error.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device, torch_dtype
except ModuleNotFoundError:
    from crbc_background_scan import resolve_device, torch_dtype  # type: ignore[no-redef]


@dataclass(frozen=True)
class ScanConfig:
    points: int
    seed: int
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cpu-reference-points", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_kessence_no_go_scan.json"))
    return parser.parse_args()


def sample(config: ScanConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    k = -2.0 + 4.0 * torch.rand(config.points, generator=generator, device=device, dtype=dtype)
    l = -2.0 + 4.0 * torch.rand(config.points, generator=generator, device=device, dtype=dtype)
    x = 1.0e-4 + 2.0 * torch.rand(config.points, generator=generator, device=device, dtype=dtype)
    return k, l, x


def classify(k: torch.Tensor, l: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    p_x = k + 2.0 * l * x
    rho_x = k + 6.0 * l * x
    sound_speed_sq = p_x / rho_x
    nec_violation = p_x < 0.0
    no_ghost = rho_x > 0.0
    no_gradient_instability = sound_speed_sq > 0.0
    viable = nec_violation & no_ghost & no_gradient_instability
    return {
        "nec_violation": nec_violation,
        "no_ghost": no_ghost,
        "no_gradient_instability": no_gradient_instability,
        "viable": viable,
        "sound_speed_sq": sound_speed_sq,
    }


def run(config: ScanConfig, device: torch.device, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    dtype = torch_dtype(config.dtype)
    if inputs is None:
        inputs = sample(config, device, dtype)
    started = time.perf_counter()
    result = classify(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "device": str(device),
        "dtype": config.dtype,
        "points": config.points,
        "elapsed_seconds": elapsed,
        "nec_violation_count": int(result["nec_violation"].sum().item()),
        "no_ghost_count": int(result["no_ghost"].sum().item()),
        "no_gradient_instability_count": int(result["no_gradient_instability"].sum().item()),
        "viable_count": int(result["viable"].sum().item()),
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return summary, inputs, result


def agreement(config: ScanConfig, reference_points: int, device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"performed": False, "reason": "CUDA unavailable"}
    reference = ScanConfig(min(config.points, reference_points), config.seed, "float64")
    cpu_summary, cpu_inputs, cpu_result = run(reference, torch.device("cpu"))
    gpu_inputs = tuple(value.to(device) for value in cpu_inputs)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_summary, _, gpu_result = run(reference, device, gpu_inputs)
    return {
        "performed": True,
        "cpu_reference": cpu_summary,
        "gpu_reference": gpu_summary,
        "same_viability_mask": bool(torch.equal(cpu_result["viable"], gpu_result["viable"].cpu())),
        "max_abs_sound_speed_difference": float((cpu_result["sound_speed_sq"] - gpu_result["sound_speed_sq"].cpu()).abs().nan_to_num().max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.points < 1:
        raise ValueError("points must be positive")
    device = resolve_device(args.device)
    config = ScanConfig(args.points, args.seed, args.dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    summary, _, _ = run(config, device)
    report = {
        "config": asdict(config),
        "model": "P(X,phi)=KX+LX^2-V, minimally coupled single field in flat GR",
        "interpretation": "viable_count must be zero; otherwise inspect implementation assumptions",
        "scan": summary,
        "cpu_gpu_agreement": agreement(config, args.cpu_reference_points, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
