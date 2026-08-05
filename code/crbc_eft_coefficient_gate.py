#!/usr/bin/env python3
"""GPU stability gate for coefficient trajectories derived from a CRBC EFT.

Input is an .npz file with one-dimensional arrays: time, a, H, q_s, c_s_sq,
q_t, c_t_sq, cutoff, characteristic_energy.  The script validates only the
provided trajectory; it does not derive those coefficients from an action.
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


REQUIRED = ("time", "a", "H", "q_s", "c_s_sq", "q_t", "c_t_sq", "cutoff", "characteristic_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="EFT coefficient .npz file")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_eft_coefficient_gate.json"))
    return parser.parse_args()


def load_coefficients(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [name for name in REQUIRED if name not in data]
        if missing:
            raise ValueError(f"Missing required arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(data[name], dtype=np.float64) for name in REQUIRED}
    lengths = {name: array.size for name, array in arrays.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) < 3:
        raise ValueError("All arrays must be one-dimensional and have the same length >= 3.")
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError("All coefficient arrays must be one-dimensional.")
    if not np.all(np.diff(arrays["time"]) > 0):
        raise ValueError("time must be strictly increasing.")
    return arrays


def evaluate(arrays: dict[str, torch.Tensor]) -> dict[str, object]:
    ratio = arrays["characteristic_energy"] / arrays["cutoff"]
    conditions = {
        "positive_scale_factor": arrays["a"] > 0.0,
        "scalar_no_ghost": arrays["q_s"] > 0.0,
        "scalar_no_gradient_instability": arrays["c_s_sq"] > 0.0,
        "tensor_no_ghost": arrays["q_t"] > 0.0,
        "tensor_no_gradient_instability": arrays["c_t_sq"] > 0.0,
        "eft_control": ratio < 0.1,
    }
    valid = torch.ones_like(arrays["time"], dtype=torch.bool)
    report: dict[str, object] = {}
    for name, condition in conditions.items():
        valid &= condition
        report[f"{name}_violations"] = int((~condition).sum().item())
    report.update({
        "points": int(arrays["time"].numel()),
        "all_points_valid": bool(valid.all().item()),
        "first_invalid_index": None if bool(valid.all().item()) else int(torch.where(~valid)[0][0].item()),
        "min_q_s": float(arrays["q_s"].min().item()),
        "min_c_s_sq": float(arrays["c_s_sq"].min().item()),
        "min_q_t": float(arrays["q_t"].min().item()),
        "min_c_t_sq": float(arrays["c_t_sq"].min().item()),
        "max_energy_cutoff_ratio": float(ratio.max().item()),
    })
    return report


def main() -> None:
    args = parse_args()
    coefficients = load_coefficients(args.input)
    device = resolve_device(args.device)
    cpu_arrays = {name: torch.from_numpy(value) for name, value in coefficients.items()}
    gpu_arrays = {name: value.to(device) for name, value in cpu_arrays.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report = {
        "input": str(args.input),
        "device": str(device),
        "gpu": evaluate(gpu_arrays),
        "cpu_reference": evaluate(cpu_arrays),
    }
    report["cpu_gpu_same_verdict"] = report["gpu"] == report["cpu_reference"]
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
