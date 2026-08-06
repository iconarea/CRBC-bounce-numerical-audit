#!/usr/bin/env python3
"""Does the CRBC bounce dynamically produce the coherent-relic envelope f(k; k_star, sigma_k)?

The research spec (결맞은_잔재_바운스_우주론_연구명세_kr.md §3) writes the primordial curvature
perturbation as a stochastic part plus a deterministic coherent relic,

    R(k) = R_G(k) + A_R f(k; k_star, sigma_k) Y_LM(k_hat; n_hat, psi),

and §4 requires that L, k_star, sigma_k and n_hat be *derived* before any CMB fit. This
script tests the part of that requirement which is decidable by the background dynamics.

The observation that makes it decidable: a coherent relic is a classical field
configuration, so it obeys the same linear mode equation as the fluctuations. Its envelope
at read-out is therefore

    A_R f(k) = [ initial classical profile ](k) x T(k),

with T(k) the transfer function of the gated CRBC background. T(k) is computable; the
initial profile is not part of the background dynamics. So the question "is f(k) predicted
or assumed?" reduces to "does T(k) itself carry a band-limited feature?".

  * If T(k) has a peak at some dynamically generated k_star, the spec's log-normal envelope
    is a prediction and k_star, sigma_k are fixed by the bounce.
  * If T(k) is featureless or monotonic (for example a low-pass filter set by the bounce
    duration), then k_star and sigma_k are properties of the initial data, not of CRBC, and
    the spec's §4 requirement cannot be met by the background alone.

Initial data are classical and real, taken deep in the contracting phase. Two independent
choices are run — (delta_s, delta_s') = (1, 0) and (0, 1) — and the shape of T(k) is
reported for both, because a conclusion that depended on that choice would not be a
statement about the bounce.

T(k) is normalized to its small-k plateau, where modes are frozen and pass through the
bounce unprocessed, so the reported curve isolates what the bounce does.

This script does not address the angular factor Y_LM. An isotropic FLRW background cannot
generate a preferred direction; see the accompanying documentation for that separate point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
    from quantum_gravity.crbc_gpu.crbc_primordial_spectrum import build_background, post_bounce_horizon_peak
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]
    from crbc_primordial_spectrum import build_background, post_bounce_horizon_peak  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-initial", type=float, default=8.0)
    parser.add_argument("--p-final", type=float, default=2.4)
    parser.add_argument("--eta-p", type=float, default=0.7)
    parser.add_argument("--tau-p", type=float, default=1.0)
    parser.add_argument("--p-mid", type=float, default=None,
                        help="two-stage background from the three-level hierarchy")
    parser.add_argument("--eta-p2", type=float, default=-6.0)
    parser.add_argument("--tau-p2", type=float, default=4.0)
    parser.add_argument("--k1", type=float, default=12.0)
    parser.add_argument("--k2", type=float, default=5.0)
    parser.add_argument("--tau1", type=float, default=1.7)
    parser.add_argument("--tau2-sq", type=float, default=0.59)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument(
        "--entropic-lambda",
        type=float,
        default=104.0,
        help="tachyonic entropy mass coefficient m_eff^2 = -lambda H^2 from the tuned mechanism",
    )

    parser.add_argument("--extent", type=float, default=200.0)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--modes", type=int, default=160)
    parser.add_argument("--k-min", type=float, default=1e-3)
    parser.add_argument("--k-max", type=float, default=3e1)
    parser.add_argument("--plateau-decades", type=float, default=1.0,
                        help="k range at the low end used to define the frozen plateau")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_coherent_relic_transfer.json"))
    return parser.parse_args()


def propagate_classical(
    fields: dict[str, torch.Tensor], k: torch.Tensor, steps: int, lam: float, initial: tuple[float, float]
) -> torch.Tensor:
    """Evolve real classical data for u = a delta_s and return delta_s at the final time."""
    step = float(fields["step"].item()) * 2.0
    fine_step = float(fields["step"].item())
    scale, scale_dot, hubble = fields["a"], fields["a_dot"], fields["H"]

    scale_ddot = torch.empty_like(scale)
    scale_ddot[1:-1] = (scale[2:] - 2.0 * scale[1:-1] + scale[:-2]) / fine_step**2
    scale_ddot[0], scale_ddot[-1] = scale_ddot[1], scale_ddot[-2]
    a_second_conformal = scale_dot.square() + scale * scale_ddot
    comoving_hubble_sq = (scale * hubble).square()

    friction = scale_dot / scale
    inverse_scale_sq = 1.0 / scale.square()

    def acceleration(index: int, u: torch.Tensor, u_dot: torch.Tensor) -> torch.Tensor:
        potential = k.square() - lam * comoving_hubble_sq[index] - a_second_conformal[index]
        return -friction[index] * u_dot - potential * inverse_scale_sq[index] * u

    entropy_initial, entropy_rate_initial = initial
    u = torch.full_like(k, entropy_initial) * scale[0]
    u_dot = torch.full_like(k, entropy_rate_initial) * scale[0] + torch.full_like(k, entropy_initial) * scale_dot[0]

    for i in range(steps):
        left, middle, right = 2 * i, 2 * i + 1, 2 * i + 2
        k1u, k1a = u_dot, acceleration(left, u, u_dot)
        k2u, k2a = u_dot + 0.5 * step * k1a, acceleration(middle, u + 0.5 * step * k1u, u_dot + 0.5 * step * k1a)
        k3u, k3a = u_dot + 0.5 * step * k2a, acceleration(middle, u + 0.5 * step * k2u, u_dot + 0.5 * step * k2a)
        k4u, k4a = u_dot + step * k3a, acceleration(right, u + step * k3u, u_dot + step * k3a)
        u = u + step * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        u_dot = u_dot + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0

    return u / scale[-1]


def characterize(k: torch.Tensor, transfer: torch.Tensor, plateau_decades: float) -> dict[str, object]:
    """Is the normalized transfer curve a band-limited bump, or a monotonic filter?"""
    log_k = torch.log10(k)
    plateau = log_k < (log_k[0] + plateau_decades)
    normalization = transfer[plateau].abs().mean()
    normalized = transfer.abs() / normalization

    peak_index = int(torch.argmax(normalized).item())
    peak_value = float(normalized[peak_index].item())
    # A band-limited relic requires a peak that stands above the frozen plateau and above
    # the high-k end; a low-pass filter has its maximum at the low-k plateau instead.
    interior_peak = 0 < peak_index < k.numel() - 1
    enhancement_over_plateau = peak_value  # plateau is normalized to 1 by construction

    half = normalized[peak_index] / 2.0
    below = normalized[:peak_index] < half
    above = normalized[peak_index:] < half
    left_edge = float(k[: peak_index][below].max().item()) if bool(below.any().item()) else None
    right_edge = float(k[peak_index:][above].min().item()) if bool(above.any().item()) else None
    width_decades = (
        float(torch.log10(torch.tensor(right_edge / left_edge)).item())
        if left_edge is not None and right_edge is not None
        else None
    )

    # Effective cutoff: where the normalized transfer first drops below 1/e.
    cutoff_mask = normalized < (1.0 / torch.e)
    cutoff = float(k[cutoff_mask].min().item()) if bool(cutoff_mask.any().item()) else None

    return {
        "plateau_normalization": float(normalization.item()),
        "peak_k": float(k[peak_index].item()),
        "peak_over_plateau": enhancement_over_plateau,
        "peak_is_interior": interior_peak,
        "band_limited": bool(interior_peak and enhancement_over_plateau > 2.0 and width_decades is not None),
        "half_max_k_range": [left_edge, right_edge],
        "width_decades": width_decades,
        "cutoff_k_at_one_over_e": cutoff,
        "normalized_transfer_at_high_k": float(normalized[-1].item()),
        "monotonic_low_pass": bool(enhancement_over_plateau < 2.0 and cutoff is not None),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_end = post_bounce_horizon_peak(args, device)
    fields = build_background(args, 2 * args.steps + 1, device, t_end)
    k = torch.logspace(
        float(torch.log10(torch.tensor(args.k_min))),
        float(torch.log10(torch.tensor(args.k_max))),
        args.modes,
        dtype=torch.float64,
        device=device,
    )

    results = {}
    for name, initial in (("static", (1.0, 0.0)), ("rate", (0.0, 1.0))):
        transfer = propagate_classical(fields, k, args.steps, args.entropic_lambda, initial)
        results[name] = {
            "initial_data": {"delta_s": initial[0], "delta_s_dot": initial[1]},
            "shape": characterize(k, transfer, args.plateau_decades),
            "transfer_abs": transfer.abs().cpu().tolist(),
        }

    coarse = propagate_classical(fields, k, args.steps, args.entropic_lambda, (1.0, 0.0))
    half_fields = build_background(args, args.steps + 1, device, t_end)
    coarse_half = propagate_classical(half_fields, k, args.steps // 2, args.entropic_lambda, (1.0, 0.0))
    settled = k < 1.0  # convergence is judged where the mode is not violently oscillating
    convergence = float(
        ((coarse.abs()[settled] - coarse_half.abs()[settled]) / coarse.abs()[settled]).abs().max().item()
    )

    # Horizon scales, so the measured cutoff can be attributed rather than guessed.
    comoving_hubble = (fields["a"] * fields["H"]).abs()
    contracting = fields["time"] < 0.0
    horizon_scales = {
        "aH_at_t_eval": float(comoving_hubble[-1].item()),
        "sqrt_lambda_times_aH_at_t_eval": float((args.entropic_lambda**0.5 * comoving_hubble[-1]).item()),
        "max_aH_during_contraction": float(comoving_hubble[contracting].max().item()),
        "sqrt_lambda_times_max_aH_during_contraction": float(
            (args.entropic_lambda**0.5 * comoving_hubble[contracting].max()).item()
        ),
    }

    # Bounce filter scale: conformal duration of the bounce region sets the comoving cutoff.
    time = fields["time"]
    near_bounce = time.abs() < 1.0
    conformal_duration = float((1.0 / fields["a"][near_bounce]).sum().item() * float(fields["step"].item()))
    bounce_scale = 2.0 * torch.pi / conformal_duration if conformal_duration > 0.0 else float("inf")

    band_limited = any(results[name]["shape"]["band_limited"] for name in results)
    report: dict[str, object] = {
        "question": "does the gated CRBC background dynamically generate the coherent-relic "
        "envelope f(k; k_star, sigma_k) of 연구명세 §3?",
        "method": "a coherent relic is classical, so it obeys the same linear mode equation; "
        "A_R f(k) = [initial classical profile] x T(k), and only T(k) is background dynamics",
        "background": "CRBC_EFT_선정과_계수계약_kr.md §7.3 realization with the §9 entropic mass",
        "entropic_lambda": args.entropic_lambda,
        "background_stages": 2 if args.p_mid is not None else 1,
        "p_profile": {"p_initial": args.p_initial, "p_mid": args.p_mid, "p_final": args.p_final,
                      "eta_p": args.eta_p, "tau_p": args.tau_p,
                      "eta_p2": args.eta_p2, "tau_p2": args.tau_p2},
        "device": str(device),
        "grid": {"extent": args.extent, "rk4_steps": args.steps, "modes": args.modes, "t_eval": t_end},
        "k": k.cpu().tolist(),
        "initial_data_choices": {name: results[name]["shape"] for name in results},
        "transfer_curves": {name: results[name]["transfer_abs"] for name in results},
        "horizon_scales": horizon_scales,
        "bounce_filter_scale": {
            "conformal_duration_of_bounce_region": conformal_duration,
            "comoving_k_of_bounce": bounce_scale,
        },
        "convergence": {"steps_compared": [args.steps // 2, args.steps], "max_relative_difference_k_below_1": convergence},
        "verdict": {
            "band_limited_feature_found": band_limited,
            "conclusion": (
                "T(k) carries a band-limited feature, so k_star and sigma_k are dynamically generated."
                if band_limited
                else "T(k) carries no band-limited feature. The bounce acts as a filter with a single "
                "scale, not as a band-pass. k_star and sigma_k are therefore properties of the "
                "assumed initial classical profile, not predictions of CRBC, and 연구명세 §4's "
                "requirement to derive them cannot be met by the background dynamics alone."
            ),
        },
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in ("k", "transfer_curves")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
