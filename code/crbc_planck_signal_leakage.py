#!/usr/bin/env python3
"""Where does the masked sky put the quadrupole signal? Diagnostic before building a filter.

The pseudo-alm estimator recovers only 19% of the full-sky response
(CRBC_Planck_사중극_사전등록_2026_kr.md §10), and deconvolution does not fix it (§11). The
right repair depends on where the signal actually goes, which has two candidate answers:

  (a) it stays in the l, l+2 pairs but with an l-dependent efficiency, in which case a
      per-l reweighting recovers it;
  (b) it leaks into l, l+delta pairs with delta = 4, 6, ..., in which case the estimator
      must be widened in delta before any reweighting can help.

This script measures both. For each even separation delta it forms

    X_l,delta = sum_m u_l(m) Re( a_lm a*_{l+delta,m} ),

with u_l(m) the same 3j m-weighting the estimator already uses at delta = 2, and reports
the mean response per unit g_0 together with the null scatter, for the full sky and for
mask+fill. Comparing the two answers the question directly rather than by inference.

It reads only the mask and simulations, never a CMB map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from quantum_gravity.crbc_gpu.crbc_planck_mask_coupling import (
        _alm_value,
        build_response,
        reconstruct_alm,
        three_j_l_lplus2,
    )
    from quantum_gravity.crbc_gpu.crbc_planck_injection_recovery import (
        diagonal_response,
        synthesize,
    )
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_planck_mask_coupling import (  # type: ignore[no-redef]
        _alm_value,
        build_response,
        reconstruct_alm,
        three_j_l_lplus2,
    )
    from crbc_planck_injection_recovery import diagonal_response, synthesize  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=Path("data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"))
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--lmax", type=int, default=250)
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=120)
    parser.add_argument("--injection", type=float, default=2.0, help="large g_0 so the template is well measured")
    parser.add_argument("--deltas", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--tilt", type=float, default=-0.70)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_signal_leakage.json"))
    return parser.parse_args()


def m_weighting(l: int, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """The 3j m-pattern the estimator uses at delta = 2, reused for every delta."""
    m = np.arange(-l, l + 1)
    coupling = three_j_l_lplus2(np.full(m.shape, l), np.zeros_like(m), -m, m)
    return m, ((-1.0) ** m) * coupling


def statistics(alm: np.ndarray, lmax: int, lmin: int, deltas: list[int]) -> dict[int, np.ndarray]:
    """X_{l,delta} for every l and every requested even separation."""
    import healpy as hp

    out: dict[int, np.ndarray] = {}
    for delta in deltas:
        values = np.zeros(lmax + 1)
        for l in range(lmin, lmax - delta + 1):
            m, weight = m_weighting(l, lmax)
            valid = np.abs(m) <= l + delta
            m, weight = m[valid], weight[valid]
            first = _alm_value(alm, np.full(m.shape, l), m, lmax, hp)
            second = _alm_value(alm, np.full(m.shape, l + delta), m, lmax, hp)
            values[l] = float(np.sum(weight * np.real(first * np.conjugate(second))))
        out[delta] = values
    return out


def main() -> None:
    args = parse_args()
    import healpy as hp

    response = build_response(args.lmax, args.tilt)
    cl = response["cl"][: args.lmax + 3].copy()
    cl[: args.lmin] = 0.0
    diagonal = diagonal_response(args.lmax, args.tilt)

    mask = hp.ud_grade(hp.read_map(str(args.mask)), args.nside)
    mask = np.where(mask > 0.5, 1.0, 0.0)

    configurations = ("full_sky", "mask_fill")
    accumulated = {
        name: {"signal": {d: [] for d in args.deltas}, "null": {d: [] for d in args.deltas}}
        for name in configurations
    }

    # Paired simulations with common random numbers: the injected and null skies are driven
    # by identical white noise and identical mask filler, so their difference isolates the
    # deterministic signal response instead of measuring it against realization scatter.
    # Without this the diagnostic is noise dominated, and the full-sky delta = 4, 6 channels
    # — which must vanish identically — come out nonzero.
    for realization in range(args.simulations):
        for injected in (args.injection, 0.0):
            key = "signal" if injected > 0.0 else "null"
            rng = np.random.default_rng(args.seed + realization)
            alm_true = synthesize(injected, args.lmin, args.lmax, cl, response["response"], diagonal, rng, hp)
            sky = hp.alm2map(alm_true, args.nside, lmax=args.lmax)
            np.random.seed(args.seed + realization)
            filled = reconstruct_alm(sky, mask, cl, args.nside, args.lmax, "fill", 30, 40, hp, np)
            for name, alm in (("full_sky", alm_true), ("mask_fill", filled)):
                values = statistics(alm, args.lmax, args.lmin, args.deltas)
                for delta in args.deltas:
                    accumulated[name][key][delta].append(values[delta])

    edges = np.linspace(args.lmin, args.lmax, args.bins + 1).astype(int)
    report_configurations = {}
    for name in configurations:
        per_delta = {}
        for delta in args.deltas:
            signal = np.array(accumulated[name]["signal"][delta])
            null = np.array(accumulated[name]["null"][delta])
            response_per_l = (signal - null).mean(axis=0) / args.injection  # paired difference
            scatter_per_l = null.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(scatter_per_l > 0, response_per_l / scatter_per_l, 0.0)
            binned = []
            for start, stop in zip(edges[:-1], edges[1:]):
                block = ratio[start:stop]
                binned.append({"l_range": [int(start), int(stop)], "mean_response_over_scatter": float(block.mean())})
            per_delta[str(delta)] = {
                "total_snr_per_unit_g0": float(np.sqrt(np.sum(ratio**2))),
                "binned": binned,
            }
        report_configurations[name] = per_delta

    full = report_configurations["full_sky"]
    masked = report_configurations["mask_fill"]

    # Noise floor, calibrated internally: on the full sky the modulation couples only
    # l' = l and l' = l +/- 2, so every full-sky channel with delta >= 4 is exactly zero and
    # whatever it returns is the finite-simulation noise of this statistic.
    floor_channels = [d for d in args.deltas if d >= 4]
    noise_floor = (
        float(np.sqrt(np.mean([full[str(d)]["total_snr_per_unit_g0"] ** 2 for d in floor_channels])))
        if floor_channels
        else 0.0
    )

    def debias(value: float) -> float:
        return float(np.sqrt(max(value**2 - noise_floor**2, 0.0)))

    for name in configurations:
        for d in args.deltas:
            report_configurations[name][str(d)]["total_snr_debiased"] = debias(
                report_configurations[name][str(d)]["total_snr_per_unit_g0"]
            )

    delta2_ratio = (
        masked["2"]["total_snr_debiased"] / full["2"]["total_snr_debiased"]
        if full["2"]["total_snr_debiased"] > 0
        else float("nan")
    )
    masked_total = np.sqrt(sum(masked[str(d)]["total_snr_debiased"] ** 2 for d in args.deltas))
    masked_delta2 = masked["2"]["total_snr_debiased"]
    widening_gain = masked_total / masked_delta2 if masked_delta2 > 0 else float("nan")

    report = {
        "question": "does the mask reduce the l,l+2 response, or scatter the signal to larger separations?",
        "configuration": {
            "nside": args.nside,
            "lmin": args.lmin,
            "lmax": args.lmax,
            "simulations_per_case": args.simulations,
            "injection": args.injection,
            "deltas": args.deltas,
        },
        "per_configuration": report_configurations,
        "diagnosis": {
            "noise_floor_from_full_sky_delta_ge_4": noise_floor,
            "noise_floor_note": "full-sky delta >= 4 vanishes identically, so it measures this "
            "statistic's finite-simulation noise and calibrates the debiasing",
            "delta2_snr_masked_over_full_sky": float(delta2_ratio),
            "gain_from_widening_delta_masked": float(widening_gain),
            "reading": (
                "Widening delta recovers little; the loss is in the l,l+2 response itself, so a "
                "per-l reweighting is the repair."
                if widening_gain < 1.2
                else "A significant part of the signal sits at delta > 2; the estimator must be "
                "widened in delta, not merely reweighted."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "per_configuration"}
    summary["totals_raw"] = {
        name: {d: report_configurations[name][d]["total_snr_per_unit_g0"] for d in map(str, args.deltas)}
        for name in configurations
    }
    summary["totals_debiased"] = {
        name: {d: report_configurations[name][d]["total_snr_debiased"] for d in map(str, args.deltas)}
        for name in configurations
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
