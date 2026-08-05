#!/usr/bin/env python3
"""Simulation-trained matched filter for the CRBC quadrupole, widened in multipole separation.

Background. The pseudo-alm estimator keeps only 19% of the full-sky response
(사전등록 §10) and deconvolution does not repair it (§11). The leakage diagnostic
(crbc_planck_signal_leakage.py) located the cause: on the full sky the modulation couples
only l' = l, l +/- 2, but the mask scatters that signal to larger separations. With
mask+fill the debiased per-channel response splits as

    delta = 2: 17.9,  delta = 4: 35.4,  delta = 6: 26.0,  delta = 8: 18.7

against 49.1 at delta = 2 and ~0 elsewhere for the full sky. An estimator restricted to
delta = 2 therefore throws most of the signal away no matter how it is weighted.

This script builds the repair. For each l and each even separation delta it forms

    X_l,delta = sum_m u_l(m) Re( a_lm a*_{l+delta,m} ),

trains the response R_l,delta and the null scatter V_l,delta on simulations, and combines
them with the matched-filter weights

    g_hat = sum (R X / V^2) / sum (R^2 / V^2).

Training and evaluation use disjoint simulation sets, so the reported bias and response are
not the ones the weights were fitted to. The channel covariance is approximated as
diagonal, which costs optimality but not correctness: the weights enter both numerator and
normalization, so any fixed choice yields an unbiased estimator once calibrated.

The script reads only the mask and simulations, never a CMB map. Its output is the same
pair of numbers the earlier estimators were judged on — the mask-coupling gate and the
injection-recovery slope — so the three can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from quantum_gravity.crbc_gpu.crbc_planck_mask_coupling import build_response, reconstruct_alm
    from quantum_gravity.crbc_gpu.crbc_planck_injection_recovery import diagonal_response, synthesize
    from quantum_gravity.crbc_gpu.crbc_planck_signal_leakage import statistics
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_planck_mask_coupling import build_response, reconstruct_alm  # type: ignore[no-redef]
    from crbc_planck_injection_recovery import diagonal_response, synthesize  # type: ignore[no-redef]
    from crbc_planck_signal_leakage import statistics  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=Path("data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"))
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--lmax", type=int, default=250)
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument("--deltas", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--train", type=int, default=80, help="paired realizations used to fit the weights")
    parser.add_argument("--gate", type=int, default=300, help="null realizations for the mask-coupling gate")
    parser.add_argument("--calibrate", type=int, default=80, help="realizations per injection level")
    parser.add_argument("--injections", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--train-injection", type=float, default=2.0)
    parser.add_argument("--noise-uk-arcmin", type=float, default=0.0,
                        help="white noise level; 0 disables noise entirely")
    parser.add_argument("--noise-anisotropy", type=float, default=0.0,
                        help="fractional depth modulation of the noise across the sky, following "
                        "Planck's scan pattern (deeper near the ecliptic poles)")
    parser.add_argument("--foreground-residual", type=float, default=0.0,
                        help="amplitude of a Galactic-plane residual, as a fraction of the rms "
                        "CMB signal, applied only outside the mask")
    parser.add_argument("--beam-arcmin", type=float, default=5.0)
    parser.add_argument("--tilt", type=float, default=-0.70)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_matched_filter.json"))
    return parser.parse_args()


def instrument_templates(args, mask, hp):
    """Noise depth and foreground-residual patterns, built once."""
    npix = hp.nside2npix(args.nside)
    directions = np.arange(npix)
    theta, phi = hp.pix2ang(args.nside, directions)

    # Planck scans on great circles roughly perpendicular to the ecliptic, so the depth is
    # highest near the ecliptic poles. Modelled as a quadrupolar depth modulation about the
    # ecliptic axis, which is the part that matters here: it couples l to l +/- 2 exactly
    # like the mask and the signal do.
    rotator = hp.Rotator(coord=["G", "E"])
    ecliptic_theta, _ = rotator(theta, phi)
    depth = 1.0 + args.noise_anisotropy * (3.0 * np.cos(ecliptic_theta) ** 2 - 1.0) / 2.0
    depth = np.clip(depth, 0.05, None)

    # Foreground residual. A first attempt put this within +/- 15 deg of the Galactic plane
    # and found no effect — correctly, because that is exactly what the common mask removes,
    # and residuals inside the mask are replaced by the fill. The dangerous residual is the
    # one that survives masking, so the template is broadened and multiplied by the mask to
    # keep only the part the estimator actually sees.
    # A latitude-only profile is axisymmetric in Galactic coordinates and therefore populates
    # m = 0 alone; the estimator sums over 2l+1 values of m, so such a template is diluted by
    # ~1/(2l+1) and wrongly looks harmless. Real Galactic residuals have longitude structure
    # (centre, spurs, loops) that fills every m, so the template is a fixed red-spectrum
    # realization modulated by the latitude profile. Fixed, not redrawn per realization,
    # because a residual is deterministic — that is what makes it dangerous.
    latitude = np.exp(-((np.pi / 2 - theta) ** 2) / (2.0 * np.radians(45.0) ** 2))
    ell = np.arange(3 * args.nside)
    red_spectrum = np.where(ell >= 2, (ell + 1.0) ** -3.0, 0.0)
    saved_state = np.random.get_state()
    np.random.seed(987654321)  # fixed template: the residual is the same on every realization
    structure = hp.synfast(red_spectrum, args.nside, lmax=min(3 * args.nside - 1, 2 * args.lmax), new=True)
    np.random.set_state(saved_state)
    structure = structure / np.std(structure)
    galactic_profile = latitude * structure * mask
    galactic_profile = galactic_profile - galactic_profile[mask > 0].mean() * mask
    return {"depth": depth, "foreground": galactic_profile}


def realize(args, cl, off_diagonal, diag, mask, hp, g0: float, seed: int, templates=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alm_true = synthesize(g0, args.lmin, args.lmax, cl, off_diagonal, diag, rng, hp)
    sky = hp.alm2map(alm_true, args.nside, lmax=args.lmax)

    if templates is not None and (args.noise_uk_arcmin > 0.0 or args.foreground_residual > 0.0):
        signal_rms = float(np.std(sky))
        if args.noise_uk_arcmin > 0.0:
            pixel_area = hp.nside2pixarea(args.nside, degrees=True) * 3600.0
            level = (args.noise_uk_arcmin / np.sqrt(pixel_area)) / 2.7255e6
            sky = sky + level * np.sqrt(templates["depth"]) * rng.standard_normal(sky.size)
        if args.foreground_residual > 0.0:
            # deterministic, not stochastic: a residual is the same on every realization,
            # which is what makes it dangerous for an off-diagonal estimator
            sky = sky + args.foreground_residual * signal_rms * templates["foreground"]

    np.random.seed(seed % (2**31 - 1))
    return reconstruct_alm(sky, mask, cl, args.nside, args.lmax, "fill", 30, 40, hp, np)


def flatten(values: dict[int, np.ndarray], deltas: list[int]) -> np.ndarray:
    return np.concatenate([values[d] for d in deltas])


def main() -> None:
    args = parse_args()
    import healpy as hp

    response = build_response(args.lmax, args.tilt)
    cl = response["cl"][: args.lmax + 3].copy()
    cl[: args.lmin] = 0.0
    diag = diagonal_response(args.lmax, args.tilt)
    mask = hp.ud_grade(hp.read_map(str(args.mask)), args.nside)
    mask = np.where(mask > 0.5, 1.0, 0.0)

    templates = instrument_templates(args, mask, hp)

    def observe(g0: float, seed: int) -> np.ndarray:
        alm = realize(args, cl, response["response"], diag, mask, hp, g0, seed, templates)
        return flatten(statistics(alm, args.lmax, args.lmin, args.deltas), args.deltas)

    # --- training: paired common-random-number realizations isolate the response ---
    train_seed = args.seed
    paired, nulls = [], []
    for index in range(args.train):
        seed = train_seed + index
        null = observe(0.0, seed)
        paired.append(observe(args.train_injection, seed) - null)
        nulls.append(null)
    response_vector = np.mean(paired, axis=0) / args.train_injection
    scatter_vector = np.std(nulls, axis=0, ddof=1)
    usable = scatter_vector > 0
    weights = np.zeros_like(response_vector)
    weights[usable] = response_vector[usable] / scatter_vector[usable] ** 2
    normalization = float(np.sum(weights * response_vector))
    if normalization <= 0.0:
        raise SystemExit("matched filter normalization is not positive; refusing to continue")

    def estimate(vector: np.ndarray) -> float:
        return float(np.sum(weights * vector) / normalization)

    # --- gate on independent null realizations ---
    gate_seed = train_seed + 10_000
    gate_values = np.array([estimate(observe(0.0, gate_seed + index)) for index in range(args.gate)])
    half = args.gate // 2
    first, second = gate_values[:half], gate_values[half:]
    bias_first, bias_second = float(first.mean()), float(second.mean())
    error_first = float(first.std(ddof=1) / np.sqrt(first.size))
    error_second = float(second.std(ddof=1) / np.sqrt(second.size))
    scatter = float(gate_values.std(ddof=1))
    stability = abs(bias_first - bias_second) / np.sqrt(error_first**2 + error_second**2)
    closure = abs(float((second - bias_first).mean())) / error_second
    bias_over_scatter = abs(bias_first) / scatter
    gate_passed = bool(bias_over_scatter < 1.0 and stability < 3.0 and closure < 3.0)

    # --- calibration on independent injected realizations ---
    calibration_seed = train_seed + 50_000
    means, errors = [], []
    for level, g0 in enumerate(args.injections):
        values = np.array(
            [estimate(observe(g0, calibration_seed + level * 1000 + index)) for index in range(args.calibrate)]
        )
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1) / np.sqrt(values.size)))
    x = np.array(args.injections)
    y = np.array(means)
    w = 1.0 / np.array(errors) ** 2
    centre = np.average(x, weights=w)
    slope = float(np.sum(w * (x - centre) * y) / np.sum(w * (x - centre) ** 2))
    slope_error = float(np.sum(w * (x - centre) ** 2) ** -0.5)
    sensitivity = scatter / slope if slope > 0 else float("nan")

    report = {
        "estimator": "matched filter over (l, delta) with delta in " + str(args.deltas),
        "weights_trained_on": args.train,
        "training_note": "paired common-random-number realizations; training and evaluation sets are disjoint",
        "covariance_approximation": "channels treated as independent; suboptimal but not biasing",
        "configuration": {
            "noise_uk_arcmin": args.noise_uk_arcmin,
            "noise_anisotropy": args.noise_anisotropy,
            "foreground_residual": args.foreground_residual,
            "nside": args.nside,
            "lmin": args.lmin,
            "lmax": args.lmax,
            "deltas": args.deltas,
            "gate_realizations": args.gate,
            "calibration_realizations_per_level": args.calibrate,
            "injections": args.injections,
        },
        "gate": {
            "bias_first_half": bias_first,
            "bias_second_half": bias_second,
            "scatter": scatter,
            "bias_over_scatter": float(bias_over_scatter),
            "stability_sigma": float(stability),
            "closure_sigma": float(closure),
            "gate_passed": gate_passed,
        },
        "calibration": {
            "recovered_mean_per_injection": dict(zip(map(str, args.injections), means)),
            "slope": slope,
            "slope_error": slope_error,
        },
        "sensitivity": {
            "sigma_g0": float(sensitivity),
            "reference_fill_delta2_only": 0.116,
            "improvement_over_fill": float(0.116 / sensitivity) if sensitivity > 0 else None,
        },
        "verdict": (
            "Matched filter passes the mask-coupling gate and has a measured response."
            if gate_passed
            else "Matched filter fails the mask-coupling gate; it is not used on data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
