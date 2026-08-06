#!/usr/bin/env python3
"""Injection-recovery calibration for the CRBC quadrupole estimator.

Passing the mask-coupling gate (pre-registration §9) shows the estimator is unbiased when
there is no signal. It does not say what a nonzero reading means, because filling the mask
with an independent realization removes signal along with bias. This script measures that:
it injects a known g_0 into simulated skies, runs the full analysis chain, and reports the
recovered slope d<g_hat>/dg_0.

Sky generation is exact rather than perturbative. With the axis along z only M = 0 is
excited, and the primordial modulation

    P(k, k_hat) = P_iso(k) [1 + g_*(k) mu^2],   mu^2 = 1/3 + (2/3) sqrt(4pi/5) Y_20(k_hat)

gives a covariance that, at fixed m, is tridiagonal in l with steps of 2:

    <a_lm a*_lm>   = C_l + g_0 [ D_ll/3 + (2/3) sqrt(4pi/5) D_ll G_ll(m) ]
    <a_lm a*_l+2m> = -g_0 (2/3) sqrt(4pi/5) D_l,l+2 G_l,l+2(m)
    G_ll'(m)       = (-1)^m sqrt(5(2l+1)(2l'+1)/(4pi)) ThreeJ(2,l,l';0,0,0) ThreeJ(2,l,l';0,-m,m)

with D per unit g_0 from the CAMB transfer functions. Even and odd l form separate chains,
each of which is Cholesky-factored and driven with white noise, so the injected sky has
exactly the intended covariance to machine precision.

The chain carries a sharp internal check. In the estimator's normalization the recovered
quantity is g_20, and for an axis along z

    g_20 = g_0 (2/3) sqrt(4pi/5) = 1.05683 g_0,

so a full-sky run with no mask must return a slope of 1.05683. Any departure means the
formalism, not the mask treatment, is wrong. The masked runs are only interpretable once
that control passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from quantum_gravity.crbc_gpu.crbc_planck_mask_coupling import (
        apply_estimator,
        build_response,
        estimator_weights,
        reconstruct_alm,
        three_j_l_lplus2,
    )
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_planck_mask_coupling import (  # type: ignore[no-redef]
        apply_estimator,
        build_response,
        estimator_weights,
        reconstruct_alm,
        three_j_l_lplus2,
    )

EXPECTED_FULL_SKY_SLOPE = (2.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=Path("data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"))
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--lmax", type=int, default=250)
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--injections", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--tilt", type=float, default=-0.70)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cg-iterations", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_injection_recovery.json"))
    return parser.parse_args()


def three_j_diagonal(l: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(2, l, l; 0, -m, m). Table form; validated against sympy to 2.8e-17."""
    l = np.asarray(l, dtype=float)
    m = np.asarray(m, dtype=float)
    denominator = (2 * l + 3) * (2 * l + 2) * (2 * l + 1) * (2 * l) * (2 * l - 1)
    return (-1.0) ** np.abs(l - m) * 2.0 * (3.0 * m * m - l * (l + 1.0)) / np.sqrt(denominator)


def diagonal_response(lmax: int, tilt: float) -> np.ndarray:
    """D_ll per unit g_0: the g_*-weighted C_l."""
    import camb

    fiducial = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "tau": 0.0544, "As": 2.1e-9, "ns": 0.9649}
    params = camb.set_params(lmax=lmax + 60, **fiducial)
    params.set_accuracy(AccuracyBoost=2, lSampleBoost=50)
    results = camb.get_results(params)
    data = results.get_cmb_transfer_data("scalar")
    ells = np.array(data.L, dtype=int)
    k = np.array(data.q)
    delta = np.array(data.delta_p_l_k)[0]
    log_k = np.log(k)
    primordial = fiducial["As"] * (k / 0.05) ** (fiducial["ns"] - 1.0)
    shape = (k / 0.05) ** tilt
    position = {int(value): index for index, value in enumerate(ells)}
    out = np.zeros(lmax + 3)
    for l in range(2, lmax + 1):
        if l in position:
            out[l] = 4.0 * np.pi * np.trapezoid(primordial * shape * delta[position[l]] ** 2, log_k)
    return out


def geometry_factor(l: np.ndarray, lp: np.ndarray, m: np.ndarray, parity: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    return ((-1.0) ** m) * np.sqrt(5.0 * (2 * l + 1) * (2 * lp + 1) / (4.0 * np.pi)) * parity * coupling


def synthesize(
    g0: float,
    lmin: int,
    lmax: int,
    cl: np.ndarray,
    off_diagonal: np.ndarray,
    diagonal: np.ndarray,
    rng: np.random.Generator,
    hp,
) -> np.ndarray:
    """Exact Gaussian realization of the modulated covariance (axis along z, M = 0)."""
    prefactor = (2.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0)
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)

    for m in range(0, lmax + 1):
        for start in (0, 1):
            chain = np.arange(max(lmin, m) + ((max(lmin, m) + start) % 2), lmax + 1, 2)
            if chain.size == 0:
                continue
            size = chain.size
            matrix = np.zeros((size, size))
            l = chain.astype(float)
            parity_dd = three_j_diagonal(l, np.zeros_like(l) + 0.0)  # (2 l l; 0 0 0)
            coupling_dd = three_j_diagonal(l, np.full_like(l, float(m)))
            g_dd = geometry_factor(l, l, np.full_like(l, float(m)), parity_dd, coupling_dd)
            matrix[np.arange(size), np.arange(size)] = (
                cl[chain] + g0 * (diagonal[chain] / 3.0 + prefactor * diagonal[chain] * g_dd)
            )
            if size > 1:
                head = chain[:-1]
                lh = head.astype(float)
                parity_od = three_j_l_lplus2(lh, np.zeros_like(lh), np.zeros_like(lh), np.zeros_like(lh))
                coupling_od = three_j_l_lplus2(lh, np.zeros_like(lh), -np.full_like(lh, float(m)), np.full_like(lh, float(m)))
                g_od = geometry_factor(lh, lh + 2, np.full_like(lh, float(m)), parity_od, coupling_od)
                value = -g0 * prefactor * off_diagonal[head] * g_od
                matrix[np.arange(size - 1), np.arange(1, size)] = value
                matrix[np.arange(1, size), np.arange(size - 1)] = value
            factor = np.linalg.cholesky(matrix)

            if m == 0:
                values = factor @ rng.standard_normal(size)
                draw = values.astype(np.complex128)
            else:
                real = factor @ rng.standard_normal(size)
                imaginary = factor @ rng.standard_normal(size)
                draw = (real + 1j * imaginary) / np.sqrt(2.0)
            alm[hp.Alm.getidx(lmax, chain, np.full(size, m))] = draw
    return alm


def main() -> None:
    args = parse_args()
    import healpy as hp

    response = build_response(args.lmax, args.tilt)
    cl = response["cl"][: args.lmax + 3].copy()
    cl[: args.lmin] = 0.0
    diagonal = diagonal_response(args.lmax, args.tilt)
    weights = estimator_weights(args.lmin, args.lmax, response["cl"], response["response"])

    mask = hp.read_map(str(args.mask))
    mask = hp.ud_grade(mask, args.nside)
    mask = np.where(mask > 0.5, 1.0, 0.0)
    f_sky = float(mask.mean())

    rng = np.random.default_rng(args.seed)
    spectrum = cl.copy()

    configurations = ("full_sky", "mask_none", "mask_fill", "mask_deconv")
    collected = {name: {value: [] for value in args.injections} for name in configurations}

    for g0 in args.injections:
        for _ in range(args.simulations):
            alm_true = synthesize(g0, args.lmin, args.lmax, cl, response["response"], diagonal, rng, hp)
            sky = hp.alm2map(alm_true, args.nside, lmax=args.lmax)
            collected["full_sky"][g0].append(apply_estimator(alm_true, args.lmax, weights)[2])
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            for name, mode in (("mask_none", "none"), ("mask_fill", "fill"), ("mask_deconv", "deconv")):
                alm = reconstruct_alm(
                    sky, mask, spectrum, args.nside, args.lmax, mode, 30, args.cg_iterations, hp, np
                )
                collected[name][g0].append(apply_estimator(alm, args.lmax, weights)[2])

    report_configurations = {}
    for name in configurations:
        means, errors = [], []
        for g0 in args.injections:
            values = np.array(collected[name][g0])
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1) / np.sqrt(values.size)))
        x = np.array(args.injections)
        y = np.array(means)
        weight = 1.0 / np.array(errors) ** 2
        slope = (np.sum(weight * (x - np.average(x, weights=weight)) * y)
                 / np.sum(weight * (x - np.average(x, weights=weight)) ** 2))
        slope_error = float(np.sum(weight * (x - np.average(x, weights=weight)) ** 2) ** -0.5)
        report_configurations[name] = {
            "recovered_mean_per_injection": dict(zip(map(str, args.injections), means)),
            "error_per_injection": dict(zip(map(str, args.injections), errors)),
            "slope": float(slope),
            "slope_error": slope_error,
        }

    control = report_configurations["full_sky"]
    control_pull = abs(control["slope"] - EXPECTED_FULL_SKY_SLOPE) / control["slope_error"]
    control_ok = bool(control_pull < 3.0)
    fill_slope = report_configurations["mask_fill"]["slope"]
    efficiency = fill_slope / EXPECTED_FULL_SKY_SLOPE

    report = {
        "purpose": "calibrate what a nonzero estimator reading means in units of g_0",
        "configuration": {
            "nside": args.nside,
            "lmin": args.lmin,
            "lmax": args.lmax,
            "simulations_per_injection": args.simulations,
            "injections": args.injections,
            "tilt": args.tilt,
            "f_sky": f_sky,
        },
        "expected_full_sky_slope": float(EXPECTED_FULL_SKY_SLOPE),
        "expected_from": "g_20 = g_0 (2/3) sqrt(4 pi/5) for an axis along z",
        "results": report_configurations,
        "full_sky_control": {
            "measured_slope": control["slope"],
            "expected": float(EXPECTED_FULL_SKY_SLOPE),
            "pull": float(control_pull),
            "formalism_validated": control_ok,
        },
        "calibration": {
            "mask_fill_slope": float(fill_slope),
            "efficiency_vs_full_sky": float(efficiency),
            "usage": "divide a measured g_hat by this slope to obtain g_0",
        },
        "verdict": (
            "Formalism validated on the full sky; the mask+fill chain has a measured, stable "
            "response and can be converted to g_0."
            if control_ok
            else "Full-sky control failed: the formalism or the estimator normalization is wrong, "
            "and no calibration should be used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
