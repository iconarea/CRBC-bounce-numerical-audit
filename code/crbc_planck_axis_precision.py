#!/usr/bin/env python3
"""Direct measurement of the single-sky axis precision, free of the grid floor.

사전등록 §13.7 fitted the axis error to a power law in l_max and extrapolated; §13.8
retracted that, because the local slope of the Fisher forecast ranges from -0.01 to -1.32
and no single power law describes it. The remaining number therefore has to be measured,
not extrapolated, and measured on a grid fine enough that the answer is not the grid
spacing. At l <= 250 the earlier run returned 4.41 deg against a 3.66 deg grid, so it was
grid limited and only an upper bound.

This script measures it directly. Two things that do not depend on the realization are
precomputed once and reused, which is what makes high l affordable:

  * the Cholesky factors of the injected covariance, per (m, parity chain);
  * the (M, delta, l) index and 3j weight arrays of the estimator.

Per realization the work is then a white-noise draw, one alm2map/map2alm pair, and
vectorized gathers — instead of thousands of Cholesky factorizations and 3j evaluations.

The grid is refined until the measured error stops changing, so the reported value is the
cosmic-variance floor of the l range rather than the pixel size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from quantum_gravity.crbc_gpu.crbc_planck_mask_coupling import build_response, three_j_l_lplus2
    from quantum_gravity.crbc_gpu.crbc_planck_injection_recovery import diagonal_response, three_j_diagonal
    from quantum_gravity.crbc_gpu.crbc_planck_axis_search import real_spherical_basis, PREFACTOR
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_planck_mask_coupling import build_response, three_j_l_lplus2  # type: ignore[no-redef]
    from crbc_planck_injection_recovery import diagonal_response, three_j_diagonal  # type: ignore[no-redef]
    from crbc_planck_axis_search import real_spherical_basis, PREFACTOR  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=250)
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument("--deltas", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--grid-nsides", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--injection", type=float, default=3.0)
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--null", type=int, default=150)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--tilt", type=float, default=-0.70)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_axis_precision.json"))
    return parser.parse_args()


def build_cholesky(g0: float, lmin: int, lmax: int, cl: np.ndarray, off: np.ndarray, diag: np.ndarray):
    """Cholesky factor of the modulated covariance per (m, chain). Realization independent."""
    factors = {}
    for m in range(lmax + 1):
        for start in (0, 1):
            first = max(lmin, m)
            chain = np.arange(first + ((first + start) % 2), lmax + 1, 2)
            if chain.size == 0:
                continue
            size = chain.size
            l = chain.astype(float)
            matrix = np.zeros((size, size))
            parity = three_j_diagonal(l, np.zeros_like(l))
            coupling = three_j_diagonal(l, np.full_like(l, float(m)))
            geometry = ((-1.0) ** m) * np.sqrt(5.0 * (2 * l + 1) ** 2 / (4.0 * np.pi)) * parity * coupling
            # covariance prefactor is (2/3) sqrt(4 pi/5); PREFACTOR above is the different
            # (2/3)(4 pi/5) that relates g_2M to g_0, and the two must not be confused
            covariance_prefactor = (2.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0)
            matrix[np.arange(size), np.arange(size)] = (
                cl[chain] + g0 * diag[chain] / 3.0 + g0 * covariance_prefactor * diag[chain] * geometry
            )
            if size > 1:
                head = chain[:-1]
                lh = head.astype(float)
                parity_off = three_j_l_lplus2(lh, np.zeros_like(lh), np.zeros_like(lh), np.zeros_like(lh))
                coupling_off = three_j_l_lplus2(lh, np.zeros_like(lh), -np.full_like(lh, float(m)), np.full_like(lh, float(m)))
                geometry_off = (
                    ((-1.0) ** m)
                    * np.sqrt(5.0 * (2 * lh + 1) * (2 * lh + 5) / (4.0 * np.pi))
                    * parity_off
                    * coupling_off
                )
                value = -g0 * covariance_prefactor * off[head] * geometry_off
                matrix[np.arange(size - 1), np.arange(1, size)] = value
                matrix[np.arange(1, size), np.arange(size - 1)] = value
            factors[(m, start)] = (chain, np.linalg.cholesky(matrix))
    return factors


def draw(factors, lmax: int, rng: np.random.Generator, hp) -> np.ndarray:
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    for (m, _), (chain, factor) in factors.items():
        size = chain.size
        if m == 0:
            values = (factor @ rng.standard_normal(size)).astype(np.complex128)
        else:
            real = factor @ rng.standard_normal(size)
            imaginary = factor @ rng.standard_normal(size)
            values = (real + 1j * imaginary) / np.sqrt(2.0)
        alm[hp.Alm.getidx(lmax, chain, np.full(size, m))] = values
    return alm


def build_estimator_index(lmin: int, lmax: int, deltas: list[int], hp):
    """Precomputed gather indices, conjugation signs and 3j weights, per M."""
    packed = {}
    for M in (0, 1, 2):
        first_index, second_index = [], []
        first_sign, second_sign, first_neg, second_neg = [], [], [], []
        weight, segment = [], []
        channel = 0
        for delta in deltas:
            for l in range(lmin, lmax - delta + 1):
                m = np.arange(-l, l + 1)
                mp = m - M
                keep = np.abs(mp) <= l + delta
                m, mp = m[keep], mp[keep]
                coupling = three_j_l_lplus2(np.full(m.shape, l), np.full(m.shape, M), -m, mp)
                first_index.append(hp.Alm.getidx(lmax, np.full(m.shape, l), np.abs(m)))
                second_index.append(hp.Alm.getidx(lmax, np.full(m.shape, l + delta), np.abs(mp)))
                # a_{l,-m} = (-1)^m conj(a_lm). Whether to conjugate is decided by the sign of
                # m, never by the value of (-1)^|m|, which is +1 for even |m| and would then
                # silently skip the conjugation for half of the negative modes.
                first_neg.append(m < 0)
                second_neg.append(mp < 0)
                first_sign.append((-1.0) ** np.abs(m))
                second_sign.append((-1.0) ** np.abs(mp))
                weight.append(((-1.0) ** m) * coupling)
                segment.append(np.full(m.shape, channel))
                channel += 1
        packed[M] = {
            "first": np.concatenate(first_index),
            "second": np.concatenate(second_index),
            "first_negative": np.concatenate(first_neg),
            "second_negative": np.concatenate(second_neg),
            "first_sign": np.concatenate(first_sign),
            "second_sign": np.concatenate(second_sign),
            "weight": np.concatenate(weight),
            "segment": np.concatenate(segment),
            "channels": channel,
        }
    return packed


def per_channel(alm: np.ndarray, packed: dict, M: int) -> np.ndarray:
    entry = packed[M]
    first = alm[entry["first"]]
    first = np.where(entry["first_negative"], entry["first_sign"] * np.conjugate(first), first)
    second = alm[entry["second"]]
    second = np.where(entry["second_negative"], entry["second_sign"] * np.conjugate(second), second)
    products = entry["weight"] * first * np.conjugate(second)
    real = np.bincount(entry["segment"], weights=products.real, minlength=entry["channels"])
    imaginary = np.bincount(entry["segment"], weights=products.imag, minlength=entry["channels"])
    return real + 1j * imaginary


def five_vector(alm: np.ndarray, packed: dict, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(5)
    for M in (0, 1, 2):
        value = np.sum(weights * per_channel(alm, packed, M))
        if M == 0:
            out[2] = float(value.real)
        else:
            out[2 - M] = float(value.real)
            out[2 + M] = float(value.imag)
    return out


def main() -> None:
    args = parse_args()
    import healpy as hp

    response = build_response(args.lmax, args.tilt)
    cl = response["cl"][: args.lmax + 3].copy()
    cl[: args.lmin] = 0.0
    diag = diagonal_response(args.lmax, args.tilt)

    packed = build_estimator_index(args.lmin, args.lmax, args.deltas, hp)
    null_factors = build_cholesky(0.0, args.lmin, args.lmax, cl, response["response"], diag)
    signal_factors = build_cholesky(args.injection, args.lmin, args.lmax, cl, response["response"], diag)

    rng = np.random.default_rng(args.seed)

    # inverse-variance (l, delta) weights, trained at M = 0 with paired draws
    paired, nulls = [], []
    for _ in range(args.train):
        state = rng.bit_generator.state
        signal = per_channel(draw(signal_factors, args.lmax, rng, hp), packed, 0).real
        rng.bit_generator.state = state
        null = per_channel(draw(null_factors, args.lmax, rng, hp), packed, 0).real
        paired.append(signal - null)
        nulls.append(null)
    channel_response = np.mean(paired, axis=0) / args.injection
    channel_scatter = np.std(nulls, axis=0, ddof=1)
    weights = np.where(channel_scatter > 0, channel_response / np.maximum(channel_scatter, 1e-300) ** 2, 0.0)

    # 5x5 response, from injections rotated to independent axes
    axes = [(0.62, 0.41), (1.13, 1.32), (0.94, 2.37), (1.42, 0.73), (0.48, 3.05), (1.25, 4.19)]
    truth, measured = [], []
    for theta, phi in axes:
        collected = []
        for _ in range(args.train):
            state = rng.bit_generator.state
            alm = draw(signal_factors, args.lmax, rng, hp)
            hp.rotate_alm(alm, 0.0, theta, phi)
            rng.bit_generator.state = state
            reference = draw(null_factors, args.lmax, rng, hp)
            collected.append(five_vector(alm, packed, weights) - five_vector(reference, packed, weights))
        measured.append(np.mean(collected, axis=0))
        truth.append(args.injection * PREFACTOR * real_spherical_basis(np.array(theta), np.array(phi)))
    response_matrix = np.linalg.pinv(np.stack(truth)) @ np.stack(measured)
    inverse_response = np.linalg.pinv(response_matrix.T)

    null_vectors = np.array(
        [inverse_response @ five_vector(draw(null_factors, args.lmax, rng, hp), packed, weights)
         for _ in range(args.null)]
    )
    precision = np.linalg.pinv(np.cov(null_vectors, rowvar=False))

    target = (1.2132, 1.1)
    target_vector = hp.ang2vec(*target)
    signal_vectors = []
    for _ in range(args.trials):
        alm = draw(signal_factors, args.lmax, rng, hp)
        hp.rotate_alm(alm, 0.0, target[0], target[1])
        signal_vectors.append(inverse_response @ five_vector(alm, packed, weights))

    results = {}
    for grid_nside in args.grid_nsides:
        theta_grid, phi_grid = hp.pix2ang(grid_nside, np.arange(hp.nside2npix(grid_nside)))
        templates = PREFACTOR * real_spherical_basis(theta_grid, phi_grid)
        denominator = np.einsum("ij,jk,ik->i", templates, precision, templates)
        errors = []
        for vector in signal_vectors:
            numerator = templates @ (precision @ vector)
            best = int(np.argmax(numerator**2 / denominator))
            cosine = abs(float(np.dot(hp.pix2vec(grid_nside, best), target_vector)))
            errors.append(float(np.degrees(np.arccos(min(cosine, 1.0)))))
        results[str(grid_nside)] = {
            "grid_resolution_deg": float(np.degrees(hp.nside2resol(grid_nside))),
            "median_error_deg": float(np.median(errors)),
            "percentile_68_deg": float(np.percentile(errors, 68)),
        }

    values = [results[str(n)]["median_error_deg"] for n in args.grid_nsides]
    converged = bool(len(values) > 1 and abs(values[-1] - values[-2]) < 0.1 * values[-1])

    report = {
        "purpose": "measure the single-sky axis precision directly, refining the grid until the "
        "answer stops depending on it",
        "configuration": {
            "lmin": args.lmin,
            "lmax": args.lmax,
            "deltas": args.deltas,
            "injection": args.injection,
            "trials": args.trials,
            "null_realizations": args.null,
            "full_sky": True,
        },
        "by_grid": results,
        "grid_converged": converged,
        "measured_axis_error_deg": values[-1],
        "note": "full sky; the mask reduces the effective mode count and will widen this",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
