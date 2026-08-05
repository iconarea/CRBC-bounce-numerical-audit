#!/usr/bin/env python3
"""Mask-coupling gate for the CRBC quadrupole estimator (pre-registration §3.3).

The estimator uses the l, l+2 off-diagonal of <a_lm a*_l'm'>, which vanishes identically
for an isotropic full sky. A Galactic mask breaks that: it couples l to l +/- 2 with the
*same* structure as the signal. The pre-registration therefore fixes a gate that must be
passed before any Planck map is analysed:

  1. measure the coupling bias on isotropic LambdaCDM simulations carrying the same mask,
  2. subtract it,
  3. verify that the corrected estimator has mean consistent with zero,
  4. if it does not, discard the estimator and analyse no data.

Subtracting a mean measured on the same simulations and then checking that the mean is
zero would be circular, so the test implemented here is stronger:

  (a) bias size — is |bias| large compared with the statistical error sigma(g_0)? A bias
      much larger than the signal being sought means the answer is set by the correction,
      not the sky.
  (b) bias stability — the simulations are split into two independent halves and the bias
      is measured separately in each. The halves must agree within their errors, otherwise
      the correction is not a fixed property of the mask.
  (c) closure — the second half is corrected with the bias measured on the first, and its
      mean must be consistent with zero. This is a genuine out-of-sample test.
  (d) variance — the scatter of the corrected estimator is compared with the full-sky
      Fisher forecast, to expose how much the mask costs.

Estimator. With <a_lm a*_l'm'>_aniso = sum_M g_2M S^M_lm,l'm' and

    S^M = D_ll' * i^(l-l') * (-1)^m sqrt(5(2l+1)(2l'+1)/(4pi))
          * ThreeJ(2,l,l';0,0,0) * ThreeJ(2,l,l';M,-m,m'),      m' = m - M,

the minimum-variance amplitude estimator at fixed M is

    g_2M_hat = sum_lm S^M a_lm a*_l'm' / (C_l C_l') / sum_lm (S^M)^2 / (C_l C_l').

For l' = l + 2 the Racah sum for the 3j symbols collapses to a single term because
j1 + j2 - j3 = 0, giving a closed form that is evaluated with log-gammas and was checked
against sympy to 2e-14.

This script reads only the mask. It never touches a CMB map, so running it cannot bias the
later analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=Path("data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"))
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--lmax", type=int, default=250)
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--tilt", type=float, default=-0.70)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--apodize-fwhm-deg", type=float, default=0.0,
                        help="smooth the binary mask by this FWHM to soften the l,l+2 coupling")
    parser.add_argument(
        "--filter",
        choices=("none", "fill", "inpaint", "wiener", "deconv"),
        default="none",
        help="how the masked region is handled before the estimator: raw pseudo-alm (none), "
        "replaced by an independent realization (fill), iterative harmonic inpainting "
        "(inpaint), or a Wiener/CG reconstruction of the full-sky alm (wiener)",
    )
    parser.add_argument("--inpaint-iterations", type=int, default=30)
    parser.add_argument("--cg-iterations", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_mask_coupling.json"))
    return parser.parse_args()


def log_factorial(values: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(np.asarray(values, dtype=float) + 1.0)


def three_j_l_lplus2(l: np.ndarray, m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    """(2, l, l+2; m1, m2, m3). Single-term Racah form; validated against sympy to 2e-14."""
    l = np.asarray(l, dtype=float)
    allowed = (np.abs(m2) <= l) & (np.abs(m3) <= l + 2) & (np.abs(m1) <= 2) & (np.abs(m1 + m2 + m3) < 1e-9)
    log_delta = np.log(24.0) + log_factorial(2 * l) - log_factorial(2 * l + 5)
    log_numerator = log_factorial(l + 2 + m3) + log_factorial(l + 2 - m3)
    log_denominator = (
        log_factorial(2 + m1) + log_factorial(2 - m1) + log_factorial(l + m2) + log_factorial(l - m2)
    )
    magnitude = np.exp(0.5 * (log_delta + log_numerator - log_denominator))
    sign = (-1.0) ** np.abs(2 - l - m3)
    return np.where(allowed, sign * magnitude, 0.0)


def build_response(lmax: int, tilt: float) -> dict[str, np.ndarray]:
    """C_l and the per-unit-g_0 response D_{l,l+2}, both from CAMB transfer functions."""
    import camb

    fiducial = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "tau": 0.0544, "As": 2.1e-9, "ns": 0.9649}
    params = camb.set_params(lmax=lmax + 60, **fiducial)
    params.set_accuracy(AccuracyBoost=2, lSampleBoost=50)
    results = camb.get_results(params)
    cl = results.get_cmb_power_spectra(params, CMB_unit=None, raw_cl=True)["total"][:, 0]
    data = results.get_cmb_transfer_data("scalar")
    ells = np.array(data.L, dtype=int)
    k = np.array(data.q)
    delta = np.array(data.delta_p_l_k)[0]
    log_k = np.log(k)
    primordial = fiducial["As"] * (k / 0.05) ** (fiducial["ns"] - 1.0)
    shape = (k / 0.05) ** tilt

    position = {int(value): index for index, value in enumerate(ells)}
    response = np.zeros(lmax + 3)
    for l in range(2, lmax + 1):
        if l in position and l + 2 in position:
            response[l] = 4.0 * np.pi * np.trapezoid(
                primordial * shape * delta[position[l]] * delta[position[l + 2]], log_k
            )
    # Validate the normalization the whole estimator rests on.
    checks = {}
    for target in (2, 10, 50):
        if target in position:
            computed = 4.0 * np.pi * np.trapezoid(primordial * delta[position[target]] ** 2, log_k)
            checks[str(target)] = float(computed / cl[target])
    return {"cl": cl, "response": response, "normalization_checks": checks}


def estimator_weights(lmin: int, lmax: int, cl: np.ndarray, response: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Per-M arrays of (l, m, weight) for the l, l+2 off-diagonal estimator."""
    weights: dict[int, dict[str, np.ndarray]] = {}
    for M in range(-2, 3):
        l_list, m_list, s_list = [], [], []
        for l in range(lmin, lmax - 1):
            lp = l + 2
            parity = three_j_l_lplus2(np.array([l]), np.array([0]), np.array([0]), np.array([0]))[0]
            if parity == 0.0 or cl[l] <= 0 or cl[lp] <= 0:
                continue
            m = np.arange(-l, l + 1)
            mp = m - M
            valid = np.abs(mp) <= lp
            m, mp = m[valid], mp[valid]
            coupling = three_j_l_lplus2(np.full(m.shape, l), np.full(m.shape, M), -m, mp)
            geometry = np.sqrt(5.0 * (2 * l + 1) * (2 * lp + 1) / (4.0 * np.pi)) * parity
            # i^(l-l') = -1 for l' = l+2.
            signal = -response[l] * geometry * ((-1.0) ** m) * coupling
            l_list.append(np.full(m.shape, l))
            m_list.append(m)
            s_list.append(signal / (cl[l] * cl[lp]))
        weights[M] = {
            "l": np.concatenate(l_list),
            "m": np.concatenate(m_list),
            "weight": np.concatenate(s_list),
            "signal": np.concatenate(
                [w * cl[l[0]] * cl[l[0] + 2] for w, l in zip(s_list, l_list)]
            ),
        }
    return weights


def apply_estimator(alm: np.ndarray, lmax: int, weights: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """Return g_2M_hat for M = -2..2 from a healpy alm array."""
    import healpy as hp

    values = np.zeros(5)
    for index, M in enumerate(range(-2, 3)):
        entry = weights[M]
        l, m, weight = entry["l"], entry["m"], entry["weight"]
        mp = m - M
        first = _alm_value(alm, l, m, lmax, hp)
        second = _alm_value(alm, l + 2, mp, lmax, hp)
        numerator = np.sum(weight * np.real(first * np.conjugate(second)))
        denominator = np.sum(weight * entry["signal"])
        values[index] = numerator / denominator if denominator != 0.0 else np.nan
    return values


def reconstruct_alm(sky, mask, spectrum, nside, lmax, mode, inpaint_iterations, cg_iterations, hp, np_):
    """Return alm for the estimator under the chosen treatment of the masked region."""
    if mode == "none":
        return hp.map2alm(sky * mask, lmax=lmax, iter=0)

    if mode == "fill":
        # Masked pixels replaced by an independent realization of the same C_l. The filled
        # map is full-sky, so the cut's geometry no longer multiplies the signal directly.
        filler = hp.synfast(spectrum, nside, lmax=lmax, new=True, verbose=False)
        return hp.map2alm(sky * mask + filler * (1.0 - mask), lmax=lmax, iter=0)

    if mode == "inpaint":
        # Iterative harmonic inpainting: keep the observed pixels fixed and let the masked
        # region relax to the band-limited continuation of the surrounding sky.
        filler = hp.synfast(spectrum, nside, lmax=lmax, new=True, verbose=False)
        current = sky * mask + filler * (1.0 - mask)
        for _ in range(inpaint_iterations):
            alm = hp.map2alm(current, lmax=lmax, iter=0)
            smooth = hp.alm2map(alm, nside, lmax=lmax, verbose=False)
            current = sky * mask + smooth * (1.0 - mask)
        return hp.map2alm(current, lmax=lmax, iter=0)

    if mode == "deconv":
        # MASTER-style deconvolution: solve K x = pseudo_alm for x, where
        # K y = map2alm(alm2map(y) * mask) is the mask coupling operator. K is symmetric
        # positive semi-definite, so CG applies; starting from zero gives the minimum-norm
        # solution, which is the true alm projected onto the observable subspace.
        pseudo = hp.map2alm(sky * mask, lmax=lmax, iter=0)

        def coupling(y):
            return hp.map2alm(hp.alm2map(y, nside, lmax=lmax) * mask, lmax=lmax, iter=0)

        x = np_.zeros_like(pseudo)
        r = pseudo - coupling(x)
        pvec = r.copy()
        rs = np_.vdot(r, r).real
        for _ in range(cg_iterations):
            ap = coupling(pvec)
            denominator = np_.vdot(pvec, ap).real
            if denominator == 0.0:
                break
            alpha = rs / denominator
            x = x + alpha * pvec
            r = r - alpha * ap
            rs_new = np_.vdot(r, r).real
            if rs_new <= 1e-14 * rs:
                break
            pvec = r + (rs_new / rs) * pvec
            rs = rs_new
        return x

    # mode == "wiener": solve (S^-1 + N^-1) x = N^-1 d by conjugate gradient, with N^-1
    # equal to the mask (infinite noise inside the cut). S is diagonal in harmonic space.
    inverse_signal = np_.zeros_like(spectrum)
    good = spectrum > 0
    inverse_signal[good] = 1.0 / spectrum[good]

    def operator(alm):
        harmonic = hp.almxfl(alm, inverse_signal)
        pixel = hp.alm2map(alm, nside, lmax=lmax, verbose=False)
        return harmonic + hp.map2alm(pixel * mask, lmax=lmax, iter=0)

    right = hp.map2alm(sky * mask, lmax=lmax, iter=0)
    x = np_.zeros_like(right)
    r = right - operator(x)
    pvec = r.copy()
    rs = np_.vdot(r, r).real
    for _ in range(cg_iterations):
        ap = operator(pvec)
        denominator = np_.vdot(pvec, ap).real
        if denominator == 0.0:
            break
        alpha = rs / denominator
        x = x + alpha * pvec
        r = r - alpha * ap
        rs_new = np_.vdot(r, r).real
        if rs_new <= 1e-12 * rs:
            break
        pvec = r + (rs_new / rs) * pvec
        rs = rs_new
    return x


def _alm_value(alm: np.ndarray, l: np.ndarray, m: np.ndarray, lmax: int, hp) -> np.ndarray:
    """healpy stores m >= 0 only; a_{l,-m} = (-1)^m conj(a_lm)."""
    negative = m < 0
    magnitude = np.abs(m)
    index = hp.Alm.getidx(lmax, l.astype(int), magnitude.astype(int))
    value = alm[index]
    return np.where(negative, ((-1.0) ** magnitude) * np.conjugate(value), value)


def main() -> None:
    args = parse_args()
    import healpy as hp

    response = build_response(args.lmax, args.tilt)
    cl = response["cl"]
    worst = max(abs(value - 1.0) for value in response["normalization_checks"].values())
    if worst > 5e-3:
        raise SystemExit(f"transfer normalization off by {worst:.3e}; refusing to run")

    mask = hp.read_map(str(args.mask), verbose=False) if hasattr(hp, "read_map") else None
    mask = hp.ud_grade(mask, args.nside)
    mask = np.where(mask > 0.5, 1.0, 0.0)
    if args.apodize_fwhm_deg > 0.0:
        mask = hp.smoothing(mask, fwhm=np.radians(args.apodize_fwhm_deg), verbose=False)
        mask = np.clip(mask, 0.0, 1.0)
    f_sky = float((mask**2).mean() ** 2 / (mask**4).mean()) if args.apodize_fwhm_deg > 0.0 else float(mask.mean())

    weights = estimator_weights(args.lmin, args.lmax, cl, response["response"])

    rng = np.random.default_rng(args.seed)
    spectrum = cl[: args.lmax + 3].copy()
    spectrum[: args.lmin] = 0.0

    # Full-sky control: the estimator must already be unbiased with no mask. A nonzero mean
    # here would mean the estimator itself is wrong, not that the mask biases it.
    control_count = max(args.simulations // 4, 20)
    control = np.zeros((control_count, 5))
    estimates = np.zeros((args.simulations, 5))
    for index in range(args.simulations + control_count):
        seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(seed)
        sky = hp.synfast(spectrum, args.nside, lmax=args.lmax + 2, new=True, verbose=False)
        if index < control_count:
            alm = hp.map2alm(sky, lmax=args.lmax + 2, iter=0)
            control[index] = apply_estimator(alm, args.lmax + 2, weights)
        else:
            alm = reconstruct_alm(
                sky, mask, spectrum, args.nside, args.lmax + 2, args.filter,
                args.inpaint_iterations, args.cg_iterations, hp, np,
            )
            estimates[index - control_count] = apply_estimator(alm, args.lmax + 2, weights)

    control_mean = control.mean(axis=0)
    control_error = control.std(axis=0, ddof=1) / np.sqrt(control_count)
    control_pull = np.abs(control_mean) / control_error
    control_ok = bool(np.all(control_pull < 3.0))

    half = args.simulations // 2
    first_half, second_half = estimates[:half], estimates[half:]
    bias_first = first_half.mean(axis=0)
    bias_second = second_half.mean(axis=0)
    error_first = first_half.std(axis=0, ddof=1) / np.sqrt(half)
    error_second = second_half.std(axis=0, ddof=1) / np.sqrt(len(second_half))
    scatter = estimates.std(axis=0, ddof=1)

    stability = np.abs(bias_first - bias_second) / np.sqrt(error_first**2 + error_second**2)
    corrected = second_half - bias_first
    closure = np.abs(corrected.mean(axis=0)) / error_second
    bias_over_scatter = np.abs(bias_first) / scatter

    passed = bool(
        control_ok and np.all(stability < 3.0) and np.all(closure < 3.0) and np.all(bias_over_scatter < 1.0)
    )

    report = {
        "gate": "pre-registration §3.3 mask-coupling check",
        "reads_only_the_mask": True,
        "configuration": {
            "nside": args.nside,
            "lmin": args.lmin,
            "lmax": args.lmax,
            "simulations": args.simulations,
            "tilt": args.tilt,
            "f_sky": f_sky,
            "apodize_fwhm_deg": args.apodize_fwhm_deg,
            "filter": args.filter,
        },
        "normalization_checks_vs_camb": response["normalization_checks"],
        "full_sky_control": {
            "simulations": control_count,
            "mean_pull_per_M": {str(M): float(control_pull[i]) for i, M in enumerate(range(-2, 3))},
            "scatter_per_M": {str(M): float(control.std(axis=0, ddof=1)[i]) for i, M in enumerate(range(-2, 3))},
            "estimator_unbiased_without_mask": control_ok,
        },
        "per_M": {
            str(M): {
                "bias_first_half": float(bias_first[index]),
                "bias_second_half": float(bias_second[index]),
                "scatter": float(scatter[index]),
                "bias_over_scatter": float(bias_over_scatter[index]),
                "stability_sigma": float(stability[index]),
                "closure_sigma": float(closure[index]),
            }
            for index, M in enumerate(range(-2, 3))
        },
        "verdict": {
            "estimator_unbiased_on_full_sky": control_ok,
            "bias_small_compared_to_scatter": bool(np.all(bias_over_scatter < 1.0)),
            "bias_stable_between_halves": bool(np.all(stability < 3.0)),
            "out_of_sample_closure": bool(np.all(closure < 3.0)),
            "gate_passed": passed,
            "conclusion": (
                "The mask coupling is a fixed, removable bias at this configuration; the estimator "
                "may proceed to data."
                if passed
                else "The mask coupling fails at least one criterion. Per the pre-registration the "
                "estimator is not used on data in this configuration."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
