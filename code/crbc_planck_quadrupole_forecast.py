#!/usr/bin/env python3
"""Fisher forecast for the CRBC quadrupole template, used to fix the analysis before data.

CRBC_EFT_선정과_계수계약_kr.md §11 predicts a multiplicative quadrupolar modulation whose
*shape* is fixed and whose only free parameters are an overall amplitude and an axis:

    P(k, k_hat) = P_iso(k) [ 1 + g_*(k) (k_hat . n_hat)^2 ],
    g_*(k) = g_0 (k / k_0)^(-0.70),      k_0 = 0.05 Mpc^-1.

This script computes what Planck can say about g_0, using only the model, the fiducial
cosmology and an instrument noise model. It touches no data. Its purpose is to let the
pre-registration fix ell_max, the estimator and the decision rule on principled grounds
rather than after seeing the sky.

Formalism. With a_lm = 4 pi i^l int d^3k/(2pi)^3 R(k) Delta_l(k) Y*_lm(k_hat), expanding
mu^2 = 1/3 + (2/3) sqrt(4 pi/5) Y_20(k_hat) in the frame where n_hat is the z axis gives

    <a_lm a*_l'm'>_aniso = D_ll' * F_ll'(m) delta_mm',
    D_ll' = 4 pi int dlnk P_R(k) g_*(k) Delta_l(k) Delta_l'(k),
    F_ll'(m) = (2/3) sqrt(4pi/5) (-1)^m i^(l-l') sqrt(5(2l+1)(2l'+1)/(4pi))
               * ThreeJ(2,l,l';0,0,0) * ThreeJ(2,l,l';0,-m,m),

and the isotropic spectrum is validated first against CAMB's own C_l, since the whole
forecast rests on that normalization.

Only l' = l + 2 is used for the forecast. Those elements vanish identically for an
isotropic sky, so they carry the anisotropy signal with no cosmic-variance-limited
subtraction, unlike the l' = l diagonal which is degenerate with C_l itself.

The reported sigma(g_0) is a forecast, not a measurement. It assumes full sky; a mask of
sky fraction f_sky degrades it roughly as 1/sqrt(f_sky), which the report includes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIDUCIAL = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "tau": 0.0544, "As": 2.1e-9, "ns": 0.9649}
PIVOT = 0.05  # Mpc^-1
CRBC_TILT = -0.70  # d ln g_* / d ln k, from CRBC_EFT_선정과_계수계약_kr.md §11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=200)
    parser.add_argument("--tilt", type=float, default=CRBC_TILT)
    parser.add_argument("--noise-uk-arcmin", type=float, default=45.0, help="Planck-like temperature noise")
    parser.add_argument("--beam-arcmin", type=float, default=5.0)
    parser.add_argument("--f-sky", type=float, default=0.78, help="Planck common mask sky fraction")
    parser.add_argument("--accuracy-boost", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_planck_quadrupole_forecast.json"))
    return parser.parse_args()


def build_transfer(lmax: int, accuracy_boost: float) -> dict[str, np.ndarray]:
    import camb

    params = camb.set_params(lmax=lmax + 40, **FIDUCIAL)
    params.set_accuracy(AccuracyBoost=accuracy_boost, lSampleBoost=50)
    results = camb.get_results(params)
    spectra = results.get_cmb_power_spectra(params, CMB_unit=None, raw_cl=True)["total"][:, 0]
    data = results.get_cmb_transfer_data("scalar")
    return {
        "L": np.array(data.L, dtype=int),
        "k": np.array(data.q),
        "delta_T": np.array(data.delta_p_l_k)[0],
        "cl_camb": spectra,
    }


def main() -> None:
    args = parse_args()
    from sympy.physics.wigner import wigner_3j

    transfer = build_transfer(args.lmax, args.accuracy_boost)
    ells, k, delta = transfer["L"], transfer["k"], transfer["delta_T"]
    log_k = np.log(k)
    primordial = FIDUCIAL["As"] * (k / PIVOT) ** (FIDUCIAL["ns"] - 1.0)

    # Validate the normalization C_l = 4 pi int dlnk P_R Delta_l^2 against CAMB.
    index = {int(value): position for position, value in enumerate(ells)}
    checks = []
    for target in (2, 10, 50, 100):
        if target in index:
            computed = 4.0 * np.pi * np.trapezoid(primordial * delta[index[target]] ** 2, log_k)
            checks.append({"ell": target, "ratio_to_camb": float(computed / transfer["cl_camb"][target])})
    worst = max(abs(entry["ratio_to_camb"] - 1.0) for entry in checks)
    if worst > 5e-3:
        raise SystemExit(f"transfer normalization off by {worst:.3e}; refusing to forecast")

    # Signal and noise spectra.
    usable = [int(value) for value in ells if 2 <= value <= args.lmax]
    cl_signal = transfer["cl_camb"]
    beam_sigma = np.radians(args.beam_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    cmb_temperature = 2.7255e6  # muK
    noise_amplitude = (np.radians(args.noise_uk_arcmin / 60.0) / cmb_temperature) ** 2

    def total_cl(l: int) -> float:
        return cl_signal[l] + noise_amplitude * np.exp(l * (l + 1) * beam_sigma**2)

    # Response per unit g_0 for the l, l+2 off-diagonal.
    g_shape = (k / PIVOT) ** args.tilt
    fisher_cumulative = []
    fisher_total = 0.0
    prefactor = (2.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0)
    for l in usable:
        lp = l + 2
        if lp not in index or lp > args.lmax:
            continue
        response = 4.0 * np.pi * np.trapezoid(primordial * g_shape * delta[index[l]] * delta[index[lp]], log_k)
        parity = float(wigner_3j(2, l, lp, 0, 0, 0))
        geometry = np.sqrt(5.0 * (2 * l + 1) * (2 * lp + 1) / (4.0 * np.pi)) * parity
        # i^(l-l') = i^-2 = -1 for l' = l+2; the sign cancels in |.|^2.
        # 3j orthogonality at fixed m1: sum_m (2 l l'; 0 -m m)^2 = 1/(2*2+1) = 1/5 exactly,
        # verified against sympy to 3e-17 for l = 2..40.
        weight = (prefactor * geometry) ** 2 / 5.0
        fisher_total += weight * response**2 / (total_cl(l) * total_cl(lp))
        fisher_cumulative.append({"ell": l, "cumulative_sigma_g0_full_sky": float(fisher_total**-0.5)})

    sigma_full_sky = fisher_total**-0.5
    sigma_masked = sigma_full_sky / np.sqrt(args.f_sky)

    # Where does the information sit? Report the ell below which a given fraction is reached.
    milestones = {}
    for fraction in (0.5, 0.9, 0.99):
        target = sigma_full_sky / np.sqrt(fraction)
        reached = [entry["ell"] for entry in fisher_cumulative if entry["cumulative_sigma_g0_full_sky"] <= target]
        milestones[f"ell_for_{int(fraction * 100)}pc_of_information"] = min(reached) if reached else None

    report = {
        "purpose": "fix the analysis configuration before looking at data",
        "template": {
            "form": "P(k, k_hat) = P_iso(k) [1 + g_*(k) (k_hat . n_hat)^2]",
            "g_star": "g_0 (k/k_0)^tilt",
            "tilt": args.tilt,
            "pivot_mpc_inverse": PIVOT,
            "source": "CRBC_EFT_선정과_계수계약_kr.md §11",
        },
        "fiducial_cosmology": FIDUCIAL,
        "normalization_checks": checks,
        "worst_normalization_error": worst,
        "instrument": {
            "noise_uk_arcmin": args.noise_uk_arcmin,
            "beam_arcmin": args.beam_arcmin,
            "f_sky": args.f_sky,
        },
        "forecast": {
            "sigma_g0_full_sky": float(sigma_full_sky),
            "sigma_g0_with_mask": float(sigma_masked),
            "lmax_used": args.lmax,
            "information_milestones": milestones,
            "planck_published_sigma_g_star": 0.016,
            "note": "the published 0.016 assumes scale-independent g_*; this forecast is for the "
            "CRBC tilt and is not directly comparable, which is why the reanalysis is needed",
        },
        "cumulative": fisher_cumulative,
        "estimator": "l, l+2 off-diagonal of <a_lm a*_l'm'>, which vanishes identically for an "
        "isotropic sky",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "cumulative"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
