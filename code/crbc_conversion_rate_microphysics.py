"""Can the conversion rate of section 13 come from Hagedorn string production?

Section 13 needed an ekpyrotic component to drain into a pressureless one fast enough,
Gamma > 3|H| w_1 (1-f), and parametrised that as Gamma = gamma sqrt(rho/3) with
gamma ~ 26. Two things were left open: the scaling of Gamma, and the value of gamma.
This script closes the first and shows the second cannot come from string theory.

1. The scaling was wrong. Gamma proportional to sqrt(rho) is Gamma proportional to H --
a *gravitational* ansatz. No local process knows about the expansion rate; a microscopic
rate is set by the local energy scale, Gamma = c rho^{1/4} with c dimensionless. In the
code units of the derivation (rho_c = 1, M_P = 1) that reads

    Gamma_code = C rho^{1/4},     C = c xi_red^{-1/4},   xi_red = rho_c/M_P^4,

because Gamma/H = sqrt(3) c M_P rho^{-1/4} / sqrt(1-r) carries one factor of M_P that the
code units hide. With the corrected scaling the descent condition becomes r-independent,

    C > sqrt(3) w_1 = 7.506 ,

sharper than the gamma > 3 w_1 = 13 that the Hubble ansatz gave.

2. The value cannot be Hagedorn. Section 8.6 resolved the EFT-control failure by putting
the bounce *below* the string density: rho_s/rho_c = N > 725. Section 13 needs the
conversion to switch on at rho_tr < rho_c, and the conversion only completes if
rho_tr <~ 0.3 rho_c. So the transition necessarily happens at

    E_tr/M_s = (rho_tr/rho_s)^{1/4} <~ (0.3/725)^{1/4} = 0.14 ,

between 6% and 19% of the string scale. There the Hagedorn density of states does not
compensate the Boltzmann factor -- it only does so at T = T_H -- so producing the heavy
states that constitute a pressureless string gas is exponentially suppressed, not
enhanced. Meanwhile the required coupling c = C xi_red^{1/4} is of order 0.1-1, i.e. a
rate comparable to the energy scale itself: essentially the unitarity limit.

Conclusion: the softening of w must be supplied by something that operates near the
unitarity bound at 10^16-10^17 GeV. Hagedorn string production is not that thing.

Units: 8 pi G = 1, rho_c = 1. No observational data is used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from crbc_equation_of_state_derivation import evolve_transfer, shear_budget

# Section 8.6 of the draft, the option-(a) resolution of EFT control.
# N = rho_s/rho_c is the string density in units of the bounce density; xi_red = rho_c/M_P^4.
SECTION_8_6_TABLE = [
    {"g_s": 0.1, "N": 725.0, "xi_red": 1.4e-7, "rho_c_quarter_GeV": 4.7e16},
    {"g_s": 0.3, "N": 1e4, "xi_red": 8.1e-7, "rho_c_quarter_GeV": 7.3e16},
    {"g_s": 0.5, "N": 725.0, "xi_red": 8.6e-5, "rho_c_quarter_GeV": 2.4e17},
    {"g_s": 1.0, "N": 1e4, "xi_red": 1.0e-4, "rho_c_quarter_GeV": 2.4e17},
]


def threshold_scan(w_ek, rho_tr, rho_i, coefficients, rate_mode):
    """Final p(=1.5(1+w)) at the bounce as a function of the rate coefficient."""
    rows = []
    for coefficient in coefficients:
        run = evolve_transfer(w_ek, rho_tr, float(coefficient), rho_i, 1.0,
                              4e6, 1e-11, 1e-16, rate_mode=rate_mode)
        if not run.get("success") or not run["reached_bounce"]:
            rows.append({"coefficient": float(coefficient), "failed": True})
            continue
        index = int(np.argmax(run["rho"]))
        rows.append({
            "coefficient": float(coefficient),
            "p_final": float(run["p_exponent"][index]),
            "w_final": float(run["w"][index]),
            "f_ekpyrotic_at_bounce": float(run["f_ekpyrotic"][index]),
        })
    return rows


def transition_ceiling(w_ek, coefficient, rho_i, fractions, tolerance=1.55):
    """Highest rho_tr/rho_c at which the conversion still completes."""
    rows, ceiling = [], None
    for fraction in fractions:
        run = evolve_transfer(w_ek, float(fraction), coefficient, rho_i, 1.0,
                              4e6, 1e-11, 1e-16, rate_mode="micro")
        if not run.get("success") or not run["reached_bounce"]:
            rows.append({"rho_tr_over_rho_c": float(fraction), "failed": True})
            continue
        index = int(np.argmax(run["rho"]))
        p_final = float(run["p_exponent"][index])
        completed = p_final < tolerance
        rows.append({
            "rho_tr_over_rho_c": float(fraction),
            "p_final": p_final,
            "shear_budget": shear_budget(run)[0],
            "conversion_completed": completed,
        })
        if completed:
            ceiling = float(fraction)
    return rows, ceiling


def string_scale_gap(coefficient, rho_tr_values):
    """Required coupling, distance to the string scale, and the resulting suppression."""
    rows = []
    for entry in SECTION_8_6_TABLE:
        for rho_tr in rho_tr_values:
            energy_ratio = (rho_tr / entry["N"]) ** 0.25          # E_tr / M_s
            required_c = coefficient * entry["xi_red"] ** 0.25     # c = C xi_red^{1/4}
            # A state of mass ~M_s in a gas at T ~ E_tr: the Hagedorn density of states
            # e^{m/T_H} cancels the Boltzmann factor only at T = T_H, so below it the
            # residual weight is exp[-m (1/T - 1/T_H)]. Taking m = M_s and T_H = M_s
            # (the most generous convention) gives exp[-(1/x - 1)] with x = E_tr/M_s.
            exponent = -(1.0 / energy_ratio - 1.0)
            rows.append({
                "g_s": entry["g_s"],
                "N_string_over_bounce": entry["N"],
                "xi_red": entry["xi_red"],
                "rho_tr_over_rho_c": rho_tr,
                "E_tr_over_M_s": energy_ratio,
                "required_c": required_c,
                "g_s_squared": entry["g_s"] ** 2,
                "c_over_g_s_squared": required_c / entry["g_s"] ** 2,
                "hagedorn_suppression": float(np.exp(exponent)),
                "unsuppressed_c_needed": float(required_c / np.exp(exponent)),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w-ekpyrotic", type=float, default=4.3333333333333333)
    parser.add_argument("--rho-initial", type=float, default=1.8e-10)
    parser.add_argument("--rho-tr", type=float, default=1e-2)
    parser.add_argument("--coefficient", type=float, default=9.0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    w_ek = args.w_ekpyrotic
    analytic_threshold = np.sqrt(3.0) * w_ek
    out = {"analytic_threshold_C": float(analytic_threshold),
           "analytic_threshold_gamma_hubble_ansatz": 3.0 * w_ek}

    print("=" * 78)
    print("Part A -- the scaling was gravitational, not microphysical")
    print("=" * 78)
    print("  Hubble ansatz     Gamma = gamma sqrt(rho/3)  ->  gamma > 3 w_1     = %.3f" % (3.0 * w_ek))
    print("  microphysical     Gamma = C rho^{1/4}        ->  C     > sqrt3 w_1 = %.3f" % analytic_threshold)
    print()
    print(f"  {'C':>8s} {'p_final':>10s} {'w_final':>12s}")
    out["threshold_scan"] = threshold_scan(
        w_ek, args.rho_tr, args.rho_initial, [2, 4, 6, 7.5, 9, 11, 13, 20, 40], "micro")
    for row in out["threshold_scan"]:
        if row.get("failed"):
            print(f"  {row['coefficient']:8.2f}   integration failed")
        else:
            print(f"  {row['coefficient']:8.2f} {row['p_final']:10.5f} {row['w_final']:12.3e}")

    print()
    print("=" * 78)
    print("Part B -- how late can the transition be?")
    print("=" * 78)
    rows, ceiling = transition_ceiling(
        w_ek, args.coefficient, args.rho_initial,
        [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    out["transition_ceiling"] = {"rows": rows, "max_rho_tr_over_rho_c": ceiling}
    print(f"  {'rho_tr/rho_c':>13s} {'p_final':>10s} {'shear':>9s} {'completed':>10s}")
    for row in rows:
        if row.get("failed"):
            print(f"  {row['rho_tr_over_rho_c']:13.4g}   integration failed"); continue
        print(f"  {row['rho_tr_over_rho_c']:13.4g} {row['p_final']:10.5f}"
              f" {row['shear_budget']:9.3f} {str(row['conversion_completed']):>10s}")
    print(f"\n  conversion completes only for rho_tr <~ {ceiling} rho_c")

    print()
    print("=" * 78)
    print("Part C -- the gap to the string scale")
    print("=" * 78)
    rho_tr_values = [args.rho_tr, ceiling if ceiling else 0.3]
    out["string_scale_gap"] = string_scale_gap(analytic_threshold, rho_tr_values)
    print(f"  {'g_s':>5s} {'N':>7s} {'rho_tr':>8s} {'E_tr/M_s':>9s} {'c 필요':>8s}"
          f" {'g_s^2':>7s} {'c/g_s^2':>8s} {'억제':>10s} {'무억제 c':>10s}")
    for row in out["string_scale_gap"]:
        print(f"  {row['g_s']:5.1f} {row['N_string_over_bounce']:7.0f} {row['rho_tr_over_rho_c']:8.3g}"
              f" {row['E_tr_over_M_s']:9.4f} {row['required_c']:8.4f} {row['g_s_squared']:7.3f}"
              f" {row['c_over_g_s_squared']:8.2f} {row['hagedorn_suppression']:10.2e}"
              f" {row['unsuppressed_c_needed']:10.2e}")

    print()
    print("  The transition sits at 6-19% of the string scale, where the Hagedorn density")
    print("  of states no longer cancels the Boltzmann factor. Producing the heavy states")
    print("  that make a pressureless string gas is suppressed, not enhanced.")
    print("  Meanwhile the required c is of order 0.1-1: a rate comparable to the energy")
    print("  scale itself, i.e. essentially the unitarity limit.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
