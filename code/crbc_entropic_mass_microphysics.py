"""Where could the entropic mass lambda ~ 104 come from?

Section 9 (and its re-test in section 16) needs a tachyonic entropy mass
m_eff^2 = -lambda H^2 with lambda tuned to 0.29%. This script asks what supplies it.

First, lambda is not a free parameter. Demanding an exactly scale-invariant entropy
spectrum, nu = 3/2 in nu^2 = 1/4 + beta(beta-1) + lambda beta^2 with beta = 1/(p_i - 1),
gives a closed form,

    lambda_SI = [2 - beta(beta-1)] / beta^2 = 2(p_i - 1)^2 + p_i - 2 ,

verified against the numeric solver to machine precision. Choosing the ekpyrotic
steepness p_i therefore *fixes* lambda. The tuning is not "which lambda to pick" but
"does the actual m_eff^2 land on that value to 0.29%".

Second, the ansatz m_eff^2 proportional to H^2 is not arbitrary -- it is what a
non-minimal coupling gives for free. In ekpyrotic contraction a ~ |t|^{1/p}, so
H = 1/(pt), Hdot = -p H^2 and

    R = 6(Hdot + 2H^2) = 6(2 - p) H^2 ,

which is negative for p > 2. A coupling xi R s^2 therefore produces a mass that is
automatically tachyonic and automatically tracks H^2. The form is derived.

The value is not, and the p-scaling is why. Every natural source is *linear* in p,

    |R|/H^2 = 6(p-2),     sigma_dot^2/(H^2 M_P^2) = 2p,     |V|/(H^2 M_P^2) = p-3 ,

while the requirement lambda_SI ~ 2p^2 is *quadratic*. The coefficient needed therefore
grows linearly with p: steeper ekpyrotic contraction, which is what shear suppression
wants, makes the entropy sector less natural in exact proportion.

This script quantifies that trade-off and the resulting joint constraint with the shear
budget of section 13.

Units: 8 pi G = 1 (M_P = 1). No observational data is used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from crbc_entropic_mechanism import PLANCK_NS, PLANCK_NS_SIGMA, lambda_for_tilt

# Section 17: the conversion completes only below this threshold, and the shear budget
# improves as the threshold rises, so this is the optimum for both.
RHO_TRANSITION_OVER_RHO_C = 0.3


def lambda_scale_invariant(p_initial):
    """Closed form of the lambda that makes the entropy spectrum exactly scale invariant."""
    return 2.0 * (p_initial - 1.0) ** 2 + p_initial - 2.0


def natural_sources(p_initial):
    """|m_eff^2|/H^2 available from each natural source, in the ekpyrotic scaling solution.

    a ~ |t|^{1/p}, H = 1/(pt), Hdot = -p H^2, so with 8 pi G = 1:
        R          = 6(2-p) H^2                (negative for p > 2: tachyonic for xi > 0)
        sigma_dot^2 = (rho + P) = 2p H^2       (the adiabatic kinetic energy)
        V          = (rho - P)/2 = (3-p) H^2   (negative for p > 3)
    """
    return {
        "ricci_scalar": 6.0 * abs(2.0 - p_initial),
        "adiabatic_kinetic": 2.0 * p_initial,
        "potential_depth": abs(3.0 - p_initial),
    }


def required_coefficients(p_initial):
    """The dimensionless coefficient each source needs in order to supply lambda_SI."""
    lam = lambda_scale_invariant(p_initial)
    sources = natural_sources(p_initial)
    return {
        "lambda_SI": lam,
        "xi_non_minimal": lam / sources["ricci_scalar"],
        "field_space_curvature": lam / sources["adiabatic_kinetic"],
        "potential_transverse": lam / sources["potential_depth"],
    }


def shear_start_density(p_initial, target_budget, rho_tr_over_rho_c, p_final=1.5):
    """rho_i/rho_c needed to reach a given net anisotropy budget.

    Stage exponents: a = 1 - 3/p_i (ekpyrotic, suppresses), b = 1 - 3/p_f = -1 (soft,
    gives it back). The net is  -a ln(rho_tr/rho_i) - b ln(rho_c/rho_tr).
    """
    a = 1.0 - 3.0 / p_initial
    b = 1.0 - 3.0 / p_final
    if a <= 0.0:
        return None
    soft = -b * np.log(1.0 / rho_tr_over_rho_c)
    # target = -a ln(rho_tr/rho_i) + soft  ->  ln(rho_tr/rho_i) = (soft - target)/a
    return float(rho_tr_over_rho_c * np.exp(-(soft - target_budget) / a))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-values", type=float, nargs="+",
                        default=[4, 5, 6, 8, 10, 16, 30])
    parser.add_argument("--shear-target", type=float, default=-6.55,
                        help="net anisotropy budget to match (the section 7 value)")
    parser.add_argument("--rho-tr", type=float, default=RHO_TRANSITION_OVER_RHO_C)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    out = {"shear_target": args.shear_target, "rho_tr_over_rho_c": args.rho_tr}

    print("=" * 82)
    print("Part A -- lambda is fixed by p_i, not chosen")
    print("=" * 82)
    print("  lambda_SI = [2 - beta(beta-1)]/beta^2 = 2(p-1)^2 + p - 2,   beta = 1/(p-1)")
    print()
    print(f"  {'p_i':>5s} {'beta':>10s} {'numeric':>12s} {'closed form':>12s} {'difference':>12s}")
    rows = []
    for p in args.p_values:
        beta = 1.0 / (p - 1.0)
        numeric = lambda_for_tilt(beta, 1.0)
        closed = lambda_scale_invariant(p)
        rows.append({"p": p, "beta": beta, "numeric": numeric, "closed": closed})
        print(f"  {p:5.0f} {beta:10.6f} {numeric:12.6f} {closed:12.1f} {numeric - closed:12.2e}")
    out["closed_form_check"] = rows

    print()
    print("=" * 82)
    print("Part B -- the form is derived, the value is not")
    print("=" * 82)
    print("  A coupling xi R s^2 gives m_eff^2 = xi R = 6 xi (2-p) H^2: automatically")
    print("  tachyonic for p > 2 and automatically proportional to H^2. The ansatz is free.")
    print()
    print(f"  {'p_i':>5s} {'lambda_SI':>10s} {'|R|/H^2':>9s} {'sig^2/H^2':>10s} {'|V|/H^2':>9s}"
          f" {'xi':>8s} {'|R_fs|':>8s} {'kappa':>8s}")
    coefficient_rows = []
    for p in args.p_values:
        sources = natural_sources(p)
        need = required_coefficients(p)
        coefficient_rows.append({"p": p, **need})
        print(f"  {p:5.0f} {need['lambda_SI']:10.1f} {sources['ricci_scalar']:9.1f}"
              f" {sources['adiabatic_kinetic']:10.1f} {sources['potential_depth']:9.1f}"
              f" {need['xi_non_minimal']:8.3f} {need['field_space_curvature']:8.3f}"
              f" {need['potential_transverse']:8.2f}")
    out["required_coefficients"] = coefficient_rows
    print()
    print("  Requirement ~ 2p^2 (quadratic); every source ~ p (linear). The needed")
    print("  coefficient therefore grows linearly in p -- steeper contraction, which is")
    print("  what shear suppression wants, costs naturalness in exact proportion.")

    print()
    print("=" * 82)
    print("Part C -- the trade-off against the shear budget")
    print("=" * 82)
    print(f"  net anisotropy target {args.shear_target}, transition at rho_tr = {args.rho_tr} rho_c")
    print()
    print(f"  {'p_i':>5s} {'shear exp.':>11s} {'xi needed':>10s} {'xi tolerance':>13s}"
          f" {'rho_i/rho_c needed':>19s}")
    trade = []
    for p in args.p_values:
        need = required_coefficients(p)
        exponent = 1.0 - 3.0 / p
        rho_i = shear_start_density(p, args.shear_target, args.rho_tr)
        # The 0.29% tuning on lambda transfers directly to whatever coefficient supplies it.
        beta = 1.0 / (p - 1.0)
        nu = (3.0 - (PLANCK_NS - 1.0)) / 2.0
        tolerance = PLANCK_NS_SIGMA * nu / beta**2
        fractional = tolerance / lambda_for_tilt(beta, PLANCK_NS)
        trade.append({"p": p, "shear_exponent": exponent, "xi": need["xi_non_minimal"],
                      "xi_fractional_tolerance": fractional, "rho_i_over_rho_c": rho_i})
        print(f"  {p:5.0f} {exponent:11.4f} {need['xi_non_minimal']:10.3f}"
              f" {fractional:12.4%} {rho_i:19.3e}")
    out["trade_off"] = trade

    print()
    print("  Lowering p makes xi more ordinary but forces the contraction to start far")
    print("  earlier; raising p relaxes the start but drives xi up without bound.")
    print("  Neither end produces a value with an independent reason to be what it is.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
