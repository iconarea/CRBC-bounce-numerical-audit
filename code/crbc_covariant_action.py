"""A covariant action for the equation-of-state transition: a candidate, and why it fails.

Sections 13 and 17 specified the transfer by hand, writing
rho_1' + 3H(1+w_1)rho_1 = -Gamma rho_1 with Gamma given and a density threshold rho_tr
at which it switches on. That is a phenomenological source term, not an action, which is
why the draft named a covariant action as the single thing most needed. This script
supplies the natural candidate, tests it, and reports two obstructions.

The candidate is standard coupled quintessence: an ekpyrotic scalar plus a pressureless
sector coupled to it conformally,

    S = int d^4x sqrt(-g) [ R/2 - (1/2)(grad phi)^2 - V(phi) ] + S_m[psi, A^2(phi) g_mn],
    A(phi) = exp(beta phi),

giving, with 8 pi G = 1,

    phi'' + 3H phi' + V_phi = -beta rho_2 ,
    rho_2' + 3H rho_2       = +beta phi' rho_2 ,

whose sum conserves the total energy identically.

What works. The transfer rate is *derived* rather than posited: Gamma_eff = beta phi',
and in the ekpyrotic scaling solution phi'^2 = 2 p rho/3, so Gamma_eff is proportional
to sqrt(rho), i.e. to H. The Hubble scaling of section 13 -- which section 17 called a
gravitational ansatz -- is exactly what a conformal coupling produces. Section 17's
exclusion is therefore narrower than it looked: it rules out a *thermal* origin for the
rate, not a covariant one. No density threshold is needed either, since rho_2 grows
relative to rho_phi whenever beta < (c/2)(3/p - 2) = -3.25 for the canonical p = 8.

What does not work, in two provable steps.

Obstruction 1 -- an unbounded potential. With V = -V_0 exp(-c phi) the field rolls to
-infinity and the coupling does make rho_2 dominate the *energy*, but p_phi = K - V >= -V
diverges with it. Draining rho_phi = K + V does not drain p_phi: at beta = -8 the
pressureless sector holds 94% of the energy while w_eff is still 9.16. This is precisely
what the two-fluid parametrisation of section 13 hid, because writing p_1 = w_1 rho_1
makes suppressing rho_1 suppress p_1 automatically. A field does not obey that.

Obstruction 2 -- bounding the potential. Adding V_1 exp(-2 c phi) caps p_phi - rho_phi
at 2|V_min|, which is necessary for w_eff to fall. But then the field passes the minimum
and turns around, exp(beta phi) stops growing with beta < 0, the transfer reverses, and
the scalar kinates: every combination tested lands on w = 1 exactly, with the
pressureless sector never taking over.

The two are not independently fixable. p_phi >= -V forces V bounded below for w_eff to
fall, and V bounded below forces phi to turn around, which switches off the very coupling
that was to drive the transfer.

Scope of the negative result. This tests one class: a canonical scalar conformally
coupled to a pressureless sector, with unbounded and bounded exponential potentials.
Not tested: derivative or disformal couplings, non-canonical kinetic terms, more than
two fields, or the possibility that the beyond-Horndeski sector itself carries the
transition. No claim is made about those.

Units: 8 pi G = 1, rho_c = 1. No observational data is used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp


def coupling_threshold(p_initial, c):
    """beta below which the pressureless sector overtakes the ekpyrotic one."""
    return 0.5 * c * (3.0 / p_initial - 2.0)


def evolve(beta, c, w_ek, rho_initial, rho_2_fraction, rho_c, t_max, rtol, atol,
           rho_stop_expanding=None, v_min=None):
    """Integrate the coupled system from the ekpyrotic scaling attractor."""
    # Scaling-solution initial data for the scalar alone.
    kinetic = rho_initial * (1.0 + w_ek) / 2.0
    potential_value = rho_initial * (1.0 - w_ek) / 2.0
    phi_dot = -np.sqrt(2.0 * kinetic)          # rolling toward -infinity
    v0 = 1.0
    phi_0 = -np.log(-potential_value / v0) / c
    rho_2_0 = rho_2_fraction * rho_initial
    total = rho_initial + rho_2_0
    hubble_0 = -np.sqrt(total * (1.0 - total / rho_c) / 3.0)

    # A pure negative exponential is unbounded below, and then p_phi = K - V diverges
    # even as rho_phi = K + V is diluted away: draining the field does not drain its
    # pressure. Bounding V below with a V_1 exp(-2 c phi) term caps p_phi - rho_phi
    # at 2|V_min|, which is what makes the descent of w_eff possible at all.
    v1 = 0.0 if v_min is None else v0 * v0 / (4.0 * v_min)

    def potential(phi):
        u = np.exp(-c * phi)
        return v1 * u * u - v0 * u

    def potential_gradient(phi):
        u = np.exp(-c * phi)
        return -2.0 * c * v1 * u * u + c * v0 * u

    def rhs(_t, y):
        phi, phi_dot_, rho_2, hubble, _ln_a = y
        rho_2 = max(rho_2, 0.0)
        rho = 0.5 * phi_dot_ * phi_dot_ + potential(phi) + rho_2
        pressure = 0.5 * phi_dot_ * phi_dot_ - potential(phi)
        return [
            phi_dot_,
            -3.0 * hubble * phi_dot_ - potential_gradient(phi) - beta * rho_2,
            -3.0 * hubble * rho_2 + beta * phi_dot_ * rho_2,
            -0.5 * (rho + pressure) * (1.0 - 2.0 * rho / rho_c),
            hubble,
        ]

    if rho_stop_expanding is None:
        def stop(_t, y):
            return y[3] - 0.6 * np.sqrt(rho_c / 12.0)
        stop.direction = 1.0
    else:
        def stop(_t, y):
            rho = 0.5 * y[1] ** 2 + potential(y[0]) + max(y[2], 0.0)
            return rho - rho_stop_expanding if y[3] > 0.0 else 1.0
        stop.direction = -1.0
    stop.terminal = True

    solution = solve_ivp(
        rhs, (0.0, t_max), [phi_0, phi_dot, rho_2_0, hubble_0, 0.0],
        method="Radau", rtol=rtol, atol=atol, events=stop,
        max_step=t_max / 4000.0, dense_output=True,
    )

    t = solution.t
    phi, phi_dot_, rho_2, hubble, ln_a = solution.y
    v = potential(phi)
    kinetic_density = 0.5 * phi_dot_ * phi_dot_
    rho_phi = kinetic_density + v
    rho = rho_phi + rho_2
    pressure = kinetic_density - v
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(rho) > 1e-300, pressure / rho, np.nan)

    constraint = np.abs(hubble**2 - rho * (1.0 - rho / rho_c) / 3.0)
    scale = np.maximum(np.abs(rho) / 3.0, 1e-300)

    return {
        "t": t, "phi": phi, "phi_dot": phi_dot_, "rho_2": rho_2, "rho_phi": rho_phi,
        "rho": rho, "H": hubble, "ln_a": ln_a, "w": w,
        "p_exponent": 1.5 * (1.0 + w),
        "f_ekpyrotic": rho_phi / rho,
        "transfer_rate": beta * phi_dot_,
        "constraint_relative": constraint / scale,
        "success": bool(solution.status >= 0),
        "reached_bounce": bool(np.any(rho >= 0.999 * rho_c)),
    }


def shear_budget(run):
    contracting = run["H"] < 0.0
    budget = -6.0 * run["ln_a"][contracting] - np.log(run["rho"][contracting])
    return float(budget[-1] - budget[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w-ekpyrotic", type=float, default=4.3333333333333333)
    parser.add_argument("--rho-initial", type=float, default=1e-6)
    parser.add_argument("--rho-2-fraction", type=float, default=1e-8)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[-1.0, -2.0, -3.0, -3.25, -3.5, -4.0, -5.0, -6.5, -8.0, -12.0])
    parser.add_argument("--t-max", type=float, default=4e6)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)
    parser.add_argument("--v-mins", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    p_initial = 1.5 * (1.0 + args.w_ekpyrotic)
    c = np.sqrt(2.0 * p_initial)
    threshold = coupling_threshold(p_initial, c)

    print("=" * 84)
    print("Covariant action:  S = int sqrt(-g)[R/2 - (grad phi)^2/2 - V(phi)] + S_m[A^2(phi) g]")
    print("=" * 84)
    print(f"  V = -V_0 exp(-c phi),  A = exp(beta phi)")
    print(f"  p_i = {p_initial:.4f},  c = sqrt(2 p_i) = {c:.4f}")
    print(f"  analytic takeover condition  beta < (c/2)(3/p - 2) = {threshold:.4f}")
    print()
    print(f"  {'beta':>7s} {'p_final':>9s} {'w_final':>11s} {'f_ek at bounce':>15s}"
          f" {'shear':>8s} {'|constraint|':>13s} {'takeover':>9s}")

    rows = []
    for beta in args.betas:
        run = evolve(beta, c, args.w_ekpyrotic, args.rho_initial, args.rho_2_fraction,
                     1.0, args.t_max, args.rtol, args.atol)
        if not run["success"] or not run["reached_bounce"]:
            print(f"  {beta:7.2f}   did not reach the bounce")
            rows.append({"beta": beta, "failed": True})
            continue
        index = int(np.argmax(run["rho"]))
        entry = {
            "beta": beta,
            "p_final": float(run["p_exponent"][index]),
            "w_final": float(run["w"][index]),
            "f_ekpyrotic_at_bounce": float(run["f_ekpyrotic"][index]),
            "shear_budget": shear_budget(run),
            "constraint_max_relative": float(np.max(run["constraint_relative"])),
            "predicted_takeover": bool(beta < threshold),
        }
        rows.append(entry)
        print(f"  {beta:7.2f} {entry['p_final']:9.5f} {entry['w_final']:11.3e}"
              f" {entry['f_ekpyrotic_at_bounce']:15.3e} {entry['shear_budget']:8.3f}"
              f" {entry['constraint_max_relative']:13.2e} {str(entry['predicted_takeover']):>9s}")

    print()
    print("  Obstruction 1: rho_2 can dominate the energy while p_phi = K - V >= -V still")
    print("  diverges, so w_eff does not fall. Draining rho_phi does not drain p_phi.")

    print()
    print("=" * 84)
    print("Part B -- bounding the potential, which is necessary and also fatal")
    print("=" * 84)
    print(f"  {'|V_min|/rho_c':>13s} {'beta':>7s} {'p_final':>9s} {'w_final':>10s}"
          f" {'f_ek':>10s} {'shear':>8s}")
    bounded = []
    for v_min in args.v_mins:
        for beta in args.betas:
            run = evolve(beta, c, args.w_ekpyrotic, args.rho_initial, args.rho_2_fraction,
                         1.0, args.t_max, args.rtol, args.atol, v_min=v_min)
            if not run["success"] or not run["reached_bounce"]:
                continue
            index = int(np.argmax(run["rho"]))
            entry = {"v_min_over_rho_c": v_min, "beta": beta,
                     "p_final": float(run["p_exponent"][index]),
                     "w_final": float(run["w"][index]),
                     "f_ekpyrotic_at_bounce": float(run["f_ekpyrotic"][index]),
                     "shear_budget": shear_budget(run)}
            bounded.append(entry)
            print(f"  {v_min:13.4g} {beta:7.1f} {entry['p_final']:9.5f}"
                  f" {entry['w_final']:10.3e} {entry['f_ekpyrotic_at_bounce']:10.3e}"
                  f" {entry['shear_budget']:8.3f}")
    print()
    print("  Obstruction 2: the field passes the minimum and turns around, exp(beta phi)")
    print("  stops growing, the transfer reverses, and the scalar kinates to w = 1.")
    print("  Every combination lands there, with the pressureless sector never taking over.")

    out = {
        "p_initial": p_initial, "c": c, "analytic_beta_threshold": threshold,
        "unbounded_potential": rows, "bounded_potential": bounded,
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
