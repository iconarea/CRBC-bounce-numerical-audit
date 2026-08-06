"""A derivative coupling for the equation-of-state transition: a candidate, not a completion.

Section 19 tested a conformal coupling A(phi) and found two obstructions that block each
other:

  1. p_phi = K - V >= -V diverges with an unbounded potential, so draining rho_phi does
     not drain p_phi and w_eff never falls;
  2. bounding V below caps p_phi but makes phi turn around, after which exp(beta phi)
     stops growing and the transfer switches off.

A coupling to the kinetic term rather than to the field value is aimed exactly at the
second. Take the matter sector to couple through A(X) with X = phi_dot^2,

    S = int d^4x sqrt(-g)[R/2 - X/2 - V(phi)] + S_m[psi, A^2(X) g_mn],
    ln A = (nu/2) X ,

so that d ln A/dt = nu phi_dot phi_ddot. The matter conservation equation and the
scalar equation follow from total energy conservation:

    rho_2' + 3H rho_2 = nu phi_dot phi_ddot rho_2 ,
    phi_ddot (1 + nu rho_2) = -3H phi_dot - V_phi .

Two things are then different from the A(phi) case. The transfer is driven by
phi_dot phi_ddot, which is positive whenever the kinetic energy grows -- and it grows
throughout contraction whether or not phi turns around, so obstruction 2 is evaded by
construction. And the X-dependence appears as an effective kinetic normalisation
(1 + nu rho_2), which is positive for nu > 0: no ghost from this term.

Obstruction 1 still requires V bounded below, which is now allowed, since the transfer
no longer depends on phi continuing in one direction. With V bounded and the field
kinating, p_phi -> rho_phi, so w_eff = p_phi/(rho_phi + rho_2) -> 0 provided the coupling
can drive rho_2 >> rho_phi. Whether it can is what this script measures.

Caveat, and it is a large one. The equations below are effective background equations
*inspired by* that action, not obtained by varying it. Given Q, total energy conservation
fixes the phi equation uniquely and reproduces the form used here, so the system is
internally consistent -- but the coupling term one would expect from varying S_m changes
the denominator from (1 + nu rho_2) to (1 + nu rho_2 + nu^2 rho_2 X), and near the bounce
nu X ~ 55, so the two differ by about fifty-five fold. X depends on the metric and on
grad phi, so varying S_m[psi, A^2(X) g] produces terms beyond a simple two-fluid transfer
which are not pinned down here. The stability gate is therefore withheld: with the
background unsettled at that magnitude, Q_s, c_s^2, Q_T and c_T^2 would not say what they
belong to. This is a candidate realization -- a first viable derivative-coupling proxy --
not a covariant completion.

Units: 8 pi G = 1, rho_c = 1. No observational data is used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp


def evolve(nu, c, w_ek, rho_initial, rho_2_fraction, v_min, rho_c,
           t_max, rtol, atol):
    """Background evolution with the derivative coupling."""
    kinetic = rho_initial * (1.0 + w_ek) / 2.0
    potential_value = rho_initial * (1.0 - w_ek) / 2.0
    phi_dot_0 = -np.sqrt(2.0 * kinetic)
    v0 = 1.0
    phi_0 = -np.log(-potential_value / v0) / c
    v1 = 0.0 if v_min is None else v0 * v0 / (4.0 * v_min)
    rho_2_0 = rho_2_fraction * rho_initial
    total = rho_initial + rho_2_0
    hubble_0 = -np.sqrt(total * (1.0 - total / rho_c) / 3.0)

    def potential(phi):
        u = np.exp(-c * phi)
        return v1 * u * u - v0 * u

    def potential_gradient(phi):
        u = np.exp(-c * phi)
        return -2.0 * c * v1 * u * u + c * v0 * u

    def rhs(_t, y):
        phi, phi_dot, rho_2, hubble, _ln_a = y
        rho_2 = max(rho_2, 0.0)
        normalisation = 1.0 + nu * rho_2          # effective kinetic normalisation
        phi_ddot = (-3.0 * hubble * phi_dot - potential_gradient(phi)) / normalisation
        rho = 0.5 * phi_dot * phi_dot + potential(phi) + rho_2
        pressure = 0.5 * phi_dot * phi_dot - potential(phi)
        return [
            phi_dot,
            phi_ddot,
            -3.0 * hubble * rho_2 + nu * phi_dot * phi_ddot * rho_2,
            -0.5 * (rho + pressure) * (1.0 - 2.0 * rho / rho_c),
            hubble,
        ]

    def stop(_t, y):
        return y[3] - 0.6 * np.sqrt(rho_c / 12.0)
    stop.terminal = True
    stop.direction = 1.0

    solution = solve_ivp(
        rhs, (0.0, t_max), [phi_0, phi_dot_0, rho_2_0, hubble_0, 0.0],
        method="Radau", rtol=rtol, atol=atol, events=stop,
        max_step=t_max / 4000.0, dense_output=True,
    )

    t = solution.t
    phi, phi_dot, rho_2, hubble, ln_a = solution.y
    v = potential(phi)
    kinetic_density = 0.5 * phi_dot * phi_dot
    rho_phi = kinetic_density + v
    rho = rho_phi + rho_2
    pressure = kinetic_density - v
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(rho) > 1e-300, pressure / rho, np.nan)

    constraint = np.abs(hubble**2 - rho * (1.0 - rho / rho_c) / 3.0)
    scale = np.maximum(np.abs(rho) / 3.0, 1e-300)

    return {
        "t": t, "phi": phi, "phi_dot": phi_dot, "rho_2": rho_2, "rho_phi": rho_phi,
        "rho": rho, "H": hubble, "ln_a": ln_a, "w": w,
        "p_exponent": 1.5 * (1.0 + w),
        "f_ekpyrotic": rho_phi / rho,
        "kinetic_normalisation": 1.0 + nu * rho_2,
        "ln_A": 0.5 * nu * phi_dot * phi_dot,
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
    parser.add_argument("--nus", type=float, nargs="+",
                        default=[0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    parser.add_argument("--v-mins", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--t-max", type=float, default=4e6)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    p_initial = 1.5 * (1.0 + args.w_ekpyrotic)
    c = np.sqrt(2.0 * p_initial)

    print("=" * 86)
    print("Derivative coupling  ln A = (nu/2) X,  X = phi_dot^2")
    print("=" * 86)
    print(f"  p_i = {p_initial:.4f},  c = {c:.4f}")
    print("  transfer  rho_2' + 3H rho_2 = nu phi_dot phi_ddot rho_2")
    print("  scalar    phi_ddot (1 + nu rho_2) = -3H phi_dot - V_phi")
    print("  the driver phi_dot phi_ddot is positive whenever the kinetic energy grows,")
    print("  so it does not switch off when phi turns around -- unlike A(phi).")
    print()
    print(f"  {'|V_min|/rho_c':>13s} {'nu':>8s} {'p_final':>9s} {'w_final':>11s}"
          f" {'f_ek':>10s} {'shear':>8s} {'max 1+nu rho_2':>15s} {'max ln A':>9s}")

    rows = []
    for v_min in args.v_mins:
        for nu in args.nus:
            run = evolve(nu, c, args.w_ekpyrotic, args.rho_initial, args.rho_2_fraction,
                         v_min, 1.0, args.t_max, args.rtol, args.atol)
            if not run["success"] or not run["reached_bounce"]:
                print(f"  {v_min:13.4g} {nu:8.1f}   did not reach the bounce")
                continue
            index = int(np.argmax(run["rho"]))
            entry = {
                "v_min_over_rho_c": v_min, "nu": nu,
                "p_final": float(run["p_exponent"][index]),
                "w_final": float(run["w"][index]),
                "f_ekpyrotic_at_bounce": float(run["f_ekpyrotic"][index]),
                "shear_budget": shear_budget(run),
                "min_kinetic_normalisation": float(np.min(run["kinetic_normalisation"])),
                "max_kinetic_normalisation": float(np.max(run["kinetic_normalisation"])),
                "max_ln_A": float(np.max(run["ln_A"])),
                "constraint_max_relative": float(np.max(run["constraint_relative"])),
            }
            rows.append(entry)
            print(f"  {v_min:13.4g} {nu:8.1f} {entry['p_final']:9.5f}"
                  f" {entry['w_final']:11.3e} {entry['f_ekpyrotic_at_bounce']:10.3e}"
                  f" {entry['shear_budget']:8.3f} {entry['max_kinetic_normalisation']:15.4g}"
                  f" {entry['max_ln_A']:9.2f}")

    successes = [r for r in rows if r["p_final"] < 1.55 and r["f_ekpyrotic_at_bounce"] < 0.1]
    print()
    if successes:
        best = min(successes, key=lambda r: abs(r["p_final"] - 1.5))
        print(f"  DESCENT ACHIEVED: p_final -> 3/2 with the pressureless sector dominating.")
        print(f"  best: |V_min|/rho_c = {best['v_min_over_rho_c']:g}, nu = {best['nu']:g}, "
              f"p_final = {best['p_final']:.5f}, f_ek = {best['f_ekpyrotic_at_bounce']:.3e}, "
              f"shear = {best['shear_budget']:.3f}")
    else:
        print("  No descent: no combination reaches p_final -> 3/2 with rho_2 dominating.")

    out = {"p_initial": p_initial, "c": c, "runs": rows,
           "descent_achieved": bool(successes)}
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
