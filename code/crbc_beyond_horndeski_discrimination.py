"""Does the beyond-Horndeski sector *carry* the w(t) transition, or merely accommodate it?

Section 19 named the beyond-Horndeski completion as the most promising remaining route
to a covariant action, on the grounds that section 14 had confirmed such a completion
exists for the derived background. That reasoning does not survive checking.

The Ye-Piao construction is a *reconstruction*: it takes H(t) and a(t) as input and
solves algebraically for the Lagrangian functions. It does not predict H(t). An action
therefore exists for the derived background in exactly the sense that one exists for any
background fed to the algebra, so existence carries no information. What would make the
sector informative is a *selective* stability gate -- healthy coefficients for the
transition CRBC claims and not for others.

The comparison has to be controlled. Swapping p(eta) inside the closed-form ansatz
H = eta/(p(1+eta^2)) also changes H(t), because that ansatz fixes rho/rho_c = 1/(1+eta^2)
as well; a verdict obtained that way is not attributable to w(t). Here the comparison
backgrounds are built by integrating the same equations the derived background solves,
with only w(rho) replaced, and mapped to the realization's units by the same rule.

Result at extent 200, where the transition lies inside the grid for every case:

    (a) derived background            6.19% of candidates viable
    (c) reversed transition w: 0 -> 4.33    0.21% viable
    (d) constant w = 4.33                   0    viable
    (e) constant w = 0                      8.01% viable

The gate is not vacuous -- constant w = 4.33 fails everywhere tested -- but it does not
select the claimed transition. A background with *no transition at all*, constant w = 0,
passes more easily than the derived background, and the reversed transition passes too.
Passing the gate is therefore not evidence for the transition asserted, and the
beyond-Horndeski sector cannot be its origin.

What does discriminate is the shear budget, which is separate: constant w = 0 has
1 - 3/p = -1 and grows shear throughout, so it is excluded by anisotropy rather than by
stability. The selection comes from the shear requirement, not from this sector.

Caveat on case (b). The prescribed switch is a function of density alone and is therefore
reversible: on the expanding branch w returns to its low-density value, whereas the
derived background's conversion is irreversible and ends at p = 3/2. Case (b) is not a
clean forward control and nothing here rests on it; the conclusion rests on (c) and (e).

Units: alpha = 1 (the realization's units). No observational data is used.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from crbc_derived_background_gate import (as_background_dict, derived_background_on_grid,
                                          scan)


def prescribed_w_background(eta_grid, w_of_rho, w_reference, rho_initial, rho_stop,
                            rho_c, t_max, rtol, atol, device):
    """Integrate the CRBC equations with w prescribed as a function of density.

    The comparison must be controlled: the closed-form ansatz H = eta/(p(1+eta^2)) fixes
    rho/rho_c = 1/(1+eta^2) as well, so swapping its p(eta) changes H(t) too and the
    resulting gate verdict is not attributable to w(t). Here rho, H and a are obtained by
    integrating the same equations the derived background solves, with only w(rho)
    replaced, and mapped to the realization's units by the same rule.
    """
    rho_c_gate = 4.0 / (3.0 * (1.0 + w_reference) ** 2)
    scale_h = np.sqrt(rho_c_gate)

    def rhs(_t, y):
        rho, hubble, _ln_a = y
        rho = max(rho, 0.0)
        w = w_of_rho(rho / rho_c)
        return [
            -3.0 * hubble * (1.0 + w) * rho,
            -0.5 * rho * (1.0 + w) * (1.0 - 2.0 * rho / rho_c),
            hubble,
        ]

    def stop(_t, y):
        return (y[0] - rho_stop) if y[1] > 0.0 else 1.0
    stop.terminal = True
    stop.direction = -1.0

    hubble_0 = -np.sqrt(rho_initial * (1.0 - rho_initial / rho_c) / 3.0)
    solution = solve_ivp(rhs, (0.0, t_max), [rho_initial, hubble_0, 0.0],
                         method="Radau", rtol=rtol, atol=atol, events=stop,
                         max_step=t_max / 4000.0, dense_output=True)
    if solution.status < 0:
        raise RuntimeError("prescribed-w integration failed")

    hubble_all = solution.y[1]
    crossing = np.where(np.diff(np.sign(hubble_all)) > 0)[0]
    if crossing.size == 0:
        raise RuntimeError("no bounce")
    t_bounce = brentq(lambda t: solution.sol(t)[1], solution.t[crossing[0]],
                      solution.t[crossing[0] + 1], xtol=1e-14, rtol=1e-15)
    ln_a_bounce = solution.sol(t_bounce)[2]

    wanted = t_bounce + eta_grid * scale_h
    if wanted[0] < solution.t[0] or wanted[-1] > solution.t[-1]:
        raise RuntimeError("trajectory does not cover the grid")
    rho, hubble, ln_a = solution.sol(wanted)
    w = np.array([w_of_rho(r / rho_c) for r in rho])

    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(hubble != 0.0, 0.5 / (hubble * scale_h), 0.0)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device=device)

    return {
        "H": tensor(hubble * scale_h),
        "a": tensor(np.exp(ln_a - ln_a_bounce)),
        "p": tensor(1.5 * (1.0 + w)),
        "inv_two_hubble": tensor(inverse),
        "rho_over_rhoc": tensor(rho / rho_c),
        "varying_p": True,
        "p_constant": float(np.max(1.5 * (1.0 + w))),
    }


def switch(w_low, w_high, rho_tr, sharpness):
    """w = w_low at rho << rho_tr, w_high at rho >> rho_tr. Coordinate-free in density."""
    def f(r):
        return w_high + (w_low - w_high) / (1.0 + (max(r, 1e-300) / rho_tr) ** sharpness)
    return f


def build_cases(eta, a, device):
    """Five backgrounds built the same way; only w(rho) differs."""
    common = dict(w_reference=a.w, rho_initial=a.rho_initial, rho_stop=a.rho_stop,
                  rho_c=1.0, t_max=a.t_max, rtol=a.rtol, atol=a.atol, device=device)
    w_ek, rho_tr, n = a.w, a.rho_h_over_rho_c, a.sharpness

    cases = [{
        "name": "(a) derived background (section 13)", "claimed": True,
        "background": as_background_dict(derived_background_on_grid(a, eta), device),
    }]
    for name, claimed, w_of_rho in [
        ("(b) prescribed forward  w: %.2f -> 0" % w_ek, True, switch(w_ek, 0.0, rho_tr, n)),
        ("(c) prescribed reversed w: 0 -> %.2f" % w_ek, False, switch(0.0, w_ek, rho_tr, n)),
        ("(d) constant w = %.2f (no transition)" % w_ek, False, lambda r: w_ek),
        ("(e) constant w = 0 (no transition)", False, lambda r: 0.0),
    ]:
        try:
            background = prescribed_w_background(eta, w_of_rho, **common)
        except RuntimeError as error:
            cases.append({"name": name, "claimed": claimed, "error": str(error)})
            continue
        cases.append({"name": name, "claimed": claimed, "background": background})

    for case in cases:
        if "background" in case:
            p = case["background"]["p"]
            case["p_start"], case["p_end"] = float(p[0]), float(p[-1])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w", type=float, default=4.3333333333333333)
    parser.add_argument("--gamma", type=float, default=26.0)
    parser.add_argument("--rho-h-over-rho-c", type=float, default=1e-2)
    parser.add_argument("--rho-initial", type=float, default=1.8e-10)
    parser.add_argument("--rho-stop", type=float, default=1e-5)
    parser.add_argument("--t-max", type=float, default=4e6)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)
    parser.add_argument("--sharpness", type=float, default=4.0,
                        help="exponent of the density switch in w(rho)")
    parser.add_argument("--extent", type=float, default=40.0)
    parser.add_argument("--scan-points", type=int, default=6001)
    parser.add_argument("--k1-max", type=float, default=25.0)
    parser.add_argument("--k2-max", type=float, default=15.0)
    parser.add_argument("--tau1-max", type=float, default=40.0)
    parser.add_argument("--tau2sq-max", type=float, default=40.0)
    parser.add_argument("--k1-grid", type=int, default=21)
    parser.add_argument("--k2-grid", type=int, default=21)
    parser.add_argument("--tau1-grid", type=int, default=13)
    parser.add_argument("--tau2sq-grid", type=int, default=13)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--require-subluminal", action="store_true", default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    eta = np.linspace(-args.extent, args.extent, args.scan_points)

    print("=" * 88)
    print("Does the stability gate discriminate between the claimed w(t) and others?")
    print("=" * 88)
    print("  The Ye-Piao construction is a reconstruction: it takes H(t) and returns")
    print("  coefficients. Existence of an action is therefore not in question. What is")
    print("  in question is whether the gate prefers the transition CRBC asserts.")
    print()

    cases = build_cases(eta, args, device)
    print(f"  {'background':<36s} {'p start':>8s} {'p end':>8s} {'viable':>9s}"
          f" {'fraction':>9s} {'best margin':>12s} {'claimed':>8s}")
    rows = []
    for case in cases:
        if "background" not in case:
            print(f"  {case['name']:<36s}   {case['error']}")
            rows.append({"name": case["name"], "claimed": case["claimed"], "error": case["error"]})
            continue
        result = scan(args, device, eta, case["background"])
        rows.append({
            "name": case["name"], "claimed": case["claimed"],
            "p_start": case["p_start"], "p_end": case["p_end"],
            "viable_subluminal": result["viable_subluminal"],
            "viable_subluminal_fraction": result["viable_subluminal_fraction"],
            "best_margin": result["best_margin"],
            "best_parameters": result["best_parameters"],
            "candidates": result["candidates"],
        })
        print(f"  {case['name']:<36s} {case['p_start']:8.3f} {case['p_end']:8.3f}"
              f" {result['viable_subluminal']:9d} {result['viable_subluminal_fraction']:9.2%}"
              f" {result['best_margin']:12.6f} {str(case['claimed']):>8s}")

    claimed = [r for r in rows if r["claimed"] and "error" not in r]
    others = [r for r in rows if not r["claimed"] and "error" not in r]
    claimed_pass = all(r["viable_subluminal"] > 0 for r in claimed) and bool(claimed)
    others_pass = [r for r in others if r["viable_subluminal"] > 0]
    discriminates = claimed_pass and not others_pass

    print()
    print(f"  claimed backgrounds passing : {sum(1 for r in claimed if r['viable_subluminal'] > 0)}/{len(claimed)}")
    print(f"  unclaimed backgrounds passing: {len(others_pass)}/{len(others)}")
    if discriminates:
        print("  The gate DISCRIMINATES: only the claimed transition admits a healthy completion.")
    elif others_pass:
        print("  The gate does NOT discriminate on direction. Backgrounds the model does not")
        print("  claim also admit ghost-free, gradient-stable, subluminal completions, so")
        print("  passing it is not by itself evidence for the transition asserted.")
        for r in others_pass:
            print(f"    {r['name']}  ({r['viable_subluminal_fraction']:.2%} viable)")
    else:
        print("  Inconclusive: a claimed background failed, so the comparison is not clean.")

    out = {"discriminates": bool(discriminates), "cases": rows}
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
