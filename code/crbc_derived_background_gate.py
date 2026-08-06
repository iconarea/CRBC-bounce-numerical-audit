"""Re-run the EFT coefficient gate on the *derived* background.

Sections 7 and 8 of the EFT document gated a background whose equation of state was
supplied as a tanh profile p(eta). Section 13 replaced that with a derived background:
an ekpyrotic component draining into a pressureless one above a Hagedorn density,
which fixes p_f = 3/2 rather than the 2.4 that was assumed. Since c_s^2(+infinity)=p/3
depends on p_f directly, the earlier stability verdict does not transfer, and this
script asks the question again on the background the derivation actually produces.

Units. The derivation works with 8 pi G = 1 and rho_c = 1; the realization works in
alpha = 1 units, where 8 pi G = 1 and rho_c = 4/[3(1+w_ref)^2]. The two differ by a
pure choice of density unit, so the map is

    t_gate = (t - t_bounce)/sqrt(rho_c_gate),    H_gate = H sqrt(rho_c_gate),

with a normalised to 1 at the bounce. The check that this is right is that
max|H_gate| comes out at 1/(2 p_ref) to machine precision, which the closed-form
background satisfies identically.

The construction itself (Ye-Piao Eqs. 24-25) is algebraic in H(eta) and a(eta), so it
applies unchanged; only the 1/(2H) that regularises M_cal at the bounce had to be
generalised, since for the closed form it was written as p(1+eta^2)/(2 eta).

No observational data is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

import crbc_beyond_horndeski_realization as realization
from crbc_equation_of_state_derivation import evolve_transfer


def derived_background_on_grid(args, eta_grid):
    """Integrate the transfer model and resample it onto the gate's time grid."""
    rho_c_gate = 4.0 / (3.0 * (1.0 + args.w) ** 2)
    scale_h = np.sqrt(rho_c_gate)

    run = evolve_transfer(
        args.w, args.rho_h_over_rho_c, args.gamma, args.rho_initial, 1.0,
        args.t_max, args.rtol, args.atol, rho_stop_expanding=args.rho_stop,
    )
    if not run.get("success"):
        raise RuntimeError("transfer integration failed below the Hagedorn threshold")

    t_all = run["t"]
    below, above = run["segments"]

    # The bounce must sit exactly at eta = 0. The Ye-Piao construction centres the free
    # function c2 = exp(-eta^2/tau2^2) on the bounce, and M_cal = (1-c2)/(2H) - ... is
    # regular there only because both factors vanish at the same point. Taking the
    # bounce from the nearest sampled maximum of rho leaves an offset of order the
    # solver's step, at which (1-c2) no longer vanishes where H does -- which turns the
    # removable 0/0 into an actual pole. Locate H = 0 by root-finding instead.
    bounce_index = int(np.argmax(run["rho"]))
    hubble_all = run["H"]
    left = np.searchsorted(above.t, t_all[max(bounce_index - 2, 0)])
    bracket_lo = above.t[max(left - 1, 0)]
    bracket_hi = above.t[min(left + 2, above.t.size - 1)]
    if not (above.sol(bracket_lo)[2] < 0.0 < above.sol(bracket_hi)[2]):
        sign_change = np.where(np.diff(np.sign(hubble_all)) > 0)[0]
        if sign_change.size == 0:
            raise RuntimeError("no bounce (H never crosses zero upward)")
        bracket_lo, bracket_hi = t_all[sign_change[0]], t_all[sign_change[0] + 1]
    t_bounce = brentq(lambda t: above.sol(t)[2], bracket_lo, bracket_hi, xtol=1e-14, rtol=1e-15)
    ln_a_bounce = above.sol(t_bounce)[3]

    # Gate time -> derivation time.
    t_wanted = t_bounce + eta_grid * scale_h
    span_lo, span_hi = t_all[0], t_all[-1]
    if t_wanted[0] < span_lo or t_wanted[-1] > span_hi:
        covered_lo = (span_lo - t_bounce) / scale_h
        covered_hi = (span_hi - t_bounce) / scale_h
        raise RuntimeError(
            f"trajectory covers eta in [{covered_lo:.2f}, {covered_hi:.2f}], "
            f"grid needs [{eta_grid[0]:.2f}, {eta_grid[-1]:.2f}]"
        )

    # Dense output of whichever segment each sample falls in.
    split = below.t[-1]
    state = np.empty((4, t_wanted.size))
    first = t_wanted <= split
    if np.any(first):
        state[:, first] = below.sol(t_wanted[first])
    if np.any(~first):
        state[:, ~first] = above.sol(t_wanted[~first])

    rho_1, rho_2, hubble, ln_a = state
    rho = rho_1 + rho_2
    w_of_t = args.w * rho_1 / rho

    constraint = np.abs(hubble**2 - rho * (1.0 - rho) / 3.0) / np.maximum(rho / 3.0, 1e-300)

    return {
        "rho_over_rhoc": rho,
        "w": w_of_t,
        "H_gate": hubble * scale_h,
        "a_gate": np.exp(ln_a - ln_a_bounce),
        "constraint_relative": constraint,
        "rho_c_gate": rho_c_gate,
        "t_bounce": float(t_bounce),
    }


def as_background_dict(sampled, device):
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device=device)

    hubble = tensor(sampled["H_gate"])
    # 1/(2H) is supplied explicitly: the closed-form expression p(1+eta^2)/(2 eta) is
    # not valid here, and M_cal needs a regular value through the bounce.
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(sampled["H_gate"] != 0.0, 0.5 / sampled["H_gate"], 0.0)
    return {
        "H": hubble,
        "a": tensor(sampled["a_gate"]),
        "p": tensor(1.5 * (1.0 + sampled["w"])),
        "inv_two_hubble": tensor(inverse),
        "rho_over_rhoc": tensor(sampled["rho_over_rhoc"]),
        "varying_p": True,
        "p_constant": float(np.max(1.5 * (1.0 + sampled["w"]))),
    }


def scan(args, device, eta, background):
    step = float(eta[1] - eta[0])
    eta_t = torch.tensor(eta, dtype=torch.float64, device=device)

    k1_values = torch.linspace(-args.k1_max, args.k1_max, args.k1_grid, dtype=torch.float64, device=device)
    k2_values = torch.linspace(-args.k2_max, args.k2_max, args.k2_grid, dtype=torch.float64, device=device)
    tau1_values = torch.linspace(0.5, args.tau1_max, args.tau1_grid, dtype=torch.float64, device=device)
    tau2sq_values = torch.linspace(0.1, args.tau2sq_max, args.tau2sq_grid, dtype=torch.float64, device=device)
    mesh = torch.meshgrid(k1_values, k2_values, tau1_values, tau2sq_values, indexing="ij")
    flat = [component.reshape(-1) for component in mesh]
    total = flat[0].numel()

    interior = slice(1, -1)
    best_margin, best_index = -float("inf"), -1
    viable = viable_subluminal = 0

    for start in range(0, total, args.chunk):
        stop = min(start + args.chunk, total)
        shape = (stop - start, 1)
        values = realization.coefficients(
            eta_t, step, background,
            flat[0][start:stop].reshape(shape), flat[1][start:stop].reshape(shape),
            flat[2][start:stop].reshape(shape), flat[3][start:stop].reshape(shape),
        )
        u_i, cs_i, qt_i = values["U"][:, interior], values["cs_sq"][:, interior], values["q_t"][:, interior]
        healthy = torch.isfinite(cs_i) & torch.isfinite(u_i)
        floor = torch.full_like(cs_i, -1e30)

        stable = torch.where(healthy, torch.minimum(torch.minimum(u_i, cs_i), qt_i), floor).min(dim=1).values
        causal = torch.where(
            healthy, torch.minimum(torch.minimum(torch.minimum(u_i, cs_i), qt_i), 1.0 - cs_i), floor
        ).min(dim=1).values

        viable += int((stable > 0.0).sum().item())
        viable_subluminal += int((causal >= 0.0).sum().item())

        selection = causal if args.require_subluminal else stable
        chunk_best = int(torch.argmax(selection).item())
        if float(selection[chunk_best].item()) > best_margin:
            best_margin = float(selection[chunk_best].item())
            best_index = start + chunk_best

    return {
        "candidates": total,
        "viable_stable": viable,
        "viable_stable_fraction": viable / total,
        "viable_subluminal": viable_subluminal,
        "viable_subluminal_fraction": viable_subluminal / total,
        "best_margin": best_margin,
        "best_parameters": {
            "k1": float(flat[0][best_index]), "k2": float(flat[1][best_index]),
            "tau1": float(flat[2][best_index]), "tau2_sq": float(flat[3][best_index]),
        },
    }


def finalize(args, device, eta, background, sampled, parameters):
    step = float(eta[1] - eta[0])
    eta_t = torch.tensor(eta, dtype=torch.float64, device=device)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)

    values = realization.coefficients(
        eta_t, step, background,
        tensor(parameters["k1"]), tensor(parameters["k2"]),
        tensor(parameters["tau1"]), tensor(parameters["tau2_sq"]),
    )

    interior = slice(1, -1)
    u_i = values["U"][interior]
    cs_i = values["cs_sq"][interior]
    qt_i = values["q_t"][interior]
    q_s = qt_i * u_i

    hubble = background["H"]
    hubble_dot = realization.derivative(hubble, step)
    e_char = torch.sqrt(torch.maximum(hubble.square(), hubble_dot.abs()))[interior]
    cutoff = args.cutoff_over_e_char * float(e_char.max())
    control = float((e_char / cutoff).max())

    contracting = sampled["H_gate"] < 0.0
    budget = -6.0 * np.log(sampled["a_gate"]) - np.log(sampled["rho_over_rhoc"])
    shear = float(budget[contracting][-1] - budget[contracting][0])

    p_profile = 1.5 * (1.0 + sampled["w"])
    return {
        "parameters": parameters,
        "grid": {"points": eta.size, "extent": float(eta[-1]), "step": step},
        "background": {
            "source": "derived (energy conversion above rho_H)",
            "w_ekpyrotic": args.w,
            "gamma": args.gamma,
            "rho_H_over_rho_c": args.rho_h_over_rho_c,
            "rho_initial_over_rho_c": args.rho_initial,
            "p_asymptotic_past": float(p_profile[0]),
            "p_at_bounce": float(p_profile[np.argmax(sampled["rho_over_rhoc"])]),
            "p_final_expanding": float(p_profile[-1]),
            "max_abs_H_gate": float(np.max(np.abs(sampled["H_gate"]))),
            "max_abs_H_gate_expected": 1.0 / (2.0 * 1.5 * (1.0 + args.w)),
            "max_constraint_relative": float(np.max(sampled["constraint_relative"])),
            "shear_budget_contracting": shear,
            "rho_at_grid_start_over_rho_c": float(sampled["rho_over_rhoc"][0]),
            "rho_at_grid_end_over_rho_c": float(sampled["rho_over_rhoc"][-1]),
        },
        "analytic_checks": {
            "c_s_sq_plus_infinity": float(cs_i[-1]),
            "c_s_sq_plus_infinity_expected_p_over_3": float(p_profile[-1] / 3.0),
            "c_s_sq_minus_infinity": float(cs_i[0]),
            "c_s_sq_minus_infinity_expected": float(p_profile[0] / (3.0 + parameters["k1"])),
        },
        "stability": {
            "min_U": float(u_i.min()),
            "min_c_s_sq": float(cs_i.min()),
            "max_c_s_sq": float(cs_i.max()),
            "eta_at_min_c_s_sq": float(eta[interior][int(torch.argmin(cs_i))]),
            "min_Q_T": float(qt_i.min()),
            "min_Q_s": float(q_s.min()),
            "ghost_free": bool(q_s.min() > 0.0 and qt_i.min() > 0.0),
            "gradient_stable": bool(cs_i.min() > 0.0),
            "subluminal": bool(cs_i.max() <= 1.0),
            "superluminal_points": int((cs_i > 1.0).sum()),
        },
        "eft_control": {
            "mode": "assumed-ratio",
            "cutoff_over_e_char": args.cutoff_over_e_char,
            "max_e_char_over_cutoff": control,
            "passes": bool(control < 0.1),
        },
        "gate_points": int(u_i.numel()),
        "gate_all_pass": bool(
            q_s.min() > 0.0 and qt_i.min() > 0.0 and cs_i.min() > 0.0 and cs_i.max() <= 1.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w", type=float, default=4.3333333333333333, help="ekpyrotic w_1; p_i = 1.5(1+w)")
    parser.add_argument("--gamma", type=float, default=26.0)
    parser.add_argument("--rho-h-over-rho-c", type=float, default=1e-2)
    parser.add_argument("--rho-initial", type=float, default=1.8e-10)
    parser.add_argument("--rho-stop", type=float, default=1e-5)
    parser.add_argument("--extent", type=float, default=40.0)
    parser.add_argument("--scan-points", type=int, default=6001)
    parser.add_argument("--final-points", type=int, default=96001)
    parser.add_argument("--k1-max", type=float, default=25.0)
    parser.add_argument("--k2-max", type=float, default=5.0)
    parser.add_argument("--tau1-max", type=float, default=10.0)
    parser.add_argument("--tau2sq-max", type=float, default=5.0)
    parser.add_argument("--k1-grid", type=int, default=21)
    parser.add_argument("--k2-grid", type=int, default=21)
    parser.add_argument("--tau1-grid", type=int, default=9)
    parser.add_argument("--tau2sq-grid", type=int, default=11)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--require-subluminal", action="store_true", default=True)
    parser.add_argument("--cutoff-over-e-char", type=float, default=20.0)
    parser.add_argument("--t-max", type=float, default=4e6)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fixed-parameters", type=str, default="",
                        help="k1,k2,tau1,tau2_sq to skip the scan (e.g. the section 7 choice)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    scan_grid = np.linspace(-args.extent, args.extent, args.scan_points)
    final_grid = np.linspace(-args.extent, args.extent, args.final_points)

    print(f"derived background: w_1 = {args.w:.4f}, gamma = {args.gamma}, "
          f"rho_H/rho_c = {args.rho_h_over_rho_c:g}, rho_i/rho_c = {args.rho_initial:g}")

    sampled_final = derived_background_on_grid(args, final_grid)
    background_final = as_background_dict(sampled_final, device)
    print(f"  max|H| = {np.max(np.abs(sampled_final['H_gate'])):.10f} "
          f"(closed form 1/(2 p_i) = {1.0/(2*1.5*(1+args.w)):.10f})")
    print(f"  constraint (max relative) = {np.max(sampled_final['constraint_relative']):.3e}")
    print(f"  rho/rho_c over the grid: {sampled_final['rho_over_rhoc'][0]:.3e} "
          f"-> 1 -> {sampled_final['rho_over_rhoc'][-1]:.3e}")

    if args.fixed_parameters:
        k1, k2, tau1, tau2_sq = (float(v) for v in args.fixed_parameters.split(","))
        parameters = {"k1": k1, "k2": k2, "tau1": tau1, "tau2_sq": tau2_sq}
        scan_result = {"skipped": True, "reason": "fixed parameters supplied"}
        print(f"\nusing fixed parameters {parameters}")
    else:
        sampled_scan = derived_background_on_grid(args, scan_grid)
        background_scan = as_background_dict(sampled_scan, device)
        print(f"\nscanning {args.k1_grid*args.k2_grid*args.tau1_grid*args.tau2sq_grid} "
              f"candidates on {args.scan_points} points ...")
        scan_result = scan(args, device, scan_grid, background_scan)
        parameters = scan_result["best_parameters"]
        print(f"  viable (stable)      {scan_result['viable_stable']:6d}"
              f"  ({scan_result['viable_stable_fraction']:.4%})")
        print(f"  viable (subluminal)  {scan_result['viable_subluminal']:6d}"
              f"  ({scan_result['viable_subluminal_fraction']:.4%})")
        print(f"  best margin {scan_result['best_margin']:.6f} at {parameters}")

    result = finalize(args, device, final_grid, background_final, sampled_final, parameters)
    result["scan"] = scan_result

    print()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
