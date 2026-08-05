#!/usr/bin/env python3
"""Realize the CRBC baseline bounce as a c_T=1 beyond-Horndeski EFT and gate it.

The CRBC baseline background (결맞은_잔재_바운스_우주론_연구명세_kr.md §2) is

    H^2 = (8 pi G / 3) rho (1 - rho/rho_c),    rho = rho_c / (1 + alpha t^2),
    a   = a_B (1 + alpha t^2)^{1/(3(1+w))},   alpha = 6 pi G rho_c (1+w)^2,
    H   = 2 alpha t / (3 (1+w) (1 + alpha t^2)).

In units alpha = 1, a_B = 1 this is exactly

    H(t) = t / (p (1 + t^2)),   a(t) = (1 + t^2)^{1/(2p)},   p = 3(1+w)/2,

which is the Ye & Piao (2019) background ansatz (their Eq. 17) with a *constant* p and
unit lapse N = 1. The CRBC baseline is therefore inside the family for which
arXiv:1901.02202 gives an explicit c_T=1 beyond-Horndeski construction: one fixes the
background and solves algebraically for the Lagrangian functions. This script uses that
construction with the CRBC background substituted for theirs.

Because N = 1 here, the two readings of Eq. (13) that survived
ye_piao_denominator_identification.py (D = M_pl_sq/2 and D = A) coincide identically,
so the realization does not inherit that ambiguity.

Free functions of the construction (Ye-Piao Eqs. 24-25):

    c1(eta) = k1 (1 - tanh(eta/tau1)),  c2(eta) = exp(-eta^2/tau2_sq),  c3 = k2
    f1      = c2 (k2 H + 1) / (2 x^2),  x = 1/N = 1
    M_cal   = (1 - 4 f1^2 x^4) / (2 H (1 + 6 f1 x^2))
    A       = 1/2 + f1 / N^2,   M_pl_sq = 2 N A,   Q_T = M_pl_sq/2,   c_T^2 = 1
    U       = c1 + 6/N^2,       V = 2 [ N/(a A) d(a M_cal)/d(eta) - 1 ],   c_s^2 = V/U

The 0/0 of M_cal at the bounce is evaluated in closed form rather than interpolated:
with (1 - 2 f1) = 1 - c2 (k2 H + 1),

    M_cal = [ (1 - c2)/(2H) - c2 k2 / 2 ] * (1 + 2 f1) / (1 + 6 f1),

whose first bracket is regular, giving the exact bounce value M_cal(0) = -k2/4.

Analytic checks the run must reproduce:
    Friedmann residual   ~ machine epsilon (the mapping above is exact, not fitted)
    c_s^2(+inf) = p/3,   c_s^2(-inf) = p/(3 + k1)

What this does and does not establish: it tests whether the CRBC baseline background
admits a ghost-free, gradient-stable c_T=1 beyond-Horndeski completion, and exports the
coefficient trajectory for crbc_eft_coefficient_gate.py. It is not a CMB prediction, and
the EFT cutoff is an assumption supplied here, not derived (see --cutoff-over-e-char).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from quantum_gravity.crbc_gpu.crbc_background_scan import resolve_device
except ModuleNotFoundError:  # direct execution from the package directory
    from crbc_background_scan import resolve_device  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w", type=float, default=1.0, help="constant equation of state; p = 3(1+w)/2")
    parser.add_argument("--p-initial", type=float, default=None, help="if set, p(eta) runs p_i -> p_f (w(t) transition)")
    parser.add_argument("--p-final", type=float, default=3.0)
    parser.add_argument("--eta-p", type=float, default=0.7)
    parser.add_argument("--tau-p", type=float, default=1.0)
    parser.add_argument("--p-mid", type=float, default=None,
                        help="if set, p(eta) descends in two stages instead of one, which is what "
                        "a three-level phase-synchronization hierarchy implies")
    parser.add_argument("--eta-p2", type=float, default=-6.0)
    parser.add_argument("--tau-p2", type=float, default=4.0)
    parser.add_argument("--k1-max", type=float, default=2.5)
    parser.add_argument("--extent", type=float, default=40.0)
    parser.add_argument("--scan-points", type=int, default=6001)
    parser.add_argument("--final-points", type=int, default=96001)
    parser.add_argument("--k1-grid", type=int, default=11)
    parser.add_argument("--k2-grid", type=int, default=21)
    parser.add_argument("--tau1-grid", type=int, default=9)
    parser.add_argument("--tau2sq-grid", type=int, default=11)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument(
        "--require-subluminal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require 0 < c_s^2 <= 1 everywhere when selecting the candidate",
    )
    parser.add_argument(
        "--cutoff-over-e-char",
        type=float,
        default=20.0,
        help="assumed ratio Lambda / max(E_char); an input, not a derived quantity",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--npz-output", type=Path, default=Path("outputs/crbc_beyond_horndeski_trajectory.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crbc_beyond_horndeski_realization.json"))
    return parser.parse_args()


def derivative(values: torch.Tensor, step: float) -> torch.Tensor:
    """Second-order finite difference along the last axis."""
    result = torch.empty_like(values)
    result[..., 0] = (-3.0 * values[..., 0] + 4.0 * values[..., 1] - values[..., 2]) / (2.0 * step)
    result[..., -1] = (3.0 * values[..., -1] - 4.0 * values[..., -2] + values[..., -3]) / (2.0 * step)
    result[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * step)
    return result


def crbc_background(eta: torch.Tensor, w: float, args: argparse.Namespace | None = None) -> dict[str, torch.Tensor]:
    """CRBC baseline in alpha = 1, a_B = 1 units.

    Constant p is the exact closed-form CRBC solution of the spec (§2). A p(eta) profile
    generalizes it to a w(t) transition: the CRBC effective Friedmann equation still holds,
    but rho/rho_c and w must then be reconstructed from H rather than read off in closed
    form, which reconstruct_equation_of_state() does.
    """
    constant_p = 1.5 * (1.0 + w)
    if args is None or args.p_initial is None:
        p = torch.full_like(eta, constant_p)
        varying = False
    elif getattr(args, "p_mid", None) is not None:
        # Two-stage descent p_i -> p_mid -> p_f. A three-level synchronization hierarchy
        # (oscillators -> platform 1 -> platform 2) separates the coupling timescales, so the
        # effective equation of state steps down twice rather than once. The slow stage is
        # placed earlier and wider than the fast one.
        p = (
            args.p_initial
            + 0.5 * (1.0 + torch.tanh((eta - args.eta_p2) / args.tau_p2)) * (args.p_mid - args.p_initial)
            + 0.5 * (1.0 + torch.tanh((eta - args.eta_p) / args.tau_p)) * (args.p_final - args.p_mid)
        )
        varying = True
    else:
        p = args.p_initial + 0.5 * (1.0 + torch.tanh((eta - args.eta_p) / args.tau_p)) * (args.p_final - args.p_initial)
        varying = True
    one_plus = 1.0 + eta.square()
    hubble = eta / (p * one_plus)
    if varying:
        step = float((eta[1] - eta[0]).item())
        scale = torch.exp(cumulative_trapezoid(hubble, step))
    else:
        scale = one_plus.pow(1.0 / (2.0 * constant_p))
    return {
        "p": p,
        "p_constant": constant_p,
        "varying_p": varying,
        "H": hubble,
        "a": scale,
        "rho_over_rhoc": 1.0 / one_plus,
    }


def cumulative_trapezoid(values: torch.Tensor, step: float) -> torch.Tensor:
    integral = torch.zeros_like(values)
    integral[1:] = torch.cumsum(0.5 * (values[1:] + values[:-1]) * step, dim=0)
    return integral - integral[integral.numel() // 2]


def reconstruct_equation_of_state(
    eta: torch.Tensor, background: dict[str, torch.Tensor], step: float, w_reference: float
) -> dict[str, object]:
    """Recover rho/rho_c and w(t) from H(t) using the CRBC equations themselves.

        H^2 = (8 pi G/3) rho (1 - rho/rho_c)      ->  r = rho/rho_c from a quadratic
        Hdot = -4 pi G (rho + p)(1 - 2 rho/rho_c) ->  w = -Hdot/[4 pi G rho_c r (1-2r)] - 1

    The quadratic has two branches meeting at r = 1/2; the physical branch is the one on
    which r decreases away from the bounce, which is selected by the sign of (|eta| - eta_*).
    """
    eight_pi_g_rho_c = 4.0 / (3.0 * (1.0 + w_reference) ** 2)
    hubble = background["H"]
    discriminant_argument = 3.0 * hubble.square() / eight_pi_g_rho_c
    discriminant = torch.clamp(1.0 - 4.0 * discriminant_argument, min=0.0)
    root = torch.sqrt(discriminant)

    # The two roots meet at r = 1/2; locate that point separately on each side, because a
    # p(eta) profile makes the background asymmetric.
    negative, positive = eta < 0.0, eta > 0.0
    eta_star_minus = float(eta[negative][torch.argmin(discriminant[negative])].item())
    eta_star_plus = float(eta[positive][torch.argmin(discriminant[positive])].item())
    inner = (eta >= eta_star_minus) & (eta <= eta_star_plus)
    r = 0.5 * (1.0 + torch.where(inner, root, -root))

    hubble_dot = derivative(hubble, step)
    denominator = 0.5 * eight_pi_g_rho_c * r * (1.0 - 2.0 * r)
    # Both Hdot and (1 - 2r) vanish at the branch points, so w is a 0/0 there. Only points
    # comfortably away from that cancellation are reported.
    reliable = (1.0 - 2.0 * r).abs() > 0.05
    w_of_t = torch.where(
        reliable, -hubble_dot / torch.where(reliable, denominator, torch.ones_like(denominator)) - 1.0,
        torch.full_like(hubble, float("nan")),
    )

    extent = float(eta[-1].item())
    asymptotic_past = (eta < -0.8 * extent) & reliable
    contracting = (eta < eta_star_minus) & reliable
    bounce_index = int(torch.argmin(eta.abs()).item())
    return {
        "branch_points": {"past": eta_star_minus, "future": eta_star_plus},
        "max_rho_over_rhoc": float(r.max().item()),
        "w_at_bounce": float(w_of_t[bounce_index].item()),
        "w_asymptotic_past": float(w_of_t[asymptotic_past].mean().item()) if bool(asymptotic_past.any().item()) else None,
        "w_min_contracting_reliable": float(w_of_t[contracting].min().item()) if bool(contracting.any().item()) else None,
        "w_max_contracting_reliable": float(w_of_t[contracting].max().item()) if bool(contracting.any().item()) else None,
        "ekpyrotic_asymptotic_past": bool(w_of_t[asymptotic_past].min().item() > 1.0)
        if bool(asymptotic_past.any().item())
        else None,
        "excluded_near_branch_points": int((~reliable).sum().item()),
        "caveat": "w is a 0/0 at the branch points r = 1/2; points with |1-2r| <= 0.05 are excluded.",
    }


def shear_budget(eta: torch.Tensor, background: dict[str, torch.Tensor], w_reference: float) -> dict[str, object]:
    """Anisotropy growth across the contracting phase (CRBC spec falsification condition 4).

    Shear energy density scales as a^-6 and the background as rho, so

        Q(eta) = ln(sigma^2 / rho) = -6 ln a - ln(rho/rho_c) + const,

    which needs neither w nor a derivative. The contraction is shear-safe when Q decreases
    from the deep past to the bounce. For constant p this evaluates in closed form to
    (3/p - 1) ln(1 + eta_start^2), so it is negative exactly when p > 3, i.e. w > 1.
    """
    eight_pi_g_rho_c = 4.0 / (3.0 * (1.0 + w_reference) ** 2)
    argument = torch.clamp(1.0 - 4.0 * (3.0 * background["H"].square() / eight_pi_g_rho_c), min=0.0)
    root = torch.sqrt(argument)
    negative, positive = eta < 0.0, eta > 0.0
    eta_star_minus = float(eta[negative][torch.argmin(argument[negative])].item())
    eta_star_plus = float(eta[positive][torch.argmin(argument[positive])].item())
    r = 0.5 * (1.0 + torch.where((eta >= eta_star_minus) & (eta <= eta_star_plus), root, -root))

    bounce = int(torch.argmin(eta.abs()).item())
    scale = background["a"]
    change = float(
        (6.0 * torch.log(scale[0] / scale[bounce]) + torch.log(r[0] / r[bounce])).item()
    )
    return {
        "ln_shear_over_background_change_past_to_bounce": change,
        "shear_suppressed_through_contraction": change < 0.0,
        "e_folds_of_shear_growth": max(change, 0.0),
        "note": "negative means the background outpaces shear; requires w > 1 for constant w",
    }


def friedmann_residual(background: dict[str, torch.Tensor], w: float) -> torch.Tensor:
    """|H^2 - (8 pi G/3) rho (1 - rho/rho_c)| with 8 pi G rho_c = 4/(3(1+w)^2) at alpha=1.

    Meaningful only for the constant-w closed form, where rho/rho_c = 1/(1+t^2) is known
    analytically. For a p(eta) profile use reconstruct_equation_of_state() instead.
    """
    eight_pi_g_rho_c = 4.0 / (3.0 * (1.0 + w) ** 2)
    r = background["rho_over_rhoc"]
    return (background["H"].square() - (eight_pi_g_rho_c / 3.0) * r * (1.0 - r)).abs()


def coefficients(
    eta: torch.Tensor,
    step: float,
    background: dict[str, torch.Tensor],
    k1: torch.Tensor,
    k2: torch.Tensor,
    tau1: torch.Tensor,
    tau2_sq: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Scalar/tensor quadratic coefficients. Leading axis is the parameter batch."""
    hubble, scale = background["H"], background["a"]
    p = background["p"]

    c1 = k1 * (1.0 - torch.tanh(eta / tau1))
    c2 = torch.exp(-eta.square() / tau2_sq)
    f1 = 0.5 * c2 * (k2 * hubble + 1.0)  # x = 1

    # Regular evaluation of (1 - 2 f1)/(2H) through the bounce, where both vanish.
    one_minus_c2 = -torch.expm1(-eta.square() / tau2_sq)
    safe_eta = torch.where(eta == 0.0, torch.ones_like(eta), eta)
    ratio = torch.where(
        eta == 0.0,
        torch.zeros_like(eta).expand_as(one_minus_c2),
        one_minus_c2 * p * (1.0 + eta.square()) / (2.0 * safe_eta),
    )
    m_cal = (ratio - 0.5 * c2 * k2) * (1.0 + 2.0 * f1) / (1.0 + 6.0 * f1)

    a_coefficient = 0.5 + f1  # A = 1/2 + f1/N^2, N = 1
    q_t = a_coefficient  # Q_T = M_pl_sq/2 = N A
    u = c1 + 6.0  # Eq. (12) with N = 1
    v = 2.0 * (derivative(scale * m_cal, step) / (scale * a_coefficient) - 1.0)  # Eq. (13)
    cs_sq = v / u

    return {"c1": c1, "f1": f1, "M_cal": m_cal, "A": a_coefficient, "U": u, "V": v, "cs_sq": cs_sq, "q_t": q_t}


def scan(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    eta = torch.linspace(-args.extent, args.extent, args.scan_points, dtype=torch.float64, device=device)
    step = float((eta[1] - eta[0]).item())
    background = crbc_background(eta, args.w, args)

    k1_values = torch.linspace(-args.k1_max, args.k1_max, args.k1_grid, dtype=torch.float64, device=device)
    k2_values = torch.linspace(-5.0, 5.0, args.k2_grid, dtype=torch.float64, device=device)
    tau1_values = torch.linspace(0.5, 10.0, args.tau1_grid, dtype=torch.float64, device=device)
    tau2sq_values = torch.linspace(0.1, 5.0, args.tau2sq_grid, dtype=torch.float64, device=device)
    mesh = torch.meshgrid(k1_values, k2_values, tau1_values, tau2sq_values, indexing="ij")
    flat = [component.reshape(-1) for component in mesh]
    total = flat[0].numel()

    # Interior slice: the one-sided end derivatives are not part of the verdict.
    interior = slice(1, -1)
    best_margin = -float("inf")
    best_index = -1
    viable = 0
    viable_subluminal = 0
    for start in range(0, total, args.chunk):
        stop = min(start + args.chunk, total)
        shape = (stop - start, 1)
        values = coefficients(
            eta,
            step,
            background,
            flat[0][start:stop].reshape(shape),
            flat[1][start:stop].reshape(shape),
            flat[2][start:stop].reshape(shape),
            flat[3][start:stop].reshape(shape),
        )
        u_i, cs_i, qt_i = values["U"][:, interior], values["cs_sq"][:, interior], values["q_t"][:, interior]
        healthy = torch.isfinite(cs_i) & torch.isfinite(u_i)
        floor = torch.full_like(cs_i, -1e30)

        stable_margin = torch.where(healthy, torch.minimum(torch.minimum(u_i, cs_i), qt_i), floor).min(dim=1).values
        # Subluminality: 0 < c_s^2 <= 1 everywhere, so 1 - c_s^2 enters the margin.
        causal_margin = torch.where(
            healthy, torch.minimum(torch.minimum(torch.minimum(u_i, cs_i), qt_i), 1.0 - cs_i), floor
        ).min(dim=1).values

        viable += int((stable_margin > 0.0).sum().item())
        viable_subluminal += int((causal_margin >= 0.0).sum().item())

        selection = causal_margin if args.require_subluminal else stable_margin
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
        "selection_criterion": "subluminal" if args.require_subluminal else "stable_only",
        "best_margin": best_margin,
        "best_parameters": {
            "k1": float(flat[0][best_index].item()),
            "k2": float(flat[1][best_index].item()),
            "tau1": float(flat[2][best_index].item()),
            "tau2_sq": float(flat[3][best_index].item()),
        },
        "analytic_subluminality_bounds": {
            "c_s_sq_plus_infinity": "p/3 <= 1 requires w <= 1",
            "c_s_sq_minus_infinity": "p/(3+k1) <= 1 requires k1 >= p - 3",
        },
    }


def finalize(
    args: argparse.Namespace,
    device: torch.device,
    parameters: dict[str, float],
    write_npz: bool = True,
) -> dict[str, object]:
    eta = torch.linspace(-args.extent, args.extent, args.final_points, dtype=torch.float64, device=device)
    step = float((eta[1] - eta[0]).item())
    background = crbc_background(eta, args.w, args)
    p = float(background["p"].max().item()) if background["varying_p"] else background["p_constant"]

    def as_tensor(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64, device=device)

    values = coefficients(
        eta,
        step,
        background,
        as_tensor(parameters["k1"]),
        as_tensor(parameters["k2"]),
        as_tensor(parameters["tau1"]),
        as_tensor(parameters["tau2_sq"]),
    )

    interior = slice(1, -1)
    eta_i = eta[interior]
    u_i, cs_i, qt_i = values["U"][interior], values["cs_sq"][interior], values["q_t"][interior]
    q_s = qt_i * u_i  # Q_s = (M_pl_sq/2) U

    # Characteristic background frequency and the assumed EFT cutoff.
    hubble = background["H"]
    hubble_dot = derivative(hubble, step)
    e_char = torch.sqrt(torch.maximum(hubble.square(), hubble_dot.abs()))[interior]
    e_char_max = float(e_char.max().item())
    cutoff = args.cutoff_over_e_char * e_char_max

    tail = eta_i > 0.8 * args.extent
    head = eta_i < -0.8 * args.extent
    arrays = {
        "time": eta_i.cpu().numpy(),
        "a": background["a"][interior].cpu().numpy(),
        "H": hubble[interior].cpu().numpy(),
        "q_s": q_s.cpu().numpy(),
        "c_s_sq": cs_i.cpu().numpy(),
        "q_t": qt_i.cpu().numpy(),
        "c_t_sq": np.ones(eta_i.numel(), dtype=np.float64),
        "cutoff": np.full(eta_i.numel(), cutoff, dtype=np.float64),
        "characteristic_energy": e_char.cpu().numpy(),
    }
    if write_npz:
        args.npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.npz_output, **arrays)

    return {
        "parameters": parameters,
        "grid": {"points": args.final_points, "extent": args.extent, "step": step},
        "background": {
            "w": args.w,
            "p": p,
            "varying_p": background["varying_p"],
            "p_initial": args.p_initial,
            "p_final": args.p_final if background["varying_p"] else None,
            "max_friedmann_residual": None
            if background["varying_p"]
            else float(friedmann_residual(background, args.w).max().item()),
            "reconstructed_equation_of_state": reconstruct_equation_of_state(eta, background, step, args.w),
            "shear_budget": shear_budget(eta, background, args.w),
            "min_scale_factor": float(background["a"].min().item()),
            "bounce_H": float(hubble[args.final_points // 2].item()),
        },
        "analytic_checks": {
            "M_cal_at_bounce": float(values["M_cal"][args.final_points // 2].item()),
            "M_cal_at_bounce_expected": -parameters["k2"] / 4.0,
            "c_s_sq_plus_infinity": float(cs_i[tail].mean().item()),
            "c_s_sq_plus_infinity_expected": (args.p_final if background["varying_p"] else p) / 3.0,
            "c_s_sq_minus_infinity": float(cs_i[head].mean().item()),
            "c_s_sq_minus_infinity_expected": (args.p_initial if background["varying_p"] else p)
            / (3.0 + parameters["k1"]),
        },
        "stability": {
            "min_U": float(u_i.min().item()),
            "min_c_s_sq": float(cs_i.min().item()),
            "eta_at_min_c_s_sq": float(eta_i[torch.argmin(cs_i)].item()),
            "min_Q_T": float(qt_i.min().item()),
            "min_Q_s": float(q_s.min().item()),
            "ghost_free": bool((q_s > 0.0).all().item() and (qt_i > 0.0).all().item()),
            "gradient_stable": bool((cs_i > 0.0).all().item()),
            "max_c_s_sq": float(cs_i.max().item()),
            "superluminal_points": int((cs_i > 1.0).sum().item()),
            "subluminal": bool((cs_i <= 1.0).all().item()),
        },
        "eft_cutoff": {
            "max_characteristic_energy": e_char_max,
            "assumed_ratio_cutoff_over_e_char": args.cutoff_over_e_char,
            "assumed_cutoff": cutoff,
            "status": "assumption supplied to the gate, not derived from a microscopic theory",
        },
        "npz": str(args.npz_output),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scan_report = scan(args, device)
    final_report = finalize(args, device, scan_report["best_parameters"])

    cpu_final = (
        finalize(args, torch.device("cpu"), scan_report["best_parameters"], write_npz=False)
        if device.type == "cuda"
        else None
    )

    report: dict[str, object] = {
        "construction": "Ye & Piao (2019) c_T=1 beyond-Horndeski construction, arXiv:1901.02202, "
        "with the CRBC baseline background of 결맞은_잔재_바운스_우주론_연구명세_kr.md §2 substituted.",
        "background_identity": "H = t/(p(1+t^2)), a = (1+t^2)^{1/(2p)}, p = 3(1+w)/2 at alpha=1 "
        "is exactly the CRBC effective Friedmann solution and exactly the Ye-Piao ansatz with constant p, N=1.",
        "denominator_ambiguity": "N = 1 makes the two surviving readings of Eq. (13) identical; "
        "the realization does not inherit that ambiguity.",
        "device": str(device),
        "scan": scan_report,
        "selected": final_report,
        "cpu_cross_check": None
        if cpu_final is None
        else {
            "min_c_s_sq": cpu_final["stability"]["min_c_s_sq"],
            "same_gradient_verdict": cpu_final["stability"]["gradient_stable"]
            == final_report["stability"]["gradient_stable"],
            "abs_min_c_s_sq_difference": abs(
                cpu_final["stability"]["min_c_s_sq"] - final_report["stability"]["min_c_s_sq"]
            ),
        },
        "scope": "Background-level realization and stability gate input. Not a CMB prediction; "
        "the coherent-relic template of the CRBC spec is a separate, later step.",
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
