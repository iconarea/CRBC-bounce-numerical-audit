"""Derive w(t) instead of inserting it by hand.

Every CRBC background so far has taken p(eta) = 1.5(1+w) as a *given* tanh profile
(crbc_beyond_horndeski_realization.crbc_background) and reconstructed rho and w from
the resulting H. That makes the central dynamical claim of the model -- an ekpyrotic
contraction that softens into a Hagedorn-like pressureless phase before the bounce --
an assumption with four free parameters (p_i, p_f, eta_p, tau_p) rather than a result.

Three parts, in order of what they rule out.

Part A -- constant-w matter cannot do it. For any mixture of components with constant
equation-of-state parameters,

    d w_eff / d ln a = -3 Var_f(w)  <= 0     exactly,

so contraction (d ln a < 0) can only *raise* w_eff, toward max(w_i). The assumed
descent w: 4.33 -> 0 is impossible for any weighting of any number of such components.

Part B -- an oscillating scalar cannot do it either. With

    V(phi) = V1 exp(-2 c phi) - V0 exp(-c phi)

the tail is the ekpyrotic exponential (eps = c^2/2 exactly) and the V1 term turns the
potential around into a minimum, about which the field should virialize to <w> = 0.
It does not get the chance: the oscillator mass and the transition density are set by
the same potential, m^2 = 2 c^2 |V_min| and rho_transition ~ |V_min|, so the number of
oscillations available before the bounce,

    N_osc = m * Delta_t / 2 pi ,

is *independent of |V_min|* and comes out near 1. No choice of depth buys more. The
pressureless phase never forms; the field kinates to w -> 1 instead.

Part C -- what is left is energy transfer. Relaxing conservation of the individual
components to

    rho_1' + 3H(1+w_1) rho_1 = -Gamma rho_1,    rho_2' + 3H rho_2 = +Gamma rho_1

gives d w_eff/dt = w_1 f [3H(w_eff - w_1) - Gamma], so the descent requires

    Gamma > 3 |H| w_1 (1 - f) ,

a lower bound on the conversion rate. This is the string-gas reading of the Hagedorn
transition: the soft phase is not a field settling into a minimum, it is the ekpyrotic
energy being *converted* into string excitations once the density crosses rho_H. The
resulting w(t) is then fitted against the tanh that was assumed, to report which of
its four parameters survive as predictions.

Units: 8 pi G = 1, rho_c = 1. No observational data is used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


# ---------------------------------------------------------------- Part A: the no-go

def multifluid_effective_w(w_components, fractions_at_reference, ln_a):
    """w_eff(ln a) for constant-w components with rho_i ~ a^{-3(1+w_i)}."""
    w = np.asarray(w_components, dtype=float)[:, None]
    f0 = np.asarray(fractions_at_reference, dtype=float)[:, None]
    rho = f0 * np.exp(-3.0 * (1.0 + w) * ln_a[None, :])
    total = rho.sum(axis=0)
    return (w * rho).sum(axis=0) / total, rho / total


def check_no_go():
    """Verify d w_eff / d ln a = -3 Var_f(w) and the resulting monotonicity."""
    ln_a = np.linspace(-2.0, 2.0, 40001)
    w_components = [4.3333333333333333, 0.0, 1.0 / 3.0]
    w_eff, f = multifluid_effective_w(w_components, [1.0, 1.0, 1.0], ln_a)

    step = ln_a[1] - ln_a[0]
    numeric = np.gradient(w_eff, step)
    w = np.asarray(w_components)[:, None]
    variance = (f * w**2).sum(axis=0) - w_eff**2
    predicted = -3.0 * variance

    interior = slice(2, -2)
    residual = np.max(np.abs(numeric[interior] - predicted[interior]))
    scale = np.max(np.abs(predicted[interior]))

    return {
        "identity_max_residual": float(residual),
        "identity_relative_residual": float(residual / scale),
        "derivative_all_nonpositive": bool(np.all(numeric[interior] <= 1e-9)),
        # ln_a runs -2 -> +2, so index 0 is the *small* scale factor.
        "w_eff_at_small_a": float(w_eff[0]),
        "w_eff_at_large_a": float(w_eff[-1]),
        "contraction_limit_is_max_w": float(max(w_components)),
        "expansion_limit_is_min_w": float(min(w_components)),
    }


# ------------------------------------------- Part B: the oscillating-scalar attempt

class EkpyroticMinimumPotential:
    """V(phi) = V1 exp(-2 c phi) - V0 exp(-c phi).

    Tail (phi large): V -> -V0 exp(-c phi), the ekpyrotic exponential with eps = c^2/2.
    Minimum at exp(-c phi_min) = V0/(2 V1), depth |V_min| = V0^2/(4 V1),
    curvature m^2 = V''(phi_min) = 2 c^2 |V_min|.
    """

    def __init__(self, c, v0, v1):
        self.c, self.v0, self.v1 = float(c), float(v0), float(v1)

    def __call__(self, phi):
        u = np.exp(-self.c * phi)
        return self.v1 * u * u - self.v0 * u

    def gradient(self, phi):
        u = np.exp(-self.c * phi)
        return -2.0 * self.c * self.v1 * u * u + self.c * self.v0 * u

    @property
    def phi_min(self):
        return -np.log(self.v0 / (2.0 * self.v1)) / self.c

    @property
    def depth(self):
        return self.v0 * self.v0 / (4.0 * self.v1)

    @property
    def mass_squared(self):
        return 2.0 * self.c * self.c * self.depth


def evolve_scalar(potential, rho_initial, rho_c, w_initial, t_max, rtol, atol):
    """Integrate the CRBC scalar background from the ekpyrotic attractor."""
    kinetic = rho_initial * (1.0 + w_initial) / 2.0          # = 0.5 phi'^2
    potential_value = rho_initial * (1.0 - w_initial) / 2.0  # = V  (negative)
    phi_dot = -np.sqrt(2.0 * kinetic)                        # rolling toward -infinity
    phi_0 = -np.log(-potential_value / potential.v0) / potential.c
    hubble_0 = -np.sqrt(rho_initial * (1.0 - rho_initial / rho_c) / 3.0)

    def rhs(_t, y):
        phi, phi_dot_, hubble, _ln_a = y
        rho = 0.5 * phi_dot_ * phi_dot_ + potential(phi)
        return [
            phi_dot_,
            -3.0 * hubble * phi_dot_ - potential.gradient(phi),
            -0.5 * phi_dot_ * phi_dot_ * (1.0 - 2.0 * rho / rho_c),
            hubble,
        ]

    def past_bounce(_t, y):
        return y[2] - 0.6 * np.sqrt(rho_c / 12.0)

    past_bounce.terminal = True
    past_bounce.direction = 1.0

    solution = solve_ivp(
        rhs, (0.0, t_max), [phi_0, phi_dot, hubble_0, 0.0],
        method="Radau", rtol=rtol, atol=atol, dense_output=True,
        events=past_bounce, max_step=t_max / 4000.0,
    )

    t = solution.t
    phi, phi_dot_, hubble, ln_a = solution.y
    v = potential(phi)
    kinetic_density = 0.5 * phi_dot_ * phi_dot_
    rho = kinetic_density + v
    pressure = kinetic_density - v
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(rho) > 1e-30, pressure / rho, np.nan)

    # Normalise the constraint by rho/3, not by H^2 -- H vanishes at the bounce.
    constraint = hubble * hubble - rho * (1.0 - rho / rho_c) / 3.0
    scale = np.maximum(np.abs(rho) / 3.0, 1e-30)

    return {
        "t": t, "phi": phi, "phi_dot": phi_dot_, "H": hubble, "ln_a": ln_a,
        "rho": rho, "p": pressure, "w": w, "V": v,
        "constraint_relative": np.abs(constraint) / scale,
        "success": bool(solution.status >= 0),
        "reached_bounce": bool(np.any(rho >= 0.999 * rho_c)),
    }


def count_oscillations(t, phi, phi_min, hubble):
    """Half-turns of phi about the minimum, from first crossing to the bounce."""
    contracting = hubble < 0.0
    offset = phi - phi_min
    crossings = np.where(np.diff(np.sign(offset[contracting])) != 0)[0]
    if crossings.size == 0:
        return 0.0, np.nan
    first = crossings[0]
    turns = np.diff(np.sign(np.diff(phi[contracting][first:])))
    return float(np.count_nonzero(turns) / 2.0), float(t[contracting][first])


def cycle_average(t, rho, pressure, window):
    """<w> over a sliding window, from integrated p and rho (not pointwise ratios)."""
    averaged = np.full(t.size, np.nan)
    for i in range(t.size):
        lo = np.searchsorted(t, t[i] - 0.5 * window)
        hi = np.searchsorted(t, t[i] + 0.5 * window)
        if hi - lo < 4:
            continue
        span = slice(lo, hi)
        rho_bar = np.trapezoid(rho[span], t[span])
        if abs(rho_bar) > 1e-30:
            averaged[i] = np.trapezoid(pressure[span], t[span]) / rho_bar
    return averaged


def analytic_oscillation_budget(c, depth, rho_c):
    """N_osc = m Delta_t / 2 pi with m = sqrt(2 c^2 |V_min|) and rho_tr ~ |V_min|.

    In the matter-like phase rho ~ a^-3 and |H| = sqrt(rho/3), so
        Delta_t = (2/sqrt 3) (rho_tr^{-1/2} - rho_c^{-1/2}).
    The sqrt(|V_min|) in m cancels the 1/sqrt(rho_tr) in Delta_t, leaving N_osc
    independent of the depth.
    """
    mass = np.sqrt(2.0 * c * c * depth)
    rho_transition = depth
    delta_t = (2.0 / np.sqrt(3.0)) * (rho_transition**-0.5 - rho_c**-0.5)
    return float(mass * delta_t / (2.0 * np.pi)), float(c / np.pi * np.sqrt(2.0 / 3.0))


# ------------------------------------------------------- Part C: energy conversion

def evolve_transfer(w_ek, rho_h, gamma, rho_initial, rho_c, t_max, rtol, atol,
                    rho_stop_expanding=None, rate_mode="hubble"):
    """Ekpyrotic component draining into a pressureless component above rho_H.

        rho_1' + 3H(1+w_1) rho_1 = -Gamma rho_1
        rho_2' + 3H       rho_2 = +Gamma rho_1,     Gamma = gamma sqrt(rho/3) for rho > rho_H

    The threshold is on *density*, not on time: it is the Hagedorn density. There is no
    width parameter -- the width of the resulting w(t) is a prediction.
    """
    rho_1_0 = rho_initial
    rho_2_0 = 1e-12 * rho_initial
    hubble_0 = -np.sqrt(rho_initial * (1.0 - rho_initial / rho_c) / 3.0)

    def make_rhs(active):
        def rhs(_t, y):
            rho_1, rho_2, hubble, _ln_a = y
            rho_1 = max(rho_1, 0.0)
            rho_2 = max(rho_2, 0.0)
            rho = rho_1 + rho_2
            if not active:
                rate = 0.0
            elif rate_mode == "hubble":
                # Gamma proportional to H. This is a gravitational ansatz: it ties the
                # conversion rate to the expansion rate, which no local process does.
                rate = gamma * np.sqrt(max(rho, 0.0) / 3.0)
            else:
                # Gamma proportional to the local energy scale, Gamma = C rho^{1/4},
                # which is what a microphysical rate obeys. In code units (rho_c = 1,
                # M_P = 1) the coefficient is C = c * xi_red^{-1/4} with c the physical
                # dimensionless coupling and xi_red = rho_c/M_P^4.
                rate = gamma * max(rho, 0.0) ** 0.25
            return [
                -3.0 * hubble * (1.0 + w_ek) * rho_1 - rate * rho_1,
                -3.0 * hubble * rho_2 + rate * rho_1,
                -0.5 * (rho + w_ek * rho_1) * (1.0 - 2.0 * rho / rho_c),
                hubble,
            ]
        return rhs

    # The Hagedorn threshold is a discontinuity in Gamma; a stiff solver stalls on it.
    # Split the integration at rho = rho_H with an event instead of smoothing the step,
    # which would reintroduce a width parameter.
    def crosses_threshold(_t, y):
        return y[0] + y[1] - rho_h

    crosses_threshold.terminal = True
    crosses_threshold.direction = 1.0

    if rho_stop_expanding is None:
        def past_bounce(_t, y):
            return y[2] - 0.6 * np.sqrt(rho_c / 12.0)
        past_bounce.direction = 1.0
    else:
        # Stop once the expanding branch has diluted back to a chosen density.
        def past_bounce(_t, y):
            return (y[0] + y[1]) - rho_stop_expanding if y[2] > 0.0 else 1.0
        past_bounce.direction = -1.0
    past_bounce.terminal = True

    common = dict(method="Radau", rtol=rtol, atol=atol, max_step=t_max / 4000.0,
                  dense_output=True)
    below = solve_ivp(make_rhs(False), (0.0, t_max), [rho_1_0, rho_2_0, hubble_0, 0.0],
                      events=crosses_threshold, **common)
    if below.status < 0 or below.t_events[0].size == 0:
        return {"success": False, "reached_bounce": False, "stalled_below_threshold": True}
    above = solve_ivp(make_rhs(True), (below.t[-1], t_max), below.y[:, -1],
                      events=past_bounce, **common)

    solution = above
    t = np.concatenate([below.t[:-1], above.t])
    rho_1, rho_2, hubble, ln_a = np.concatenate([below.y[:, :-1], above.y], axis=1)
    rho = rho_1 + rho_2
    pressure = w_ek * rho_1
    w = pressure / rho
    constraint = hubble * hubble - rho * (1.0 - rho / rho_c) / 3.0

    return {
        "segments": (below, above),
        "t": t, "rho_1": rho_1, "rho_2": rho_2, "rho": rho, "H": hubble,
        "ln_a": ln_a, "w": w, "p_exponent": 1.5 * (1.0 + w),
        "f_ekpyrotic": rho_1 / rho,
        "constraint_relative": np.abs(constraint) / np.maximum(rho / 3.0, 1e-30),
        "success": bool(solution.status >= 0),
        "reached_bounce": bool(np.any(rho >= 0.999 * rho_c)),
    }


def fit_tanh_profile(run, rho_c):
    """Map the derived w onto the eta of the assumed parametrisation and fit a tanh.

    The assumed background has rho/rho_c = 1/(1+eta^2), so eta = -sqrt(rho_c/rho - 1)
    while contracting. Fitting p(eta) = p_i + (1+tanh((eta-eta_p)/tau_p))/2 (p_f - p_i)
    says which of the four assumed parameters the derivation actually reproduces.
    """
    contracting = run["H"] < 0.0
    rho = run["rho"][contracting]
    p_exponent = run["p_exponent"][contracting]
    usable = (rho > 1e-8 * rho_c) & (rho < 0.999 * rho_c) & np.isfinite(p_exponent)
    eta = -np.sqrt(np.maximum(rho_c / rho[usable] - 1.0, 0.0))
    values = p_exponent[usable]
    order = np.argsort(eta)
    eta, values = eta[order], values[order]

    def model(x, p_i, p_f, eta_p, tau_p):
        return p_i + 0.5 * (1.0 + np.tanh((x - eta_p) / tau_p)) * (p_f - p_i)

    try:
        popt, _ = curve_fit(
            model, eta, values,
            p0=[values[0], values[-1], np.median(eta), 0.5 * (eta.max() - eta.min())],
            maxfev=40000,
        )
    except RuntimeError:
        return None
    residual = values - model(eta, *popt)
    return {
        "p_initial": float(popt[0]),
        "p_final": float(popt[1]),
        "eta_p": float(popt[2]),
        "tau_p": float(popt[3]),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "max_residual": float(np.max(np.abs(residual))),
        "p_range": float(abs(popt[1] - popt[0])),
    }


# ------------------------------------------- Part D: what the soft phase costs in shear

def shear_budget(run):
    """Delta ln(sigma^2/rho) along the derived trajectory, up to the bounce.

    Anisotropy obeys sigma^2 ~ a^-6 exactly, so the quantity that matters is
    ln(sigma^2/rho) = -6 ln a - ln rho, evaluated with the *computed* a and rho rather
    than with idealised power laws. Negative means the contraction suppressed shear.
    """
    contracting = run["H"] < 0.0
    ln_a = run["ln_a"][contracting]
    rho = run["rho"][contracting]
    budget = -6.0 * ln_a - np.log(rho)
    return float(budget[-1] - budget[0]), budget


def analytic_shear_budget(p_ek, p_soft, rho_i, rho_h, rho_c):
    """Two power-law stages: (rho_i -> rho_h) at p_ek, then (rho_h -> rho_c) at p_soft.

    In a stage of constant p, rho ~ a^{-2p} and sigma^2/rho ~ rho^{-(1 - 3/p)}, so
        Delta ln(sigma^2/rho) = -(1 - 3/p) ln(rho_end/rho_start).
    A pressureless stage has p = 3/2, hence exponent +1: it *undoes* suppression at the
    same rate the density grows.
    """
    ekpyrotic = -(1.0 - 3.0 / p_ek) * np.log(rho_h / rho_i)
    soft = -(1.0 - 3.0 / p_soft) * np.log(rho_c / rho_h)
    return float(ekpyrotic), float(soft), float(ekpyrotic + soft)


def minimum_hagedorn_density(p_ek, p_soft, rho_i, rho_c):
    """Smallest rho_H/rho_c for which the two stages still net a suppression."""
    a = 1.0 - 3.0 / p_ek     # positive for p_ek > 3
    b = 1.0 - 3.0 / p_soft   # negative for p_soft < 3
    if a <= 0.0 or a - b <= 0.0:
        return None
    # a ln(rho_H/rho_i) + b ln(rho_c/rho_H) > 0  ->  rho_H/rho_c > (rho_i/rho_c)^{a/(a-b)}
    return float((rho_i / rho_c) ** (a / (a - b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w-initial", type=float, default=4.3333333333333333)
    parser.add_argument("--rho-initial", type=float, default=1e-6)
    parser.add_argument("--rho-c", type=float, default=1.0)
    parser.add_argument("--rho-h-over-rho-c", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.0, help="0 scans for the threshold")
    parser.add_argument("--t-max", type=float, default=40000.0)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-16)
    parser.add_argument("--npz", type=str, default="")
    args = parser.parse_args()

    out = {}
    c = np.sqrt(2.0 * 1.5 * (1.0 + args.w_initial))

    print("=" * 80)
    print("Part A -- constant-w matter cannot produce the descent")
    print("=" * 80)
    out["no_go_constant_w"] = check_no_go()
    for key, value in out["no_go_constant_w"].items():
        print(f"  {key:34s} {value}")
    print("\n  d w_eff/d ln a = -3 Var_f(w) <= 0.  Contraction raises w_eff toward max(w_i).")

    print()
    print("=" * 80)
    print("Part B -- an oscillating scalar cannot either:  N_osc is depth-independent")
    print("=" * 80)
    print(f"  c = sqrt(2 eps) = {c:.6f}  ->  eps = {c*c/2:.4f},  w_ek = {c*c/3 - 1:.6f}")
    predicted_n, asymptote = analytic_oscillation_budget(c, 1e-3, args.rho_c)
    print(f"  analytic budget  N_osc -> (c/pi) sqrt(2/3) = {asymptote:.4f}  as |V_min| -> 0")
    print()
    print(f"  {'|V_min|/rho_c':>14s} {'m/|H|max':>9s} {'N_osc':>7s} {'N_pred':>7s}"
          f" {'p_early':>8s} {'p_late':>8s} {'min rho/rho_c':>14s} {'|constraint|':>13s}")
    scan = []
    for depth_fraction in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        depth = depth_fraction * args.rho_c
        potential = EkpyroticMinimumPotential(c, 1.0, 1.0 / (4.0 * depth))
        run = evolve_scalar(potential, args.rho_initial * args.rho_c, args.rho_c,
                            args.w_initial, args.t_max, args.rtol, args.atol)
        if not run["success"]:
            print(f"  {depth_fraction:14.4g}  INTEGRATION FAILED")
            continue
        mass = np.sqrt(potential.mass_squared)
        n_osc, _ = count_oscillations(run["t"], run["phi"], potential.phi_min, run["H"])
        n_pred, _ = analytic_oscillation_budget(c, depth, args.rho_c)
        averaged = cycle_average(run["t"], run["rho"], run["p"], 2.0 * np.pi / mass)
        contracting = run["H"] < 0.0
        early = contracting & (run["rho"] < 0.05 * depth)
        late = contracting & (run["rho"] > 0.7 * args.rho_c) & np.isfinite(averaged)
        entry = {
            "V_min_over_rho_c": depth_fraction,
            "m_over_H_max": float(mass / np.max(np.abs(run["H"]))),
            "N_osc_measured": n_osc,
            "N_osc_predicted": n_pred,
            "p_early": float(1.5 * (1.0 + np.nanmean(run["w"][early]))) if np.any(early) else None,
            "p_late": float(1.5 * (1.0 + np.nanmean(averaged[late]))) if np.any(late) else None,
            "min_rho_over_rho_c": float(np.min(run["rho"]) / args.rho_c),
            "constraint_max_relative": float(np.max(run["constraint_relative"])),
        }
        scan.append(entry)
        print(f"  {depth_fraction:14.4g} {entry['m_over_H_max']:9.3f} {n_osc:7.2f} {n_pred:7.2f}"
              f" {entry['p_early'] if entry['p_early'] else float('nan'):8.4f}"
              f" {entry['p_late'] if entry['p_late'] else float('nan'):8.4f}"
              f" {entry['min_rho_over_rho_c']:14.4e} {entry['constraint_max_relative']:13.2e}")
    out["oscillating_scalar_scan"] = scan
    print()
    print("  p_late stays near 3 (w -> 1, kination), never 1.5 (w = 0).")
    print("  Raising the depth raises m but delays the transition by exactly as much.")

    print()
    print("=" * 80)
    print("Part C -- energy conversion above the Hagedorn density")
    print("=" * 80)
    bound = 3.0 * args.w_initial
    print(f"  required rate   Gamma > 3 |H| w_1 (1-f)  ->  gamma > {bound:.3f} as f -> 0")
    print(f"  Hagedorn threshold rho_H/rho_c = {args.rho_h_over_rho_c:g}")
    print()
    gammas = [1.0, 3.0, 6.5, 13.0, 26.0, 52.0] if args.gamma == 0.0 else [args.gamma]
    print(f"  {'gamma':>7s} {'p_early':>8s} {'p_final':>8s} {'f_end':>9s}"
          f" {'tanh p_i':>9s} {'tanh p_f':>9s} {'eta_p':>8s} {'tau_p':>8s} {'fit rms':>9s}")
    transfers = []
    best = None
    for gamma in gammas:
        run = evolve_transfer(args.w_initial, args.rho_h_over_rho_c * args.rho_c, gamma,
                              args.rho_initial * args.rho_c, args.rho_c,
                              args.t_max, args.rtol, args.atol)
        if not run["success"] or not run["reached_bounce"]:
            print(f"  {gamma:7.2f}  did not reach the bounce")
            continue
        contracting = run["H"] < 0.0
        early = contracting & (run["rho"] < 0.1 * args.rho_h_over_rho_c)
        at_bounce = int(np.argmax(run["rho"]))
        fit = fit_tanh_profile(run, args.rho_c)
        entry = {
            "gamma": gamma,
            "p_early": float(np.mean(run["p_exponent"][early])) if np.any(early) else None,
            "p_final": float(run["p_exponent"][at_bounce]),
            "f_ekpyrotic_at_bounce": float(run["f_ekpyrotic"][at_bounce]),
            "constraint_max_relative": float(np.max(run["constraint_relative"])),
            "tanh_fit": fit,
        }
        transfers.append(entry)
        if fit is not None and entry["p_final"] < 2.0 and best is None:
            best = (gamma, run)
        print(f"  {gamma:7.2f} {entry['p_early'] if entry['p_early'] else float('nan'):8.4f}"
              f" {entry['p_final']:8.4f} {entry['f_ekpyrotic_at_bounce']:9.3e}"
              f" {fit['p_initial'] if fit else float('nan'):9.4f}"
              f" {fit['p_final'] if fit else float('nan'):9.4f}"
              f" {fit['eta_p'] if fit else float('nan'):8.3f}"
              f" {fit['tau_p'] if fit else float('nan'):8.3f}"
              f" {fit['rms_residual'] if fit else float('nan'):9.2e}")
    out["transfer_scan"] = transfers
    out["required_gamma_bound"] = bound

    print()
    print("=" * 80)
    print("Part D -- what the pressureless phase costs in shear")
    print("=" * 80)
    p_ek = 1.5 * (1.0 + args.w_initial)
    print(f"  {'rho_H/rho_c':>12s} {'ekpyrotic':>10s} {'soft':>8s} {'net (analytic)':>15s}"
          f" {'net (trajectory)':>17s} {'suppressed':>11s}")
    budgets = []
    for rho_h_fraction in [1e-4, 1e-3, 5e-3, 1e-2, 3e-2, 0.1, 0.3]:
        run = evolve_transfer(args.w_initial, rho_h_fraction * args.rho_c, 26.0,
                              args.rho_initial * args.rho_c, args.rho_c,
                              args.t_max, args.rtol, args.atol)
        if not run.get("success") or not run["reached_bounce"]:
            print(f"  {rho_h_fraction:12.4g}  did not reach the bounce")
            continue
        measured, _ = shear_budget(run)
        ekpyrotic, soft, net = analytic_shear_budget(
            p_ek, 1.5, args.rho_initial * args.rho_c, rho_h_fraction * args.rho_c, args.rho_c)
        budgets.append({
            "rho_H_over_rho_c": rho_h_fraction,
            "ekpyrotic_stage": ekpyrotic, "soft_stage": soft,
            "net_analytic": net, "net_trajectory": measured,
            "suppressed": bool(measured < 0.0),
        })
        print(f"  {rho_h_fraction:12.4g} {ekpyrotic:10.3f} {soft:8.3f} {net:15.3f}"
              f" {measured:17.3f} {str(measured < 0.0):>11s}")
    out["shear_budget"] = budgets
    floor = minimum_hagedorn_density(p_ek, 1.5, args.rho_initial * args.rho_c, args.rho_c)
    out["minimum_rho_H_over_rho_c"] = floor
    print()
    print(f"  analytic floor  rho_H/rho_c > (rho_i/rho_c)^{{a/(a-b)}} = {floor:.4e}")
    print(f"  with a = 1-3/p_ek = {1.0-3.0/p_ek:.4f}, b = 1-3/(3/2) = -1")
    print("  A pressureless stage has exponent +1: it gives back suppression as fast as")
    print("  rho grows, so the Hagedorn phase cannot start early.")

    if args.npz and best is not None:
        gamma, run = best
        np.savez_compressed(args.npz, gamma=gamma,
                            **{k: v for k, v in run.items() if isinstance(v, np.ndarray)})
        print(f"\n  saved trajectory (gamma = {gamma}) to {args.npz}")

    print()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
