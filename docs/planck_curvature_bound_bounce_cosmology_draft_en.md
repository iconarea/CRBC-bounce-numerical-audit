# Numerical Verification of a Curvature-Bound Bounce Cosmology: Effective-Field-Theory Control Pushes the Transition Below the Planck Scale

> **Status:** Concept-paper draft. Several entries numerically verified in the 2026-08 revision (§8); no observational test has been performed.
>
> **Revision:** 2026-08 — §8 added, then extended with §8.7–8.17 (deriving the equation-of-state transition, three re-tests on the derived background, two attempts at microscopic derivation, the relation to prior work, and three routes to a covariant action); §2.2, §6 and §7 updated; title changed. The first version was titled *A Conceptual Cosmological Model of a Planck-Scale Curvature Bound and Bounce Transition*, but §8.6 places the transition six to sixteen orders of magnitude below the Planck curvature, so that title no longer describes the content. The filename is left unchanged to preserve cross-document links. No observational data was used.
>
> **Author:** Kim, Ho-Hyeong (김호형)
>
> This document does not present a completed unification of general relativity and quantum mechanics. It asks whether classical singularities can be replaced by a non-singular transition when curvature approaches a fundamental short-distance scale. It is a specification of assumptions and tests, not an established physical theory.

## Abstract

Classical general relativity can lead to singularities of divergent curvature and density in black-hole interiors and in an extrapolation of the early Universe. Quantum gravity is expected to modify a continuum spacetime description at sufficiently short distances. This draft proposes a minimal phenomenological framework: (i) a coordinate-invariant curvature scalar, rather than a coordinate-dependent “strength of gravity,” is assigned a finite effective bound; (ii) a contracting homogeneous universe is represented by an effective non-singular bounce at a critical density; and (iii) an event horizon is distinguished from the genuinely high-curvature region where quantum-gravity effects might arise. String theory is considered only as motivation for a string length scale and for the breakdown of the low-energy curvature expansion; it does not presently derive the bound assumed here. The framework permits, but does not establish, an interpretation of the hot early Universe as an expanding phase following a high-density transition.

Numerical verification carried out after the first version (§8) converts part of the framework into computable statements. The effective bounce background is realised, ghost-free, inside the \(c_T=1\) beyond-Horndeski family, and the primordial scalar spectrum and quadrupolar anisotropy of that realisation are calculated. Two routes are excluded in the process: the minimal single-field realisation, and the constant-equation-of-state baseline. The adiabatic mode of the background is strongly blue and therefore cannot be the origin of the observed fluctuations; on the derived background it moves further away still, from \(n_s=3.34\) to \(4.54\). The equation-of-state transition is no longer inserted by hand at the level of a fluid: constant-\(w\) matter is excluded because \(dw_{\mathrm{eff}}/d\ln a=-3\,\mathrm{Var}_f(w)\le0\), a scalar oscillating about a minimum is excluded because the number of oscillations available is independent of that minimum's depth and never exceeds one, and what survives — energy conversion above a density threshold — yields a lower bound \(\gamma\gtrsim3w_1=13\) on the conversion rate, the prediction \(p_f=3/2\), and the new requirement \(\rho_H/\rho_c>(\rho_i/\rho_c)^{5/13}\). No covariant action generating that transition is available, however: a conformal coupling and the beyond-Horndeski reconstruction both close, for different reasons (§8.14--8.15), and only a coupling to the kinetic term survives as a candidate whose background equations have not yet been obtained by varying an action (§8.16) --- so the conclusions of §8.7 remain conditional on a fluid description. The **location** of the bound is also computed for the first time: requiring effective-field-theory control gives \(\eta\equiv\mathcal K_{\max}\ell_\mathrm{P}^4\sim10^{-16}\)–\(10^{-6}\), placing the transition not at the Planck curvature but at \(\rho_c^{1/4}\sim10^{16}\)–\(10^{17}\) GeV. A derivation of the critical-density coefficient, a microscopic mechanism fixing the relic amplitude, and any confrontation with data remain absent. The proposal is thus not a physical theory but a falsifiable specification, several of whose entries have now been decided numerically.

**Keywords:** quantum gravity; Planck scale; string length; curvature invariant; bounce cosmology; black-hole singularity; early Universe

## 1. Problem statement

In general relativity gravity is encoded in spacetime geometry, not in a single scalar force magnitude. Classical black-hole solutions and the backward extrapolation of FLRW cosmology can develop singularities under appropriate assumptions. It is more conservative to regard this divergence as a sign that the classical description has reached the edge of its applicability than to regard an actual physical infinity as established.

An event horizon and a quantum-gravity regime must not be conflated. The horizon of a sufficiently massive black hole can have small local curvature. This draft concerns a region in which curvature invariants or matter density approach a microscopic scale, not the horizon itself.

## 2. Assumptions and scales

### 2.1 Planck-scale benchmark

Combining the reduced quantum of action \(\hbar\), Newton’s constant \(G\), and the speed of light \(c\) gives the Planck length

\[
\ell_{\mathrm{P}}=\sqrt{\frac{\hbar G}{c^3}}.
\]

The associated dimensional density is

\[
\rho_{\mathrm{P}}=\frac{c^5}{\hbar G^2}.
\]

These are dimensional benchmarks at which both quantum and gravitational effects are expected to matter. They are not, by themselves, a derivation of a complete theory or of a universal maximum curvature.

### 2.2 Curvature-bound hypothesis

A ``maximum gravity'' is properly expressed through a coordinate-invariant curvature scalar rather than an observer-dependent acceleration. For the Kretschmann scalar

\[
\mathcal K=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}
\]

we assume that a **finite bound exists**,

\[
\mathcal K\le\mathcal K_{\max}.
\]

This is the only curvature assumption of the draft, and no microscopic origin for it is supplied.

**The location of the bound is not assumed.** This is what has changed from the first version. The Planck length is a natural unit in which to express the bound, not a claim that the bound sits there. Defining the dimensionless coefficient

\[
\eta\equiv\mathcal K_{\max}\,\ell_\mathrm{P}^{4},
\]

makes \(\eta\) a quantity to be **computed**, not declared.

The first version was written so as to read as \(\eta\sim O(1)\), that is, with the bound at the Planck curvature. §8.6 shows that reading does not hold: requiring effective-field-theory control gives

\[
\eta\sim10^{-16}\text{–}10^{-6},
\]

so the transition occurs six to sixteen orders of magnitude below the Planck curvature, at an energy scale \(\rho_c^{1/4}\sim10^{16}\)–\(10^{17}\) GeV.

The proposition this section leaves is therefore not ``a bound exists at the Planck scale'' but **``a finite bound exists, and its location is fixed by the self-consistency of the effective description''** --- a narrower and more testable statement.

### 2.3 String length scale and the high-curvature crossover

Perturbative string theory introduces the characteristic string length

\[
\ell_s=\sqrt{\alpha'}.
\]

This does not mean that every string has a universal maximum length, nor that string theory has proved a universal Planck-curvature cap. It means that at short distances the extended nature of strings and additional string excitations can become important. The relation between \(\ell_s\) and \(\ell_{\mathrm{P}}\) depends on the string coupling and on compactification data.

The low-energy derivative expansion of gravity is schematically reliable only when

\[
\alpha'\,|R_{\mu\nu\rho\sigma}|\ll1.
\]

When \(\alpha'|R_{\mu\nu\rho\sigma}|\gtrsim1\), higher-curvature corrections and string degrees of freedom cannot generally be neglected. The modest string-motivated hypothesis used here is therefore not that a bounce has been proved, but that a classical singularity need not remain a trustworthy extrapolation through this crossover. A finite effective curvature or a non-singular transition is a possibility to test.

The relevant statement about observational reach is causal rather than a global maximum-length rule:

\[
L_{\mathrm{coh}}(t)\leq D_{\mathrm{causal}}(t).
\]

Here \(L_{\mathrm{coh}}\) is the coherently connected segment of a string or string network relevant to one observer, while \(D_{\mathrm{causal}}\) denotes the particle horizon or event horizon in a specified cosmological model. This is a condition on observable causal influence. It does not prohibit longer structures or structures extending outside a horizon in an expanding universe.

### 2.4 Causality and unitarity hypothesis

This framework does not assume that causality ceases beyond an event horizon. Rather, it assumes that a more fundamental quantum causal rule or unitary evolution exists in the region where a classical spacetime description fails. The form of that rule is a central missing part of the model.

## 3. An effective bounce ansatz

For a spatially flat, homogeneous and isotropic universe, use units with \(c=1\). A simple phenomenological effective equation is

\[
H^2=\frac{8\pi G}{3}\rho\left(1-\frac{\rho}{\rho_c}\right),
\qquad \rho_c=\xi\rho_{\mathrm{P}},
\]

where \(H=\dot a/a\), \(a\) is the scale factor, and \(\xi\) is an undetermined dimensionless coefficient. At \(\rho\ll\rho_c\), the usual Friedmann equation is recovered. At \(\rho=\rho_c\), \(H=0\).

Together with energy conservation,

\[
\dot\rho+3H(\rho+p)=0,
\]

the ansatz gives

\[
\dot H=-4\pi G(\rho+p)\left(1-2\frac{\rho}{\rho_c}\right).
\]

For \(\rho+p>0\), \(\dot H>0\) at \(\rho=\rho_c\); a contracting branch can then join an expanding branch. This is an illustrative effective parametrization similar to forms used in bounce cosmology. It does not derive loop quantum gravity, string theory, conformal cyclic cosmology (CCC), or any other named theory.

## 4. Black holes, the Big Bang, and a new expanding region

The minimum narrative is:

```text
gravitational contraction or high-density black-hole interior
→ approach to a microscopic curvature/density regime
→ quantum-gravity crossover rather than a classical singularity
→ non-singular transition or bounce
→ an expanding spacetime region
```

If the final region is causally separated from a parent spacetime, an interior observer could describe its early hot, dense phase as a Big Bang. This is not the assertion that Hawking evaporation seen by an external observer is directly the Big Bang of our Universe. External evaporation and any proposed interior spacetime matching are distinct problems.

More specifically, this framework does not claim that a black hole directly turns into a white hole in the same exterior universe. The equation in Section 3 is a homogeneous cosmological ansatz and cannot simply be inserted into a black-hole interior. A black-hole application would require a separate spherically symmetric or rotating spacetime solution, a transition timescale, energy and information boundary conditions, and a causal structure showing whether exterior signals are absent. At this stage the defensible statement is only that a microscopic transition may occur before a classical singularity is reached.

Likewise, the scenario should not be phrased as matter created from nothing. A physical model must describe conversion among fields, radiation, and particle mass, and calculate particle production. Whether a stable, weakly interacting remnant could contribute to dark matter is a separate quantum-field-theoretic question, not a consequence of the bounce ansatz.

## 5. Relation to other proposals

CCC, associated with Roger Penrose, relates the remote conformal future of one aeon to the Big Bang of the next. The present bounce ansatz instead focuses on a dynamical high-density transition at finite effective curvature or density and does not assume a conformal matching surface. It should therefore be treated as a distinct hypothesis, not as a derivation or restatement of CCC.

Ekpyrotic and other cyclic scenarios are closer in their use of a contracting phase before a transition. This draft nevertheless assumes neither brane collisions nor a particular higher-dimensional construction. At this stage it is best regarded as a conceptual checklist of the ingredients needed to avoid a singularity and specify a transition rule.

## 6. Falsifiability and observational program

To become a scientific model, the framework must make at least one quantitative prediction that differs from standard inflationary cosmology. Relevant targets include:

1. **Primordial CMB fluctuations:** a specified low-multipole feature, oscillatory spectrum, or non-Gaussian signal produced by the transition. **(Partly supplied in §8.** The adiabatic \(n_s\simeq3.35\) is an exclusion; the quadrupolar modulation \(g_*(k)\propto k^{-0.70}\) of the anisotropic realisation is a signal whose *shape* is predicted. Its amplitude is still not.**)**
2. **Primordial gravitational waves:** a parameter-independent or tightly constrained prediction for the tensor-to-scalar ratio, tilt, or polarization spectrum. **(Not supplied.** The construction has \(c_T^2=1\) and so is compatible with GW170817, but that is a constraint, not a prediction.**)**
3. **Nucleosynthesis and structure formation:** a reheating history compatible with light-element abundances and the observed matter distribution.
4. **Black-hole observations:** an explicitly calculated mass remnant, ringdown deviation, or echo signature. Such signals cannot be claimed without a specified matching rule.

UHECR source-distance estimates do not directly test this proposal. They could at most be a secondary constraint if a particular compact-object model predicted correlated and repeatable UHECR, neutrino, and gravitational-wave signals beyond standard propagation expectations. Conventional jet acceleration, composition, and magnetic deflection would first need to be excluded.

### 6.1 Indirect JWST route

JWST cannot directly observe Planck-scale curvature or the region beyond a cosmological event horizon. It can, however, constrain early black-hole seeds and their host galaxies. A practical first step is a uniform census of \(z\gtrsim4\) Little Red Dot candidates in public NIRCam and NIRSpec data, jointly fitting spectroscopic redshift, line width, host stellar mass, and lensing-selection effects.

This program addresses the empirical question of how often apparently overmassive early black holes occur in low-mass hosts. It can compare stellar-remnant seeds, direct-collapse seeds, and other astrophysical formation channels. By itself it cannot establish a bounce, a wormhole, CCC, or a quantum-gravity curvature bound. Those ideas require independent, quantitative CMB or primordial-gravitational-wave predictions.

## 7. Limitations and next steps

The decisive missing items are:

- **open** — a microscopic derivation of \(\eta\) and \(\xi\). The effective-field-theory cutoff \(\Lambda\) belongs here too: it is the one entry of the §8 stability gate that remains an assumption.
- **partly settled** — the dynamics of the equation-of-state transition. Constant-\(w\) matter and a scalar oscillating about a minimum are both excluded, leaving energy conversion above a density threshold (§8.7); that route predicts \(p_f=3/2\) and squeezes the threshold from both sides, but does not fix the rate.
- **open** — the microscopic origin of the conversion rate. Local thermal Hagedorn production is excluded (§8.11); what remains is the requirement of a process near the unitarity limit at \(10^{16}\)–\(10^{17}\) GeV, for which no candidate exists.
- **partly settled** — the entropic mass. Its *form* follows from a non-minimal coupling \(\xi Rs^2\) (§8.12); its *value* \(\xi=2.889\) has no independent justification, must be tuned to 0.29%, and must be re-solved whenever the background changes (§8.10).
- **open** — a covariant action generating the specific \(w(t)\). The coefficient gate of §8.8 is an internal consistency check rather than a derivation, and §8.15 shows it passes a background with no transition more easily than the derived one. Of three routes tested, the conformal coupling (§8.14) and the beyond-Horndeski sector (§8.15) close; only a derivative coupling (§8.16) survives as a candidate, with four checks outstanding — varying the full action, the DHOST degeneracy conditions, the quadratic action of perturbations, and strong coupling at \(1+\nu\rho_2\sim10^3\).
- **open** — the reliance of §8.7, §8.10 and §8.11 on a **fluid** description. Writing \(p=w\rho\) per component defines a fluid but is not a property of a field, and §8.14 showed that difference decides whether the transition occurs. Section 8.16 gives a first indication that the descent can also happen for a field, but this is not settled.
- **partly settled** — a stability calculation including anisotropies and inhomogeneous perturbations. Scalar and tensor second-order perturbations on the isotropic background are decided over the whole range, and anisotropy is computed at the level of a perturbative test shear (§8.2). A non-perturbative Bianchi I re-derivation is still owed.
- **open** — global matching conditions from a black-hole interior, if invoked, to an expanding region. §8.5 quantifies the contraction such a route would require; it does not supply the matching.
- **open** — an account of information, entropy, and Hawking radiation.
- **partly settled** — numerical observables comparable with data. The scalar tilt and the quadrupole shape are computed (§8.3–8.4); the amplitude is not predicted, and the gravitational-wave side remains absent. On the derived background the adiabatic tilt *worsens*, from \(n_s=3.34\) to \(4.54\) (§8.9).

The tests added in §8 sharpened this list rather than shortening it. What remains unknown is no longer a set of free functions but three numbers — the transition threshold \(\rho_{\mathrm{tr}}\) (squeezed from both sides), the conversion rate (its requirement quantified), and the coupling \(\xi\) (its form fixed, its value not) — at the cost of two derivation routes closing.

The next step should be a model-selection exercise, not further verbal extrapolation: specify one **covariant action**; derive \(w(t)\), \(\rho_c\), perturbation equations, and observables from it; then compare the permitted parameter space with public data. Failure to outperform or remain compatible with the standard model is evidence against the hypothesis.

## 8. Numerical verification (2026-08)

This section reports only those items of §6 and §7 that were actually computed. Every number is backed by reproducible code in the verification records of the repository. No observational data has been used.

### 8.1 The background lies inside a known stable family

In units \(\alpha=1\) the effective bounce background of §3 is \(H(t)=t/[p(1+t^2)]\) with \(p=\tfrac32(1+w)\), which is **exactly** the background ansatz for which Ye and Piao exhibit a \(c_T=1\) beyond-Horndeski construction, at constant \(p\) and unit lapse. The residual of the effective Friedmann equation is \(2.8\times10^{-17}\): this is an identity, not a fit.

The ansatz of §3 is therefore not arbitrary phenomenology but a member of a family for which a stable microscopic realisation exists in the literature.

### 8.2 Stability: passed, but conditionally

Two candidates are excluded. The minimally coupled single field \(P(X)=KX+LX^2-V\) yields not a single point, out of a million sampled, that violates the null energy condition while remaining ghost-free and gradient-stable. The constant-equation-of-state baseline fails for a subtler reason: scalar subluminality requires \(c_s^2(+\infty)=p/3\le1\), that is \(w\le1\), while suppressing anisotropy through the contraction requires \(\Delta\ln(\sigma^2/\rho)=(3/p-1)\ln(1+\eta_i^2)<0\), that is \(w>1\). The two conditions do not overlap.

A \(w(t)\) transition background (\(w_i\to4.33\), \(w_f=0.6\)) passes. It is ghost-free throughout (\(\min Q_s=3.000\), \(\min Q_T=0.500\)), gradient-stable (\(\min c_s^2=0.114\)), subluminal (\(\max c_s^2=0.849\)), and suppresses shear (\(\Delta\ln(\sigma^2/\rho)=-6.56\)).

**Caveat.** The effective-field-theory cutoff is not derived. The gate's EFT-control entry passes only because \(\Lambda\) is assumed, so it is not counted as passed.

### 8.3 Primordial spectrum: the adiabatic mode is excluded

Integrating the Mukhanov–Sasaki equation with the \(z''/z\) that follows from the coefficient trajectory which passed the gate gives \(n_s\simeq3.35\), incomparable with Planck's \(n_s=0.9649\pm0.0042\). This is a known property of ekpyrotic contraction. The pipeline was validated against the analytic contracting-phase tilt \(n_s-1=3-2|\beta-\tfrac12|\), which it reproduces to within 0.07 over a factor five in \(p_i\).

**The adiabatic mode of the bounce therefore cannot originate the observed fluctuations.** The standard remedy, an entropic mechanism, was tested on the same background: a tachyonic entropy mass \(m_{\mathrm{eff}}^2=-\lambda H^2\) gives \(n_s=0.9925\) at \(\lambda=104\) and \(0.9572\) at \(\lambda=106.6\). The price is tuning — matching Planck within its error bar fixes \(\lambda\) to **0.29%**.

### 8.4 The coherent relic: shape predicted, amplitude not

Whether the bounce imprints a preferred band of wavenumbers was measured directly. A classical relic obeys the same linear mode equation as the fluctuations, so its envelope is the initial profile times the transfer function, and only the transfer function is background dynamics. The measured transfer function carries **no band-limited feature**: its cutoff lies two orders of magnitude below the bounce scale, so every observable mode is far longer than the transition and carries no record of having crossed it. Splitting the transition into two timescales changes nothing.

The angular structure is a stronger statement. On an isotropic FLRW background the mode equation depends on \(|k|\) alone, so no direction dependence can be generated at all. Within this framework the only dynamical source of a direction is residual shear, and it **forces**:

- a quadrupole, \(L=2\); no other multipole is available;
- a multiplicative modulation \(P(k)\left[1+g_*(k)(\hat{\mathbf k}\cdot\hat{\mathbf n})^2\right]\), not an additive component;
- a scale dependence \(d\ln g_*/d\ln k=-0.70\), consistent across two shear amplitudes, so the shape is a prediction.

The free parameters of the observational template drop from six to two, an overall size and an axis. **The initial shear amplitude is not fixed by the theory, so only an upper limit follows; detectability is not predicted.** Using the Planck quadrupole constraint of Kim and Komatsu, \(\Delta g_*=0.016\), the shear amplitude obeys \(C\lesssim2.3\times10^{-6}\). That published limit assumes a scale-independent \(g_*\), so the number is indicative pending a dedicated reanalysis.

### 8.5 What a black-hole-interior origin would require

If the reading of §4 is to be maintained, the level of anisotropy becomes the obstacle. A black-hole interior is Kantowski–Sachs rather than FLRW, so it starts at \(\sigma^2/\rho\sim O(1)\). Reaching the regime verified in §8.2 (\(\sim10^{-7}\)) demands \(\Delta\ln(\sigma^2/\rho)\simeq-16\), which at \(p_i=8\) corresponds to a contraction roughly two hundred times longer than the one computed. The stability verdict of §8.2 would also have to be re-established non-perturbatively in that regime.

**The black-hole–Big-Bang reading of §4 is therefore neither excluded nor supported.** What has changed is that the requirement is now quantified.

### 8.6 Joining ekpyrotic and Hagedorn phases, and a route to \(\xi\)

After §8.2 excluded constant \(w\), the open question was what supplies \(w(t)\). Joining two string-theoretic ingredients makes the requirements fit.

Flattening and smoothing during contraction operate only for \(w>1\), since they are a competition between curvature \(a^{-2}\), shear \(a^{-6}\) and the background \(a^{-3(1+w)}\). An ekpyrotic modulus with a steep negative potential supplies this. A Hagedorn phase, by contrast, puts energy into oscillator and winding modes and has \(w\simeq0\), so it **cannot** flatten — the same point as the known criticism that string gas cosmology does not by itself solve the flatness problem.

Joined **in sequence**, each phase covers the other's deficit: the ekpyrotic phase flattens, and once the density reaches the Hagedorn scale that phase takes over and provides the shallow equation of state near the bounce. Tested at \(w_f=0\) (that is, \(p_f=1.5\)), the configuration passes: \(\min Q_s=3.000\), \(\min Q_T=0.500\), \(\min c_s^2=0.098\), \(\max c_s^2=0.820\), \(w(-\infty)=4.33\), \(\Delta\ln(\sigma^2/\rho)=-6.55\), and the coefficient gate is passed at all 47,999 points.

A possible by-product --- a route to deriving \(\xi\) --- was examined and **closed**. If the Hagedorn scale fixes when the transition occurs, the EFT cutoff sits at the same place, \(\Lambda=M_s=\rho_c^{1/4}\), and is then not adjustable. Feeding that \(\Lambda\) into the gate gives \(E_{\mathrm{char}}/\Lambda=0.519\), five times the 0.1 criterion, with 8\,833 of 47\,999 points violating EFT control. For constant \(w\) the analytic value is \(E_{\mathrm{char}}/\Lambda=(4/3)^{1/4}/\sqrt2=0.760\), independent of \(w\). The result is physically unsurprising: if the bounce occurs at the string scale, the characteristic energy of the bounce *is* the string scale, which is not where an effective description is controlled.

Control would require \(\rho_H/\rho_c>725\), that is, a bounce some seven hundred times below the string density. But then the Hagedorn scale no longer fixes \(\rho_c\), and \(\xi\) is a free parameter again.

What survives and what does not: **survives** --- the stability of the joined background; ghost and gradient violations remain zero for both scalar and tensor sectors even in the run above. **Dies** --- the identification \(\rho_c=\rho_H\), and with it the route to \(\xi\). **Newly exposed** --- if the bounce really is at the string scale, then the stability verdict of §8.2 was itself obtained with an effective description in a regime where that description is not controlled.

**Option (a) is adopted here:** accept $\rho_H/\rho_c>725$ as a requirement and return $\xi$ to being a free parameter. The gate then passes. At $\rho_H/\rho_c=725$ one finds $E_{\mathrm{char}}/\Lambda=0.1000$ (marginal) and at $10^4$, $0.0519$ (comfortable), with zero EFT violations. Stability is untouched in all three cases.

**The price is the curvature scale, and it changes what this draft is called.** With $\rho_H/\rho_\mathrm{P}\sim g_s^4$ one has $\xi_{\mathrm{red}}\equiv\rho_c/M_\mathrm{P}^4=g_s^4/N$, and since $H=0$ at the bounce, $\mathcal K=12\dot H^2$, so the $\eta$ of §2.2 is $\eta=3\xi_{\mathrm{red}}^2(1+w)^2$.

| $g_s$ | $N=\rho_H/\rho_c$ | $\xi_{\mathrm{red}}$ | $\eta$ | $\rho_c^{1/4}$ [GeV] |
| --- | --- | --- | --- | --- |
| 0.1 | 725 | $1.4\times10^{-7}$ | $5.7\times10^{-14}$ | $4.7\times10^{16}$ |
| 0.3 | $10^4$ | $8.1\times10^{-7}$ | $2.0\times10^{-12}$ | $7.3\times10^{16}$ |
| 0.5 | 725 | $8.6\times10^{-5}$ | $2.2\times10^{-8}$ | $2.4\times10^{17}$ |
| 1.0 | $10^4$ | $1.0\times10^{-4}$ | $3.0\times10^{-8}$ | $2.4\times10^{17}$ |

Since $\eta$ lands between $10^{-16}$ and $10^{-6}$, **the curvature bound does not saturate at the Planck curvature** but six to sixteen orders of magnitude below it. The picture of §2.2, which tacitly assumes $\eta\sim O(1)$, does not hold in this construction; the actual transition scale is $\rho_c^{1/4}\sim10^{16}$–$10^{17}$ GeV. That is the same decade that tensor-mode observations probe, though the bounce is not inflation and the bound on $r$ does not transfer directly.

Gained and lost: **gained** --- EFT control, so that the stability verdict of §8.2 is obtained in a regime where the description is trustworthy. **Lost** --- $\xi$ remains a free parameter, and the original picture of a Planck-scale curvature bound does not survive.

Three caveats remain. (i) \(w_f=0\) approximates the Hagedorn value; the string-gas dynamics was not computed. (ii) **No equation drives \(w\) from 4.33 down to 0**; the test inserts a \(\tanh\) profile by hand. (iii) The cutoff remains an assumption, and it is now known that the assumption is not harmless.

### 8.7 The dynamics that drives the transition: two no-gos and one mechanism

The joining of §8.6 *supplies* the equation of state $w(t)$ as a tanh profile and reconstructs the rest from it. With four free parameters ($p_i,p_f,\eta_p,\tau_p$), "the ekpyrotic phase softens into a Hagedorn phase" was an assumption rather than a result. This subsection reports what happens when it is computed instead.

**First, matter with constant equation-of-state parameters cannot do it.** If each component obeys $\rho_i\propto a^{-3(1+w_i)}$ then

\[
\frac{d w_{\mathrm{eff}}}{d\ln a}=-3\,\mathrm{Var}_f(w)\ \le 0
\]

holds exactly, the variance being taken with weights $f_i=\rho_i/\rho$. Contraction has $d\ln a<0$, so $w_{\mathrm{eff}}$ can only *increase*, toward $\max_i w_i$. No number of components, in any proportion, produces the descent $w:4.33\to0$. Verified on a 40,001-point grid to a relative residual of $1.3\times10^{-7}$.

**Second, a scalar field oscillating about a minimum cannot either.** The potential $V(\phi)=V_1e^{-2c\phi}-V_0e^{-c\phi}$ has the ekpyrotic exponential tail ($\epsilon=c^2/2$) at large $\phi$ and turns over into a minimum, about which the virial theorem would give $\langle w\rangle\to0$. But the oscillator mass and the transition density are set by the *same* potential: $m\propto\sqrt{|V_{\min}|}$ while the transition occurs at $\rho\sim|V_{\min}|$, leaving a time $\propto|V_{\min}|^{-1/2}$ before the bounce. The number of oscillations available,

\[
N_{\mathrm{osc}}=\frac{m\,\Delta t}{2\pi}\ \longrightarrow\ \frac{c}{\pi}\sqrt{\frac{2}{3}}=1.04,
\]

is therefore *independent of the depth of the minimum*. Integrating seven cases spanning a factor of 100 in depth gave a single turning point in every one; the end state was kination ($p\simeq3$), not the pressureless $p=3/2$, and deepening the minimum made it worse.

**Third, what remains is energy conversion.** Relaxing conservation of the individual components to

\[
\dot\rho_1+3H(1+w_1)\rho_1=-\Gamma\rho_1,\qquad \dot\rho_2+3H\rho_2=+\Gamma\rho_1
\]

makes the descent require $\Gamma>3|H|\,w_1(1-f)$, that is $\gamma\equiv\Gamma/\sqrt{\rho/3}\gtrsim3w_1=13$. This is the string-gas reading of the Hagedorn transition: the soft phase arises not from a field settling into a minimum but from ekpyrotic energy being *converted into string excitations* once the density crosses $\rho_H$. The threshold is on density rather than on time, so it carries no width parameter. The numerics confirm the analytic bound: the final $w$ is $2.6\times10^{-3}$ at $\gamma=13$ and $6\times10^{-13}$ at $\gamma=26$, with the Hamiltonian constraint held to $1.1\times10^{-12}$.

**What is gained and what it costs.** Four free parameters become three physical ones ($c,\rho_H,\gamma$); $p_f=3/2$ becomes a *prediction* (measured 1.5000000000009); and the width $\tau_p$ is fixed by $\gamma$ alone ($\gamma\tau_p\simeq44$, unchanged when the initial density is varied over six orders of magnitude). The tanh turns out to be a 1.1% approximation to the derived shape.

The cost is anisotropy. A pressureless stage has $1-3/p=-1$, so it gives back ekpyrotic suppression by a factor $\rho_c/\rho_H$. A net suppression survives only if

\[
\frac{\rho_H}{\rho_c}>\left(\frac{\rho_i}{\rho_c}\right)^{5/13},
\]

a bound that did not exist before, and recovering the anisotropy budget $-6.55$ of §8.6 requires the ekpyrotic contraction to begin at $\rho_i\lesssim2\times10^{-10}\rho_c$. A genuinely pressureless Hagedorn phase therefore demands a contraction starting ten orders of magnitude below $\rho_c$; the six assumed in §8.6 are not enough.

**Consequently $p_f=2.4$ is discarded.** The derivation gives $3/2$; keeping 2.4 would require the soft component to have $w=0.6$, which a string gas does not.


### 8.8 Re-running the coefficient gate on the derived background

Because $c_s^2(+\infty)=p/3$ depends on $p_f$ directly, the result above does not leave the stability verdict of §8.1--8.6 intact, so the coefficient gate was re-run on the derived background. The unit map between the two calculations checks out: $\max|H|$ agrees with the closed-form $1/(2p_i)=1/16$ to eleven decimals.

The first run returned $\min c_s^2=-134.3$, but **it was tested for a pole before being recorded, and it was one**: the two extrema sat two grid points apart at $H\simeq0$. Taking the bounce from the sampled maximum of $\rho$ left it offset from the true bounce, at which the free function $1-c_2$ and $H$ vanish at different points and the removable $0/0$ becomes an actual pole. After locating $H=0$ by root-finding, $\mathcal M(0)$ matched the analytic $-k_2/4$ to ten decimals.

Even after that alignment, the coefficients of §8.6, $(k_1,k_2,\tau_1,\tau_2^2)=(12,5,1.7,0.59)$, **fail** on the derived background with $\min c_s^2=-2.162$ --- a genuine gradient instability this time, not a pole. Re-deriving them recovers a pass: at $(7.5,-6.0,26.833,6.75)$ one finds $\min Q_s=3.0000$, $\min c_s^2=0.09258$ and $\max c_s^2=0.7838$ over all 480,003 grid points, with 6.19% of 74,529 candidates viable, so the healthy region has finite volume. The substantive change is that $k_2$ reverses sign. Widening the window by a factor of 25, covering four further decades in density, leaves $\min c_s^2$ unchanged to five decimals.

**Moving $p_f$ from 2.4 to $3/2$ therefore preserves the existence of a stable completion**, though not the coefficients that realise it; the primordial spectrum has not yet been re-integrated on the derived background.


### 8.9 Re-integrating the primordial spectrum

The $n_s\simeq3.35$ of §8.3 came from the $z''/z$ of the §8.6 background, so it was re-integrated on the coefficient trajectory above.

The first thing encountered was a closed mode window, for a physical reason. The pressureless stage grows the scale factor by $(\rho_c/\rho_H)^{1/3}=4.64$ between $\rho_H$ and $\rho_c$, so at a given density $|z''/z|$ is 33 times its §8.6 value; the smallest wavenumber for which the WKB vacuum is legitimate rises above the super-horizon bound at read-out. Pulling the contracting grid ten times further into the past opens a window comparable to §8.6's (0.62 decade against 0.43). This is independent evidence pointing the same way as the requirement $\rho_i\lesssim2\times10^{-10}\rho_c$ of §8.7.

With the window open, the result is that **the blue tilt gets worse**. On the same grid (extent 4000) the §8.6 background gives $n_s=3.343$ and the derived background $n_s=4.541$. Changing the grid window by a factor of two or ten moves $n_s$ by about 0.05, four per cent of the difference of 1.198, so this is not a grid artefact. The convergence diagnostic is $6.2\times10^{-6}$ over 26 valid modes.

The first stored GPU JSON retained the tanh-branch metadata field `p_final: 2.4`. That field is not read by the derived branch: a direct code-path check gives $p_i=8$ and $p_f=1.5$ at the bounce and evaluation time. The reporting schema was corrected to record these actual values; the archived GPU output itself should be regenerated under the corrected schema before a new archival release.

Measured against Planck 2018's $n_s=0.9649\pm0.0042$, $n_s=4.54$ is **further away** than 3.34. The conclusion of §8.3 stands and is strengthened: the adiabatic mode cannot be the origin of the observed fluctuations, and a second field is not optional but required.


### 8.10 Re-testing the entropic mechanism

Since the adiabatic mode became bluer, the remedy of §8.3 was re-tested on the derived background.

The mode window narrows for the third time for the same reason: the pressureless stage grows the scale factor by 4.64, raising the WKB bound for the entropy modes too, so the default grid leaves the window empty. Widening it eightfold opens a window comparable to §8.6's.

Scanning $\lambda$ in $m_{\mathrm{eff}}^2=-\lambda H^2$ over five values there, **the mechanism works**. The measured $n_s(\lambda)$ tracks the analytic $\nu^2=\tfrac14+\beta(\beta-1)+\lambda\beta^2$ across the whole range, with a nearly constant deviation --- an additive shift of the zero point, not an error in the slope.

The **severity** of the tuning is unchanged, 0.293% either way, because the tolerance comes from $\beta=1/(p_i-1)$ and $p_i=8$ is preserved; changing $p_f$ does not touch it.

The **central value** does not carry over, however. The $\lambda$ that reproduces Planck moves from 106.935 to 108.099, a shift of 1.16 or 3.7 tolerance units. The offset from the analytic formula also grows 4.4-fold, from $-0.0046$ to $-0.0202$, because that formula assumes pure ekpyrotic contraction --- the same reason it was off by 1.2 for the adiabatic mode above.

**Moving $p_f$ from 2.4 to $3/2$ therefore leaves the entropic mechanism able to repair the blue spectrum**, though the tuned value, like the coefficients of §8.6, must be re-solved per background, and deriving $\lambda\simeq108$ from microphysics remains open.


### 8.11 Deriving the conversion rate --- a failure

Above, the conversion rate $\Gamma$ was an input. Deriving it from string microphysics closes the route rather than opening it.

The two scalings correspond to different mechanisms. $\Gamma\propto\sqrt\rho$ is $\Gamma\propto H$, which is not a wrong ansatz but the one appropriate to **gravitational particle production**: in a time-dependent background the production rate is set by non-adiabaticity, expressed through $H$ and $\dot H$, so there the rate does know the expansion rate. A **thermal, collisional** rate is instead set by the local energy scale, $\Gamma=c\rho^{1/4}$. Only the latter is tested below; the gravitational route is not computed. With the thermal scaling the descent condition becomes density-independent, $C>\sqrt3 w_1=7.506$, sharper than the $\gamma>3w_1=13$ obtained before.

A ceiling on the threshold also appears: the conversion completes only for $\rho_{\mathrm{tr}}\lesssim0.3\rho_c$. The anisotropy budget improves as the threshold rises (from $-3.13$ to $-13.30$), so $0.3$ is best for both, and together with the lower bound of §8.7 the threshold is squeezed from both sides.

**And there the route closes.** Section 8.6 lowered the bounce below the string density to recover effective-field-theory control, so $\rho_s/\rho_c>725$. Combining the two requirements puts the transition at 3--14\% of the string scale, where the Hagedorn density of states does not cancel the Boltzmann factor --- it only does so at $T=T_H$. Producing the heavy states that constitute a pressureless string gas is exponentially suppressed, not enhanced. Even in the most favourable case, overcoming the suppression would need an unsuppressed coupling of 59; elsewhere $10^2$--$10^{13}$.

Setting the suppression aside, the required coupling is $c=0.145$--$0.751$, and since $\Gamma=cE$ this means **a process operating near the unitarity limit** (a perturbative gauge interaction gives $\Gamma\sim10^{-3}E$).

**Within the local thermal Hagedorn-production ansatz tested here, the required conversion rate is incompatible with EFT-controlled sub-string-scale transition densities.** This repeats the pattern by which the route to $\xi$ closed in §8.6, and for the same reason: the string scale sits orders of magnitude above the bounce scale. What remains is a quantified requirement --- a process near the unitarity limit at $10^{16}$--$10^{17}$ GeV --- for which no candidate has been offered.

The scope of that exclusion is stated explicitly. Gravitational particle production, non-thermal production such as parametric resonance, tachyonic instabilities at points of enhanced symmetry, winding-mode annihilation and non-perturbative effects were none of them computed. This section therefore does not exclude string-theoretic mechanisms in general, and gives no ground for asserting that string theory cannot supply one.


### 8.12 Deriving the entropic mass --- a partial success

Asking where the tuned $\lambda$ comes from gives a partial answer, unlike the conversion rate.

**$\lambda$ is not a free parameter.** Demanding exact scale invariance gives the closed form $\lambda_{\mathrm{SI}}=2(p_i-1)^2+p_i-2$, matching the numeric solver to machine precision (exactly 104 at $p_i=8$). Choosing $p_i$ fixes $\lambda$.

**The form is derived.** In ekpyrotic contraction $R=6(2-p)H^2$ is negative for $p>2$, so a standard non-minimal coupling $\xi Rs^2$ gives $m_{\mathrm{eff}}^2=6\xi(2-p)H^2$, **automatically tachyonic and automatically proportional to $H^2$** for $\xi>0$. Of what §8.3's remedy was said to insert by hand, the form of the ansatz did not need inserting.

**The value is not derived, because the scalings disagree.** Every natural source is linear in $p$ ($|R|/H^2=6(p-2)$, $\dot\sigma^2/H^2=2p$, $|V|/H^2=p-3$) while the requirement $\simeq2p^2$ is quadratic, so the needed coefficient grows linearly with $p$: at $p_i=8$, $\xi=26/9=2.889$, seventeen times the conformal value. Lowering $p$ makes $\xi$ ordinary but drives the starting density down eight orders of magnitude; raising it does the reverse. Neither end supplies a reason for the value, and the 0.29% tuning transfers intact to whatever coefficient provides it.

**One incidental gain.** The requirement $\rho_i\lesssim2\times10^{-10}\rho_c$ of §8.7 assumed a threshold at $10^{-2}\rho_c$; at the optimum $\rho_{\mathrm{tr}}=0.3\rho_c$ established above, the same anisotropy budget needs only $\rho_i=1.2\times10^{-6}\rho_c$ --- four orders of magnitude of relief.

Unlike the conversion rate, the route did not close. But a tuned $O(1)$ number remains, and it must be re-solved per background, so **the claim that CRBC is simpler than inflation does not hold.**


### 8.13 Relation to prior work

None of the above establishes a new law. That vacuum-initial-condition ekpyrotic contraction yields a strongly blue adiabatic spectrum is an established property of the scenario, not a finding here: Lyth showed the collapsing-phase spectrum to be strongly scale-dependent (Lyth, *The primordial curvature perturbation in the ekpyrotic Universe*, Phys. Lett. B524 (2002) 1, arXiv:hep-ph/0106153), and the point is standard in the subsequent literature (Lehners, *Ekpyrotic and Cyclic Cosmology*, Phys. Rept. 465 (2008) 223, arXiv:0806.1245; Cai, Easson and Brandenberger, JCAP 08 (2012) 020, arXiv:1206.2382). What is added is the quantification that the derived background makes it worse. That stability and effective-field-theory control are the central difficulties for non-singular bounces is the subject of an existing literature (Ijjas, JCAP 02 (2018) 007, arXiv:1710.05990; Cai and Wilson-Ewing, JCAP 03 (2014) 026, arXiv:1402.3009), and the second of these is the closest precedent, treating exactly the loop-quantum-cosmology-like effective Friedmann equation used here. The coefficient gate above is an internal consistency check; no covariant action generating the specific $w(t)$ has been derived. That a Hagedorn phase does not by itself solve flatness is consistent with existing string-gas discussions. What is added is the combination of $\rho_s/\rho_c>725$ with $\rho_{\mathrm{tr}}\lesssim0.3\rho_c$ to exclude the thermal Hagedorn route quantitatively; nothing beyond that scope is claimed.

### 8.14 A candidate covariant action, and two obstructions

This is the item the conclusion names as most needed. The natural candidate was constructed and tested.

It is standard coupled quintessence: an ekpyrotic scalar with a pressureless sector coupled to it conformally, $A(\phi)=e^{\beta\phi}$ in $S_m[\psi,A^2(\phi)g_{\mu\nu}]$.

**A covariant origin for the transfer rate does exist.** $\Gamma_{\mathrm{eff}}=\beta\dot\phi$, and on the attractor $\dot\phi^2=2p\rho/3$, so $\Gamma_{\mathrm{eff}}\propto H$. The Hubble scaling used in §8.7 is what a conformal coupling supplies for free, and §8.11 went too far in calling it a gravitational ansatz without local justification: **§8.11's exclusion applies only to a *thermal* origin for the rate.** The density threshold also disappears --- the crossover happens by itself whenever $\beta<(c/2)(3/p-2)=-3.25$, so $\rho_{\mathrm{tr}}$ ceases to be a free parameter.

**But two obstructions block each other.** With an unbounded potential the pressureless sector can hold 94% of the energy (at $\beta=-8$) while $p_\phi=K-V\ge-V$ diverges alongside it, leaving $w_{\mathrm{eff}}=9.16$: draining $\rho_\phi=K+V$ does not drain $p_\phi$. Bounding the potential below caps $p_\phi$ but makes the field pass the minimum and turn around, so $e^{\beta\phi}$ stops growing --- all fifteen combinations tested land on $w=1.000$, kination. Since $p_\phi\ge-V$, a falling $w$ requires $V$ bounded below, and $V$ bounded below switches off the coupling that was to drive the transfer.

**Something more important surfaces here.** The two-fluid parametrisation of §8.7 is **not faithful to a scalar realisation.** Writing $p=w\rho$ per component is a definition for a fluid but does not hold for a field, and that difference decides whether the transition happens: in the fluid model suppressing $\rho_1$ suppresses $p_1$ automatically, and a field does not oblige. **The conclusions of §8.7, §8.10 and §8.11 must be re-read as conditional on such a fluid existing.**

What was tested is a canonical scalar with a conformal coupling and two families of exponential potential, bounded and unbounded. Derivative or disformal couplings, non-canonical kinetic terms, more than two fields, and **the possibility that the beyond-Horndeski sector itself carries the transition** were not tested. The last is the most promising: §8.8 has already confirmed that such a completion exists, so writing its action down is the natural next target.

### 8.15 The beyond-Horndeski sector does not carry the transition

This was the most promising route left by §8.14. It does not hold up.

**A reconstruction guarantees existence and therefore carries no information.** The Ye--Piao construction takes $H(t)$ and $a(t)$ as input and solves algebraically for the Lagrangian functions; it does not predict $H(t)$. That an action exists for the derived background therefore means only that one exists for *any* background fed to the algebra. For the sector to be informative, the stability gate would have to be selective.

**The comparison must be controlled.** Swapping $p(\eta)$ inside the closed-form ansatz $H=\eta/[p(1+\eta^2)]$ also fixes $\rho/\rho_c=1/(1+\eta^2)$ and so changes $H(t)$ as well: fitting a tanh to the derived background's own $p(t)$ to 1.2% and feeding it through that ansatz drops the viable count from 4,611 to **zero**. The comparison backgrounds were therefore built by integrating the same equations, with only $w(\rho)$ replaced.

| background | viable | fraction | claimed here |
|---|---|---|---|
| derived (§8.7) | 4,611 | **6.19%** | yes |
| reversed transition $w:0\to4.33$ | 158 | 0.21% | no |
| constant $w=4.33$ | 0 | 0.00% | no |
| constant $w=0$ | 5,970 | **8.01%** | no |

**The gate is not vacuous but neither is it selective.** Constant $w=4.33$ fails everywhere tested, yet a background with *no transition at all*, constant $w=0$, passes more easily than the derived one. Passing the gate of §8.8 is therefore not evidence that the background is right, and this sector cannot be the origin of $w(t)$.

**What does discriminate is anisotropy.** Constant $w=0$ has $1-3/p=-1$ and grows shear throughout the contraction, so it is excluded by the shear requirement rather than by stability --- the same ground on which §8.2 set aside the constant-$w$ baseline.

This route closes for a different reason than $\xi$ in §8.6 or $\gamma$ in §8.11. There a mismatch of scales blocked the way; here the obstruction is **methodological**. A reconstruction returns what was put in, and cannot derive what was not.

### 8.16 A derivative coupling: a promising candidate, not a covariant completion

Obstruction 2 of §8.14 was that bounding the potential makes the field turn around, after which $e^{\beta\phi}$ stops growing. Coupling to the **kinetic term** rather than to the field value removes that problem, since the kinetic energy keeps growing through contraction whether or not $\phi$ turns around:

\[
S=\int d^4x\sqrt{-g}\left[\frac{R}{2}-\frac{X}{2}-V(\phi)\right]+S_m[\psi,A^2(X)g_{\mu\nu}],\qquad \ln A=\frac{\nu}{2}X,\quad X=\dot\phi^2 .
\]

Total energy conservation then gives $\dot\rho_2+3H\rho_2=\nu\dot\phi\ddot\phi\rho_2$ and $\ddot\phi(1+\nu\rho_2)=-3H\dot\phi-V_{,\phi}$. The driver $\dot\phi\ddot\phi$ is positive whenever the kinetic energy grows, so **obstruction 2 is evaded by construction**, and the $X$-dependence appears as an effective kinetic normalisation $(1+\nu\rho_2)>0$. The bounded potential that obstruction 1 requires is now permitted.

**The descent happens.** Raising $\nu$ from 100 to $10^4$ drives the final $p$ monotonically from 3.000 to **1.504**, with the pressureless sector dominating ($f_{\mathrm{ek}}$ down to 0.25%). What failed in §8.14 succeeds here.

**Three costs.** Anisotropy trades against the depth of the descent: at $\nu=10^4$ the budget reaches $+0.401$ and suppression is gone, the compromise being $|V_{\min}|=10^{-2}$, $\nu=10^3$ with $p=1.541$ and a budget of $-3.360$. At the bounce $1+\nu\rho_2\sim10^3$, so the coupling is not perturbative. And the required $\ln A\simeq27$ is large but is set by the **initial pressureless abundance**: at a 10% initial fraction $\ln A=10.4$ suffices, though the budget then degrades to $-2.291$.

**But the background equations have not been derived from the action.** The pair above are *effective background equations inspired by* a candidate action, not the result of varying it. Symbolic computation established two things. Given $Q$, total energy conservation fixes the $\phi$ equation uniquely and reproduces the form above, so the system is **internally consistent**. However, the coupling term one would expect from varying the matter action changes the denominator from $(1+\nu\rho_2)$ to $(1+\nu\rho_2+\nu^2\rho_2X)$, and near the bounce $\nu X\simeq55$, so the two forms differ by about **fifty-five fold**. Since $X=-g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi$ already depends on the metric and on $\partial\phi$, varying $S_m[\psi,A^2(X)g_{\mu\nu}]$ produces terms beyond a simple two-fluid transfer, and those terms were not pinned down.

**The stability gate is therefore not run.** With the background unsettled at that magnitude, computing $Q_s$, $c_s^2$, $Q_T$, $c_T^2$ would not say what those values belong to. Four verifications are required: (1) re-derive the exact background equations by varying the full action; (2) check the DHOST degeneracy conditions so that no higher-derivative Ostrogradsky ghost is present; (3) obtain the quadratic action of scalar and tensor perturbations on that background and compute $Q_s$, $c_s^2$, $Q_T$, $c_T^2$; (4) check that large $\nu$ and $1+\nu\rho_2\sim10^3$ do not break strong coupling or the EFT cutoff. Items (2) and (4) are already flagged by cost (ii) above.

**What can and cannot be said.** It can be said that a bounded potential with a kinetic-dependent coupling permits a $p_f\simeq3/2$ transition, that it evades the obstruction of the $\phi$-conformal coupling, and that this is the **first numerical indication** that the fluid model may carry over to a field realisation. It cannot be said that a covariant action generating the $w(t)$ transition has been found. Until the four checks are passed, this result is recorded as a **candidate realisation --- a first viable derivative-coupling proxy**. The deeper problem exposed in §8.14 is likewise not resolved, only shown to be resolvable in principle.

### 8.17 What this section does not claim

1. The effective-field-theory cutoff is not derived.
2. No microscopic mechanism fixes the relic amplitude; only its shape is predicted.
3. The entropy field and its 0.29% tuning are inserted by hand and are not derived within this framework.
4. Section 8.8 narrows the *class* of dynamics that can drive the transition; it does not fix the conversion rate. The values $\gamma\simeq26$ and $\rho_H$ remain inputs, not quantities computed from a string-excitation production rate, and there is as yet no basis for assigning that role to phase synchronisation.
5. The coefficient gate, primordial spectrum, and entropy mechanism have all been re-run on the derived background (§8.8). These are conditional background-level checks: the conversion rate \(\gamma\), Hagedorn density \(\rho_H\), and retuned \(\lambda\simeq108\) remain inputs rather than quantities derived from microphysics.
6. No data analysis has been carried out. The analysis design is pre-registered, but no CMB map has been opened.

## Conclusion

A finite curvature or density scale can be a useful research hypothesis for replacing a classical singularity with a non-singular transition. String theory offers motivation for caution about the low-energy treatment of sufficiently high curvature, but it does not, in the form used here, prove a universal curvature maximum or a bounce.

The verification of §8 moved this draft in two directions at once. It advanced, in that the effective background belongs to a family with a stable microscopic realisation, and that a primordial spectrum and a quadrupole shape follow from it by calculation. It narrowed, in that the same calculation excluded the minimal single-field realisation, the constant-equation-of-state baseline, and the standing of the bounce's adiabatic mode as the origin of the CMB. What survives is a single quadrupolar signal whose shape is predicted and whose size is not — and even that rests on an anisotropy put in by hand.

Sections 8.7--8.13 then record an attempt to replace, one by one, the things that had been put in by hand. **Three things were gained.** Two exact arguments show that the equation-of-state transition is impossible for constant-$w$ matter and for a scalar oscillating about a minimum alike, and the route that survives predicts $p_f=3/2$. On that background the coefficient gate passes again and the entropic mechanism still works. And the *form* of the entropic-mass ansatz comes free from a non-minimal coupling.

**Three were lost.** On the derived background the adiabatic tilt gets *worse*, from $n_s=3.34$ to $4.54$. The route to the conversion rate through local thermal Hagedorn production closes, because effective-field-theory control pushed the bounce below the string scale and the transition must occur further below still. And, as in §8.6, neither the coefficients nor the tuned value carry over when the background changes.

What remains unknown is no longer a set of free functions but three numbers: the transition threshold $\rho_{\mathrm{tr}}$ (squeezed from both sides), the conversion rate (only its requirement quantified), and the coupling $\xi$ (its form fixed, its value not). That is progress, but while a tuned $O(1)$ number remains, **the claim that this framework is simpler than inflation does not hold.**

**And §8.14 exposed a condition prior to all three.** Testing a candidate covariant action showed that the two-fluid parametrisation of §8.7 is not faithful to a scalar realisation: $p=w\rho$ per component is a definition for a fluid but not for a field, and the $p_1$ that a fluid model suppresses automatically along with $\rho_1$ is not suppressed for a field. Within the class tested, two obstructions block each other and leave no way through. The conclusions of §8.7, §8.10 and §8.11 are therefore **conditional on such a fluid existing**, and that condition is not yet secured. Something was gained as well: the Hubble scaling of the transfer rate is supplied covariantly by a conformal coupling, so §8.11's exclusion applies only to a *thermal* origin.

The value of this draft lies not in declaring an answer to quantum gravity, but in identifying the equations and observations required before such an answer could be claimed. §8 records which of those items turned out to be computable, and which did not survive being computed. The single thing most needed is still a covariant action generating the specific $w(t)$. The most natural candidate failed in §8.14, and the beyond-Horndeski route that looked promising closed in §8.15 for a **methodological** reason: a reconstruction returns what was put in, and its gate passes a background with no transition more easily than the derived one. From that list, **a derivative coupling remains a promising candidate** (§8.16): coupling to the kinetic term keeps the transfer alive after the field turns around, resolving both obstructions of §8.14 and driving the final $p$ to $3/2$. But its background equations have not been obtained by varying an action, and they differ from the form one would expect by about fifty-five fold near the bounce. **A covariant action has therefore not been found**, and the stability gate is withheld until the background is settled. Without it, every result above remains conditional on a background that was supplied rather than derived --- and now also on that background admitting a fluid description.

## References

1. NIST CODATA, *Recommended Values of the Fundamental Physical Constants*. https://physics.nist.gov/cuu/pdf/RevModPhys.93.025010.pdf
2. R. Penrose, “On the Gravitization of Quantum Mechanics 2: Conformal Cyclic Cosmology,” *Foundations of Physics* 44, 873–890 (2014). https://doi.org/10.1007/s10701-013-9763-z
3. M. Novello and S. E. P. Bergliaffa, “Bouncing Cosmologies,” *Physics Reports* 463, 127–213 (2008). https://arxiv.org/abs/0802.1634
4. D. Battefeld and P. Peter, “A Critical Review of Classical Bouncing Cosmologies,” *Physics Reports* 571, 1–66 (2015). https://arxiv.org/abs/1406.2790
5. L. N. Chang, Z. Lewis, D. Minic and T. Takeuchi, “On the Minimal Length Uncertainty Relation and the Foundations of String Theory” (2011). https://arxiv.org/abs/1106.0068
6. L. McAllister and E. Silverstein, “String Cosmology: A Review,” *General Relativity and Gravitation* 40, 565–605 (2008). https://arxiv.org/abs/0710.2951
7. NASA, “Webb Reveals Black Hole That Formed Before Its Galaxy” (2026). https://science.nasa.gov/missions/webb/nasas-webb-reveals-black-hole-that-formed-before-its-galaxy/
8. G. Ye and Y.-S. Piao, “Implication of GW170817 for cosmological bounces,” *Communications in Theoretical Physics* 71, 427 (2019) — the \(c_T=1\) beyond-Horndeski construction used in §8.1. https://arxiv.org/abs/1901.02202
9. T. Kobayashi, “Generic instabilities of non-singular cosmologies in Horndeski theory: a no-go theorem” (2016) — background to the exclusions of §8.2. https://arxiv.org/abs/1606.05831
10. J. Kim and E. Komatsu, “Limits on anisotropic inflation from the Planck data” — the quadrupolar-anisotropy constraint used in §8.4. https://arxiv.org/abs/1310.1605
11. Planck Collaboration, “Planck 2018 results. VII. Isotropy and statistics of the CMB” (2020) — observational reference for §8.3–8.4. https://arxiv.org/abs/1906.02552
