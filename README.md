# Curvature-Bound Bounce Cosmology: Numerical Verification

Reproducibility package for:

> **Numerical Verification of a Curvature-Bound Bounce Cosmology:
> Effective-Field-Theory Control Pushes the Transition Below the Planck Scale**

Author: Ho Hyung Kim, Independent Researcher, Seoul, Republic of Korea.

Latest archived preprint (v1.2): [10.5281/zenodo.21813977](https://doi.org/10.5281/zenodo.21813977). Version 1.3 is the current source release and will receive its DOI when the Zenodo record is published.

## Scope

This is a numerical audit of selected consequences of a phenomenological bounce
ansatz. It is **not** a validated quantum-gravity theory, a derivation of a
CRBC-specific covariant action, or an analysis of observed CMB maps.

The package documents:

- reproduction of a constant-$p$ background correspondence to the published
  Ye--Piao $c_T=1$ beyond-Horndeski family;
- coefficient-gate checks for a supplied phenomenological $w(t)$ profile;
- exclusion of a minimal single-field k-essence route;
- a blue single-field adiabatic spectrum and the requirements of an added
  entropic sector;
- re-integration of the scalar and entropy spectra on a derived
  energy-transfer background;
- explicit negative results for the tested thermal Hagedorn, conformal-coupling,
  and background-reconstruction routes; and
- a kinetic-dependent derivative-coupling proxy, retained only as a candidate
  pending a variational derivation and a complete perturbative stability audit;
- simulation-only tests of a perturbative Bianchi-I quadrupole estimator.

## What remains required

1. Derive a covariant, degenerate action for the specific $w(t)$ profile by
   varying the full action, rather than reconstructing a supplied background.
2. Establish DHOST degeneracy and derive the scalar and tensor quadratic
   actions, including $Q_s$, $c_s^2$, $Q_T$, and $c_T^2$.
3. Derive the EFT cutoff and test strong coupling where the candidate proxy has
   $1+\nu\rho_2\sim10^3$, as well as non-linear Bianchi-I stability.
4. Supply reheating, amplitude normalization, and any black-hole-interior
   matching conditions.
5. Run a blinded, likelihood-based analysis of actual CMB maps after completing
  realistic foreground, mask, and polarization validation.

## Version 1.3 — derived-background and covariant-action audit

Version 1.3 makes the manuscript more falsifiable by preserving negative and
inconclusive results instead of treating the supplied fluid profile as a field
theory. On the derived energy-transfer background, the adiabatic result becomes
more strongly blue ($n_s=4.54$, versus $3.34$ for the earlier tanh background).
The entropy mechanism remains numerically available, but its central tuning
must be recalibrated for that background. The tested local thermal Hagedorn
production route, a canonical-scalar conformal coupling, and a
beyond-Horndeski background reconstruction do not provide a covariant origin
for the transition. The remaining derivative-coupling calculation is a
background-level proxy only: its equations have not been obtained by varying a
complete covariant action, and its DHOST, perturbative-stability, and
strong-coupling tests have deliberately not been claimed.

See [the v1.3 English revision record](REVISION_2026-08-06_v1.3_en.md) and
[its Korean counterpart](REVISION_2026-08-06_v1.3_kr.md).

## Version 1.2 — sub-Planck transition scale

The direct Hagedorn/string-density identification is rejected by the declared
EFT-control gate: \(\max(E_{\rm char}/\Lambda)=0.519\), above the 0.1
criterion. Control requires \(\rho_H/\rho_c>725\); at \(10^4\), the
independent CPU check gives 0.05188 with no gate violations. Consequently,
the dimensionless curvature-bound coefficient is not assumed to be unity:
the manuscript derives \(\eta=\mathcal K_{\max}\ell_P^4\sim10^{-16}\)--\(10^{-6}\).
The transition is therefore six to sixteen orders below Planck curvature. See
[Korean revision record](REVISION_2026-08-06_kr.md) and
[English revision record](REVISION_2026-08-06_en.md).

## Contents

- `paper/` — English manuscript PDF and LaTeX source.
- `code/` — selected Python scripts for the numerical gates and simulations.
- `outputs/` — trajectory and JSON reports used by the manuscript.
- `docs/` — Korean audit record, EFT coefficient contract, and Planck-analysis
  preregistration.

## Reproduction

Install the dependencies listed in `code/requirements-gpu.txt`. CPU execution
is supported for small checks; CUDA is required to reproduce the archived GPU
performance results. Begin with:

```bash
python code/crbc_background_scan.py --device cpu --points 128 --time-steps 257
python code/crbc_kessence_no_go_scan.py --device cpu --points 10000
python code/ye_piao_2019_corrected_reproduction.py --device cpu --points 3001 --extent 20
```

The published background correspondence is based on Ye and Piao,
[arXiv:1901.02202](https://arxiv.org/abs/1901.02202). The Planck tilt benchmark
is from [Planck 2018 X](https://arxiv.org/abs/1807.06211).

## Citation and license

This release is licensed under [CC BY 4.0](LICENSE). Until the v1.3 DOI is
registered, cite v1.2, [10.5281/zenodo.21813977](https://doi.org/10.5281/zenodo.21813977).
