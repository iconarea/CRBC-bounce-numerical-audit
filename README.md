# CRBC Numerical Audit

Reproducibility package for:

> **Numerical Audit of a Phenomenological Curvature-Bound Bounce Ansatz:
> Stability Gates, Exclusions, and Observational Requirements**

Author: Ho Hyung Kim, Independent Researcher, Seoul, Republic of Korea.

Archived preprint DOI: [10.5281/zenodo.21813443](https://doi.org/10.5281/zenodo.21813443)

## Scope

This is a numerical audit of a phenomenological bounce ansatz. It is **not** a
validated quantum-gravity theory, a derivation of a CRBC-specific covariant
action, or an analysis of observed CMB maps.

The package documents:

- reproduction of a constant-$p$ background correspondence to the published
  Ye--Piao $c_T=1$ beyond-Horndeski family;
- coefficient-gate checks for a supplied phenomenological $w(t)$ profile;
- exclusion of a minimal single-field k-essence route;
- a blue single-field adiabatic spectrum and the requirements of an added
  entropic sector;
- simulation-only tests of a perturbative Bianchi-I quadrupole estimator.

## What remains required

1. Derive one covariant beyond-Horndeski/DHOST action for the $w(t)$ profile and
   verify its background and perturbation equations independently.
2. Derive the EFT cutoff and examine strong coupling and non-linear Bianchi-I
   stability.
3. Supply reheating, amplitude normalization, and any black-hole-interior
   matching conditions.
4. Run a blinded, likelihood-based analysis of actual CMB maps after completing
  realistic foreground, mask, and polarization validation.

## Revision 2026-08-06

The revised string-scale cutoff test rejects a controlled bounce at the direct
Hagedorn/string-density identification: \(\max(E_{\rm char}/\Lambda)=0.519\)
exceeds the declared 0.1 criterion. The gate passes only when the bounce density
is set four orders of magnitude below the assumed Hagedorn density. See
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

This release is licensed under [CC BY 4.0](LICENSE). Cite
[10.5281/zenodo.21813443](https://doi.org/10.5281/zenodo.21813443).
