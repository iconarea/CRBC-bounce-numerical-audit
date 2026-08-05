# Revision verification record — 2026-08-06

The revision replaces the arbitrary cutoff ratio with the explicit test
\(\Lambda=\rho_H^{1/4}\), introduces a phenomenological ekpyrotic-to-Hagedorn
\(w(t)\) profile, and records the result in the audit manuscript.

Independent CPU reruns reproduce the key distinction:

| Setup | \(\max(E_{\rm char}/\Lambda)\) | EFT gate |
| --- | ---: | --- |
| \(\rho_H/\rho_c=1\) | 0.5188 | fails; 1,840 of 9,999 points violate the declared criterion |
| \(\rho_H/\rho_c=10^4\) | 0.05188 | passes |

The scalar and tensor coefficient inequalities pass for the supplied profile in
both runs. This is a quadratic coefficient-gate result only. It does not derive
a covariant beyond-Horndeski/DHOST action, the phase-transition dynamics, or an
observational CMB likelihood.

Thus the revision excludes a controlled bounce placed directly at the
Hagedorn/string density within this EFT test. The passing case instead places
the phenomenological bounce four orders of magnitude below that density.
