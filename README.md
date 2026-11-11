# Python code for "Softplus and Neural Architectures for Enhanced Negative Binomial INGARCH Modeling" (revised version)

This folder replaces the R scripts of the first submission. It reproduces every
table and figure of the revised manuscript and of the Supplementary Material.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `pandas`, `matplotlib`. Nothing else is needed.

`ingarch_core.py` compiles the small C file `_kernel.c` on first import if a C
compiler (`cc`, `gcc` or `clang`) is on the path. If none is found it silently
falls back to a pure NumPy implementation that gives the same numbers but is
much slower (the simulation study and the neural fits then take hours rather
than minutes). `ingarch_core.BACKEND` tells you which one is in use; set the
environment variable `INGARCH_NO_C=1` to force the NumPy version.

## Files

| File | Purpose | Output |
|---|---|---|
| `ingarch_core.py` | Model specification (`Spec`), likelihood and exact gradient, CML fitting with standard errors (`fit`, `fit_neural`), LR test of the Poisson hypothesis, simulation, residual diagnostics, PIT, scoring rules | – |
| `_kernel.c` | Optional C accelerator for the likelihood/gradient recursion | – |
| `plotting.py` | Plot helpers | – |
| `01_moments.py` | Accuracy of the linear moment approximations | Table 1, Tables S1–S2 |
| `02_simulation.py` | Finite-sample study of the CML estimator with estimated dispersion; size and power of the LR test | Tables 2–3, Tables S3–S6 |
| `03_syphilis.py` | Syphilis counts: softplus and neural fits, diagnostics, marginal distribution | Table 4, Figure 1, Tables S7–S8, Figures S1–S2 |
| `04_ed.py` | ED arrivals: seasonal softplus and neural models estimated in one step, forecasts | Table 5, Figures 2–3, Tables S9–S11, Figures S3–S5 |
| `05_stability.py` | Neural fits with and without the contraction constraint, several seeds | Table S12 |

Run the scripts from this folder, in any order:

```
python 01_moments.py          # about 1 minute
python 02_simulation.py       # about 45 minutes on 2 cores with the C kernel
python 03_syphilis.py         # about 15 minutes (many starting values)
python 04_ed.py               # about 25 minutes (many starting values)
python 05_stability.py        # about 10 minutes
```

They write LaTeX table bodies to `../tex/tables/`, figures to
`../tex/figures/` and machine-readable results to `../results/`. The
manuscript and the supplement `\input` the table bodies, so recompiling the
TeX files after a rerun updates all numbers in the tables. Numbers quoted in
the running text are not updated automatically.

`02_simulation.py` caches every (setting, sample size) task in
`../results/sim/` and can be interrupted and resumed. `03_syphilis.py` and
`04_ed.py` cache the fitted models (`../results/*_fits.pkl`); pass `--refit`
to ignore the cache.

## Parametrisation

The NB conditional distribution is parametrised by its mean `lam_t` and the
dispersion parameter `delta = 1/r` (NB size `r`), so that
`Var(X_t | past) = lam_t (1 + delta lam_t)` and `delta = 0` is the Poisson
model. `delta` is always the last element of the parameter vector.

## Data

* `../data/syphilis_wsc.csv` – weekly syphilis counts, West South Central
  states, 2007–2010 (series `a43` of the data set `syph` in the R package ZIM).
* `../data/ed_h2_hourly.csv` – hourly ED arrivals (hospital "h2",
  https://doi.org/10.5281/zenodo.7874721). The analysis uses 1 Dec 2018 00:00
  to 1 Feb 2019 23:00 UTC (1512 hours).

## Contraction constraint for neural models

Neural models with feedback (`L > 0` and lags of `lam` among the inputs) are
fitted subject to `contraction(par, spec) <= 0.95` (Proposition 1 and
equation (10) of the paper), enforced by a quadratic penalty. This guarantees
that the filtered conditional means forget their starting value. Pass
`constrain=False` to `fit` to switch it off; `05_stability.py` shows what
happens then. `Fit.contraction` returns the value at the optimum.

## Notes on reproducibility

All random number generators are seeded. The likelihood of neural INGARCH
models is multimodal; the scripts use several hundred starting values per
model, pooled over several seeds, and keep the best optimum. Table S12
(`05_stability.py`) shows how far independent searches agree, with and without
the contraction constraint.
