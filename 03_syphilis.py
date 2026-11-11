"""
Weekly syphilis counts, West South Central states of the US, 2007-2010
(series a43 of the data set `syph` in the R package ZIM; 209 observations).

  * softplus Poisson / NB INGARCH (1,0), (2,0), (1,1); delta = 1/r estimated
  * LR test of the Poisson hypothesis delta = 0
  * neural NB-INGARCH fits with L = 1, 2, 3 hidden neurons
  * residual diagnostics, PIT histograms
  * observed versus model-implied (simulated) marginal distribution

Usage:  python 03_syphilis.py [--refit]
(fitted models are cached in results/syphilis_fits.pkl; --refit ignores the cache)
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import plotting as P
from ingarch_core import (Spec, acf, fit, fit_neural, ljung_box, warm_start, lr_test_poisson,
                          pearson_residuals, pit_histogram, simulate_fitted)

plt = P.plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "syphilis_wsc.csv")
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tex", "tables")
FIG = os.path.join(HERE, "..", "tex", "figures")
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)

y = pd.read_csv(DATA)["count"].values.astype(float)
T = len(y)
SEEDS = (11, 12, 13)
START = 2                      # common effective sample: t = 3, ..., 209
ORDERS = {"(1,0)": ((1,), ()), "(2,0)": ((1, 2), ()), "(1,1)": ((1,), (1,))}
out = {"n": T, "mean": y.mean(), "disp": y.var(ddof=1) / y.mean(),
       "acf": acf(y, 3)[1:].tolist(), "zero_frac": float(np.mean(y == 0))}

# ---------------------------------------------------------------- fits
CACHE = os.path.join(RES, "syphilis_fits.pkl")       # delete to refit everything
if os.path.exists(CACHE) and "--refit" not in sys.argv:
    fits = pickle.load(open(CACHE, "rb"))
else:
    fits = {}
    for od, (xl, fl) in ORDERS.items():
        for dist in ("poisson", "nb"):
            fits[("sp", od, dist, 0)] = fit(y, Spec(xl, fl, 0, dist), start=START,
                                            n_random=10)
        # The neural likelihood is strongly multimodal for T = 209: pool the best
        # optimum per L over several independent multi-start searches.
        mk = lambda L, xl=xl, fl=fl: Spec(xl, fl, L, "nb", scale=y.mean())
        for seed in SEEDS:
            nf = fit_neural(y, mk, (1, 2, 3), start=START, n_random=300, n_warm=100,
                            n_screen=50000, seed=seed)
            for L, f in nf.items():
                key = ("neu", od, "nb", L)
                if key not in fits or f.loglik > fits[key].loglik + 1e-6:
                    fits[key] = f
        rng_w = np.random.default_rng(99)
        for L in (2, 3):                 # final warm-start pass from the pooled best
            key, prev = ("neu", od, "nb", L), fits[("neu", od, "nb", L - 1)]
            st_w = [warm_start(prev, mk(L), rng_w, j) for j in np.resize([0.3, 1.0, 3.0], 150)]
            f = fit(y, mk(L), start=START, starts=st_w + [fits[key].par])
            if f.loglik > fits[key].loglik - 1e-9:
                fits[key] = f
        print("order", od, "done", flush=True)
    pickle.dump(fits, open(CACHE, "wb"))


def diag(f):
    r = pearson_residuals(y[START:], f.lam[START:], f.delta)
    q, p = ljung_box(r, 10)
    return dict(mean=r.mean(), var=r.var(ddof=1), acf1=acf(r, 2)[1], lb=q, lbp=p, res=r)


# ---------------------------------------------------------------- Table: softplus
lines = []
for dist, lab in (("poisson", "sp PINGARCH"), ("nb", "sp NB-INGARCH")):
    for od in ORDERS:
        f = fits[("sp", od, dist, 0)]
        d = diag(f)
        nm = 3 if od != "(1,0)" else 2
        est = [f"{f.par[i]:.3f}" for i in range(nm)] + ["--"] * (3 - nm)
        ses = [f"({f.se[i]:.3f})" for i in range(nm)] + [""] * (3 - nm)
        if dist == "nb":
            est.append(f"{f.par[-1]:.3f}")
            ses.append(f"({f.se[-1]:.3f})")
            lr, p = lr_test_poisson(f, fits[("sp", od, "poisson", 0)])
            lrs = f"{lr:.1f}"
        else:
            est.append("0")
            ses.append("")
            lrs = "--"
        lines.append(f"{lab}{od} & " + " & ".join(est)
                     + f" & {f.loglik:.2f} & {f.aic:.2f} & {f.bic:.2f} & {lrs} & "
                     f"{d['var']:.2f} & {d['lbp']:.2f}" + r" \\")
        lines.append(" & " + " & ".join(ses) + r" & & & & & & \\")
    if dist == "poisson":
        lines.append(r"\midrule")
open(os.path.join(TAB, "syph_softplus.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: neural
la, lb = [], []
for od in ORDERS:
    cell = lambda f: f"{f.k} & {f.loglik:.2f} & {f.aic:.2f} & {f.bic:.2f}"
    la.append(f"{od} & " + cell(fits[("sp", od, "nb", 0)]) + " & "
              + cell(fits[("neu", od, "nb", 1)]) + r" \\")
    lb.append(f"{od} & " + cell(fits[("neu", od, "nb", 2)]) + " & "
              + cell(fits[("neu", od, "nb", 3)]) + r" \\")
open(os.path.join(TAB, "syph_neural_a.tex"), "w").write("\n".join(la) + "\n")
open(os.path.join(TAB, "syph_neural_b.tex"), "w").write("\n".join(lb) + "\n")

best_sp = min((fits[("sp", od, "nb", 0)] for od in ORDERS), key=lambda f: f.bic)
best_neu_key = min(((od, L) for od in ORDERS for L in (1, 2, 3)),
                   key=lambda k: fits[("neu", k[0], "nb", k[1])].bic)
best_neu_aic_key = min(((od, L) for od in ORDERS for L in (1, 2, 3)),
                       key=lambda k: fits[("neu", k[0], "nb", k[1])].aic)
out["best_neu_bic"] = list(best_neu_key)
out["best_neu_aic"] = list(best_neu_aic_key)

for k, f in fits.items():
    d = diag(f)
    out["|".join(map(str, k))] = dict(
        par=f.par.tolist(), se=f.se.tolist(), names=f.spec.names(), loglik=f.loglik,
        aic=f.aic, bic=f.bic, k=f.k, res_mean=d["mean"], res_var=d["var"],
        res_acf1=d["acf1"], lb10=d["lb"], lb10_p=d["lbp"],
        max_abs_input_weight=(float(np.abs(f.par[f.spec.L:f.spec.n_mean]).max())
                              if f.spec.L else None))
for od in ORDERS:
    lr, p = lr_test_poisson(fits[("sp", od, "nb", 0)], fits[("sp", od, "poisson", 0)])
    out[f"LR|{od}"] = dict(lr=lr, p=p)

# ---------------------------------------------------------------- marginal distributions
rng = np.random.default_rng(2025)
NSIM = 2000
cmp_models = {"sp PINGARCH(1,1)": fits[("sp", "(1,1)", "poisson", 0)],
              "sp NB-INGARCH(1,1)": fits[("sp", "(1,1)", "nb", 0)],
              f"neu NB-INGARCH{best_neu_aic_key[0]}, $L={best_neu_aic_key[1]}$":
                  fits[("neu", best_neu_aic_key[0], "nb", best_neu_aic_key[1])]}
sims = {k: simulate_fitted(f, y, T, NSIM, rng)[:, START:] for k, f in cmp_models.items()}
obs_sum = P.marginal_summary(y[START:])
lines = ["Observed & " + " & ".join(
    f"{v:.2f}" if i not in (4, 5, 6) else f"{v:.0f}" for i, v in enumerate(obs_sum)) + r" \\"]
out["marginal"] = {"observed": obs_sum.tolist()}
for k, S in sims.items():
    M = np.array([P.marginal_summary(s) for s in S])
    lo, hi, me = np.quantile(M, .025, 0), np.quantile(M, .975, 0), M.mean(0)
    out["marginal"][k] = dict(mean=me.tolist(), lo=lo.tolist(), hi=hi.tolist())
    lines.append(k + " & " + " & ".join(
        f"{me[i]:.2f}" if i not in (4, 5, 6) else f"{me[i]:.1f}" for i in range(7)) + r" \\")
    lines.append(" & " + " & ".join(
        (f"[{lo[i]:.2f}, {hi[i]:.2f}]" if i not in (4, 5, 6) else f"[{lo[i]:.0f}, {hi[i]:.0f}]")
        for i in range(7)) + r" \\")
open(os.path.join(TAB, "syph_marginal.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figures
fig = plt.figure(figsize=(6.3, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
ax.plot(np.arange(1, T + 1), y, color=P.C_OBS, lw=0.8)
ax.set_xlabel("Week")
ax.set_ylabel("Syphilis counts")
P.plot_acf(fig.add_subplot(gs[1, 0]), y, 20)
P.plot_acf(fig.add_subplot(gs[1, 1]), y, 20, partial=True)
fig.savefig(os.path.join(FIG, "syph_series.pdf"))
plt.close(fig)

edges = np.array([0, 1, 3, 6, 9, 12, 15, 20, 25, 30, 40, 80]) - 0.5
labels = ["0", "1-2", "3-5", "6-8", "9-11", "12-14", "15-19", "20-24", "25-29",
          "30-39", r"$\geq$40"]
obs_h = np.histogram(y[START:], bins=edges)[0] / (T - START)
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1), gridspec_kw={"width_ratios": [2.3, 1, 1],
                                                              "wspace": 0.38})
ax = axs[0]
xpos = np.arange(len(labels))
ax.bar(xpos, obs_h, width=0.8, color="#d5d9de", label="Observed")
for (k, S), col, off, mk in zip(sims.items(), (P.C_B, P.C_A, P.C_C), (-0.22, 0, 0.22), "sov"):
    m, lo, hi = P.marginal_bands(S, edges)
    ax.errorbar(xpos + off, m, yerr=[np.maximum(m - lo, 0), np.maximum(hi - m, 0)], fmt=mk, ms=3, color=col, lw=0.8,
                capsize=1.5, label=k)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Relative frequency")
ax.set_xlabel("Count")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.12, 1.0), ncol=2,
          handlelength=1.0, columnspacing=0.8, fontsize=6.3)
fp, fn = fits[("sp", "(1,1)", "poisson", 0)], fits[("sp", "(1,1)", "nb", 0)]
P.plot_pit(axs[1], pit_histogram(y[START:], fp.lam[START:], 0.0), "sp PINGARCH(1,1)")
P.plot_pit(axs[2], pit_histogram(y[START:], fn.lam[START:], fn.delta), "sp NB-INGARCH(1,1)")
axs[2].set_ylabel("")
fig.savefig(os.path.join(FIG, "syph_marginal_pit.pdf"))
plt.close(fig)

# residual diagnostics (supplement)
fig, axs = plt.subplots(6, 2, figsize=(6.0, 8.6), gridspec_kw={"hspace": 1.35, "wspace": 0.3})
i = 0
for dist, lab in (("poisson", "sp PINGARCH"), ("nb", "sp NB-INGARCH")):
    for od in ORDERS:
        r = diag(fits[("sp", od, dist, 0)])["res"]
        P.plot_acf(axs[i, 0], r, 20, f"{lab}{od}")
        P.plot_cpgram(axs[i, 1], r, f"{lab}{od}")
        i += 1
fig.savefig(os.path.join(FIG, "syph_residuals.pdf"))
plt.close(fig)

json.dump(out, open(os.path.join(RES, "syphilis.json"), "w"), indent=1, default=float)
print(open(os.path.join(TAB, "syph_softplus.tex")).read())
print(open(os.path.join(TAB, "syph_neural_a.tex")).read())
print(open(os.path.join(TAB, "syph_neural_b.tex")).read())
print(open(os.path.join(TAB, "syph_marginal.tex")).read())
print("best neural (BIC/AIC):", best_neu_key, best_neu_aic_key)
