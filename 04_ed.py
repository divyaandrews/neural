"""
Hourly emergency department (ED) arrivals, 1 Dec 2018 - 1 Feb 2019
(hospital "h2" of Rostami-Tabar et al., https://doi.org/10.5281/zenodo.7874721).

Seasonality and calendar effects are modelled *inside* the INGARCH recursion
(one-step likelihood) instead of a second-stage SARIMAX model for residuals:

  input set A : X_{t-1}, lam_{t-1}                         (non-seasonal)
  input set B : A + X_{t-24}
  input set C : B + Monday / weekend / public-holiday indicators
  input set D : C + hour-of-day harmonics (periods 24, 12, 8 hours)

Each input set is used in a softplus INGARCH model (Poisson and NB) and in a
neural NB-INGARCH model with L = 1, 2, 3 hidden neurons.
Training sample: up to 25 Jan 2019; test sample: the last 7 days (168 hours),
one-step-ahead predictive distributions with parameters fixed at the training
estimates.

Usage:  python 04_ed.py [--refit]
(fitted models are cached in results/ed_fits.pkl; --refit ignores the cache)
"""
import copy
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import plotting as P
from ingarch_core import (Spec, acf, count_cdf, filter_lambda, fit, fit_neural,
                          ljung_box, lr_test_poisson, pearson_residuals,
                          pit_histogram, scores, simulate_fitted)
from scipy import stats

plt = P.plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "ed_h2_hourly.csv")
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tex", "tables")
FIG = os.path.join(HERE, "..", "tex", "figures")
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(DATA, parse_dates=["arrival_1h"])
df = df[(df.arrival_1h >= "2018-12-01 00:00+00:00")
        & (df.arrival_1h <= "2019-02-01 23:00+00:00")].reset_index(drop=True)
ts = df.arrival_1h
y = df.n_attendance.values.astype(float)
T = len(y)
NTEST = 168
SEEDS = (7, 8, 9)
START = 24
ytr = y[:T - NTEST]
hour, dow = ts.dt.hour.values, ts.dt.dayofweek.values
monday = (dow == 0).astype(float)
weekend = (dow >= 5).astype(float)
holiday = ts.dt.strftime("%m-%d").isin(["12-25", "12-26", "01-01"]).values.astype(float)
CAL = np.column_stack([monday, weekend, holiday])
CALN = ("Mon", "Wkd", "Hol")
HARM = np.column_stack([f(2 * np.pi * k * hour / 24) for k in (1, 2, 3)
                        for f in (np.sin, np.cos)])
HARMN = ("s1", "c1", "s2", "c2", "s3", "c3")
SETS = {"A": dict(xlags=(1,), Z=None, zn=()),
        "B": dict(xlags=(1, 24), Z=None, zn=()),
        "C": dict(xlags=(1, 24), Z=CAL, zn=CALN),
        "D": dict(xlags=(1, 24), Z=np.column_stack([CAL, HARM]), zn=CALN + HARMN)}
SCALE = float(ytr.mean())
out = dict(n=T, n_train=T - NTEST, n_test=NTEST, mean=y.mean(),
           disp=y.var(ddof=1) / y.mean(), acf=acf(y, 48)[[1, 12, 24, 48]].tolist(),
           mean_by_dow=[float(y[dow == k].mean()) for k in range(7)],
           mean_holiday=float(y[holiday == 1].mean()))


def make_spec(s, L, dist):
    d = SETS[s]
    return Spec(d["xlags"], (1,), L, dist, Z=d["Z"], znames=d["zn"], scale=SCALE)


def train_spec(spec):
    sp = copy.copy(spec)
    if sp.Z is not None:
        sp.Z = spec.Z[:T - NTEST]
    return sp


def evaluate(f, spec):
    lam = filter_lambda(y, f, spec)                 # one-step-ahead means, all t
    r = pearson_residuals(ytr[START:], f.lam[START:], f.delta)
    te = slice(T - NTEST, T)
    e = y[te] - lam[te]
    sc = scores(y[te], lam[te], f.delta)
    a = acf(r, 48)
    return dict(k=f.k, loglik=f.loglik, aic=f.aic, bic=f.bic, delta=f.delta,
                delta_se=f.se[-1] if f.spec.nb else np.nan,
                res_mean=r.mean(), res_var=r.var(ddof=1), acf1=a[1], acf2=a[2],
                acf24=a[24], acf48=a[48], lb48=ljung_box(r, 48)[0],
                rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)),
                logs=sc["logs"], rps=sc["rps"], lam=lam, res=r)


# ---------------------------------------------------------------- fits
CACHE = os.path.join(RES, "ed_fits.pkl")             # delete to refit everything
fits, evals = {}, {}
if os.path.exists(CACHE) and "--refit" not in sys.argv:
    fits = pickle.load(open(CACHE, "rb"))
else:
    for s in SETS:
        for dist in ("poisson", "nb"):
            fits[("sp", s, dist, 0)] = fit(ytr, train_spec(make_spec(s, 0, dist)),
                                           start=START, n_random=10)
        mk = lambda L, s=s: train_spec(make_spec(s, L, "nb"))
        for seed in SEEDS:           # pool the best optimum per L over seeds
            nf = fit_neural(ytr, mk, (1, 2, 3), start=START, n_random=60, n_warm=30,
                            n_screen=3000, seed=seed)
            for L, f in nf.items():
                key = ("neu", s, "nb", L)
                if key not in fits or f.loglik > fits[key].loglik + 1e-6:
                    fits[key] = f
        print("input set", s, "done", flush=True)
    pickle.dump(fits, open(CACHE, "wb"))
for k, f in fits.items():
    evals[k] = evaluate(f, make_spec(k[1], k[3], k[2]))

for s in SETS:
    lr, p = lr_test_poisson(fits[("sp", s, "nb", 0)], fits[("sp", s, "poisson", 0)])
    out[f"LR|{s}"] = dict(lr=lr, p=p)
for k, e in evals.items():
    f = fits[k]
    out["|".join(map(str, k))] = dict(
        {kk: vv for kk, vv in e.items() if kk not in ("lam", "res")},
        par=f.par.tolist(), se=f.se.tolist(), names=f.spec.names())

# ---------------------------------------------------------------- model comparison table
SETLAB = {"A": r"$X_{t-1}$", "B": r"A + $X_{t-24}$", "C": "B + calendar",
          "D": "C + harmonics"}
lines = []
best_bic = min(e["bic"] for e in evals.values())
for s in SETS:
    keys = [("sp", s, "poisson", 0), ("sp", s, "nb", 0)] + [("neu", s, "nb", L) for L in (1, 2, 3)]
    for j, k in enumerate(keys):
        e = evals[k]
        name = {("sp", "poisson"): "sp Poisson", ("sp", "nb"): "sp NB"}.get(
            (k[0], k[2]), f"neu NB, $L={k[3]}$")
        dl = "0" if k[2] == "poisson" else f"{e['delta']:.3f}"
        bic = f"\\textbf{{{e['bic']:.1f}}}" if e["bic"] == best_bic else f"{e['bic']:.1f}"
        lines.append((f"{s}: {SETLAB[s]}" if j == 0 else "") + f" & {name} & {e['k']} & "
                     f"{e['loglik']:.1f} & {e['aic']:.1f} & {bic} & {dl} & "
                     f"{e['res_var']:.2f} & {e['acf24']:.2f} & {e['rmse']:.2f} & "
                     f"{e['mae']:.2f} & {e['logs']:.3f} & {e['rps']:.3f}" + r" \\")
    if s != "D":
        lines.append(r"\addlinespace[2pt]")
open(os.path.join(TAB, "ed_models.tex"), "w").write("\n".join(lines) + "\n")

# selected models: best softplus and best neural model by BIC
k_sp = min((k for k in evals if k[0] == "sp"), key=lambda k: evals[k]["bic"])
k_neu = min((k for k in evals if k[0] == "neu"), key=lambda k: evals[k]["bic"])
out["best_sp"], out["best_neu"] = list(k_sp), list(k_neu)
f_sp, f_neu = fits[k_sp], fits[k_neu]

# Poisson counterpart of the selected neural model -> LR test of delta = 0
sp_p = train_spec(make_spec(k_neu[1], k_neu[3], "poisson"))
rng_p = np.random.default_rng(3)
st_p = [f_neu.par[:-1]] + [f_neu.par[:-1] + rng_p.normal(0, 0.05, f_neu.k - 1) for _ in range(20)]
f_neu_p = fit(ytr, sp_p, start=START, starts=st_p, hessian=False)
lr_n, p_n = lr_test_poisson(f_neu, f_neu_p)
out["LR|neural_selected"] = dict(lr=lr_n, p=p_n, loglik_poisson=f_neu_p.loglik)
e_p = evaluate(f_neu_p, make_spec(k_neu[1], k_neu[3], "poisson"))
out["neural_selected_poisson"] = {kk: vv for kk, vv in e_p.items() if kk not in ("lam", "res")}

# coefficient table of the softplus models B, C, D (NB)
names_all = ["alpha0", "alpha1", "alpha24", "beta1", "gamma_Mon", "gamma_Wkd", "gamma_Hol"] \
    + [f"gamma_{h}" for h in HARMN] + ["delta"]
tex = {"alpha0": r"$\alpha_0$", "alpha1": r"$\alpha_1$", "alpha24": r"$\alpha_{24}$",
       "beta1": r"$\beta_1$", "gamma_Mon": r"$\gamma_{\mathrm{Mon}}$",
       "gamma_Wkd": r"$\gamma_{\mathrm{Wkd}}$", "gamma_Hol": r"$\gamma_{\mathrm{Hol}}$",
       "delta": r"$\delta$"}
for i in (1, 2, 3):
    tex[f"gamma_s{i}"] = rf"$\gamma_{{s{i}}}$"
    tex[f"gamma_c{i}"] = rf"$\gamma_{{c{i}}}$"
lines = []
for nm in names_all:
    row = tex[nm]
    for s in "ABCD":
        f = fits[("sp", s, "nb", 0)]
        d = dict(zip(f.spec.names(), zip(f.par, f.se)))
        row += f" & {d[nm][0]:.3f} & ({d[nm][1]:.3f})" if nm in d else " & & "
    lines.append(row + r" \\")
open(os.path.join(TAB, "ed_coef_softplus.tex"), "w").write("\n".join(lines) + "\n")

# weights of the selected neural model
f = f_neu
L, width = f.spec.L, 1 + f.spec.K + f.spec.Q
ins = ["1"] + [f"$X_{{t-{l}}}/s$" for l in f.spec.xlags] + \
      [{"Mon": "Monday", "Wkd": "Weekend", "Hol": "Holiday"}.get(z, z) for z in f.spec.znames] + \
      [r"$\lambda_{t-1}/s$"]
W = f.par[L:L + L * width].reshape(L, width)
SEW = f.se[L:L + L * width].reshape(L, width)
has_se = bool(np.all(np.isfinite(f.se)))
cell = (lambda v, e: f"{v:.3f} & ({e:.3f})") if has_se else (lambda v, e: f"{v:.3f}")
ncol = (2 if has_se else 1) * L
lines = [r"$u^1_l$ (output weight) & " + " & ".join(cell(f.par[l], f.se[l]) for l in range(L))
         + r" \\", r"\midrule"]
for i, nm in enumerate(ins):
    for a_, b_ in (("s1", r"$\sin(2\pi h_t/24)$"), ("c1", r"$\cos(2\pi h_t/24)$"),
                   ("s2", r"$\sin(4\pi h_t/24)$"), ("c2", r"$\cos(4\pi h_t/24)$"),
                   ("s3", r"$\sin(6\pi h_t/24)$"), ("c3", r"$\cos(6\pi h_t/24)$")):
        if nm == a_:
            nm = b_
    lines.append(nm + " & " + " & ".join(cell(W[l, i], SEW[l, i]) for l in range(L)) + r" \\")
dtxt = f"{f.par[-1]:.4f}" + (f" ({f.se[-1]:.4f})" if has_se else "")
lines += [r"\midrule", r"$\delta$ & \multicolumn{" + str(ncol) + r"}{c}{" + dtxt + r"} \\"]
out["neural_has_se"] = has_se
open(os.path.join(TAB, "ed_weights_neural.tex"), "w").write("\n".join(lines) + "\n")
out["neural_L"] = L

# ---------------------------------------------------------------- marginal distributions
rng = np.random.default_rng(2025)
NSIM = 1000
ntr = T - NTEST
cmp_models = {"sp NB, set A": fits[("sp", "A", "nb", 0)],
              f"sp NB, set {k_sp[1]}": f_sp,
              f"neu NB, set {k_neu[1]}, $L={k_neu[3]}$": f_neu}
sims = {k: simulate_fitted(f, ytr, ntr, NSIM, rng)[:, START:] for k, f in cmp_models.items()}
obs_sum = P.marginal_summary(ytr[START:])
fm = lambda v, i: (f"{v:.2f}" if i < 4 else f"{v:.1f}")
lines = ["Observed & " + " & ".join(fm(v, i) for i, v in enumerate(obs_sum)) + r" \\"]
out["marginal"] = {"observed": obs_sum.tolist()}
for k, S in sims.items():
    M = np.array([P.marginal_summary(s) for s in S])
    lo, hi, me = np.quantile(M, .025, 0), np.quantile(M, .975, 0), M.mean(0)
    out["marginal"][k] = dict(mean=me.tolist(), lo=lo.tolist(), hi=hi.tolist())
    lines.append(k + " & " + " & ".join(fm(me[i], i) for i in range(7)) + r" \\")
    lines.append(" & " + " & ".join(
        (f"[{lo[i]:.2f}, {hi[i]:.2f}]" if i < 4 else f"[{lo[i]:.0f}, {hi[i]:.0f}]")
        for i in range(7)) + r" \\")
open(os.path.join(TAB, "ed_marginal.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figures
fig = plt.figure(figsize=(6.3, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.6, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
ax.plot(ts, y, color=P.C_OBS, lw=0.45)
ax.axvline(ts.iloc[T - NTEST], color=P.C_B, ls="--", lw=0.8)
ax.set_ylabel("Hourly ED arrivals")
ax.xaxis.set_major_formatter(P.matplotlib.dates.DateFormatter("%d %b"))
P.plot_acf(fig.add_subplot(gs[1, 0]), y, 50)
P.plot_acf(fig.add_subplot(gs[1, 1]), y, 50, partial=True)
fig.savefig(os.path.join(FIG, "ed_series.pdf"))
plt.close(fig)

# forecasts for the test week
te = slice(T - NTEST, T)
lam_n, lam_s = evals[k_neu]["lam"][te], evals[k_sp]["lam"][te]
r_ = 1 / f_neu.delta if f_neu.delta > 1e-10 else None
q = (lambda p: stats.nbinom.ppf(p, r_, r_ / (r_ + lam_n))) if r_ else (lambda p: stats.poisson.ppf(p, lam_n))
lo, hi = q(0.05), q(0.95)
out["pi90_coverage_neu"] = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
cdf_ = (lambda k: stats.nbinom.cdf(k, r_, r_ / (r_ + lam_n))) if r_ else (lambda k: stats.poisson.cdf(k, lam_n))
out["pi90_expected_coverage_neu"] = float(np.mean(cdf_(hi) - cdf_(lo - 1)))   # > 0.90 by discreteness
out["lam_max_neu"] = float(np.logaddexp(0, np.abs(f_neu.par[:f_neu.spec.L]).sum()))
out["largest_fitted_mean"] = dict(neu=float(evals[k_neu]["lam"].max()), sp=float(evals[k_sp]["lam"].max()),
                                  y_max=float(y.max()))


def diebold_mariano(d, lag=3):
    """DM statistic for a loss differential series d (Bartlett kernel)."""
    n, m = len(d), d.mean()
    g = [np.sum((d[k:] - m) * (d[:n - k] - m)) / n for k in range(lag + 1)]
    v = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))
    t = m / np.sqrt(v / n)
    return dict(mean_diff=float(m), stat=float(t), p=float(2 * stats.norm.sf(abs(t))))


from ingarch_core import count_logpmf
yt = y[te]
out["DM_sp_vs_neu"] = dict(
    squared_error=diebold_mariano((yt - lam_s) ** 2 - (yt - lam_n) ** 2),
    absolute_error=diebold_mariano(np.abs(yt - lam_s) - np.abs(yt - lam_n)),
    log_score=diebold_mariano(count_logpmf(yt, lam_n, f_neu.delta)
                              - count_logpmf(yt, lam_s, f_sp.delta)))
fig, ax = plt.subplots(figsize=(6.3, 1.9))
tt = ts.iloc[te]
ax.fill_between(tt, lo, hi, color="#c9dcef", lw=0, label="90% prediction interval")
ax.plot(tt, y[te], "o", ms=1.8, color=P.C_OBS, label="Observed")
ax.plot(tt, lam_n, color=P.C_A, lw=1.0, label="neu NB-INGARCH (one-step mean)")
ax.plot(tt, lam_s, color=P.C_B, lw=0.8, ls="--", label="sp NB-INGARCH (one-step mean)")
ax.set_ylabel("Hourly ED arrivals")
ax.xaxis.set_major_formatter(P.matplotlib.dates.DateFormatter("%d %b"))
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          handlelength=1.5, columnspacing=1.0)
fig.savefig(os.path.join(FIG, "ed_forecast.pdf"))
plt.close(fig)

# marginal distribution + PIT
edges = np.arange(0, 45, 3) - 0.5
edges[-1] = 200
labels = [f"{int(a + .5)}-{int(b - .5)}" for a, b in zip(edges[:-2], edges[1:-1])] + [r"$\geq$39"]
obs_h = np.histogram(ytr[START:], bins=edges)[0] / (ntr - START)
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1), gridspec_kw={"width_ratios": [2.3, 1, 1],
                                                              "wspace": 0.38})
ax = axs[0]
xpos = np.arange(len(labels))
ax.bar(xpos, obs_h, width=0.8, color="#d5d9de", label="Observed")
for (k, S), col, off, mk in zip(sims.items(), (P.C_B, P.C_C, P.C_A), (-0.22, 0, 0.22), "sov"):
    m, l_, h_ = P.marginal_bands(S, edges)
    ax.errorbar(xpos + off, m, yerr=[np.maximum(m - l_, 0), np.maximum(h_ - m, 0)], fmt=mk,
                ms=2.5, color=col, lw=0.8, capsize=1.5, label=k)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Relative frequency")
ax.set_xlabel("Count")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.12, 1.0), ncol=2,
          handlelength=1.0, columnspacing=0.8, fontsize=6.3)
fA = fits[("sp", "A", "nb", 0)]
P.plot_pit(axs[1], pit_histogram(ytr[START:], fA.lam[START:], fA.delta), "sp NB, set A")
P.plot_pit(axs[2], pit_histogram(ytr[START:], f_neu.lam[START:], f_neu.delta),
           f"neu NB, set {k_neu[1]}")
axs[1].set_ylim(0, 1.6)
axs[2].set_ylim(0, 1.6)
axs[2].set_ylabel("")
fig.savefig(os.path.join(FIG, "ed_marginal_pit.pdf"))
plt.close(fig)

# residual diagnostics (supplement)
show = [("sp", "A", "nb", 0), ("sp", "B", "nb", 0), ("sp", "C", "nb", 0), k_sp, k_neu]
show = list(dict.fromkeys(show))
fig, axs = plt.subplots(len(show), 2, figsize=(6.0, 1.5 * len(show)),
                        gridspec_kw={"hspace": 1.35, "wspace": 0.3})
for i, k in enumerate(show):
    lab = ("sp" if k[0] == "sp" else "neu") + f" NB-INGARCH, set {k[1]}" + \
          (f", L={k[3]}" if k[0] == "neu" else "")
    P.plot_acf(axs[i, 0], evals[k]["res"], 50, lab)
    P.plot_cpgram(axs[i, 1], evals[k]["res"], lab)
fig.savefig(os.path.join(FIG, "ed_residuals.pdf"))
plt.close(fig)

# exploratory box plots (supplement)
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.0), gridspec_kw={"width_ratios": [3, 1.6, 0.8],
                                                              "wspace": 0.3})
bp = dict(widths=0.6, patch_artist=True, flierprops=dict(marker=".", ms=2),
          medianprops=dict(color=P.C_OBS), boxprops=dict(facecolor="#c9dcef", lw=0.5),
          whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
axs[0].boxplot([y[hour == h] for h in range(24)], **bp)
axs[0].set_xticks(range(1, 25, 2))
axs[0].set_xticklabels(range(0, 24, 2))
axs[0].set_xlabel("Hour of day")
axs[0].set_ylabel("Hourly ED arrivals")
axs[1].boxplot([y[dow == k] for k in range(7)], **bp)
axs[1].set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
axs[1].set_xlabel("Day of week")
axs[2].boxplot([y[holiday == 0], y[holiday == 1]], **bp)
axs[2].set_xticklabels(["No", "Yes"])
axs[2].set_xlabel("Public holiday")
fig.savefig(os.path.join(FIG, "ed_boxplots.pdf"))
plt.close(fig)

json.dump(out, open(os.path.join(RES, "ed.json"), "w"), indent=1, default=float)
print(open(os.path.join(TAB, "ed_models.tex")).read())
print("selected:", k_sp, k_neu, "PI coverage", out["pi90_coverage_neu"])
