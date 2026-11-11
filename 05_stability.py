"""
Effect of the contraction constraint on neural NB-INGARCH fits
(Supplementary Table S12).

For the neu NB-INGARCH(1,1) model with L = 2 (syphilis counts) and for the
seasonal neural model with input set D and L = 2 (ED arrivals), the model is
fitted with and without the constraint contraction(u) <= 0.95 of
Proposition 1, for several independent multi-start searches (seeds).  For each
fit the script reports

  * the maximised log-likelihood,
  * the contraction constant kappa(u),
  * the sensitivity of the filtered conditional means to their starting value:
    max_t |lam_t(lam_0) - lam_t(1.5 lam_0)| over the last quarter of the
    sample (zero, up to rounding, if the filter forgets its starting value).

Usage:  python 05_stability.py
"""
import json
import os

import numpy as np
import pandas as pd

import ingarch_core as ic
from ingarch_core import Spec, default_starts, fit, nll, warm_start

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tex", "tables")


def sensitivity(f, x):
    x = np.ascontiguousarray(x, float)
    V = np.ascontiguousarray(f.spec.design(x))
    l1 = nll(f.par, x, V, f.spec, f.start, f.lam0, want_grad=False)[2]
    l2 = nll(f.par, x, V, f.spec, f.start, 1.5 * f.lam0, want_grad=False)[2]
    n = len(x)
    return float(np.abs(l1 - l2)[n - n // 4:].max())


def search(x, mk, start, seed, constrain, n_random, n_screen, n_warm):
    """One multi-start search for L = 1 and then L = 2 (same recipe as
    ingarch_core.fit_neural, with the constraint switched on or off)."""
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(x, float)
    prev = None
    for L in (1, 2):
        spec = mk(L)
        V = np.ascontiguousarray(spec.design(x))
        cand = default_starts(x, spec, rng, n_screen)
        if constrain:
            cand = [ic._shrink_feedback(p, spec) for p in cand]
        val = np.array([nll(p, x, V, spec, start, float(x.mean()), want_grad=False)[0]
                        for p in cand])
        val = np.where(np.isfinite(val), val, np.inf)
        starts = [cand[i] for i in np.argsort(val)[:n_random]]
        if prev is not None:
            starts += [warm_start(prev, spec, rng, j) for j in np.resize([0.3, 1.0, 3.0], n_warm)]
        prev = fit(x, spec, start=start, starts=starts, constrain=constrain, hessian=False)
    return prev


def main():
    rows = []
    # ---- syphilis
    y = pd.read_csv(os.path.join(HERE, "..", "data", "syphilis_wsc.csv"))["count"].values.astype(float)
    mk = lambda L: Spec((1,), (1,), L, "nb", scale=y.mean())
    for constrain in (True, False):
        for seed in (1, 2, 3):
            f = search(y, mk, 2, seed, constrain, 150, 30000, 60)
            rows.append(dict(data="Syphilis, order $(1,1)$", constrain=constrain, seed=seed,
                             loglik=f.loglik, kappa=f.contraction, sens=sensitivity(f, y),
                             converged=f.converged))
            print(rows[-1], flush=True)
    # ---- ED arrivals, input set D
    df = pd.read_csv(os.path.join(HERE, "..", "data", "ed_h2_hourly.csv"), parse_dates=["arrival_1h"])
    df = df[(df.arrival_1h >= "2018-12-01 00:00+00:00")
            & (df.arrival_1h <= "2019-02-01 23:00+00:00")].reset_index(drop=True)
    ts = df.arrival_1h
    x = df.n_attendance.values.astype(float)[:len(df) - 168]
    hour, dow = ts.dt.hour.values, ts.dt.dayofweek.values
    Z = np.column_stack([(dow == 0), (dow >= 5),
                         ts.dt.strftime("%m-%d").isin(["12-25", "12-26", "01-01"]).values]
                        + [f(2 * np.pi * k * hour / 24) for k in (1, 2, 3)
                           for f in (np.sin, np.cos)]).astype(float)[:len(x)]
    zn = ("Mon", "Wkd", "Hol", "s1", "c1", "s2", "c2", "s3", "c3")
    mk = lambda L: Spec((1, 24), (1,), L, "nb", Z=Z, znames=zn, scale=float(x.mean()))
    for constrain in (True, False):
        for seed in (1, 2, 3):
            f = search(x, mk, 24, seed, constrain, 60, 3000, 30)
            rows.append(dict(data="ED arrivals, input set D", constrain=constrain, seed=seed,
                             loglik=f.loglik, kappa=f.contraction, sens=sensitivity(f, x),
                             converged=f.converged))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RES, "stability.json"), "w"), indent=1, default=float)
    lines = []
    for i, r in enumerate(rows):
        first = i % 6 == 0
        lines.append((r["data"] if first else "") + " & "
                     + (("yes" if r["constrain"] else "no") if i % 3 == 0 else "")
                     + f" & {r['seed']} & {r['loglik']:.2f} & {r['kappa']:.2f} & "
                     + (f"{r['sens']:.3f}" if r["sens"] >= 5e-4 else "$<0.001$")
                     + " & " + ("yes" if r["converged"] else "no") + r" \\")
        if i % 6 == 5 and i < len(rows) - 1:
            lines.append(r"\addlinespace[3pt]")
    open(os.path.join(TAB, "stability.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
