"""
Finite-sample behaviour of the conditional ML estimator of the softplus
NB-INGARCH(1,1) model when the dispersion parameter delta = 1/r is estimated
jointly with (alpha0, alpha1, beta1), and of the likelihood-ratio test of
H0: delta = 0 (Poisson) against delta > 0.

  design "main"     : 16 settings (mu = 2.5, 10; r = 3),  c in {1, 0.5}
  design "boundary" : 8 settings with |alpha1| + |beta1| = 0.99 (r = 10)
  design "test"     : 4 settings, delta in {0, 0.05, 0.1, 1/3}  (size / power)

Each (setting, sample size) task is cached in results/sim/, so the script can
be interrupted and resumed.

Usage:  python 02_simulation.py [--reps 1000] [--jobs 2] [--tables-only]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from ingarch_core import Spec, fit, simulate_sp11

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SIM = os.path.join(RES, "sim")
TAB = os.path.join(HERE, "..", "tex", "tables")
SIZES = [100, 500, 1000]

MAIN = [(0.75, .25, .45), (2, -.25, .45), (3, .25, -.45), (4.25, -.25, -.45),
        (0.75, .45, .25), (3, -.45, .25), (2, .45, -.25), (4.25, -.45, -.25),
        (3, .25, .45), (8, -.25, .45), (12, .25, -.45), (17, -.25, -.45),
        (3, .45, .25), (12, -.45, .25), (8, .45, -.25), (17, -.45, -.25)]
BOUND = [(0.05, .75, .24), (7.55, -.75, .24), (2.45, .75, -.24), (9.95, -.75, -.24),
         (0.05, .24, .75), (2.45, -.24, .75), (7.55, .24, -.75), (9.95, -.24, -.75)]
TEST = [(0.75, .25, .45), (4.25, -.25, -.45), (3, .25, .45), (17, -.25, -.45)]
TEST_DELTAS = [0.0, 0.05, 0.1, 1 / 3]


def tasks(reps):
    out = []
    for c in (1.0, 0.5):
        for i, (a0, a1, b1) in enumerate(MAIN):
            for s in SIZES:
                out.append(("main", i + 1, a0, a1, b1, 1 / 3, c, s, reps))
        for i, (a0, a1, b1) in enumerate(BOUND):
            for s in SIZES:
                out.append(("boundary", i + 1, a0, a1, b1, 0.1, c, s, reps))
    for i, (a0, a1, b1) in enumerate(TEST):
        for d in TEST_DELTAS:
            for s in SIZES:
                out.append(("test", i + 1, a0, a1, b1, d, 1.0, s, reps))
    return out


def fname(task):
    design, i, a0, a1, b1, d, c, s, reps = task
    return os.path.join(SIM, f"{design}_m{i}_d{d:.3f}_c{c}_s{s}_R{reps}.npz")


def run_task(task):
    design, i, a0, a1, b1, d, c, s, reps = task
    fn = fname(task)
    if os.path.exists(fn):
        return fn
    seed = (1000003 * i + 7919 * s + int(1e4 * d) + int(100 * c)
            + {"main": 1, "boundary": 2, "test": 3}[design])
    rng = np.random.default_rng(seed)
    X = simulate_sp11(a0, a1, b1, d, s, c=c, n_paths=reps, rng=rng)
    nbs = Spec((1,), (1,), 0, "nb", c=c)
    pos = Spec((1,), (1,), 0, "poisson", c=c)
    par = np.full((reps, 4), np.nan)
    se = np.full((reps, 4), np.nan)
    ll = np.full((reps, 2), np.nan)
    for k in range(reps):
        x = X[k].astype(float)
        if x.max() == 0:
            continue
        try:
            f = fit(x, nbs)
            starts = [f.par[:3].copy(), np.array([x.mean() * 0.5, 0.2, 0.3])]
            g = fit(x, pos, starts=starts, hessian=False)
            par[k], se[k], ll[k] = f.par, f.se, (f.loglik, g.loglik)
        except Exception:
            pass
    np.savez(fn, par=par, se=se, ll=ll, true=np.array([a0, a1, b1, d]), c=c, s=s)
    return fn


def summarise(task):
    z = np.load(fname(task))
    par, se, ll, true = z["par"], z["se"], z["ll"], z["true"]
    ok = np.isfinite(par).all(1)
    par, se, ll = par[ok], se[ok], ll[ok]
    mean = par.mean(0)
    rmse = np.sqrt(((par - true) ** 2).mean(0))
    mad = np.abs(par - true).mean(0)
    lo, hi = par - 1.96 * se, par + 1.96 * se
    cover = np.mean((lo <= true) & (true <= hi) & np.isfinite(se), 0)
    lr = np.maximum(2 * (ll[:, 0] - ll[:, 1]), 0)
    return dict(n_ok=int(ok.sum()), mean=mean, rmse=rmse, mad=mad, cover=cover,
                rej05=float(np.mean(lr > 2.7055)), rej01=float(np.mean(lr > 5.4119)),
                zero=float(np.mean(par[:, 3] < 1e-6)), true=true)


def f3(v):
    return f"{v:.3f}"


def write_tables(reps):
    os.makedirs(TAB, exist_ok=True)
    allres = {}
    for design, cfgs, d in (("main", MAIN, 1 / 3), ("boundary", BOUND, 0.1)):
        for c in (1.0, 0.5):
            lines = []
            for i, (a0, a1, b1) in enumerate(cfgs):
                for s in SIZES:
                    r = summarise((design, i + 1, a0, a1, b1, d, c, s, reps))
                    allres[f"{design}|{i+1}|{c}|{s}"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                        for k, v in r.items()}
                    head = (f"{i+1} & {a0:g} & {a1:g} & {b1:g}" if s == SIZES[0]
                            else " & & & ")
                    lines.append(head + f" & {s} & "
                                 + " & ".join(f3(v) for v in r["mean"]) + " & "
                                 + " & ".join(f3(v) for v in r["rmse"]) + " & "
                                 + " & ".join(f"{v:.2f}" for v in r["cover"])
                                 + f" & {r['rej05']:.2f}" + r" \\")
                if i < len(cfgs) - 1:
                    lines.append(r"\addlinespace[2pt]")
            tag = "c1" if c == 1.0 else "c05"
            open(os.path.join(TAB, f"sim_{design}_{tag}.tex"), "w").write("\n".join(lines) + "\n")
    # condensed table for the main text: four representative settings, c = 1
    lines = []
    for i in (1, 4, 9, 12):
        a0, a1, b1 = MAIN[i - 1]
        for s in SIZES:
            r = summarise(("main", i, a0, a1, b1, 1 / 3, 1.0, s, reps))
            head = f"{a0:g} & {a1:g} & {b1:g}" if s == SIZES[0] else " & & "
            lines.append(head + f" & {s} & " + " & ".join(f3(v) for v in r["mean"]) + " & "
                         + " & ".join(f3(v) for v in r["rmse"]) + " & "
                         + " & ".join(f"{v:.2f}" for v in r["cover"]) + r" \\")
        if i != 12:
            lines.append(r"\addlinespace[2pt]")
    open(os.path.join(TAB, "sim_main_condensed.tex"), "w").write("\n".join(lines) + "\n")
    # size and power of the LR test
    lines = []
    for i, (a0, a1, b1) in enumerate(TEST):
        row = f"{a0:g} & {a1:g} & {b1:g}"
        for d in TEST_DELTAS:
            for s in SIZES:
                r = summarise(("test", i + 1, a0, a1, b1, d, 1.0, s, reps))
                allres[f"test|{i+1}|{d:.3f}|{s}"] = dict(rej05=r["rej05"], rej01=r["rej01"],
                                                         zero=r["zero"])
                row += f" & {r['rej05']:.3f}"
        lines.append(row + r" \\")
    open(os.path.join(TAB, "sim_lrtest.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(allres, open(os.path.join(RES, "simulation_summary.json"), "w"), indent=1)
    return allres


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(SIM, exist_ok=True)
    T = tasks(a.reps)
    if not a.tables_only:
        T = sorted(T, key=lambda t: t[7])
        with Pool(a.jobs) as pool:
            for n, fn in enumerate(pool.imap_unordered(run_task, T), 1):
                if n % 20 == 0:
                    print(f"{n}/{len(T)} tasks done", flush=True)
    write_tables(a.reps)
    print("tables written")
