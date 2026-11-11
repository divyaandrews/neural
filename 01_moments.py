"""
Accuracy of the linear-model moment approximations for the softplus
NB-INGARCH(1,1) model (Supplementary Tables S1-S2, summary in Section 2.3).

For 24 parameter settings (stationary means 2, 6, 12; r = 3) and
c in {1, 0.5, 0.25, ->0 (ReLU)} one path of length N is simulated and the
empirical mean, dispersion ratio and ACF(1..3) are compared with the closed
form moments of the linear NB-INGARCH(1,1) model.

Usage:  python 01_moments.py [--N 1000000] [--seed 2025]
"""
import argparse
import json
import os

import numpy as np

from ingarch_core import linear_moments

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tex", "tables")

SHAPES = [(0.3, 0.4), (0.3, -0.4), (-0.3, 0.4), (-0.3, -0.4),
          (0.4, 0.3), (0.4, -0.3), (-0.4, 0.3), (-0.4, -0.3)]
MEANS = [2, 6, 12]
CVALS = [1.0, 0.5, 0.25, 0.0]        # 0.0 encodes the ReLU limit c -> 0
R = 3.0


def settings():
    out = []
    for a1, b1 in SHAPES:
        for mu in MEANS:
            out.append((round(mu * (1 - a1 - b1), 4), a1, b1))
    return out


def run(N, seed):
    rng = np.random.default_rng(seed)
    S = settings()
    a0 = np.array([s[0] for s in S] * len(CVALS))
    a1 = np.array([s[1] for s in S] * len(CVALS))
    b1 = np.array([s[2] for s in S] * len(CVALS))
    c = np.repeat(CVALS, len(S))
    relu = c == 0
    cs = np.where(relu, 1.0, c)

    def link(eta):
        return np.where(relu, np.maximum(eta, 1e-10), cs * np.logaddexp(0, eta / cs))

    p_of = lambda lam: R / (R + lam)
    lam = link(a0 / (1 - a1 - b1))
    x = rng.negative_binomial(R, p_of(lam)).astype(float)
    hist = [x.copy()] * 3                       # x_{t-1}, x_{t-2}, x_{t-3}
    s1 = np.zeros_like(lam)
    s2 = np.zeros_like(lam)
    sc = np.zeros((3, len(lam)))
    burn = 2000
    for t in range(-burn, N):
        lam = link(a0 + a1 * x + b1 * lam)
        x = rng.negative_binomial(R, p_of(lam)).astype(float)
        if t >= 0:
            s1 += x
            s2 += x * x
            for h in range(3):
                sc[h] += x * hist[h]
        hist = [x, hist[0], hist[1]]
    mean = s1 / N
    var = s2 / N - mean ** 2
    rho = (sc / N - mean ** 2) / var
    rows = []
    for i in range(len(lam)):
        mu_l, disp_l, rho_l = linear_moments(a0[i], a1[i], b1[i], 1 / R)
        rows.append(dict(model=i % len(S) + 1, c=float(c[i]), alpha0=float(a0[i]),
                         alpha1=float(a1[i]), beta1=float(b1[i]),
                         mean=float(mean[i]), disp=float(var[i] / mean[i]),
                         rho=[float(v) for v in rho[:, i]],
                         mean_lin=float(mu_l), disp_lin=float(disp_l),
                         rho_lin=[float(v) for v in rho_l]))
    return rows


def write_tables(rows):
    os.makedirs(TAB, exist_ok=True)

    def block(cv):
        lines = []
        for r in [q for q in rows if q["c"] == cv]:
            lin = ""
            if cv in (1.0, 0.25):       # linear values do not depend on c
                lin = (f" & {r['mean_lin']:.3f} & {r['disp_lin']:.3f} & "
                       + " & ".join(f"{v:.3f}" for v in r["rho_lin"]))
            lines.append(f"{r['model']} & {r['alpha0']:g} & {r['alpha1']:g} & "
                         f"{r['beta1']:g} & {r['mean']:.3f} & {r['disp']:.3f} & "
                         + " & ".join(f"{v:.3f}" for v in r["rho"]) + lin + r" \\")
            if r["model"] % 3 == 0 and r["model"] < 24:
                lines.append(r"\addlinespace[2pt]")
        return lines

    # one table per pair of c values: simulated (c_a), simulated (c_b), linear
    for name, (ca, cb) in {"S1_moments_c1_c05": (1.0, 0.5),
                           "S2_moments_c025_relu": (0.25, 0.0)}.items():
        ra = [q for q in rows if q["c"] == ca]
        rb = [q for q in rows if q["c"] == cb]
        with open(os.path.join(TAB, name + ".tex"), "w") as f:
            for A, B in zip(ra, rb):
                f.write(f"{A['model']} & {A['alpha0']:g} & {A['alpha1']:g} & {A['beta1']:g} & "
                        + f"{A['mean']:.2f} & {A['disp']:.2f} & "
                        + " & ".join(f"{v:.3f}" for v in A["rho"]) + " & "
                        + f"{B['mean']:.2f} & {B['disp']:.2f} & "
                        + " & ".join(f"{v:.3f}" for v in B["rho"]) + " & "
                        + f"{A['mean_lin']:.2f} & {A['disp_lin']:.2f} & "
                        + " & ".join(f"{v:.3f}" for v in A["rho_lin"]) + r" \\" + "\n")
                if A["model"] % 3 == 0 and A["model"] < 24:
                    f.write(r"\addlinespace[2pt]" + "\n")

    # summary used in the main text: maximal / mean absolute deviations
    summ = {}
    for cv in CVALS:
        sub = [q for q in rows if q["c"] == cv]
        rel_mu = np.array([abs(q["mean"] - q["mean_lin"]) / q["mean_lin"] for q in sub])
        rel_d = np.array([abs(q["disp"] - q["disp_lin"]) / q["disp_lin"] for q in sub])
        d_rho = np.array([np.abs(np.array(q["rho"]) - np.array(q["rho_lin"])).max()
                          for q in sub])
        low = np.array([q["mean_lin"] == 2 or abs(q["mean_lin"] - 2) < 1e-6 for q in sub])
        summ[str(cv)] = dict(
            mean_rel_all=float(rel_mu.mean()), max_rel_mean=float(rel_mu.max()),
            max_rel_mean_mu2=float(rel_mu[low].max()),
            max_rel_mean_mu6_12=float(rel_mu[~low].max()),
            max_rel_disp=float(rel_d.max()), max_rel_disp_mu6_12=float(rel_d[~low].max()),
            max_abs_rho=float(d_rho.max()), max_abs_rho_mu6_12=float(d_rho[~low].max()))
    with open(os.path.join(TAB, "moments_summary.tex"), "w") as f:
        for cv, lab in zip(CVALS, ["1", "0.5", "0.25", r"\to 0"]):
            s = summ[str(cv)]
            f.write(f"${lab}$ & {100*s['max_rel_mean_mu2']:.1f} & "
                    f"{100*s['max_rel_mean_mu6_12']:.1f} & "
                    f"{100*s['max_rel_disp']:.1f} & {100*s['max_rel_disp_mu6_12']:.1f} & "
                    f"{s['max_abs_rho']:.3f} & {s['max_abs_rho_mu6_12']:.3f}" + r" \\" + "\n")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    rows = run(a.N, a.seed)
    summ = write_tables(rows)
    json.dump(dict(rows=rows, summary=summ, N=a.N, seed=a.seed),
              open(os.path.join(RES, "moments.json"), "w"), indent=1)
    for k, v in summ.items():
        print("c =", k, {kk: round(vv, 4) for kk, vv in v.items()})
