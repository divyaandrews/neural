"""
Softplus and neural INGARCH models with Poisson / negative binomial
conditional distribution -- core routines.

Model
-----
    X_t | F_{t-1} ~ Poisson(lam_t)   or   NB(r, p_t) with mean lam_t,
    Var(X_t | F_{t-1}) = lam_t (1 + delta * lam_t),      delta = 1/r  >= 0,
    lam_t = sp_c(eta_t),   sp_c(x) = c * log(1 + exp(x / c)).

  * softplus INGARCH (L = 0):
        eta_t = a0 + sum_i a_i X_{t-i} + gamma'z_t + sum_j b_j lam_{t-j}
  * neural INGARCH (L hidden neurons, logistic hidden activation):
        eta_t = sum_l u_l * logistic(w_l0 + w_l'(X-lags/s, z_t) + wl_l'lam-lags/s)

The dispersion parameter is estimated jointly with the mean parameters and is
parametrised as delta = 1/r, so that delta = 0 is the Poisson model.

Everything needed by the scripts 01_...04_ lives here: likelihood + exact
gradient, conditional ML fitting with standard errors, simulation, residual
diagnostics and scoring rules.  A small C kernel (_kernel.c) is compiled
automatically if a C compiler is found; otherwise an equivalent pure NumPy
implementation is used (identical results, only slower).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# optional C kernel
# --------------------------------------------------------------------------
def _load_kernel():
    if os.environ.get("INGARCH_NO_C"):
        return None
    src = os.path.join(HERE, "_kernel.c")
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    lib_path = os.path.join(HERE, "_kernel" + ext)
    try:
        if (not os.path.exists(lib_path)
                or os.path.getmtime(lib_path) < os.path.getmtime(src)):
            tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=HERE, delete=False)
            tmp.close()
            for cc in ("cc", "gcc", "clang"):
                try:
                    subprocess.run([cc, "-O2", "-shared", "-fPIC", src, "-o",
                                    tmp.name, "-lm"], check=True,
                                   capture_output=True)
                    os.replace(tmp.name, lib_path)
                    break
                except Exception:
                    continue
            else:
                os.unlink(tmp.name)
                return None
        lib = ctypes.CDLL(lib_path)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int)
        lib.ingarch_nll.restype = ctypes.c_double
        lib.ingarch_nll.argtypes = [dp, ctypes.c_int, dp, dp, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ip,
                                    ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_double, ctypes.c_double, dp, dp,
                                    ctypes.c_int]
        return lib
    except Exception:
        return None


_LIB = _load_kernel()
BACKEND = "C kernel" if _LIB is not None else "pure NumPy"


def softplus(eta, c=1.0):
    return c * np.logaddexp(0.0, np.asarray(eta, float) / c)


# --------------------------------------------------------------------------
# model specification
# --------------------------------------------------------------------------
@dataclass
class Spec:
    """Specification of a softplus (L=0) or neural (L>0) INGARCH model."""
    xlags: tuple = (1,)        # lags of X entering eta_t
    flags: tuple = ()          # lags of lam entering eta_t (feedback)
    L: int = 0                 # hidden neurons (0 = softplus INGARCH)
    dist: str = "nb"           # "nb" or "poisson"
    c: float = 1.0             # softplus tuning parameter
    Z: np.ndarray | None = None  # (T, m) exogenous regressors, aligned with x
    znames: tuple = ()
    scale: float = 1.0         # input scale s for neural models
    label: str = ""

    @property
    def K(self):
        return len(self.xlags) + (0 if self.Z is None else self.Z.shape[1])

    @property
    def Q(self):
        return len(self.flags)

    @property
    def nb(self):
        return self.dist == "nb"

    @property
    def n_mean(self):
        if self.L == 0:
            return 1 + self.K + self.Q
        return self.L * (2 + self.K + self.Q)

    @property
    def npar(self):
        return self.n_mean + (1 if self.nb else 0)

    @property
    def maxlag(self):
        return max(tuple(self.xlags) + tuple(self.flags) + (0,))

    def names(self):
        if self.L == 0:
            nm = (["alpha0"] + [f"alpha{l}" for l in self.xlags]
                  + [f"gamma_{z}" for z in self.znames]
                  + [f"beta{l}" for l in self.flags])
        else:
            ins = (["1"] + [f"X(t-{l})" for l in self.xlags] + list(self.znames)
                   + [f"lam(t-{l})" for l in self.flags])
            nm = [f"u1_{l+1}" for l in range(self.L)]
            for l in range(self.L):
                nm += [f"u0_{l+1}[{i}]" for i in ins]
        return nm + (["delta"] if self.nb else [])

    def design(self, x):
        """Regressor matrix V (T x K); rows before maxlag are zero."""
        x = np.asarray(x, float)
        T = len(x)
        V = np.zeros((T, self.K))
        sc = self.scale if self.L > 0 else 1.0
        for k, l in enumerate(self.xlags):
            V[l:, k] = x[:T - l] / sc
        if self.Z is not None:
            V[:, len(self.xlags):] = self.Z[:T]
        return V


# --------------------------------------------------------------------------
# negative log-likelihood and gradient
# --------------------------------------------------------------------------
def _nll_numpy(par, x, V, spec, start, lam0, want_grad=True):
    """Pure NumPy twin of the C kernel."""
    T, K, L, Q = len(x), spec.K, spec.L, spec.Q
    nm = spec.n_mean
    delta = par[-1] if spec.nb else 0.0
    s, c, fl = spec.scale, spec.c, spec.flags
    lam = np.full(T, lam0, float)
    D = np.zeros((T, nm)) if want_grad else None
    sig = np.zeros(T)
    width = 1 + K + Q
    if L > 0:
        u = par[:L]
        W = par[L:L + L * width].reshape(L, width)
    for t in range(start, T):
        lf = np.array([lam[t - j] if t - j >= 0 else lam0 for j in fl])
        if L == 0:
            inp = np.concatenate(([1.0], V[t], lf))
            eta = par[:nm] @ inp
            if want_grad:
                deta, dfb = inp, par[1 + K:1 + K + Q]
        else:
            inp = np.concatenate(([1.0], V[t], lf / s))
            h = special.expit(W @ inp)
            eta = u @ h
            if want_grad:
                g = u * h * (1 - h)
                deta = np.concatenate((h, (g[:, None] * inp[None, :]).ravel()))
                dfb = (g @ W[:, 1 + K:]) / s if Q else np.zeros(0)
        lam[t] = max(softplus(eta, c), 1e-12)
        if want_grad:
            d = deta.copy()
            for j, lag in enumerate(fl):
                if t - lag >= start:
                    d = d + dfb[j] * D[t - lag]
            sig[t] = special.expit(eta / c)
            D[t] = sig[t] * d
    xs, ls = x[start:], lam[start:]
    grad = np.zeros(len(par))
    if spec.nb and delta > 1e-8:
        kmax = int(xs.max())
        ks = np.arange(1, max(kmax, 1))
        cl = np.concatenate(([0.0, 0.0], np.cumsum(np.log1p(delta * ks))))
        ck = np.concatenate(([0.0, 0.0], np.cumsum(ks / (1 + delta * ks))))
        xi = xs.astype(int)
        ll = (xs * np.log(ls) - (xs + 1 / delta) * np.log1p(delta * ls)
              - special.gammaln(xs + 1) + cl[xi])
        dl = (xs - ls) / (ls * (1 + delta * ls))
        gd = (ck[xi] + np.log1p(delta * ls) / delta ** 2
              - (xs + 1 / delta) * ls / (1 + delta * ls)).sum()
    else:
        ll = xs * np.log(ls) - ls - special.gammaln(xs + 1)
        dl = (xs - ls) / ls
        gd = 0.5 * ((xs - ls) ** 2 - xs).sum()
    if want_grad:
        grad[:nm] = -(dl[:, None] * D[start:]).sum(0)
        if spec.nb:
            grad[-1] = -gd
    return -ll.sum(), grad, lam


def nll(par, x, V, spec, start, lam0, want_grad=True):
    """Negative conditional log-likelihood, gradient and lambda path."""
    par = np.ascontiguousarray(par, float)
    if _LIB is None:
        return _nll_numpy(par, x, V, spec, start, lam0, want_grad)
    T = len(x)
    grad = np.zeros(len(par))
    lam = np.zeros(T)
    fl = (ctypes.c_int * max(spec.Q, 1))(*spec.flags)
    dp = ctypes.POINTER(ctypes.c_double)
    val = _LIB.ingarch_nll(par.ctypes.data_as(dp), len(par), x.ctypes.data_as(dp),
                           V.ctypes.data_as(dp), T, spec.K, spec.L, spec.Q, fl,
                           spec.c, int(spec.nb), start, lam0, spec.scale,
                           grad.ctypes.data_as(dp), lam.ctypes.data_as(dp),
                           int(want_grad))
    return val, grad, lam


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
KAPPA = 0.95      # upper bound imposed on the contraction constant of neural models


def contraction(par, spec, want_grad=False):
    """Lipschitz constant of lam_t with respect to past conditional means.

    softplus model : sum_j |beta_j|
    neural model   : sum_j sum_l |u1_l * u0_{l,lam_j}| / (4 s)   (Proposition 1)

    A value below one guarantees that the recursion is a contraction, i.e. a
    stationary ergodic solution exists and the influence of the starting value
    lam_0 on the filtered conditional means dies out geometrically."""
    par = np.asarray(par, float)
    g = np.zeros(len(par))
    if spec.Q == 0:
        return (0.0, g) if want_grad else 0.0
    if spec.L == 0:
        b = par[1 + spec.K:1 + spec.K + spec.Q]
        val = np.abs(b).sum()
        g[1 + spec.K:1 + spec.K + spec.Q] = np.sign(b)
        return (val, g) if want_grad else val
    L, width = spec.L, 1 + spec.K + spec.Q
    u = par[:L]
    W = par[L:L + L * width].reshape(L, width)
    wl = W[:, 1 + spec.K:]
    val = (np.abs(u)[:, None] * np.abs(wl)).sum() / (4 * spec.scale)
    if want_grad:
        g[:L] = np.sign(u) * np.abs(wl).sum(1) / (4 * spec.scale)
        G = np.zeros((L, width))
        G[:, 1 + spec.K:] = np.abs(u)[:, None] * np.sign(wl) / (4 * spec.scale)
        g[L:L + L * width] = G.ravel()
        return val, g
    return val


def _shrink_feedback(p, spec, target=0.5):
    """Rescale the feedback weights of a neural start so that the contraction
    constant does not exceed `target`."""
    p = np.array(p, float)
    c = contraction(p, spec)
    if spec.L > 0 and spec.Q > 0 and c > target:
        L, width = spec.L, 1 + spec.K + spec.Q
        W = p[L:L + L * width].reshape(L, width)
        W[:, 1 + spec.K:] *= target / c
    return p


@dataclass
class Fit:
    spec: Spec
    par: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    n: int
    start: int
    lam0: float
    lam: np.ndarray
    converged: bool
    extra: dict = field(default_factory=dict)

    @property
    def k(self):
        return len(self.par)

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self):
        return -2 * self.loglik + np.log(self.n) * self.k

    @property
    def delta(self):
        return self.par[-1] if self.spec.nb else 0.0

    @property
    def contraction(self):
        return contraction(self.par, self.spec)


def _bounds(spec, delta_max=50.0):
    if spec.L == 0:
        b = [(None, None)] + [(-0.9999, 0.9999)] * len(spec.xlags)
        b += [(None, None)] * (spec.K - len(spec.xlags))
        b += [(-0.9999, 0.9999)] * spec.Q
    else:
        b = [(None, None)] * spec.n_mean
    if spec.nb:
        b += [(0.0, delta_max)]
    return b


def _hessian(par, x, V, spec, start, lam0):
    """Observed information by central differences of the exact gradient."""
    k = len(par)
    H = np.zeros((k, k))
    for i in range(k):
        h = 1e-5 * max(1.0, abs(par[i]))
        up, dn = par.copy(), par.copy()
        up[i] += h
        dn[i] -= h
        if spec.nb and i == k - 1 and dn[i] < 0:      # one-sided at delta = 0
            dn[i] = par[i]
            H[i] = (nll(up, x, V, spec, start, lam0)[1]
                    - nll(dn, x, V, spec, start, lam0)[1]) / h
        else:
            H[i] = (nll(up, x, V, spec, start, lam0)[1]
                    - nll(dn, x, V, spec, start, lam0)[1]) / (2 * h)
    return 0.5 * (H + H.T)


def default_starts(x, spec, rng=None, n_random=0):
    """Starting values.  Softplus models: a small deterministic grid built
    from the linear-INGARCH moment relations; neural models: random weights."""
    xbar = float(np.mean(x))
    d0 = max((np.var(x) / xbar - 1) / xbar, 0.05) if spec.nb else None
    starts = []
    if spec.L == 0:
        nx, nz = len(spec.xlags), spec.K - len(spec.xlags)
        grid = [(0.2, 0.3), (-0.2, 0.3), (0.2, -0.3), (-0.2, -0.3)] if spec.Q \
            else [(0.2, 0.0), (-0.2, 0.0)]
        for a, b in grid:
            al = [a] + [a / 2] * (nx - 1)
            be = [b] + [0.0] * (spec.Q - 1) if spec.Q else []
            a0 = xbar * (1 - sum(al) - sum(be))
            p = [a0] + al + [0.0] * nz + be
            starts.append(np.array(p + ([d0] if spec.nb else [])))
    rng = rng or np.random.default_rng(1)
    for _ in range(n_random):
        if spec.L == 0:
            p = starts[0].copy()
            p[1:spec.n_mean] += rng.normal(0, 0.15, spec.n_mean - 1)
            p[1:1 + len(spec.xlags)] = np.clip(p[1:1 + len(spec.xlags)], -.9, .9)
            if spec.Q:
                p[-spec.Q - int(spec.nb):len(p) - int(spec.nb)] = np.clip(
                    p[-spec.Q - int(spec.nb):len(p) - int(spec.nb)], -.9, .9)
        else:
            width = 1 + spec.K + spec.Q
            u = rng.normal(1.5 * xbar, 1.5 * xbar, spec.L) / max(spec.L ** 0.5, 1)
            u[0] = abs(u[0]) + xbar
            W = rng.normal(0, rng.choice([0.3, 1.0, 3.0]), (spec.L, width))
            p = np.concatenate((u, W.ravel(), [d0] if spec.nb else []))
        starts.append(p)
    return starts


def fit(x, spec, start=None, starts=None, n_random=0, seed=1, hessian=True,
        lam0=None, maxiter=20000, coarse_iter=400, n_polish=5, constrain=True):
    """Conditional ML fit.

    `start` is the first time index entering the likelihood (use a common
    value to compare models on the same sample).

    Optimisation is done in two stages: every starting value is run for at
    most `coarse_iter` L-BFGS-B iterations, then the `n_polish` best candidates
    are iterated to convergence (at most `maxiter` iterations).

    For neural models with feedback (L > 0, q > 0) the likelihood is maximised
    subject to contraction(par) <= KAPPA (Proposition 1 of the paper), enforced
    by a quadratic penalty; `constrain=False` switches this off.  Without the
    constraint the fitted recursion may fail to be a contraction, in which case
    the filtered conditional means depend on the arbitrary starting value and
    the likelihood surface becomes erratic."""
    x = np.ascontiguousarray(x, float)
    start = spec.maxlag if start is None else start
    lam0 = float(np.mean(x)) if lam0 is None else lam0
    V = np.ascontiguousarray(spec.design(x))
    rng = np.random.default_rng(seed)
    if starts is None:
        starts = default_starts(x, spec, rng, n_random)
    pen_on = constrain and spec.L > 0 and spec.Q > 0
    M = 100.0 * (len(x) - start)

    def fun(p):
        val, g, _ = nll(p, x, V, spec, start, lam0)
        if pen_on:
            c, gc = contraction(p, spec, want_grad=True)
            if c > KAPPA:
                val += M * (c - KAPPA) ** 2
                g = g + 2 * M * (c - KAPPA) * gc
        return val, g

    def run(p0, iters):
        return optimize.minimize(fun, p0, jac=True, method="L-BFGS-B",
                                 bounds=_bounds(spec),
                                 options={"maxiter": iters, "maxfun": 4 * iters,
                                          "ftol": 1e-13, "gtol": 1e-7, "maxcor": 20})

    cands = []
    for p0 in starts:
        if pen_on:
            p0 = _shrink_feedback(p0, spec)
        try:
            res = run(p0, coarse_iter)
        except Exception:
            continue
        if np.isfinite(res.fun):
            cands.append(res)
    if not cands:
        raise RuntimeError("all optimisations failed")
    cands.sort(key=lambda r: r.fun)
    best = None
    for res in cands[:max(n_polish, 1)]:
        if not res.success or res.nit >= coarse_iter:
            for _ in range(3):                      # iterate to convergence
                new = run(res.x, maxiter)
                done = new.success and res.fun - new.fun < 1e-8
                res = new
                if done:
                    break
        if best is None or res.fun < best.fun - 1e-9:
            best = res
    par = best.x
    val, _, lam = nll(par, x, V, spec, start, lam0, want_grad=False)
    k = len(par)
    cov = np.full((k, k), np.nan)
    se = np.full(k, np.nan)
    if hessian:
        H = _hessian(par, x, V, spec, start, lam0)
        free = np.ones(k, bool)
        if spec.nb and par[-1] < 1e-6:          # delta on the Poisson boundary
            free[-1] = False
        try:
            ci = np.linalg.inv(H[np.ix_(free, free)])
            if np.all(np.diag(ci) > 0):
                cov[np.ix_(free, free)] = ci
                se[free] = np.sqrt(np.diag(ci))
        except np.linalg.LinAlgError:
            pass
    return Fit(spec, par, se, cov, -val, len(x) - start, start, lam0, lam,
               bool(best.success), extra={"contraction": contraction(par, spec)})


def warm_start(prev, spec, rng, jitter=0.5):
    """Start for a neural model with L hidden neurons built from a fitted
    model with L-1 neurons (same inputs): the extra neuron gets a zero output
    weight, so the optimiser starts from the likelihood of the smaller model."""
    Lp, width = prev.spec.L, 1 + spec.K + spec.Q
    u = np.concatenate((prev.par[:Lp], [0.0]))
    W = np.vstack((prev.par[Lp:Lp + Lp * width].reshape(Lp, width),
                   rng.normal(0, jitter, (1, width))))
    tail = [prev.par[-1]] if spec.nb else []
    return np.concatenate((u, W.ravel(), tail))


def fit_neural(x, spec_fn, Ls=(1, 2, 3), start=None, n_random=60, n_warm=20,
               n_screen=0, seed=1):
    """Fit neural models for an increasing number of hidden neurons.

    The likelihood of neural INGARCH models is multimodal, so three kinds of
    starting values are combined: (i) random weights; if n_screen > 0, the
    n_random random starts are the best of n_screen random candidates ranked
    by their likelihood value ("screening"); (ii) warm starts from the fitted
    network with one neuron less; (iii) jittered copies of those warm starts.
    `spec_fn(L)` must return the Spec with L hidden neurons."""
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(x, float)
    fits, prev = {}, None
    for L in Ls:
        spec = spec_fn(L)
        st = spec.maxlag if start is None else start
        if n_screen > n_random:
            V = np.ascontiguousarray(spec.design(x))
            cand = [_shrink_feedback(p, spec) for p in default_starts(x, spec, rng, n_screen)]
            val = [nll(p, x, V, spec, st, float(x.mean()), want_grad=False)[0]
                   for p in cand]
            val = np.where(np.isfinite(val), val, np.inf)
            starts = [cand[i] for i in np.argsort(val)[:n_random]]
        else:
            starts = default_starts(x, spec, rng, n_random)
        if prev is not None:
            starts += [warm_start(prev, spec, rng, jitter=j)
                       for j in np.resize([0.3, 1.0, 3.0], n_warm)]
        fits[L] = prev = fit(x, spec, start=start, starts=starts)
    return fits


def lr_test_poisson(fit_nb, fit_pois):
    """LR test of H0: delta = 0 (Poisson) against delta > 0.  The null value
    lies on the boundary, so the reference law is 0.5*chi2_0 + 0.5*chi2_1."""
    lr = max(2 * (fit_nb.loglik - fit_pois.loglik), 0.0)
    p = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0
    return lr, p


def filter_lambda(x, fit_obj, spec=None):
    """Run the fitted recursion over a (possibly longer) series x, e.g. to
    obtain one-step-ahead predictions on a hold-out sample."""
    spec = spec or fit_obj.spec
    x = np.ascontiguousarray(x, float)
    V = np.ascontiguousarray(spec.design(x))
    return nll(fit_obj.par, x, V, spec, fit_obj.start, fit_obj.lam0,
               want_grad=False)[2]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def rcount(rng, lam, delta):
    if delta <= 1e-10:
        return rng.poisson(lam)
    r = 1.0 / delta
    return rng.negative_binomial(r, r / (r + lam))


def simulate_sp11(alpha0, alpha1, beta1, delta, T, c=1.0, n_paths=1, burn=500,
                  rng=None, relu=False):
    """Simulate softplus (or ReLU, c->0) NB-INGARCH(1,1) paths, vectorised over
    paths.  Returns an array of shape (n_paths, T)."""
    rng = rng or np.random.default_rng()
    link = (lambda e: np.maximum(e, 1e-10)) if relu else (lambda e: softplus(e, c))
    lam = np.full(n_paths, max(link(alpha0 / max(1 - alpha1 - beta1, 0.05)), 0.1))
    x = rcount(rng, lam, delta)
    out = np.empty((n_paths, T), dtype=np.int64)
    for t in range(-burn, T):
        lam = link(alpha0 + alpha1 * x + beta1 * lam)
        x = rcount(rng, lam, delta)
        if t >= 0:
            out[:, t] = x
    return out


def simulate_fitted(fit_obj, x_init, T, n_paths=1000, rng=None, Z=None):
    """Simulate n_paths series of length T from a fitted model (parametric
    bootstrap), conditional on the first `start` observations and on the
    exogenous regressors."""
    spec, par = fit_obj.spec, fit_obj.par
    rng = rng or np.random.default_rng(1)
    st = fit_obj.start
    Z = spec.Z if Z is None else Z
    K, Q, L = spec.K, spec.Q, spec.L
    nx = len(spec.xlags)
    sc = spec.scale if L > 0 else 1.0
    delta = fit_obj.delta
    X = np.zeros((n_paths, T))
    X[:, :st] = np.asarray(x_init, float)[:st]
    lam = np.full((n_paths, T), fit_obj.lam0)
    width = 1 + K + Q
    for t in range(st, T):
        cols = [np.ones(n_paths)] + [X[:, t - l] / sc for l in spec.xlags]
        if Z is not None:
            cols += [np.full(n_paths, z) for z in Z[t]]
        cols += [(lam[:, t - j] if t - j >= 0 else np.full(n_paths, fit_obj.lam0))
                 / sc for j in spec.flags]
        inp = np.column_stack(cols)
        if L == 0:
            eta = inp @ par[:spec.n_mean]
        else:
            W = par[L:L + L * width].reshape(L, width)
            eta = special.expit(inp @ W.T) @ par[:L]
        lam[:, t] = np.maximum(softplus(eta, spec.c), 1e-12)
        X[:, t] = rcount(rng, lam[:, t], delta)
    return X


# --------------------------------------------------------------------------
# moments of the linear NB-INGARCH(1,1) model (used as approximations)
# --------------------------------------------------------------------------
def linear_moments(a0, a1, b1, delta, hmax=3):
    """Mean, dispersion ratio and ACF of the *linear* NB-INGARCH(1,1) model
    evaluated at (a0, a1, b1); used as approximation for the softplus model."""
    mu = a0 / (1 - a1 - b1)
    den = 1 - (1 + delta) * a1 ** 2 - 2 * a1 * b1 - b1 ** 2
    num = 1 - 2 * a1 * b1 - b1 ** 2
    var = mu * (1 + delta * mu) * num / den if den > 0 else np.nan
    rho = [a1 * (a1 + b1) ** (h - 1) * (1 - a1 * b1 - b1 ** 2) / num
           for h in range(1, hmax + 1)]
    return mu, var / mu, np.array(rho)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def acf(z, nlags):
    z = np.asarray(z, float) - np.mean(z)
    d = z @ z
    return np.array([z[h:] @ z[:len(z) - h] / d for h in range(nlags + 1)])


def pacf(z, nlags):
    """Partial ACF by the Durbin-Levinson recursion."""
    r = acf(z, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    for k in range(1, nlags + 1):
        num = r[k] - phi[k - 1, 1:k] @ r[1:k][::-1]
        den = 1 - phi[k - 1, 1:k] @ r[1:k]
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out[k] = phi[k, k]
    return out


def ljung_box(z, nlags, df_adj=0):
    n = len(z)
    r = acf(z, nlags)[1:]
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, nlags + 1)))
    return q, stats.chi2.sf(q, nlags - df_adj)


def cum_periodogram(z):
    """Cumulative periodogram with 95% Kolmogorov-Smirnov band half-width."""
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    I = np.abs(np.fft.rfft(z)) ** 2 / n
    I = I[1:(n - 1) // 2 + 1]
    m = len(I)
    freq = np.arange(1, m + 1) / n
    crit = 1.358 / (np.sqrt(m - 1) + 0.12 + 0.11 / np.sqrt(m - 1))
    return freq, np.cumsum(I) / I.sum(), crit


def pearson_residuals(x, lam, delta):
    return (x - lam) / np.sqrt(lam * (1 + delta * lam))


def count_cdf(k, lam, delta):
    k = np.asarray(k, float)
    if delta <= 1e-10:
        return np.where(k < 0, 0.0, stats.poisson.cdf(k, lam))
    r = 1.0 / delta
    return np.where(k < 0, 0.0, stats.nbinom.cdf(k, r, r / (r + lam)))


def count_logpmf(k, lam, delta):
    if delta <= 1e-10:
        return stats.poisson.logpmf(k, lam)
    r = 1.0 / delta
    return stats.nbinom.logpmf(k, r, r / (r + lam))


def pit_histogram(x, lam, delta, bins=10):
    """Non-randomised PIT histogram for counts (Czado, Gneiting & Held 2009)."""
    Fu = count_cdf(x, lam, delta)
    Fl = count_cdf(np.asarray(x) - 1, lam, delta)
    u = np.linspace(0, 1, bins + 1)
    Fbar = np.array([np.mean(np.clip((ui - Fl) / np.maximum(Fu - Fl, 1e-300), 0, 1))
                     for ui in u])
    return np.diff(Fbar) * bins


def scores(x, lam, delta, kmax=None):
    """Mean logarithmic, quadratic and ranked probability scores."""
    x = np.asarray(x, float)
    kmax = kmax or int(max(x.max(), lam.max()) * 4 + 50)
    ks = np.arange(kmax + 1)
    P = np.exp(count_logpmf(ks[None, :], lam[:, None], delta))
    logs = -count_logpmf(x, lam, delta)
    px = P[np.arange(len(x)), x.astype(int)]
    qs = -2 * px + (P ** 2).sum(1)
    cdf = np.cumsum(P, 1)
    rps = ((cdf - (ks[None, :] >= x[:, None])) ** 2).sum(1)
    return {"logs": logs.mean(), "qs": qs.mean(), "rps": rps.mean()}


def fmt(v, d=4):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
