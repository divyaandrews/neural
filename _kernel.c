/* Optional C accelerator for ingarch_core.py.
 *
 * Negative conditional log-likelihood and its exact gradient (forward-mode
 * recursion) for softplus / neural INGARCH models with Poisson or negative
 * binomial conditional distribution.  The Python module compiles this file
 * on the fly when a C compiler is available and otherwise falls back to an
 * equivalent pure NumPy implementation (same results, slower).
 *
 *   eta_t    = a0 + a'v_t + sum_j b_j lam_{t-f_j}                    (L = 0)
 *   eta_t    = sum_l u_l * logistic(w_l0 + w_l'v_t
 *                                   + sum_j wl_lj lam_{t-f_j}/s)     (L > 0)
 *   lam_t    = c * log(1 + exp(eta_t / c))
 *   X_t|F    ~ Poisson(lam_t)  or  NB(mean lam_t, dispersion delta = 1/r)
 *
 * Parameter layout
 *   L = 0 : [a0, a_1..a_K, b_1..b_Q, (delta)]
 *   L > 0 : [u_1..u_L, (w_l0, w_l1..w_lK, wl_l1..wl_lQ) for l = 1..L, (delta)]
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double softplus(double e, double c) {
    double z = e / c;
    if (z > 35.0) return e;
    if (z < -35.0) return c * exp(z);
    return c * log1p(exp(z));
}
static double logistic(double z) {
    if (z >= 0) return 1.0 / (1.0 + exp(-z));
    double e = exp(z);
    return e / (1.0 + e);
}

double ingarch_nll(const double *par, int npar, const double *x, const double *V,
                   int T, int K, int L, int Q, const int *flags, double c,
                   int nb, int start, double lam0, double s,
                   double *grad, double *lam, int want_grad) {
    int nm = npar - (nb ? 1 : 0);
    double delta = nb ? par[npar - 1] : 0.0;
    double nll = 0.0, gdelta = 0.0;
    double *D = NULL, *deta = NULL, *dlamfb = NULL;
    int width = 1 + K + Q;
    if (want_grad) {
        D = (double *)calloc((size_t)T * nm, sizeof(double));
        deta = (double *)malloc(sizeof(double) * nm);
        dlamfb = (double *)malloc(sizeof(double) * (Q > 0 ? Q : 1));
        for (int i = 0; i < npar; i++) grad[i] = 0.0;
    }
    for (int t = 0; t < start; t++) lam[t] = lam0;
    for (int t = start; t < T; t++) {
        const double *v = V + (size_t)t * K;
        double eta = 0.0;
        if (L == 0) {
            eta = par[0];
            for (int k = 0; k < K; k++) eta += par[1 + k] * v[k];
            for (int j = 0; j < Q; j++) {
                int tl = t - flags[j];
                double lf = (tl >= 0) ? lam[tl] : lam0;
                eta += par[1 + K + j] * lf;
                if (want_grad) { deta[1 + K + j] = lf; dlamfb[j] = par[1 + K + j]; }
            }
            if (want_grad) {
                deta[0] = 1.0;
                for (int k = 0; k < K; k++) deta[1 + k] = v[k];
            }
        } else {
            if (want_grad) for (int j = 0; j < Q; j++) dlamfb[j] = 0.0;
            for (int l = 0; l < L; l++) {
                const double *w = par + L + (size_t)l * width;
                double z = w[0];
                for (int k = 0; k < K; k++) z += w[1 + k] * v[k];
                for (int j = 0; j < Q; j++) {
                    int tl = t - flags[j];
                    double lf = (tl >= 0) ? lam[tl] : lam0;
                    z += w[1 + K + j] * lf / s;
                }
                double h = logistic(z);
                eta += par[l] * h;
                if (want_grad) {
                    double g = par[l] * h * (1.0 - h);
                    int base = L + l * width;
                    deta[l] = h;
                    deta[base] = g;
                    for (int k = 0; k < K; k++) deta[base + 1 + k] = g * v[k];
                    for (int j = 0; j < Q; j++) {
                        int tl = t - flags[j];
                        double lf = (tl >= 0) ? lam[tl] : lam0;
                        deta[base + 1 + K + j] = g * lf / s;
                        dlamfb[j] += g * w[1 + K + j] / s;
                    }
                }
            }
        }
        double lt = softplus(eta, c);
        if (lt < 1e-12) lt = 1e-12;
        lam[t] = lt;
        double xt = x[t];
        double dl;
        if (nb && delta > 1e-8) {
            double ll = xt * log(lt) - (xt + 1.0 / delta) * log1p(delta * lt) - lgamma(xt + 1.0);
            double sk = 0.0;
            int xi = (int)(xt + 0.5);
            for (int k = 1; k < xi; k++) { ll += log1p(delta * k); sk += k / (1.0 + delta * k); }
            nll -= ll;
            dl = (xt - lt) / (lt * (1.0 + delta * lt));
            if (want_grad)
                gdelta += sk + log1p(delta * lt) / (delta * delta)
                          - (xt + 1.0 / delta) * lt / (1.0 + delta * lt);
        } else {
            nll -= xt * log(lt) - lt - lgamma(xt + 1.0);
            dl = (xt - lt) / lt;
            if (want_grad && nb) gdelta += 0.5 * ((xt - lt) * (xt - lt) - xt);
        }
        if (want_grad) {
            double sig = logistic(eta / c);
            double *Dt = D + (size_t)t * nm;
            for (int i = 0; i < nm; i++) Dt[i] = deta[i];
            for (int j = 0; j < Q; j++) {
                int tl = t - flags[j];
                if (tl >= start) {
                    const double *Dl = D + (size_t)tl * nm;
                    double bj = dlamfb[j];
                    for (int i = 0; i < nm; i++) Dt[i] += bj * Dl[i];
                }
            }
            for (int i = 0; i < nm; i++) { Dt[i] *= sig; grad[i] -= dl * Dt[i]; }
        }
    }
    if (want_grad) {
        if (nb) grad[npar - 1] = -gdelta;
        free(D); free(deta); free(dlamfb);
    }
    return nll;
}
