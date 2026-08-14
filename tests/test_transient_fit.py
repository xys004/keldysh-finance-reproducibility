"""
Tests del ajuste transiente (exp. 05a): recuperación, causalidad y banderas.

La pieza central es el CONTROL POSITIVO DE LA MEDICIÓN (regla 11): un proceso
sintético donde ρ(τ)=exp[−(τ/τ_c)^β] es verdad por construcción, y el ajuste
tiene que recuperar (τ_c, β). Sin esto, unos parámetros ruidosos en datos
reales serían ambiguos entre "no hay estructura" y "el ajuste está roto".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.two_time import two_time_surface
from keldysh_finance.transient import (fit_stretched_exp, rolling_transient_fit,
                                       rolling_teff, synthetic_series_with_acf)


# --- el generador sintético hace lo que dice --------------------------------

def test_sintetico_tiene_la_acf_prescrita():
    """La ACF empírica del proceso generado sigue el objetivo KWW."""
    tau_c, beta = 8.0, 0.6
    x = synthetic_series_with_acf(200_000, tau_c, beta, seed=1)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    lags = np.arange(1, 41)
    acf = np.array([np.dot(xc[:-k], xc[k:]) / denom for k in lags])
    objetivo = np.exp(-(lags / tau_c) ** beta)
    assert np.max(np.abs(acf - objetivo)) < 0.05, \
        "el embedding circulante no reproduce la ACF objetivo"


# --- recuperación de parámetros conocidos (control positivo) ----------------

def _recupera(tau_c, beta, seed, n=60_000, window=4000, max_lag=60):
    x = synthetic_series_with_acf(n, tau_c, beta, seed=seed)
    # field='raw': el proceso sintético ES el campo a correlacionar
    s = two_time_surface(x, window=window, max_lag=max_lag, step=window // 2,
                         field="raw")
    traj = rolling_transient_fit(s)
    assert traj.valido.sum() >= 5, "casi ningún ajuste válido sobre datos con señal"
    return (float(np.median(np.exp(traj.log_tau_c[traj.valido]))),
            float(np.median(traj.beta[traj.valido])))


def test_recupera_kww_estirada():
    """El caso que importa: β<1, la firma vítrea."""
    tau_hat, beta_hat = _recupera(tau_c=8.0, beta=0.6, seed=2)
    assert abs(tau_hat - 8.0) / 8.0 < 0.25, f"tau_c estimado {tau_hat:.2f} vs 8"
    assert abs(beta_hat - 0.6) < 0.15, f"beta estimado {beta_hat:.2f} vs 0.6"


def test_recupera_exponencial_simple():
    """β=1 (Debye) es el caso anidado: no debe salir estirado por sesgo."""
    tau_hat, beta_hat = _recupera(tau_c=5.0, beta=1.0, seed=3)
    assert abs(tau_hat - 5.0) / 5.0 < 0.25, f"tau_c estimado {tau_hat:.2f} vs 5"
    assert abs(beta_hat - 1.0) < 0.15, f"beta estimado {beta_hat:.2f} vs 1.0"


# --- banderas: lo no identificable tiene que DECLARARSE ---------------------

def test_ruido_blanco_no_pasa_por_identificable():
    """Sobre ruido blanco (ACF ≈ 0) la fracción de ajustes válidos debe
    hundirse: convergencias al borde o R² ínfimo, nunca parámetros con cara
    de medición."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=30_000)
    s = two_time_surface(x, window=500, max_lag=40, step=250, field="raw")
    traj = rolling_transient_fit(s)
    frac = traj.valido.mean()
    assert frac < 0.2, f"el {frac:.0%} del ruido blanco se declaró identificable"


def test_perfil_degenerado_no_revienta():
    out = fit_stretched_exp(np.arange(1, 41), np.full(40, np.nan))
    assert not out["convergio"] and np.isnan(out["beta"])
    out2 = fit_stretched_exp(np.arange(1, 5), np.array([0.5, 0.4, 0.3, 0.2]))
    assert not out2["convergio"], "con 4 puntos no hay ajuste de 3 parámetros"


def test_errores_estandar_finitos_cuando_hay_senal():
    x = synthetic_series_with_acf(20_000, 8.0, 0.6, seed=5)
    s = two_time_surface(x, window=4000, max_lag=60, step=2000, field="raw")
    traj = rolling_transient_fit(s)
    v = traj.valido
    assert np.isfinite(traj.se_beta[v]).all()
    assert np.isfinite(traj.corr_tau_beta[v]).all()
    # la degeneración KWW existe pero no puede ser total
    assert np.median(np.abs(traj.corr_tau_beta[v])) < 0.999


# --- causalidad (regla 1: característica nueva, test de causalidad nuevo) ---

def _serie(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = 0.9 * vol[t - 1] + 0.1 * 0.01 + 0.02 * abs(rng.normal()) * 0.01
    return rng.normal(0, vol)


def test_ajuste_rodante_no_usa_futuro():
    """Cambiar la cola de la serie no puede mover los parámetros de ventanas
    que terminan antes del corte."""
    r = _serie()
    corte = 4000
    r2 = r.copy()
    r2[corte:] = _serie(len(r), seed=99)[corte:]

    t1 = rolling_transient_fit(two_time_surface(r, window=500, max_lag=60, step=100))
    t2 = rolling_transient_fit(two_time_surface(r2, window=500, max_lag=60, step=100))

    m = t1.t_w < corte
    assert m.sum() > 5
    np.testing.assert_allclose(t1.beta[m], t2.beta[m], rtol=1e-10, atol=1e-12,
                               err_msg="el ajuste transiente usa datos futuros")
    np.testing.assert_allclose(t1.log_tau_c[m], t2.log_tau_c[m],
                               rtol=1e-10, atol=1e-12)


def test_teff_rodante_es_causal():
    rng = np.random.default_rng(6)
    n, corte = 3000, 2000
    eps = rng.normal(0, 0.2, n)
    log_open = np.log(100.0) + np.cumsum(rng.normal(0, 0.01, n))
    eps2, lo2 = eps.copy(), log_open.copy()
    eps2[corte:] = rng.normal(0, 1.0, n - corte)
    lo2[corte:] = log_open[corte:] + 0.5

    t1, teff1, v1 = rolling_teff(log_open, eps, window=400, max_lag=20, step=50)
    t2, teff2, v2 = rolling_teff(lo2, eps2, window=400, max_lag=20, step=50)

    m = (t1 < corte) & np.isfinite(teff1) & np.isfinite(teff2)
    assert m.sum() > 5
    np.testing.assert_allclose(teff1[m], teff2[m], rtol=1e-10, atol=1e-12,
                               err_msg="rolling_teff usa datos futuros")
    np.testing.assert_allclose(v1[m], v2[m], rtol=1e-10, atol=1e-12)
