"""
Tests del reloj de quench (exp. 07) y del transporte por fase (exp. 08).

Controles positivos (regla 11): un ensemble sintético que SÍ envejece
(τ de relajación creciendo con la edad) y una serie con modulación de fase
plantada. Los pipelines tienen que verlos — y no ver nada en los controles
estacionarios.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.floquet import (modulacion, observables_por_fase,
                                     p_empirico, rotar_horas_por_dia)
from keldysh_finance.quench import (campo_post_shock, correlacion_por_edad,
                                    detectar_shocks, estadistico_envejecimiento,
                                    exponente_omori, tau_c_filas)

BORDES_TW = [8, 16, 32, 64, 128, 256]


# --- quench: ensembles sintéticos --------------------------------------------

def _ensemble_envejece(n_shocks=120, u_max=300, seed=0):
    """AR(1) con tiempo de relajación creciendo con la edad: envejecimiento
    por construcción, más relajación Omori de la media (p=0.3)."""
    rng = np.random.default_rng(seed)
    W = np.empty((n_shocks, u_max))
    for s in range(n_shocks):
        x = 0.0
        for u in range(u_max):
            tau = 4.0 + u / 8.0                    # la memoria crece con la edad
            phi = np.exp(-1.0 / tau)
            x = phi * x + np.sqrt(1 - phi ** 2) * rng.normal()
            W[s, u] = 0.8 + 3.0 * (1 + u) ** -0.3 + 0.5 * x
    return W


def _ensemble_estacionario(n_shocks=120, u_max=300, seed=1):
    """AR(1) de τ fijo: TTI respecto al reloj del shock (control negativo)."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-1.0 / 8.0)
    W = np.empty((n_shocks, u_max))
    for s in range(n_shocks):
        x = 0.0
        for u in range(u_max):
            x = phi * x + np.sqrt(1 - phi ** 2) * rng.normal()
            W[s, u] = 0.8 + 0.5 * x
    return W


def test_omori_exacto_sin_ruido():
    """El estimador NLS clava los tres parámetros sobre la curva exacta —
    fija que cualquier sesgo observado en ensembles viene del ruido
    correlacionado, no del ajuste."""
    u = np.arange(1, 301)
    om = exponente_omori(0.8 + 3.0 * (1 + u) ** -0.3)
    assert abs(om["p"] - 0.3) < 1e-6
    assert abs(om["m_inf"] - 0.8) < 1e-6


def test_pipeline_ve_el_envejecimiento_plantado():
    W = _ensemble_envejece()
    rho, _ = correlacion_por_edad(W, BORDES_TW, max_lag=30)
    tau_c = tau_c_filas(rho)
    assert np.isfinite(tau_c).sum() >= 4
    assert estadistico_envejecimiento(tau_c) > 0.7, \
        f"tau_c por bin: {tau_c} — debería crecer monótonamente"
    # La media relaja con el exponente plantado. Tolerancia ancha a
    # PROPÓSITO: el ruido del ensemble está correlacionado entre edades (la
    # memoria AR de cada shock) y sesga p hacia arriba ~0.1 con 120 shocks —
    # medido con 4 semillas: 0.37-0.47. Por eso el experimento reporta p con
    # bootstrap sobre shocks, no con el se del jacobiano.
    om = exponente_omori(np.nanmean(W, axis=0))
    assert abs(om["p"] - 0.3) < 0.20, f"Omori p={om['p']:.3f} vs 0.3"


def test_pipeline_no_inventa_envejecimiento():
    W = _ensemble_estacionario()
    rho, _ = correlacion_por_edad(W, BORDES_TW, max_lag=30)
    tau_c = tau_c_filas(rho)
    s = estadisticos = estadistico_envejecimiento(tau_c)
    assert abs(s) < 0.85, \
        f"Spearman={s:.2f} sobre un ensemble estacionario (tau_c={tau_c})"
    # el tau_c estimado ronda el verdadero (integral de phi^k ~ 7.5)
    assert 4.0 < np.nanmedian(tau_c) < 12.0


def test_deteccion_de_shocks():
    rng = np.random.default_rng(2)
    n = 5000
    r = rng.normal(0, 0.01, n)
    sitios = [1000, 1005, 2500, 4000]              # 1005 pega al 1000: mismo episodio
    for t in sitios:
        r[t] = 0.15
    sigma = np.full(n, 0.01)
    shocks, thr = detectar_shocks(r, sigma, umbral_q=0.999, separacion=300,
                                  margen_post=500, margen_pre=200)
    assert 1000 in shocks and 2500 in shocks and 4000 in shocks
    assert 1005 not in shocks, "dos barras del mismo episodio son UN temple"
    assert np.isfinite(thr) and thr > 3


def test_campo_normaliza_por_sigma_preshock():
    r = np.zeros(100)
    r[50] = 0.2                                    # el shock
    r[51:61] = 0.05
    sigma = np.full(100, 0.01)
    W = campo_post_shock(r, sigma, np.array([50]), u_max=10)
    np.testing.assert_allclose(W[0], 5.0)          # |0.05|/0.01, sin la barra u=0


# --- floquet: modulación plantada y nulo -------------------------------------

def _series_con_fase(n_dias=900, amplitud=0.05, seed=3):
    """ε con sesgo dependiente de la hora (seno de período 24) + precio."""
    rng = np.random.default_rng(seed)
    n = n_dias * 24
    horas = np.tile(np.arange(24), n_dias)
    sesgo = amplitud * np.sin(2 * np.pi * horas / 24.0)
    eps = np.clip(sesgo + rng.normal(0, 0.2, n), -1, 1)
    log_open = np.log(100) + np.cumsum(rng.normal(0, 0.01, n))
    eps_raw = eps * 50.0
    trades = np.full(n, 30.0)
    return eps, log_open, eps_raw, trades, horas


def _nulo_rotacion(eps, log_open, eps_raw, trades, horas, clave, K=30, seed=4):
    rng = np.random.default_rng(seed)
    dias = np.arange(len(horas)) // 24
    out = []
    for _ in range(K):
        h_rot = rotar_horas_por_dia(horas, dias, rng)
        filas = observables_por_fase(eps, log_open, eps_raw, trades,
                                     h_rot, q_barra=50.0 / 30.0)
        out.append(modulacion(filas, clave))
    return np.asarray(out)


def test_floquet_ve_la_modulacion_plantada():
    eps, lo, er, tr, h = _series_con_fase(amplitud=0.05)
    filas = observables_por_fase(eps, lo, er, tr, h, q_barra=50.0 / 30.0)
    v_obs = modulacion(filas, "sesgo")
    nulo = _nulo_rotacion(eps, lo, er, tr, h, "sesgo")
    assert p_empirico(v_obs, nulo) < 0.05, \
        "una modulación de sesgo del 5% tiene que detectarse"


def test_floquet_no_inventa_modulacion():
    eps, lo, er, tr, h = _series_con_fase(amplitud=0.0, seed=5)
    filas = observables_por_fase(eps, lo, er, tr, h, q_barra=50.0 / 30.0)
    v_obs = modulacion(filas, "sesgo")
    nulo = _nulo_rotacion(eps, lo, er, tr, h, "sesgo", seed=6)
    assert p_empirico(v_obs, nulo) > 0.05, \
        "sin modulación plantada el p empírico no puede ser significativo"


def test_bins_de_fase_cubren_todo():
    eps, lo, er, tr, h = _series_con_fase(n_dias=200)
    filas = observables_por_fase(eps, lo, er, tr, h)
    assert len(filas) == 8
    assert sum(f["n"] for f in filas) >= len(eps) - 8 * 24
    assert all(f["n"] >= 500 for f in filas)
