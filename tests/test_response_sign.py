"""
Tests de la función de respuesta.

El más importante es el del SIGNO: el impacto de mercado tiene que ser
POSITIVO (comprar empuja el precio arriba). Obtener R<0 no es una curiosidad
estadística, es la firma de un error de temporización — usar como referencia
un precio que ya incorpora el impacto del flujo que se está midiendo.

Ese bug apareció el 2026-08-08 usando log(Close) en vez de log(Open) y daba
R(1) = −4.7e−4 en vez de +1.3e−2. Estos tests lo fijan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.flow import (order_flow_imbalance, response_function,
                                  response_from_frame, flow_autocorrelation)


def _mercado_sintetico(n=4000, impacto=0.02, seed=0):
    """Mercado de juguete con impacto CONOCIDO y positivo.

    El precio de apertura de t+1 incorpora el impacto del flujo de t más
    ruido. Así el valor verdadero de R es conocido y el test puede
    comprobar que el estimador lo recupera.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.2, n)                      # flujo firmado
    ruido = rng.normal(0, 0.001, n)
    log_open = np.empty(n)
    log_open[0] = np.log(100.0)
    for t in range(1, n):
        log_open[t] = log_open[t - 1] + impacto * eps[t - 1] + ruido[t]
    vol = np.full(n, 100.0)
    tb = (eps * vol + vol) / 2.0                     # invierte 2*tb - vol = eps*vol
    df = pd.DataFrame({"Open": np.exp(log_open),
                       "High": np.exp(log_open) * 1.001,
                       "Low": np.exp(log_open) * 0.999,
                       "Close": np.exp(np.r_[log_open[1:], log_open[-1]]),
                       "Volume": vol, "tbBase": tb,
                       "trades": np.full(n, 10.0)})
    return df, eps, log_open, impacto


def test_flujo_se_reconstruye_del_taker_buy():
    df, eps, _, _ = _mercado_sintetico()
    rec = order_flow_imbalance(df, normalize="volume")
    np.testing.assert_allclose(rec, eps, atol=1e-10,
                               err_msg="2*tbBase - Volume no reconstruye el flujo")


def test_impacto_es_positivo_y_se_recupera():
    """R debe ser POSITIVO y del orden del impacto verdadero."""
    df, eps, log_open, impacto = _mercado_sintetico(impacto=0.02)
    lags, R = response_function(log_open, eps, max_lag=10)
    assert np.all(R[:5] > 0), f"el impacto salio negativo: {R[:5]}"
    # R(1) debe aproximar el impacto verdadero
    assert abs(R[0] - impacto) < 0.2 * impacto, \
        f"R(1)={R[0]:.4f} no recupera el impacto verdadero {impacto}"


def test_usar_close_invierte_el_signo():
    """Fija el bug histórico: con referencia posterior al impacto, R<0.

    Si este test empieza a fallar es que alguien cambió la semántica; hay que
    revisar la documentación de `response_function` antes de tocarlo.
    """
    df, eps, log_open, _ = _mercado_sintetico(impacto=0.02)
    log_close = np.log(df["Close"].to_numpy(float))
    _, R_open = response_function(log_open, eps, max_lag=5)
    _, R_close = response_function(log_close, eps, max_lag=5)
    assert R_open[0] > 0, "con Open el impacto debe ser positivo"
    assert R_close[0] < R_open[0], \
        "con Close el impacto medido debe ser menor (ya esta incorporado)"


def test_response_from_frame_usa_open():
    """La vía recomendada debe dar el signo correcto sin que el usuario piense."""
    df, eps, log_open, impacto = _mercado_sintetico(impacto=0.02)
    lags, R, flujo, lo = response_from_frame(df, max_lag=10)
    assert np.all(R[:5] > 0), f"response_from_frame dio impacto negativo: {R[:5]}"
    np.testing.assert_allclose(lo, log_open, atol=1e-12)


def test_autocorr_flujo_detecta_memoria():
    """Con flujo autocorrelacionado por construccion, C_eps(1) > 0."""
    rng = np.random.default_rng(1)
    n = 5000
    e = np.empty(n); e[0] = 0.0
    for t in range(1, n):                      # AR(1) con memoria fuerte
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.1)
    C = flow_autocorrelation(e, max_lag=5)
    assert C[0] > 0.4, f"deberia detectar la memoria del AR(1): C(1)={C[0]:.3f}"


def test_flujo_iid_no_tiene_memoria():
    """Control negativo: flujo sin estructura -> C_eps ~ 0."""
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.2, 5000)
    C = flow_autocorrelation(e, max_lag=5)
    assert np.all(np.abs(C) < 0.06), f"detecto memoria donde no la hay: {C}"
