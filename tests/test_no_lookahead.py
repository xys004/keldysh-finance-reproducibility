"""
Tests de causalidad. Son los más importantes del paquete.

Un experimento de predicción con look-ahead produce resultados espectaculares
y completamente falsos. Estos tests comprueban la propiedad estructural: si
se cambian los datos FUTUROS, las características calculadas en el pasado no
pueden moverse. Es una comprobación de caja negra que no depende de leer el
código con atención.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance import (two_time_surface, tti_breaking, correlation_time,
                             ewma_vol, realized_vol_forward)
from keldysh_finance.flow import rolling_fdt_violation, align_to_returns


def _serie(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    # volatilidad con clustering para que la ACF de |r| tenga estructura real
    vol = np.empty(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = 0.9 * vol[t - 1] + 0.1 * 0.01 + 0.02 * abs(rng.normal()) * 0.01
    return rng.normal(0, vol)


def test_superficie_no_usa_futuro():
    """Cambiar la cola de la serie no puede alterar ventanas anteriores."""
    r = _serie()
    corte = 800
    r2 = r.copy()
    r2[corte:] = _serie(len(r), seed=99)[corte:]      # futuro completamente distinto

    s1 = two_time_surface(r, window=250, max_lag=40, step=10)
    s2 = two_time_surface(r2, window=250, max_lag=40, step=10)

    # ventanas que TERMINAN antes del corte: deben ser idénticas
    m = s1.t_w < corte
    assert m.sum() > 5, "el test necesita varias ventanas anteriores al corte"
    np.testing.assert_allclose(s1.rho[m], s2.rho[m], rtol=1e-12, atol=1e-12,
                               err_msg="la superficie usa datos futuros")


def test_tti_expanding_es_causal():
    """D(t_w) en modo 'expanding' no puede depender del futuro."""
    r = _serie()
    corte = 800
    r2 = r.copy()
    r2[corte:] = _serie(len(r), seed=123)[corte:]

    s1, s2 = (two_time_surface(x, window=250, max_lag=40, step=10) for x in (r, r2))
    d1 = tti_breaking(s1, reference="expanding", min_windows=10)
    d2 = tti_breaking(s2, reference="expanding", min_windows=10)

    m = (s1.t_w < corte) & np.isfinite(d1) & np.isfinite(d2)
    assert m.sum() > 5
    np.testing.assert_allclose(d1[m], d2[m], rtol=1e-10, atol=1e-10,
                               err_msg="tti_breaking(expanding) mira al futuro")


def test_tti_full_es_no_causal_y_esta_declarado():
    """El modo 'full' SÍ usa el futuro. El test lo fija como comportamiento
    conocido para que nadie lo use por accidente en una evaluación."""
    r = _serie()
    corte = 800
    r2 = r.copy()
    r2[corte:] = _serie(len(r), seed=7)[corte:] * 5.0   # futuro muy distinto

    s1, s2 = (two_time_surface(x, window=250, max_lag=40, step=10) for x in (r, r2))
    d1 = tti_breaking(s1, reference="full")
    d2 = tti_breaking(s2, reference="full")

    m = (s1.t_w < corte) & np.isfinite(d1) & np.isfinite(d2)
    assert not np.allclose(d1[m], d2[m]), \
        "se esperaba que 'full' fuese no causal; si ya no lo es, actualiza la doc"


def test_ewma_es_causal():
    r = _serie()
    corte = 800
    r2 = r.copy(); r2[corte:] = 0.5                     # futuro absurdo
    v1, v2 = ewma_vol(r), ewma_vol(r2)
    np.testing.assert_allclose(v1[:corte], v2[:corte], rtol=1e-12, atol=1e-12)


def test_objetivo_si_es_futuro():
    """realized_vol_forward DEBE mirar al futuro: es la etiqueta, no una
    característica. El test fija esa asimetría."""
    r = _serie()
    corte = 800
    r2 = r.copy(); r2[corte:] *= 10.0
    rv1, rv2 = realized_vol_forward(r, 20), realized_vol_forward(r2, 20)
    # justo antes del corte, la etiqueta ya ve el cambio
    assert not np.allclose(rv1[corte - 10], rv2[corte - 10]), \
        "la etiqueta deberia mirar al futuro por definicion"


def test_max_lag_se_valida_contra_window():
    r = _serie(400)
    with pytest.raises(ValueError, match="max_lag"):
        two_time_surface(r, window=100, max_lag=60)


def test_correlation_time_positivo_en_serie_con_clustering():
    r = _serie()
    s = two_time_surface(r, window=250, max_lag=40, step=10)
    tc = correlation_time(s, method="integral")
    fin = tc[np.isfinite(tc)]
    assert len(fin) > 5
    assert np.median(fin) > 0, "una serie con clustering debe dar memoria positiva"


# --- características de flujo (experimento 02) --------------------------------

def _mercado_flujo(n=1200, seed=3):
    """Precios de apertura y flujo firmado, sin pretensión de realismo."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.2, n)
    log_open = np.log(100.0) + np.cumsum(rng.normal(0, 0.01, n))
    return log_open, eps


def test_rolling_fdt_es_causal():
    """V(t_w) sobre ventanas trailing no puede moverse si cambia el futuro."""
    log_open, eps = _mercado_flujo()
    n, corte = len(eps), 900
    eps2, lo2 = eps.copy(), log_open.copy()
    eps2[corte:] = np.random.default_rng(99).normal(0, 1.0, n - corte)
    lo2[corte:] = log_open[corte:] + 0.5

    t1, v1 = rolling_fdt_violation(log_open, eps, window=300, max_lag=20, step=10)
    t2, v2 = rolling_fdt_violation(lo2, eps2, window=300, max_lag=20, step=10)

    m = (t1 < corte) & np.isfinite(v1) & np.isfinite(v2)
    assert m.sum() > 5, "el test necesita varias ventanas anteriores al corte"
    np.testing.assert_allclose(v1[m], v2[m], rtol=1e-10, atol=1e-10,
                               err_msg="rolling_fdt_violation usa datos futuros")


def test_align_to_returns_desplaza_exactamente_una_posicion():
    """La característica de la vela j cae en el índice de RETORNO j-1.

    Fija la dirección del desfase. Si alguien 'corrige' esto quitando el último
    elemento en vez del primero, la característica se adelanta una barra y el
    experimento pasa a tener look-ahead sin que ningún test de ventana lo note.
    """
    n_klines = 10
    vals = np.arange(n_klines, dtype=float) * 10.0
    out = align_to_returns(np.arange(n_klines), vals, n_klines)
    assert len(out) == n_klines - 1
    np.testing.assert_allclose(out, vals[1:])


def test_align_rellena_hacia_adelante_no_hacia_atras():
    """Con step>1 sólo puede propagarse el último valor DISPONIBLE."""
    out = align_to_returns(np.array([2, 5]), np.array([1.0, 2.0]), n_klines=8)
    # velas:   0    1    2    3    4    5    6    7
    # valor:  nan  nan  1.0  1.0  1.0  2.0  2.0  2.0   -> se quita la primera
    np.testing.assert_allclose(out, [np.nan, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def test_pipeline_flujo_no_ve_el_futuro():
    """Caja negra sobre la composición completa que usa el experimento 02."""
    log_open, eps = _mercado_flujo()
    n, corte = len(eps), 900
    eps2, lo2 = eps.copy(), log_open.copy()
    eps2[corte:] = np.random.default_rng(7).normal(0, 1.0, n - corte)
    lo2[corte:] = log_open[corte:] + 0.5

    def pipeline(p, e):
        t_w, V = rolling_fdt_violation(p, e, window=300, max_lag=20, step=1)
        return align_to_returns(t_w, V, n_klines=n)

    a, b = pipeline(log_open, eps), pipeline(lo2, eps2)
    assert len(a) == n - 1

    # la característica en el índice de retorno i usa velas <= i+1
    ok = (np.arange(n - 1) <= corte - 2) & np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 50
    np.testing.assert_allclose(a[ok], b[ok], rtol=1e-10, atol=1e-10,
                               err_msg="la caracteristica de flujo usa velas futuras")

    # ...y el límite es JUSTO: en i = corte-1 el cambio ya tiene que notarse.
    # Sin esta mitad, un desfase excesivamente conservador pasaría inadvertido.
    assert np.isfinite(a[corte - 1]) and np.isfinite(b[corte - 1])
    assert not np.isclose(a[corte - 1], b[corte - 1]), \
        "el desfase es mas conservador de lo declarado: revisa align_to_returns"
