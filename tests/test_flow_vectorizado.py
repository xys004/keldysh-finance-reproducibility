r"""
Equivalencia EXACTA entre la vía rápida por FFT y la implementación directa.

POR QUÉ ESTE TEST ES OBLIGATORIO
--------------------------------
`response_function` y `flow_autocorrelation` se vectorizaron el 2026-08-08 para
poder permitirse el barrido de escalas (que multiplica el coste por ~300). Una
vectorización que cambie el estimador aunque sea en el tercer decimal
invalidaría TODAS las cifras ya depositadas en `output/*.json` sin que nada lo
delatara — los números seguirían saliendo, sólo que serían otros.

Por eso la implementación ORIGINAL vive aquí, copiada literalmente, y se
contrasta contra la de la librería. Si alguien vuelve a tocar `flow.py` en
busca de velocidad, este test es el que decide si la nueva versión es la misma
función o una distinta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.flow import (response_function, flow_autocorrelation,
                                  fdt_diagnostic)


# --- implementaciones de REFERENCIA (las originales, bucle directo) ----------

def _response_ref(p, e, max_lag):
    p = np.asarray(p, float)
    e = np.asarray(e, float)
    n = min(len(p), len(e))
    p, e = p[:n], e[:n]
    ec = e - np.nanmean(e)
    var_e = float(np.nanmean(ec ** 2))
    lags = np.arange(1, max_lag + 1)
    R = np.full(len(lags), np.nan)
    if var_e <= 0:
        return lags, R
    for k, lag in enumerate(lags):
        dp = p[lag:] - p[:-lag]
        prod = dp * ec[:-lag]
        m = np.isfinite(prod)
        if m.sum() > 30:
            R[k] = float(np.mean(prod[m])) / var_e
    return lags, R


def _autocorr_ref(e, max_lag):
    e = np.asarray(e, float)
    ec = e - np.nanmean(e)
    denom = float(np.nanmean(ec ** 2))
    lags = np.arange(1, max_lag + 1)
    C = np.full(len(lags), np.nan)
    if denom <= 0:
        return C
    for k, lag in enumerate(lags):
        prod = ec[:-lag] * ec[lag:]
        m = np.isfinite(prod)
        if m.sum() > 30:
            C[k] = float(np.mean(prod[m])) / denom
    return C


def _mercado(n=500, seed=0, escala_precio=np.log(60000.0)):
    """Precios con el orden de magnitud REAL (log 60k ≈ 11) y flujo firmado.

    La escala importa: el término cruzado de la FFT mezcla p (~11) con ε
    (~0.2), y si el centrado de p no fuese exacto la pérdida de precisión
    aparecería justo aquí y no en datos de juguete centrados en cero.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.2, n)
    p = escala_precio + np.cumsum(rng.normal(0, 0.01, n))
    return p, eps


@pytest.mark.parametrize("n,max_lag", [(500, 30), (500, 60), (1200, 40),
                                       (200, 30), (64, 30), (100, 99)])
def test_respuesta_fft_igual_que_bucle(n, max_lag):
    p, eps = _mercado(n=n, seed=n + max_lag)
    _, r_rapida = response_function(p, eps, max_lag=max_lag)
    _, r_ref = _response_ref(p, eps, max_lag)
    np.testing.assert_allclose(r_rapida, r_ref, rtol=1e-10, atol=1e-14,
                               err_msg="la via FFT no reproduce el estimador original")
    # y los NaN tienen que caer en los MISMOS desfases
    np.testing.assert_array_equal(np.isnan(r_rapida), np.isnan(r_ref))


@pytest.mark.parametrize("n,max_lag", [(500, 30), (500, 60), (1200, 40),
                                       (200, 30), (64, 30), (100, 99)])
def test_autocorr_fft_igual_que_bucle(n, max_lag):
    _, eps = _mercado(n=n, seed=2 * n + max_lag)
    c_rapida = flow_autocorrelation(eps, max_lag=max_lag)
    c_ref = _autocorr_ref(eps, max_lag)
    np.testing.assert_allclose(c_rapida, c_ref, rtol=1e-10, atol=1e-14,
                               err_msg="la via FFT no reproduce la autocorrelacion original")
    np.testing.assert_array_equal(np.isnan(c_rapida), np.isnan(c_ref))


def test_flujo_con_memoria_larga():
    """El caso que importa de verdad: flujo AR(1), no ruido blanco."""
    rng = np.random.default_rng(11)
    n = 800
    e = np.empty(n); e[0] = 0.0
    for t in range(1, n):
        e[t] = 0.85 * e[t - 1] + rng.normal(0, 0.1)
    p = np.log(3000.0) + np.cumsum(0.02 * e + rng.normal(0, 0.005, n))
    np.testing.assert_allclose(flow_autocorrelation(e, 50), _autocorr_ref(e, 50),
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(response_function(p, e, 50)[1],
                               _response_ref(p, e, 50)[1], rtol=1e-10, atol=1e-14)


def test_fallback_con_no_finitos():
    """Con NaN se cae al bucle; debe seguir dando el resultado de referencia."""
    p, eps = _mercado(n=600, seed=5)
    eps[100] = np.nan
    p[300] = np.nan
    np.testing.assert_allclose(response_function(p, eps, 30)[1],
                               _response_ref(p, eps, 30)[1],
                               rtol=1e-10, atol=1e-14, equal_nan=True)
    np.testing.assert_allclose(flow_autocorrelation(eps, 30),
                               _autocorr_ref(eps, 30),
                               rtol=1e-10, atol=1e-14, equal_nan=True)


def test_flujo_exactamente_cero_da_nan():
    """var(ε) exactamente 0: no hay respuesta definible, y ambas vías dan NaN."""
    p, _ = _mercado(n=300, seed=9)
    e = np.zeros(300)
    _, R = response_function(p, e, 20)
    assert np.all(np.isnan(R))
    assert np.all(np.isnan(flow_autocorrelation(e, 20)))


def test_flujo_casi_constante_explota_igual_en_las_dos_vias():
    """FRAGILIDAD CONOCIDA del estimador, fijada aquí para que no sorprenda.

    Con flujo constante, `ec = e − mean(e)` NO es exactamente cero: quedan
    residuos de coma flotante, y `var_e` sale ~1e−32 en vez de 0. La guarda
    `if var_e <= 0` no salta, y R(τ) se dispara a ~1e12.

    No es un defecto introducido por la vectorización — la implementación
    original hace exactamente lo mismo, y este test lo demuestra comparando
    ambas. Se documenta en vez de arreglarse porque cambiar la guarda
    cambiaría el estimador, y este fichero existe justo para impedir que eso
    ocurra de tapadillo. Si el barrido de escalas llega a activos ilíquidos
    (velas de volumen casi constante), hay que revisarlo antes.
    """
    p, _ = _mercado(n=300, seed=9)
    e = np.full(300, 0.7)
    _, r_rapida = response_function(p, e, 20)
    _, r_ref = _response_ref(p, e, 20)
    np.testing.assert_allclose(r_rapida, r_ref, rtol=1e-10, atol=1e-14,
                               equal_nan=True)
    assert np.nanmax(np.abs(r_rapida)) > 1e6, \
        "si esto deja de explotar es que alguien cambio la guarda de var_e"


def test_diagnostico_fdt_igual_extremo_a_extremo():
    """La equivalencia tiene que sobrevivir a la composición, no sólo pieza a pieza.

    `fdt_violation` es un cociente de dispersiones sobre T_eff = −dC/R, así que
    amplifica cualquier diferencia entre las dos vías: es el test que de verdad
    protege las cifras depositadas.
    """
    for seed in (0, 1, 2, 3, 4):
        p, eps = _mercado(n=500, seed=seed)
        obs = fdt_diagnostic(p, eps, max_lag=30)

        C_ref = _autocorr_ref(eps, 30)
        _, R_ref = _response_ref(p, eps, 30)
        dC = np.full_like(C_ref, np.nan)
        dC[1:-1] = (C_ref[2:] - C_ref[:-2]) / 2.0
        dC[0] = C_ref[1] - C_ref[0]
        dC[-1] = C_ref[-1] - C_ref[-2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_eff = np.where(np.abs(R_ref) > 1e-12, -dC / R_ref, np.nan)
        fin = t_eff[np.isfinite(t_eff)]
        viol_ref = float(np.std(fin) / np.median(np.abs(fin)))

        assert abs(obs.fdt_violation - viol_ref) <= 1e-8 * max(1.0, abs(viol_ref)), \
            f"semilla {seed}: violacion FDT {obs.fdt_violation} != referencia {viol_ref}"
