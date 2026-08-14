r"""
Estimación de la superficie de correlación a dos tiempos desde UNA trayectoria.

DIFERENCIA CLAVE CON QuantumTransportEOM
----------------------------------------
QTEOM resuelve el problema DIRECTO: dado un hamiltoniano y unas auto-energías,
calcula G(t,t'). Aquí el problema es el INVERSO: dada una serie temporal (una
sola realización), estimar Ĉ(t_w, τ) y contrastar sus propiedades. Por eso este
paquete no clona QTEOM — comparte el marco conceptual (descomposición a dos
tiempos, residuo tipo `scba.fdt_error`) pero la dirección del cómputo es la
opuesta.

EL PROBLEMA DEL ENSEMBLE
------------------------
En física se promedia sobre realizaciones. En un mercado hay UNA trayectoria.
La sustitución estándar es un promedio local en el tiempo: se estima la
correlación dentro de una ventana, y se desliza la ventana. Eso convierte la
dependencia en dos tiempos C(t,t') en una superficie C(t_w, τ) donde
  t_w = tiempo de observación ("waiting time"),
  τ   = desfase.

QUÉ CAMPO SE CORRELACIONA
-------------------------
NO los retornos. Los retornos crudos son casi incorrelados —eso es la línea
base de eficiencia, y el holdout pre-registrado del proyecto de trading
(0/30) lo confirma para timing direccional—. La estructura vive en el
SEGUNDO MOMENTO: |r| o r². Este módulo trabaja sobre v(t)=|r(t)| por defecto.
La elección no es cosmética: es la razón por la que este experimento puede
funcionar donde el direccional no.

SIN LOOK-AHEAD
--------------
Todas las ventanas son TRAILING (miran hacia atrás desde t_w). Una ventana
centrada usaría datos futuros y contaminaría cualquier evaluación posterior.
Hay un test dedicado en tests/test_no_lookahead.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoTimeSurface:
    """Superficie Ĉ(t_w, τ) estimada con ventanas trailing.

    Attributes
    ----------
    t_w : np.ndarray
        Índices (absolutos en la serie original) del final de cada ventana.
        La estimación en `t_w[k]` usa sólo datos con índice <= t_w[k].
    lags : np.ndarray
        Desfases τ evaluados (en número de barras).
    rho : np.ndarray
        Matriz (n_windows, n_lags) de autocorrelación local.
    window : int
        Anchura de la ventana usada.
    field : str
        Qué campo se correlacionó ('abs' | 'sq' | 'raw').
    """

    t_w: np.ndarray
    lags: np.ndarray
    rho: np.ndarray
    window: int
    field: str

    def mean_profile(self) -> np.ndarray:
        """ρ̄(τ): perfil promediado sobre todas las ventanas.

        Es el candidato a "función de correlación estacionaria". La hipótesis
        de invariancia traslacional temporal (TTI) dice que ρ(t_w,·) ≈ ρ̄(·)
        para todo t_w.
        """
        return np.nanmean(self.rho, axis=0)


def volatility_field(returns: np.ndarray, field: str = "abs") -> np.ndarray:
    """Campo fluctuante sobre el que se define la correlación.

    'abs' : |r|  — proxy robusto de volatilidad, colas menos pesadas que r²
    'sq'  : r²   — el estimador insesgado de varianza, pero muy sensible a outliers
    'raw' : r    — control negativo: si el método "encuentra" estructura aquí,
                   está encontrando ruido (los retornos son casi incorrelados).
    """
    r = np.asarray(returns, dtype=float)
    if field == "abs":
        return np.abs(r)
    if field == "sq":
        return r ** 2
    if field == "raw":
        return r
    raise ValueError(f"field desconocido: {field!r} (usa 'abs' | 'sq' | 'raw')")


def _local_acf(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Autocorrelación de una ventana, para los desfases pedidos.

    Se resta la media DE LA VENTANA (no la global): la correlación mide
    estructura alrededor del nivel local, no la deriva del nivel.
    """
    n = len(x)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0 or not np.isfinite(denom):
        return np.full(len(lags), np.nan)
    out = np.empty(len(lags), dtype=float)
    for k, lag in enumerate(lags):
        lag = int(lag)
        if lag <= 0 or lag >= n:
            out[k] = np.nan
            continue
        out[k] = float(np.dot(xc[:-lag], xc[lag:])) / denom
    return out


def two_time_surface(returns: np.ndarray,
                     window: int = 250,
                     max_lag: int = 40,
                     step: int = 1,
                     field: str = "abs") -> TwoTimeSurface:
    """Estima Ĉ(t_w, τ) deslizando una ventana TRAILING sobre la serie.

    Parameters
    ----------
    returns : array
        Log-retornos de la serie.
    window : int
        Barras por ventana. Compromiso: ventana corta reacciona antes a los
        transitorios pero estima la ACF con más ruido.
    max_lag : int
        Desfase máximo. Debe ser << window o la ACF en los desfases largos se
        estima con pocos pares.
    step : int
        Cada cuántas barras se recoloca la ventana.
    field : str
        Ver `volatility_field`.

    Returns
    -------
    TwoTimeSurface

    Notes
    -----
    La ventana que termina en el índice `t` usa `returns[t-window+1 : t+1]`.
    No se usa NINGÚN dato con índice > t.
    """
    r = np.asarray(returns, dtype=float)
    if window < 20:
        raise ValueError("window demasiado corta para estimar una ACF (mín. 20)")
    if max_lag >= window // 2:
        raise ValueError(
            f"max_lag={max_lag} debe ser < window//2={window//2}: con desfases "
            f"largos y ventana corta quedan muy pocos pares por estimación")
    v = volatility_field(r, field)
    lags = np.arange(1, max_lag + 1)

    ends = np.arange(window - 1, len(v), step)
    rho = np.empty((len(ends), len(lags)), dtype=float)
    for k, end in enumerate(ends):
        seg = v[end - window + 1: end + 1]      # TRAILING: nada de futuro
        rho[k] = _local_acf(seg, lags)

    return TwoTimeSurface(t_w=ends, lags=lags, rho=rho, window=window, field=field)
