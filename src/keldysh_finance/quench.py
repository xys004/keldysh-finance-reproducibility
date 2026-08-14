r"""
El reloj del quench: envejecimiento CONDICIONADO a shocks.

POR QUÉ EL AGING SALIÓ NULO Y QUÉ CAMBIA AQUÍ
---------------------------------------------
`aging_collapse` dio μ=0 en 22/24 configs (exp. 05a), y su propio docstring
avisaba de la causa: en un mercado no hay temple con origen temporal
definido, así que un t_w medido desde el arranque de la muestra es un reloj
ARBITRARIO. Este módulo pone el reloj donde la física lo pide: t_w = tiempo
desde el último SHOCK de volatilidad. Cada shock es un temple; la colección
de shocks es el ENSEMBLE que una sola trayectoria no daba — el promedio
⟨·⟩ sobre shocks es el promedio sobre realizaciones de Kadanoff-Baym tras el
quench, que es exactamente el objeto C(t_w, τ) del formalismo.

LAS DOS PREGUNTAS (declaradas antes de mirar)
---------------------------------------------
1. ¿Relaja la MEDIA?  m(t_w) = ⟨|r(t_s+t_w)|/σ(t_s)⟩ decayendo en ley de
   potencias es el análogo del Omori financiero (Lillo-Farmer 2003, p≈0.2-0.4).
   Esto está en la literatura: es la validación externa del pipeline, el
   papel que Lillo-Farmer jugó para flow.py.
2. ¿ENVEJECE la CORRELACIÓN?  τ_c(t_w) creciendo con la edad —el sistema
   olvida más despacio cuanto más viejo— es envejecimiento genuino, lo que
   el 05a no pudo ver sin reloj. Ésta es la parte nueva.

NORMALIZACIÓN
-------------
El campo por shock es w_s(u) = |r(t_s+u)| / σ_EWMA(t_s): normalizar por la
volatilidad PRE-shock (causal: σ_EWMA en t_s no usa r_{t_s}) hace comparables
shocks de épocas con niveles de vol distintos, que es lo que permite promediar
2022 con 2026 en el mismo ensemble.
"""
from __future__ import annotations

import numpy as np


def detectar_shocks(returns: np.ndarray, sigma: np.ndarray,
                    umbral_q: float = 0.995, separacion: int = 384,
                    margen_post: int = 300, margen_pre: int = 200
                    ) -> tuple[np.ndarray, float]:
    r"""Índices t_s de shocks: |r_t|/σ_t por encima del cuantil `umbral_q`.

    Se toma el PRIMER cruce de cada episodio (greedy con separación mínima):
    dos barras consecutivas por encima del umbral son el mismo temple, no dos.
    `separacion` garantiza ventanas post-shock sin solapar (la lección G4 del
    05a: el solape fabrica estructura); `margen_post` exige historia completa
    tras el shock y `margen_pre` deja calentar la EWMA.

    Returns
    -------
    (t_s, umbral) — índices de shock y el umbral en unidades de |r|/σ.
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma, dtype=float)
    n = len(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s > 0, np.abs(r) / s, np.nan)
    fin = np.isfinite(ratio)
    if fin.sum() < 100:
        return np.empty(0, dtype=int), float("nan")
    thr = float(np.quantile(ratio[fin], umbral_q))

    shocks = []
    ultimo = -separacion
    for t in range(margen_pre, n - margen_post):
        if fin[t] and ratio[t] > thr and t - ultimo >= separacion:
            shocks.append(t)
            ultimo = t
    return np.asarray(shocks, dtype=int), thr


def campo_post_shock(returns: np.ndarray, sigma: np.ndarray,
                     shocks: np.ndarray, u_max: int) -> np.ndarray:
    """Matriz W (n_shocks, u_max): W[s, u-1] = |r(t_s+u)| / σ(t_s), u=1..u_max.

    u empieza en 1: la barra del shock (u=0) se excluye del campo — es la
    condición inicial del temple, no parte de la relajación.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    s = np.asarray(sigma, dtype=float)
    W = np.full((len(shocks), u_max), np.nan)
    for k, t_s in enumerate(shocks):
        if s[t_s] > 0 and t_s + u_max < len(r):
            W[k] = r[t_s + 1: t_s + 1 + u_max] / s[t_s]
    return W


def correlacion_por_edad(W: np.ndarray, bordes_tw, max_lag: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    r"""ρ(bin de t_w, τ): la función a dos tiempos del ensemble de shocks.

    Se centra con la media DE ENSEMBLE por edad, m(u) = ⟨W[:,u]⟩ — no con
    medias temporales: la relajación de la media es señal (Omori), no sesgo, y
    restarla mal la colaría dentro de la correlación. Después, para cada bin
    de edades [a,b): ρ_bin(τ) = promedio sobre t_w∈bin de la correlación de
    ensemble corr(δW(t_w), δW(t_w+τ)).

    Los bins de t_w existen porque con ~50-100 shocks la celda (t_w, τ)
    individual es ruido; promediar edades dentro de un bin logarítmico es el
    estándar en análisis de envejecimiento (cuasi-estacionariedad local).

    Returns
    -------
    (rho, n_pares): rho de forma (n_bins, max_lag); n_pares por bin.
    """
    n_shocks, u_max = W.shape
    m = np.nanmean(W, axis=0)
    dW = W - m[None, :]
    sd = np.nanstd(W, axis=0, ddof=1)

    bordes = list(bordes_tw)
    n_bins = len(bordes) - 1
    rho = np.full((n_bins, max_lag), np.nan)
    n_pares = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        a, z = int(bordes[b]), int(bordes[b + 1])
        acum = np.zeros(max_lag)
        cnt = np.zeros(max_lag)
        for t_w in range(a, min(z, u_max - 1)):
            tope = min(max_lag, u_max - 1 - t_w)
            if tope <= 0 or sd[t_w] <= 0:
                continue
            for lag in range(1, tope + 1):
                if sd[t_w + lag] <= 0:
                    continue
                prod = dW[:, t_w] * dW[:, t_w + lag]
                fin = np.isfinite(prod)
                if fin.sum() < 10:
                    continue
                acum[lag - 1] += float(np.mean(prod[fin])) / (sd[t_w] * sd[t_w + lag])
                cnt[lag - 1] += 1
        con = cnt > 0
        rho[b, con] = acum[con] / cnt[con]
        n_pares[b] = int(cnt.max()) if cnt.max() > 0 else 0
    return rho, n_pares


def tau_c_filas(rho: np.ndarray) -> np.ndarray:
    """τ_c integral por fila (suma truncada en el primer cruce por cero) —
    el mismo estimador que `stationarity.correlation_time`, sobre una matriz
    cualquiera."""
    out = np.full(rho.shape[0], np.nan)
    for k in range(rho.shape[0]):
        r = rho[k]
        fin = np.isfinite(r)
        if not fin.any():
            continue
        r = np.where(fin, r, 0.0)
        neg = np.where(r <= 0)[0]
        stop = int(neg[0]) if len(neg) else len(r)
        if stop > 0:
            out[k] = float(np.sum(r[:stop]))
    return out


def estadistico_envejecimiento(tau_c: np.ndarray) -> float:
    """Spearman(índice de bin, τ_c): >0 = la memoria crece con la edad
    (envejecimiento); ~0 = TTI respecto al reloj del shock."""
    from scipy.stats import spearmanr

    fin = np.isfinite(tau_c)
    if fin.sum() < 3:
        return float("nan")
    return float(spearmanr(np.arange(len(tau_c))[fin], tau_c[fin]).statistic)


def exponente_omori(m: np.ndarray, u_min: int = 1) -> dict:
    r"""Ajuste m(u) = m_∞ + c·(1+u)^(−p): la relajación de la media post-shock.

    Los TRES parámetros se ajustan a la vez por mínimos cuadrados no
    lineales. La alternativa obvia —estimar m_∞ con la media de la cola y
    ajustar el exceso en log-log— falla exactamente en el caso financiero:
    con p pequeño la relajación no termina dentro de la ventana, la "cola"
    sigue dentro del decaimiento, y restar ese nivel inflado dispara el
    exponente (medido: p plantado 0.3 → 1.34 con el método de la resta).
    """
    from scipy.optimize import least_squares

    m = np.asarray(m, dtype=float)
    u = np.arange(1, len(m) + 1, dtype=float)
    usar = (u >= u_min) & np.isfinite(m)
    out = {"p": np.nan, "se_p": np.nan, "m_inf": np.nan,
           "n_puntos": int(usar.sum())}
    if usar.sum() < 12:
        return out
    uu, mm = u[usar], m[usar]

    def resid(x):
        m_inf, log_c, p = x
        return m_inf + np.exp(log_c) * (1.0 + uu) ** (-p) - mm

    m0 = float(np.nanmin(mm))
    c0 = max(float(mm[0] - m0), 1e-3)
    lo = np.array([0.0, np.log(1e-4), 0.05])
    hi = np.array([float(np.nanmax(mm)), np.log(1e4), 3.0])
    x0 = np.clip(np.array([m0, np.log(c0), 0.5]), lo + 1e-9, hi - 1e-9)
    try:
        res = least_squares(resid, x0, bounds=(lo, hi), max_nfev=500)
    except Exception:
        return out
    if not res.success:
        return out
    out["m_inf"], out["p"] = float(res.x[0]), float(res.x[2])
    dof = len(mm) - 3
    if dof > 0:
        ssr = float(np.sum(res.fun ** 2))
        try:
            cov = ssr / dof * np.linalg.inv(res.jac.T @ res.jac)
            out["se_p"] = float(np.sqrt(max(cov[2, 2], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return out
