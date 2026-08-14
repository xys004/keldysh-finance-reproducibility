r"""
Líneas base de predicción de volatilidad.

Son el listón que el diagnóstico de no-equilibrio tiene que superar para que
el programa valga algo. Elegirlas débiles sería hacerse trampas: EWMA es lo
que YA usa `portfolio.py` en producción, y GARCH(1,1) es el estándar de facto
desde 1986. Ganar a un promedio móvil simple no demostraría nada.

Ambas son causales por construcción: la predicción para t+1 usa sólo datos
hasta t.
"""
from __future__ import annotations

import numpy as np


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    r"""Volatilidad EWMA de RiskMetrics.

        σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}

    Devuelve σ_t para cada t, siendo σ_t la predicción hecha CON DATOS HASTA
    t-1 (es decir, ya está desplazada: no usa r_t). λ=0.94 es el valor de
    RiskMetrics para datos diarios y el que usa `portfolio.py`.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    var = np.empty(n, dtype=float)
    seed = float(np.nanvar(r[:min(50, n)]))
    var[0] = seed if np.isfinite(seed) and seed > 0 else 1e-8
    for t in range(1, n):
        prev = r[t - 1]
        var[t] = lam * var[t - 1] + (1.0 - lam) * (prev * prev if np.isfinite(prev) else 0.0)
    return np.sqrt(var)


def _garch_nll(params: np.ndarray, r: np.ndarray) -> float:
    """Log-verosimilitud negativa de GARCH(1,1) con innovaciones normales."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
        return 1e12                       # fuera de la región estacionaria
    n = len(r)
    var = np.empty(n, dtype=float)
    var[0] = omega / (1.0 - alpha - beta)
    if not np.isfinite(var[0]) or var[0] <= 0:
        return 1e12
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0 or not np.isfinite(var[t]):
            return 1e12
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + r ** 2 / var)
    return -ll if np.isfinite(ll) else 1e12


def fit_garch11(returns: np.ndarray) -> dict:
    """Ajusta GARCH(1,1) por máxima verosimilitud (scipy, sin dependencia `arch`).

    Los retornos se reescalan ×100 antes de optimizar: con retornos en
    fracción, omega es ~1e-6 y el optimizador se atasca en el mal
    condicionamiento. Se deshace el escalado al devolver.
    """
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] * 100.0
    if len(r) < 100:
        raise ValueError(f"muestra insuficiente para GARCH: {len(r)}")

    best = None
    # Varios arranques: la verosimilitud de GARCH tiene óptimos locales.
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.95)):
        x0 = np.array([np.var(r) * (1 - a0 - b0), a0, b0])
        try:
            res = minimize(_garch_nll, x0, args=(r,), method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GARCH no convergió desde ningún arranque")

    omega, alpha, beta = best.x
    return {"omega": float(omega) / 1e4,      # deshacer el ×100 (varianza: ×100²)
            "alpha": float(alpha), "beta": float(beta),
            "nll": float(best.fun), "persistence": float(alpha + beta)}


def garch_forecast_path(returns: np.ndarray, params: dict) -> np.ndarray:
    """σ_t predicha por GARCH con datos hasta t-1, para toda la serie.

    Los parámetros deben venir de una muestra de entrenamiento ANTERIOR: este
    módulo no los re-ajusta aquí, para que el walk-forward controle
    explícitamente qué información entra.
    """
    r = np.asarray(returns, dtype=float)
    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
    n = len(r)
    var = np.empty(n, dtype=float)
    denom = 1.0 - alpha - beta
    var[0] = omega / denom if denom > 0 else float(np.nanvar(r))
    for t in range(1, n):
        prev = r[t - 1]
        var[t] = omega + alpha * (prev ** 2 if np.isfinite(prev) else 0.0) + beta * var[t - 1]
        if not np.isfinite(var[t]) or var[t] <= 0:
            var[t] = var[t - 1]
    return np.sqrt(var)


def realized_vol_forward(returns: np.ndarray, horizon: int) -> np.ndarray:
    """OBJETIVO: volatilidad realizada en las `horizon` barras SIGUIENTES.

    rv[t] = sqrt( mean( r[t+1:t+1+horizon]² ) )

    Es futuro por definición — es lo que se quiere predecir. Nunca debe entrar
    como característica; sólo como etiqueta y sólo en la evaluación.
    Las últimas `horizon` posiciones quedan NaN.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n - horizon):
        seg = r[t + 1: t + 1 + horizon]
        if np.isfinite(seg).sum() >= max(2, horizon // 2):
            out[t] = float(np.sqrt(np.nanmean(seg ** 2)))
    return out
