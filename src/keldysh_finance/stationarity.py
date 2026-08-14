r"""
Diagnósticos de NO-ESTACIONARIEDAD sobre la superficie a dos tiempos.

EL OBSERVABLE CENTRAL
---------------------
En equilibrio, la función de correlación es invariante bajo traslación
temporal (TTI):

    C(t, t') = C(t - t')

Fuera de equilibrio depende de AMBOS tiempos por separado. Ésta es la
definición estructural, no una analogía: la ruptura de TTI *es* la firma de
que el sistema no ha relajado a un estado estacionario.

Traducido a la superficie estimada Ĉ(t_w, τ):
  · TTI se cumple  →  ρ(t_w, ·) ≈ ρ̄(·) para todo t_w
  · TTI se rompe   →  el perfil de correlación depende de cuándo se mire

Se definen dos medidas, ambas causales (sólo usan datos hasta t_w):

  D(t_w)   distancia normalizada entre el perfil local y el de referencia.
           Es el análogo del residuo `scba.fdt_error` de QuantumTransportEOM:
           una cantidad que vale ~0 si la relación de equilibrio se cumple y
           crece al alejarse.

  τ_c(t_w) tiempo de correlación local (memoria de la volatilidad). No mide
           distancia al equilibrio sino la ESCALA en la que el sistema
           olvida. Es la "ventana ajustable" del proyecto, pero medida en vez
           de elegida a mano.

REFERENCIA CAUSAL
-----------------
ρ̄ NO puede ser el promedio de toda la muestra: eso usaría el futuro. Se
calcula de forma expansiva (sólo con ventanas anteriores) — ver
`tti_breaking(..., reference='expanding')`, que es el modo por defecto.
El modo 'full' existe sólo para inspección exploratoria y está marcado como
NO APTO para evaluación predictiva.
"""
from __future__ import annotations

import numpy as np

from .two_time import TwoTimeSurface


def tti_breaking(surface: TwoTimeSurface,
                 reference: str = "expanding",
                 min_windows: int = 20) -> np.ndarray:
    r"""D(t_w): distancia normalizada del perfil local a la referencia.

        D(t_w) = || ρ(t_w,·) − ρ̄(·) ||₂ / (|| ρ̄(·) ||₂ + eps)

    Parameters
    ----------
    surface : TwoTimeSurface
    reference : {'expanding', 'full'}
        'expanding' (por defecto, CAUSAL): ρ̄ en la ventana k usa sólo las
            ventanas 0..k-1. Las primeras `min_windows` devuelven NaN porque
            no hay referencia suficiente.
        'full' (NO CAUSAL): ρ̄ es el promedio de toda la muestra. Sólo para
            exploración visual; usarlo en una evaluación predictiva mete
            look-ahead.
    min_windows : int
        Ventanas mínimas antes de emitir un valor en modo 'expanding'.

    Returns
    -------
    np.ndarray de longitud n_windows (con NaN al principio en modo expanding).
    """
    rho = surface.rho
    n = rho.shape[0]
    out = np.full(n, np.nan, dtype=float)

    if reference == "full":
        ref = np.nanmean(rho, axis=0)
        norm_ref = np.linalg.norm(ref[np.isfinite(ref)])
        for k in range(n):
            d = rho[k] - ref
            m = np.isfinite(d)
            if m.sum() >= 2:
                out[k] = np.linalg.norm(d[m]) / (norm_ref + 1e-12)
        return out

    if reference != "expanding":
        raise ValueError(f"reference desconocida: {reference!r}")

    # CAUSAL: la referencia en k sólo ve el pasado.
    csum = np.zeros(rho.shape[1], dtype=float)
    ccount = np.zeros(rho.shape[1], dtype=float)
    for k in range(n):
        if k >= min_windows:
            with np.errstate(invalid="ignore", divide="ignore"):
                ref = np.where(ccount > 0, csum / np.maximum(ccount, 1), np.nan)
            d = rho[k] - ref
            m = np.isfinite(d)
            if m.sum() >= 2:
                norm_ref = np.linalg.norm(ref[np.isfinite(ref)])
                out[k] = np.linalg.norm(d[m]) / (norm_ref + 1e-12)
        # acumular DESPUÉS de usar: la ventana k no entra en su propia referencia
        fin = np.isfinite(rho[k])
        csum[fin] += rho[k][fin]
        ccount[fin] += 1.0
    return out


def correlation_time(surface: TwoTimeSurface, method: str = "integral") -> np.ndarray:
    r"""τ_c(t_w): profundidad de memoria local del campo de volatilidad.

    method='integral' : τ_c = Σ_τ ρ(t_w,τ) truncada en el primer cruce por
        cero (el estimador estándar; sumar más allá del cruce añade ruido).
    method='efold'    : primer τ con ρ < 1/e, interpolado linealmente.

    Interpretación: es la escala temporal en la que el mercado "olvida" su
    propio nivel de volatilidad. Un τ_c que se mueve en el tiempo ES la
    ventana ajustable del planteamiento original — pero medida, no elegida.
    """
    rho = surface.rho
    n = rho.shape[0]
    out = np.full(n, np.nan, dtype=float)

    for k in range(n):
        r = rho[k]
        if not np.isfinite(r).any():
            continue
        if method == "integral":
            neg = np.where(r <= 0)[0]
            stop = int(neg[0]) if len(neg) else len(r)
            seg = r[:stop]
            if len(seg):
                out[k] = float(np.nansum(seg))
        elif method == "efold":
            thr = 1.0 / np.e
            below = np.where(r < thr)[0]
            if len(below) == 0:
                out[k] = float(len(r))          # no decae dentro del rango
            elif below[0] == 0:
                out[k] = 0.0
            else:
                i = int(below[0])
                r0, r1 = r[i - 1], r[i]
                frac = 0.0 if r0 == r1 else (r0 - thr) / (r0 - r1)
                out[k] = float(i + frac)
        else:
            raise ValueError(f"method desconocido: {method!r}")
    return out


def aging_collapse(surface: TwoTimeSurface, mu_grid=None) -> dict:
    r"""¿Colapsa la superficie bajo el reescalado de envejecimiento τ → τ/t_w^μ?

    En sistemas vítreos fuera de equilibrio, C(t_w, τ) no es función de τ sola
    pero SÍ de la variable de escala τ/t_w^μ. Encontrar un μ que colapse las
    curvas es la evidencia canónica de envejecimiento.

    ADVERTENCIA HONESTA: en un mercado no hay un "quench" con origen temporal
    definido, así que t_w es arbitrario respecto a cualquier reloj físico. Un
    colapso aquí sería sugerente pero NO tendría la misma fuerza que en un
    experimento con temple. Se incluye por completitud del programa, con
    μ=0 (o sea, TTI puro) como hipótesis nula anidada.

    Returns
    -------
    dict con 'mu_best', 'residual_best' y la curva 'residuals' sobre mu_grid.
    """
    if mu_grid is None:
        mu_grid = np.linspace(0.0, 1.0, 21)
    rho, lags, t_w = surface.rho, surface.lags.astype(float), surface.t_w.astype(float)
    valid = np.isfinite(rho).all(axis=1)
    if valid.sum() < 5:
        return {"mu_best": np.nan, "residual_best": np.nan,
                "mu_grid": np.asarray(mu_grid), "residuals": np.full(len(mu_grid), np.nan)}
    R, TW = rho[valid], np.maximum(t_w[valid], 1.0)

    residuals = []
    for mu in mu_grid:
        # variable de escala por ventana
        xs = np.concatenate([lags / (tw ** mu) for tw in TW])
        ys = R.ravel()
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        # dispersión alrededor de una curva maestra estimada por binning
        nb = 30
        edges = np.quantile(xs, np.linspace(0, 1, nb + 1))
        idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, nb - 1)
        resid = 0.0
        for b in range(nb):
            m = idx == b
            if m.sum() > 1:
                resid += float(np.sum((ys[m] - ys[m].mean()) ** 2))
        residuals.append(resid / max(len(ys), 1))

    residuals = np.asarray(residuals)
    best = int(np.nanargmin(residuals))
    return {"mu_best": float(np.asarray(mu_grid)[best]),
            "residual_best": float(residuals[best]),
            "mu_grid": np.asarray(mu_grid), "residuals": residuals,
            "residual_tti": float(residuals[0])}    # mu=0 == TTI puro
