r"""
Ajuste PARAMÉTRICO de la función de Green a dos tiempos, ventana a ventana.

QUÉ AÑADE ESTO SOBRE stationarity.py
------------------------------------
`tti_breaking` y `rolling_fdt_violation` reducen cada ventana a UN escalar, y
los experimentos 01-04 establecieron que esos escalares no predicen. Este
módulo hace otra cosa: ajusta una FORMA FUNCIONAL de relajación a cada perfil
local Ĉ(t_w, τ) y devuelve el VECTOR de parámetros por ventana. Dos ventanas
con la misma varianza pueden relajar distinto (β diferente a σ igual); un
escalar no puede ver eso, un vector sí.

LA FORMA FUNCIONAL
------------------
    ρ(t_w, τ) ≈ A · exp[ −(τ/τ_c)^β ]        (exponencial estirada / KWW)

Es la parametrización canónica de relajación en sistemas desordenados fuera
de equilibrio: β=1 es relajación exponencial simple (Debye, un solo tiempo);
β<1 es relajación "estirada" — superposición de escalas, la firma vítrea.
τ_c es la escala de memoria; A la amplitud de la correlación a desfase corto.

  · A es esencialmente el nivel local de correlación del campo |r| — un
    pariente cercano de σ. Por la comprobación algebraica del exp. 04,
    cualquier cosa que sea reparametrización de σ YA está probada (negativa).
    Los candidatos NUEVOS son τ_c y β. El experimento 05a mide exactamente
    eso: si τ_c y β son identificables y NO redundantes con log σ.

POR QUÉ SE AJUSTA EN (log A, log τ_c, β) Y EN EL ESPACIO DE ρ
-------------------------------------------------------------
A y τ_c son parámetros de escala positivos: en carta logarítmica la
verosimilitud es aproximadamente cuadrática y los errores del jacobiano
significan algo. Y se ajusta ρ directamente (no log ρ) porque la ACF empírica
de una ventana finita cruza cero por ruido, y el log la destruiría justo en
la cola, que es donde β<1 se distingue de β=1.

IDENTIFICABILIDAD ANTES QUE PREDICCIÓN
--------------------------------------
Con UNA realización por ventana, la pregunta previa a "¿predice?" es "¿se
puede siquiera medir?". Por eso cada ajuste devuelve, además de los
parámetros: el error estándar (jacobiano), la correlación de los estimadores
corr(log τ_c, β) — la degeneración clásica del KWW: acortar τ_c y subir β
producen curvas casi iguales —, R², y la bandera `en_borde` (un parámetro
pegado a su cota = la forma no está restringida por los datos = NO
identificable, aunque el optimizador "converja").

SIN LOOK-AHEAD
--------------
El ajuste opera sobre `TwoTimeSurface`, cuyas ventanas son TRAILING, y es
determinista por fila: la causalidad se hereda. Hay test de caja negra en
`tests/test_transient_fit.py` (regla 1 del proyecto: característica nueva,
test de causalidad nuevo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .two_time import TwoTimeSurface

# Cotas del ajuste. τ_c se capa en TAU_MAX_FACTOR·max_lag: más allá del rango
# medido la ACF no restringe τ_c y el ajuste debe declararse en_borde, no
# devolver un número enorme con cara de medición.
_LOGA_BOUNDS = (np.log(1e-3), np.log(3.0))
_BETA_BOUNDS = (0.10, 2.0)
_TAU_MAX_FACTOR = 10.0
_TAU_MIN = 0.3
_MIN_PUNTOS = 8          # puntos finitos mínimos para intentar un ajuste
_TOL_BORDE = 1e-3        # distancia a la cota por debajo de la cual es "borde"


def fit_stretched_exp(lags: np.ndarray, rho: np.ndarray) -> dict:
    r"""Ajusta ρ(τ) ≈ A·exp[−(τ/τ_c)^β] a un perfil de correlación local.

    Returns
    -------
    dict con:
      log_A, log_tau_c, beta : parámetros (NaN si no hay ajuste)
      se_log_tau_c, se_beta  : errores estándar del jacobiano (NaN si la
                               matriz normal es singular)
      corr_tau_beta          : correlación de los estimadores log τ_c y β
                               (la degeneración del KWW; |corr|→1 = no se
                               distinguen dentro del error)
      r2                     : 1 − SSR/SST sobre los puntos finitos
      convergio, en_borde    : banderas. Un ajuste sólo es utilizable si
                               convergio y NO en_borde.
      n_puntos               : puntos finitos usados
    """
    from scipy.optimize import least_squares

    out = {"log_A": np.nan, "log_tau_c": np.nan, "beta": np.nan,
           "se_log_tau_c": np.nan, "se_beta": np.nan, "corr_tau_beta": np.nan,
           "r2": np.nan, "convergio": False, "en_borde": False, "n_puntos": 0}

    t = np.asarray(lags, dtype=float)
    y = np.asarray(rho, dtype=float)
    m = np.isfinite(y) & np.isfinite(t) & (t > 0)
    out["n_puntos"] = int(m.sum())
    if m.sum() < _MIN_PUNTOS:
        return out
    t, y = t[m], y[m]
    max_lag = float(t.max())

    lo = np.array([_LOGA_BOUNDS[0], np.log(_TAU_MIN), _BETA_BOUNDS[0]])
    hi = np.array([_LOGA_BOUNDS[1], np.log(_TAU_MAX_FACTOR * max_lag),
                   _BETA_BOUNDS[1]])

    # Arranque: A0 desde el primer desfase; τ_c0 desde el cruce 1/e del perfil.
    a0 = float(np.clip(y[0], 5e-3, 2.5))
    thr = a0 / np.e
    below = np.where(y < thr)[0]
    tau0 = float(t[below[0]]) if len(below) else max_lag / 2.0
    x0 = np.clip(np.array([np.log(a0), np.log(max(tau0, _TAU_MIN)), 0.7]),
                 lo + 1e-6, hi - 1e-6)

    def resid(x):
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(x[0] - (t / np.exp(x[1])) ** x[2]) - y

    try:
        res = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                            max_nfev=200)
    except Exception:
        return out
    if not (res.success and np.all(np.isfinite(res.x))):
        return out

    out["log_A"], out["log_tau_c"], out["beta"] = (float(v) for v in res.x)
    out["convergio"] = True
    out["en_borde"] = bool(np.any(res.x - lo < _TOL_BORDE) or
                           np.any(hi - res.x < _TOL_BORDE))

    ssr = float(np.sum(res.fun ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = 1.0 - ssr / sst if sst > 0 else np.nan

    dof = len(y) - 3
    if dof > 0:
        try:
            cov = ssr / dof * np.linalg.inv(res.jac.T @ res.jac)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            out["se_log_tau_c"], out["se_beta"] = float(se[1]), float(se[2])
            if se[1] > 0 and se[2] > 0:
                out["corr_tau_beta"] = float(cov[1, 2] / (se[1] * se[2]))
        except np.linalg.LinAlgError:
            pass
    return out


@dataclass(frozen=True)
class TransientTrajectory:
    """Trayectoria de parámetros KWW sobre las ventanas de una superficie.

    Todos los arrays tienen longitud n_windows y comparten `t_w` con la
    superficie de origen (índices absolutos en la rejilla de retornos).
    `valido` = convergió, no tocó cotas y R² ≥ el mínimo pedido al construir.
    """
    t_w: np.ndarray
    log_A: np.ndarray
    log_tau_c: np.ndarray
    beta: np.ndarray
    se_log_tau_c: np.ndarray
    se_beta: np.ndarray
    corr_tau_beta: np.ndarray
    r2: np.ndarray
    valido: np.ndarray
    window: int
    field: str
    r2_min: float


def rolling_transient_fit(surface: TwoTimeSurface,
                          r2_min: float = 0.2) -> TransientTrajectory:
    """Ajusta la exponencial estirada a CADA fila ρ(t_w,·) de la superficie.

    La causalidad se hereda: la fila k de la superficie usa sólo datos con
    índice <= t_w[k], y este ajuste es determinista por fila.

    `r2_min` entra en la bandera `valido`, no filtra los arrays: los ajustes
    malos quedan a la vista (con su R²) en vez de desaparecer en silencio.
    """
    n = surface.rho.shape[0]
    cols = {k: np.full(n, np.nan) for k in
            ("log_A", "log_tau_c", "beta", "se_log_tau_c", "se_beta",
             "corr_tau_beta", "r2")}
    valido = np.zeros(n, dtype=bool)
    for k in range(n):
        f = fit_stretched_exp(surface.lags, surface.rho[k])
        for c in cols:
            cols[c][k] = f[c]
        valido[k] = (f["convergio"] and not f["en_borde"]
                     and np.isfinite(f["r2"]) and f["r2"] >= r2_min)
    return TransientTrajectory(t_w=surface.t_w.copy(), valido=valido,
                               window=surface.window, field=surface.field,
                               r2_min=r2_min, **cols)


def synthetic_series_with_acf(n: int, tau_c: float, beta: float,
                              seed: int) -> np.ndarray:
    r"""Proceso gaussiano estacionario con ACF ≈ exp[−(τ/τ_c)^β], var 1.

    Es el CONTROL POSITIVO DE LA MEDICIÓN (regla 11): datos donde la forma
    funcional es verdadera por construcción, para exigir que el ajuste
    recupere (τ_c, β) conocidos. Sin esto, "los parámetros salen ruidosos en
    datos reales" sería ambiguo entre "no hay señal" y "el ajuste está roto".

    Síntesis espectral por embedding circulante: se construye la covarianza
    circular objetivo, se pasa a autovalores por FFT y se filtra ruido blanco
    con su raíz. Los autovalores marginalmente negativos (el KWW no siempre
    es exactamente PSD al discretizarlo) se recortan a 0; el sesgo que
    introduce es pequeño y el test de recuperación lo acota empíricamente.
    """
    from scipy.fft import next_fast_len

    rng = np.random.default_rng(seed)
    m = next_fast_len(4 * int(n))          # margen contra la vuelta circular
    d = np.minimum(np.arange(m), m - np.arange(m)).astype(float)
    with np.errstate(over="ignore", under="ignore"):
        c = np.exp(-(d / float(tau_c)) ** float(beta))
    lam = np.clip(np.fft.rfft(c).real, 0.0, None)
    x = np.fft.irfft(np.fft.rfft(rng.normal(size=m)) * np.sqrt(lam), m)
    x = x[:int(n)]
    sd = x.std()
    return x / sd if sd > 0 else x


def rolling_teff(log_prices_pre: np.ndarray, flow: np.ndarray,
                 window: int = 500, max_lag: int = 30,
                 step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Nivel del cociente FDT por ventana TRAILING: mediana de |T_eff(τ)|.

    `rolling_fdt_violation` resume la ventana con la DISPERSIÓN de T_eff
    (cuánto se aleja de una constante). Esto devuelve además el NIVEL: en el
    escenario de dos temperaturas de los sistemas vítreos, la información de
    régimen estaría en dónde se sitúa T_eff, no sólo en si es constante.

    Returns
    -------
    (t_w, teff_mediana, violacion) — índices de vela y las dos series.
    """
    from .flow import fdt_diagnostic

    p = np.asarray(log_prices_pre, float)
    e = np.asarray(flow, float)
    n = min(len(p), len(e))
    ends = np.arange(window - 1, n, step)
    teff = np.full(len(ends), np.nan)
    viol = np.full(len(ends), np.nan)
    for k, end in enumerate(ends):
        sl = slice(end - window + 1, end + 1)      # TRAILING
        res = fdt_diagnostic(p[sl], e[sl], max_lag=max_lag)
        fin = res.t_eff[np.isfinite(res.t_eff)]
        if len(fin) >= 5:
            teff[k] = float(np.median(np.abs(fin)))
        viol[k] = res.fdt_violation
    return ends, teff, viol
