r"""
El contraste, en un solo sitio: de la característica cruda al veredicto DM.

POR QUÉ ESTO ES UN MÓDULO Y NO CÓDIGO DE CADA EXPERIMENTO
----------------------------------------------------------
El experimento 02 y el barrido de escalas (03) tienen que medir con la MISMA
regla, o sus cifras no son comparables y el nulo empírico de uno no calibra al
otro. Este proyecto ya se tropezó con esa clase de divergencia: la v1 del 01
comparaba contra una línea base distinta de la que creía, y sólo lo destapó el
control negativo.

Aquí viven, y sólo aquí, las tres decisiones metodológicas que se ganaron a
base de errores:

1. **`recal` vs `recal+X` es el contraste que decide.** Un modelo
   `log(rv) ~ a + b·log(σ) + c·X` bate a la σ cruda aunque X sea ruido, porque
   reajusta nivel y pendiente. Comparar contra la σ cruda atribuye a X una
   mejora que es toda de recalibración.
2. **Los dos modelos se ajustan sobre el MISMO soporte.** La máscara exige X
   finita aunque el modelo no use X. Si no, la diferencia de QLIKE mezcla el
   efecto de X con el de haber visto puntos distintos.
3. **La transformación de X usa estadísticos SÓLO de entrenamiento.**
   Winsorizar y estandarizar con momentos de toda la muestra es look-ahead, y
   del silencioso: no lo detecta ningún test de ventana.
"""
from __future__ import annotations

import numpy as np

from .baselines import (ewma_vol, fit_garch11, garch_forecast_path,
                        realized_vol_forward)
from .evaluation import diebold_mariano, diebold_mariano_losses, qlike, rmse

MIN_TRAIN = 200
_EPS_PROB = 1e-6          # recorte de probabilidades para que log-loss sea finita


def mascara_etiquetas_train(n: int, split: int, horizon: int) -> np.ndarray:
    r"""Máscara de entrenamiento con EMBARGO de `horizon` barras.

    Toda etiqueta de este proyecto mira hacia adelante: `rv[t]` usa
    `r[t+1 : t+1+horizon]`. Por tanto, para las últimas `horizon` posiciones
    del tramo de entrenamiento, la etiqueta YA CONTIENE retornos del bloque
    fuera de muestra. Entrenar con ellas —o sacar de ellas un umbral— es
    look-ahead, aunque la característica sea perfectamente causal.

    Es un efecto de frontera y pequeño (30 puntos de ~5259 en el exp. 02), pero
    es exactamente el tipo de fuga silenciosa que la regla 1 del proyecto
    prohíbe, y ningún test de ventana lo detecta: las características no se
    mueven, se mueven las ETIQUETAS.

    Detectado el 2026-08-08 por `test_umbral_del_objetivo_sale_solo_del_train`,
    que cambia el futuro y comprueba que θ no se inmuta. Sin embargo se movía
    de 1.4084 a 1.4430.

    La solución es la estándar en validación de series temporales: purgar las
    `horizon` observaciones cuya etiqueta cruza el corte.
    """
    return np.arange(n) < max(0, int(split) - int(horizon))


def transformar_caracteristica(V: np.ndarray, entrena: np.ndarray,
                               winsor: tuple[float, float] = (0.01, 0.99)):
    r"""log → winsorizar → estandarizar, con estadísticos sólo de entrenamiento.

    El log porque estas características son cocientes positivos de cola larga
    (la violación del FDT tiene denominador R(τ), que pasa cerca de cero: en
    BTC 4h su mediana es ~3 y su máximo ~8·10³). Winsorizar porque una sola
    ventana con R≈0 fijaría el coeficiente de la regresión. Y ambos límites
    salen del tramo de entrenamiento porque cualquier estadístico global
    metería futuro.

    Returns
    -------
    (Z, params) o (None, None) si la característica es degenerada.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.log(np.asarray(V, dtype=float))
    tr = X[entrena & np.isfinite(X)]
    if len(tr) < MIN_TRAIN:
        return None, None
    lo, hi = (float(q) for q in np.quantile(tr, winsor))
    tr_w = np.clip(tr, lo, hi)
    mu, sd = float(np.mean(tr_w)), float(np.std(tr_w))
    if not np.isfinite(sd) or sd <= 0:
        return None, None
    return (np.clip(X, lo, hi) - mu) / sd, {
        "q_lo": lo, "q_hi": hi, "mu": mu, "sd": sd, "n_train": int(len(tr))}


def contraste_caracteristica(retornos: np.ndarray, V: np.ndarray, *,
                             horizon: int, train_frac: float = 0.60,
                             winsor: tuple[float, float] = (0.01, 0.99),
                             lam_ewma: float = 0.94,
                             garch_params: dict | None = None) -> dict:
    """¿Aporta la característica cruda V sobre EWMA y GARCH recalibrados?

    Parameters
    ----------
    retornos : log-retornos, `np.diff(np.log(Close))`.
    V : característica CRUDA (sin transformar) en la MISMA rejilla que
        `retornos`. Si viene de una característica de flujo, tiene que haber
        pasado antes por `flow.align_to_returns`.
    garch_params : si se pasan, no se re-ajusta. Ajustar una sola vez por
        activo evita que un óptimo local distinto introduzca diferencias entre
        celdas que luego se atribuirían a la característica.

    Returns
    -------
    dict con 'modelos' (QLIKE/RMSE), 'dm_contexto' (sólo recalibrar),
    'dm_decisivo' (recal vs recal+X, el que decide) y metadatos.
    """
    r = np.asarray(retornos, dtype=float)
    n = len(r)
    split = int(n * train_frac)
    oos = np.arange(n) >= split
    entrena = ~oos

    X, par_tf = transformar_caracteristica(V, entrena, winsor)
    if X is None:
        return {"error": "caracteristica degenerada", "n": int(n)}

    sd_ewma = ewma_vol(r, lam=lam_ewma)
    par = garch_params if garch_params is not None else fit_garch11(r[:split])
    sd_garch = garch_forecast_path(r, par)
    rv = realized_vol_forward(r, horizon)
    feat_ok = np.isfinite(X)
    # Embargo: las últimas `horizon` etiquetas del train pisan el OOS.
    entrena_lab = mascara_etiquetas_train(n, split, horizon)

    def ajusta(sd_base: np.ndarray, usar_X: bool):
        base_ok = np.isfinite(sd_base) & (sd_base > 0)
        m_tr = entrena_lab & base_ok & feat_ok & np.isfinite(rv) & (rv > 0)
        if m_tr.sum() < MIN_TRAIN:
            return None
        cols = [np.ones(m_tr.sum()), np.log(sd_base[m_tr])]
        if usar_X:
            cols.append(X[m_tr])
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), np.log(rv[m_tr]), rcond=None)
        m_all = base_ok & feat_ok                      # MISMO soporte con y sin X
        lin = coef[0] + coef[1] * np.log(sd_base[m_all])
        if usar_X:
            lin = lin + coef[2] * X[m_all]
        pred = np.full(n, np.nan)
        pred[m_all] = np.exp(lin)
        return {"pred": pred, "coef": coef.tolist(), "n_train": int(m_tr.sum())}

    def corte(x):
        y = np.full(n, np.nan)
        y[oos] = np.asarray(x)[oos]
        return y

    res = {"n": int(n), "split": int(split), "horizon": int(horizon),
           "garch_params": par, "transform": par_tf,
           "modelos": {}, "dm_contexto": {}, "dm_decisivo": {}, "coef_X": {}}

    m_c = oos & feat_ok & np.isfinite(rv) & (rv > 0)
    res["corr_oos_X_logrv"] = (float(np.corrcoef(X[m_c], np.log(rv[m_c]))[0, 1])
                               if m_c.sum() > 50 else float("nan"))

    modelos = {"EWMA": sd_ewma, "GARCH": sd_garch}
    for base, sd_base in (("EWMA", sd_ewma), ("GARCH", sd_garch)):
        recal, conX = ajusta(sd_base, False), ajusta(sd_base, True)
        if recal is not None:
            modelos[f"{base}_recal"] = recal["pred"]
        if conX is not None:
            modelos[f"{base}_recal+X"] = conX["pred"]
            res["coef_X"][base] = conX["coef"][2]

    for k, sd in modelos.items():
        res["modelos"][k] = {"qlike": qlike(corte(rv), corte(sd)),
                             "rmse": rmse(corte(rv), corte(sd))}

    for base in ("EWMA", "GARCH"):
        r_key, x_key = f"{base}_recal", f"{base}_recal+X"
        if r_key in modelos:
            res["dm_contexto"][f"{base} vs {r_key}"] = diebold_mariano(
                corte(rv), corte(modelos[base]), corte(modelos[r_key]), horizon=horizon)
        if r_key in modelos and x_key in modelos:
            res["dm_decisivo"][f"{r_key} vs {x_key}"] = diebold_mariano(
                corte(rv), corte(modelos[r_key]), corte(modelos[x_key]), horizon=horizon)
    return res


# --- objetivo alternativo: TRANSICIONES de régimen (experimento 04) ----------
#
# Los experimentos 01-03 preguntaban por el NIVEL de volatilidad, y siempre a
# través de la media condicional de un modelo log-lineal. Conviene ver por qué
# eso NO agota la pregunta, porque es la unica razon por la que el 04 no es un
# re-etiquetado del 02:
#
#     log(rv) ~ a + b·log(σ) + c·X   ES ALGEBRAICAMENTE IGUAL A
#     log(rv/σ) ~ a + (b−1)·log(σ) + c·X
#
# o sea que "predecir el cambio de nivel" ya se probó, disfrazado. Lo que NO se
# probó es otro FUNCIONAL de la distribución condicional: la probabilidad de
# COLA. Una característica puede dejar la media intacta y aun así ensanchar la
# distribución —hacer más probable un reajuste grande— y eso habría sido
# invisible para los experimentos anteriores.
#
# Y es justo lo que la física pide de este diagnóstico: un sistema lejos del
# equilibrio no dice hacia dónde va, dice que es PROPENSO A REORGANIZARSE.


def objetivo_transicion(retornos: np.ndarray, sd_ref: np.ndarray, *,
                        horizon: int, entrena: np.ndarray,
                        cuantil: float = 0.80) -> tuple[np.ndarray, float]:
    r"""Evento binario: ¿salta la volatilidad por encima de su régimen actual?

        S_t = 1  si  rv_t / σ_t  >  θ

    con `rv_t` la volatilidad realizada de las próximas `horizon` barras y θ el
    cuantil `cuantil` del cociente medido SÓLO en el tramo de entrenamiento.

    Definir θ como un cuantil del train y no como una constante (1.5, 2.0…)
    elimina un parámetro libre: la tasa base queda fijada por construcción para
    cualquier activo e intervalo, así que no hay nada que ajustar mirando
    resultados, y el umbral no mete look-ahead porque sale del train.

    Returns
    -------
    (S, θ) con S en {0,1} y NaN donde el cociente no está definido.
    """
    rv = realized_vol_forward(retornos, horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((sd_ref > 0) & np.isfinite(rv), rv / sd_ref, np.nan)
    # θ sale SÓLO de etiquetas que no cruzan el corte (ver mascara_etiquetas_train)
    split = int(np.count_nonzero(entrena))
    entrena_lab = entrena & mascara_etiquetas_train(len(ratio), split, horizon)
    tr = ratio[entrena_lab & np.isfinite(ratio)]
    if len(tr) < MIN_TRAIN:
        return np.full(len(ratio), np.nan), float("nan")
    theta = float(np.quantile(tr, cuantil))
    S = np.where(np.isfinite(ratio), (ratio > theta).astype(float), np.nan)
    return S, theta


def _ajusta_logistica(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6,
                      max_iter: int = 100, tol: float = 1e-10):
    """Logística por IRLS (Newton). Devuelve los coeficientes o None.

    El ridge minúsculo no es regularización con intención estadística: es lo
    que evita que la matriz de Hesse sea singular si una columna resulta casi
    constante o si hay separación cuasi-perfecta en algún fold. Con 2-3
    columnas su efecto sobre los coeficientes es despreciable.
    """
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1.0 - p), 1e-9, None)
        grad = X.T @ (y - p) - ridge * beta
        H = (X * W[:, None]).T @ X + ridge * np.eye(k)
        try:
            paso = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + paso
        if not np.all(np.isfinite(beta)):
            return None
        if np.max(np.abs(paso)) < tol:
            break
    return beta


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pérdida logarítmica por observación. Es una regla de puntuación PROPIA:
    se minimiza en la probabilidad verdadera, así que no se puede ganar
    exagerando la confianza."""
    q = np.clip(p, _EPS_PROB, 1.0 - _EPS_PROB)
    return -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))


def contraste_transicion(retornos: np.ndarray, V: np.ndarray, *,
                         horizon: int, cuantil: float = 0.80,
                         train_frac: float = 0.60,
                         winsor: tuple[float, float] = (0.01, 0.99),
                         lam_ewma: float = 0.94,
                         garch_params: dict | None = None) -> dict:
    """¿Aporta V sobre la PROBABILIDAD de un salto de régimen de volatilidad?

    Mismo esqueleto que `contraste_caracteristica` —y a propósito: la línea
    base honesta sigue siendo el modelo que ya usa `log(σ)`, los dos modelos se
    ajustan sobre el mismo soporte, y la transformación de V usa estadísticos
    sólo de entrenamiento. Lo único que cambia es el funcional que se predice
    (probabilidad de cola en vez de media condicional) y, con él, la pérdida
    (log-loss en vez de QLIKE).

    El objetivo se define SIEMPRE con σ de EWMA, no con la línea base que se
    esté contrastando: si el evento cambiase de definición entre celdas, las
    celdas no serían comparables entre sí ni el maxT tendría sentido.
    """
    r = np.asarray(retornos, dtype=float)
    n = len(r)
    split = int(n * train_frac)
    oos = np.arange(n) >= split
    entrena = ~oos

    X, par_tf = transformar_caracteristica(V, entrena, winsor)
    if X is None:
        return {"error": "caracteristica degenerada", "n": int(n)}

    sd_ewma = ewma_vol(r, lam=lam_ewma)
    par = garch_params if garch_params is not None else fit_garch11(r[:split])
    sd_garch = garch_forecast_path(r, par)

    S, theta = objetivo_transicion(r, sd_ewma, horizon=horizon,
                                   entrena=entrena, cuantil=cuantil)
    if not np.isfinite(theta):
        return {"error": "objetivo degenerado", "n": int(n)}

    feat_ok = np.isfinite(X)
    entrena_lab = mascara_etiquetas_train(n, split, horizon)
    res = {"n": int(n), "split": int(split), "horizon": int(horizon),
           "cuantil": cuantil, "theta": theta, "garch_params": par,
           "transform": par_tf, "tasa_base_oos": float(np.nanmean(S[oos])),
           "logloss": {}, "dm_decisivo": {}, "coef_X": {}}

    for base, sd_base in (("EWMA", sd_ewma), ("GARCH", sd_garch)):
        base_ok = np.isfinite(sd_base) & (sd_base > 0)
        comun = base_ok & feat_ok & np.isfinite(S)      # MISMO soporte con y sin X
        m_tr, m_ev = entrena_lab & comun, oos & comun
        if m_tr.sum() < MIN_TRAIN or m_ev.sum() < 100:
            continue
        if len(np.unique(S[m_tr])) < 2:
            continue

        cols_tr = [np.ones(m_tr.sum()), np.log(sd_base[m_tr])]
        cols_ev = [np.ones(m_ev.sum()), np.log(sd_base[m_ev])]
        b_base = _ajusta_logistica(np.column_stack(cols_tr), S[m_tr])
        b_aug = _ajusta_logistica(np.column_stack(cols_tr + [X[m_tr]]), S[m_tr])
        if b_base is None or b_aug is None:
            continue

        p_base = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack(cols_ev) @ b_base, -30, 30)))
        p_aug = 1.0 / (1.0 + np.exp(
            -np.clip(np.column_stack(cols_ev + [X[m_ev]]) @ b_aug, -30, 30)))

        la, lb = _log_loss(S[m_ev], p_base), _log_loss(S[m_ev], p_aug)
        res["logloss"][base] = float(np.mean(la))
        res["logloss"][f"{base}+X"] = float(np.mean(lb))
        res["coef_X"][base] = float(b_aug[2])
        res["dm_decisivo"][f"{base} vs {base}+X"] = diebold_mariano_losses(
            la, lb, horizon=horizon)
    return res


def westfall_young(dm_obs: dict, draws: list[dict]) -> dict:
    r"""Corrección de multiplicidad maxT: colapsa un barrido a UNA decisión.

    Barrer celdas es la forma más fácil de engañarse: con 144 comparaciones,
    encontrar una con p<0.05 no es un hallazgo, es aritmética. Medido en el
    exp. 03, la mejor celda tenía p nominal 0.0002 y aun así el máximo del nulo
    tenía media MAYOR que el máximo observado.

    Westfall-Young responde a la pregunta correcta —¿es la mejor de N celdas
    mejor de lo que sale por azar cuando se prueban N celdas?— así:

        p_global = (1 + #{T_i >= T_obs}) / (K + 1),   T_i = max_celdas DM_i

    Cada draw tiene que haber recorrido TODAS las celdas con la misma
    realización del nulo; así la dependencia entre celdas (comparten activo,
    datos y característica) se contabiliza sola. Bonferroni sería conservador
    y además falso, porque supone independencia.

    Parameters
    ----------
    dm_obs : {clave_celda: estadístico observado}
    draws  : lista de K diccionarios con las mismas claves

    Returns
    -------
    dict con 'p_global', 'p_ajustado' por celda, 'mejor_celda' y el nulo.
    """
    claves = sorted(dm_obs)
    obs = np.array([dm_obs[k] for k in claves], dtype=float)
    if not np.any(np.isfinite(obs)):
        return {"error": "ningun estadistico observado finito"}

    maxT = np.array([np.nanmax([d.get(k, np.nan) for k in claves]) for d in draws],
                    dtype=float)
    maxT = maxT[np.isfinite(maxT)]
    if len(maxT) == 0:
        return {"error": "nulo vacio"}

    t_obs = float(np.nanmax(obs))
    return {
        "mejor_celda": claves[int(np.nanargmax(obs))],
        "dm_mejor": t_obs,
        "p_global": float((1 + np.sum(maxT >= t_obs)) / (len(maxT) + 1)),
        "p_ajustado": {k: float((1 + np.sum(maxT >= dm_obs[k])) / (len(maxT) + 1))
                       for k in claves if np.isfinite(dm_obs[k])},
        "maxT_nulo": maxT.tolist(),
        "maxT_media": float(maxT.mean()),
        "maxT_sd": float(maxT.std()),
        "maxT_q95": float(np.quantile(maxT, 0.95)),
        "n_celdas": len(claves),
        "K": len(maxT),
    }
