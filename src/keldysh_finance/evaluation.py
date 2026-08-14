r"""
Evaluación fuera de muestra de predicciones de volatilidad.

MÉTRICA
-------
QLIKE en vez de RMSE como métrica principal:

    QLIKE = mean( rv²/σ² − log(rv²/σ²) − 1 )

Razones (Patton 2011, "Volatility forecast comparison using imperfect
volatility proxies"): QLIKE es robusta a que el objetivo sea un proxy ruidoso
de la volatilidad latente —que siempre lo es—, y penaliza asimétricamente la
INFRAestimación, que es el error que arruina cuentas. RMSE sobre σ premia
predicciones sistemáticamente bajas cuando el proxy tiene ruido.
Se reporta RMSE también, pero la decisión se toma con QLIKE.

CONTRASTE
---------
Diebold-Mariano con corrección de autocorrelación (Newey-West): la diferencia
de pérdidas entre dos modelos está autocorrelacionada cuando el horizonte
solapa, y un t-test ingenuo sobreestima la significancia.
"""
from __future__ import annotations

import numpy as np


def qlike(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Pérdida QLIKE media (menor es mejor)."""
    rv = np.asarray(realized, dtype=float)
    sd = np.asarray(predicted, dtype=float)
    m = np.isfinite(rv) & np.isfinite(sd) & (rv > 0) & (sd > 0)
    if m.sum() < 10:
        return float("nan")
    ratio = (rv[m] ** 2) / (sd[m] ** 2)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse(realized: np.ndarray, predicted: np.ndarray) -> float:
    rv, sd = np.asarray(realized, float), np.asarray(predicted, float)
    m = np.isfinite(rv) & np.isfinite(sd)
    if m.sum() < 10:
        return float("nan")
    return float(np.sqrt(np.mean((rv[m] - sd[m]) ** 2)))


def _qlike_losses(realized, predicted) -> np.ndarray:
    rv, sd = np.asarray(realized, float), np.asarray(predicted, float)
    m = np.isfinite(rv) & np.isfinite(sd) & (rv > 0) & (sd > 0)
    ratio = (rv[m] ** 2) / (sd[m] ** 2)
    out = np.full(len(rv), np.nan)
    out[m] = ratio - np.log(ratio) - 1.0
    return out


def diebold_mariano(realized: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    horizon: int = 1) -> dict:
    """Contrasta H0: los modelos A y B predicen igual de bien (pérdida QLIKE).

    Estadístico DM con varianza de largo plazo Newey-West y ancho de banda
    `horizon-1` (la regla estándar cuando los horizontes se solapan).

    Returns
    -------
    dict con 'dm_stat', 'p_value', 'mean_diff' (>0 ⇒ A pierde más ⇒ B mejor).
    """
    return diebold_mariano_losses(_qlike_losses(realized, pred_a),
                                  _qlike_losses(realized, pred_b),
                                  horizon=horizon)


def diebold_mariano_losses(loss_a: np.ndarray, loss_b: np.ndarray,
                           horizon: int = 1) -> dict:
    """DM a partir de series de pérdida YA calculadas.

    Existe para que un experimento con otra función de pérdida —el 04 usa
    log-loss sobre un evento binario, no QLIKE sobre un nivel— pase por la
    MISMA maquinaria de Newey-West que los anteriores. Reimplementarla para
    cada pérdida sería la vía rápida a que dos experimentos del repo dejen de
    ser comparables sin que nadie lo note.

    ⚠ El `p_value` que devuelve es nominal y, con horizontes que solapan,
    optimista: medido en el exp. 02b, este contraste rechaza 16-18% en ETH
    donde declara 5%. Para decidir hay que usar el nulo por permutación.
    """
    la = np.asarray(loss_a, dtype=float)
    lb = np.asarray(loss_b, dtype=float)
    m = np.isfinite(la) & np.isfinite(lb)
    d = la[m] - lb[m]
    n = len(d)
    if n < 30:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": float("nan"), "n": int(n)}

    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / n)
    lrv = gamma0
    for lag in range(1, max(1, horizon)):
        if lag >= n:
            break
        cov = float(np.dot(dc[:-lag], dc[lag:]) / n)
        w = 1.0 - lag / max(horizon, 1)          # kernel de Bartlett
        lrv += 2.0 * w * cov
    if lrv <= 0:
        lrv = gamma0 if gamma0 > 0 else 1e-12

    dm = dbar / np.sqrt(lrv / n)
    # normal estándar de dos colas, sin dependencia de scipy
    from math import erfc, sqrt
    p = float(erfc(abs(dm) / sqrt(2.0)))
    return {"dm_stat": float(dm), "p_value": p, "mean_diff": dbar, "n": int(n)}


def walk_forward_split(n: int, train: int, test: int):
    """Genera (idx_train, idx_test) sin solape, avanzando en bloques.

    Devuelve rangos de índices; el llamador decide qué hacer con ellos. El
    entrenamiento SIEMPRE precede al test — no hay barajado, que en series
    temporales destruiría el sentido de la evaluación.
    """
    start = 0
    while start + train + test <= n:
        yield (np.arange(start, start + train),
               np.arange(start + train, start + train + test))
        start += test
