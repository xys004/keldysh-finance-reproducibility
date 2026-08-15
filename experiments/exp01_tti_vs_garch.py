r"""
EXPERIMENTO 01 — ¿La ruptura de TTI mejora la predicción de volatilidad?

HIPÓTESIS (declarada antes de mirar resultados)
-----------------------------------------------
H1: la característica de no-equilibrio D(t_w) —distancia del perfil de
    correlación local a su referencia causal— aporta información sobre la
    volatilidad realizada futura que EWMA(0.94) y GARCH(1,1) no capturan.

H0: no aporta nada. La diferencia de pérdida QLIKE contra las líneas base no
    es distinguible de cero (Diebold-Mariano, p >= 0.05).

CRITERIO DE DECISIÓN (fijado ex-ante, no renegociable después)
--------------------------------------------------------------
Se declara ÉXITO sólo si el modelo aumentado bate a AMBAS líneas base con
p < 0.05 en el test DM sobre QLIKE, y en AMBOS activos (BTC y ETH).
Ganar a una sola línea base, o en un solo activo, se reporta como NO
CONCLUYENTE — con dos activos y dos baselines hay 4 comparaciones y el máximo
de 4 tests sube el falso positivo a ~19% sin corregir.

CONTROL NEGATIVO
----------------
El mismo pipeline sobre el campo 'raw' (retornos, no volatilidad). Los
retornos son casi incorrelados, así que ahí no debería haber estructura. Si
el método "encuentra" mejora también en el control, está ajustando ruido y el
resultado en el campo real no es interpretable.

MODELO AUMENTADO
----------------
Deliberadamente simple: una regresión lineal de log(rv) sobre log(σ_base) y
D(t_w), ajustada SÓLO en el tramo de entrenamiento de cada fold. Un modelo
flexible (bosques, redes) podría extraer más señal, pero también podría
fabricarla; con 2 características y ajuste lineal, si aparece mejora es
atribuible a la característica y no a la capacidad del modelo.

Ejecutar:  py experiments/exp01_tti_vs_garch.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance import (two_time_surface, tti_breaking, correlation_time,
                             ewma_vol, fit_garch11, garch_forecast_path,
                             realized_vol_forward, qlike, rmse, diebold_mariano)
from keldysh_finance.flow import fetch_klines_with_flow

# Este proyecto es AUTÓNOMO: descarga sus propios datos de la API pública de
# Binance y los cachea en output/cache/. No lee del proyecto de trading — eran
# dos cosas distintas compartiendo carpeta por accidente, y esa dependencia se
# rompió el 2026-08-08 al separarlos en repos independientes.
OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]

CONF = dict(
    window=250,        # barras de 4h ≈ 42 días por ventana
    max_lag=40,
    step=1,
    horizon=30,        # predecir la vol realizada de las próximas 30 barras (~5 días)
    train_frac=0.60,   # primer 60% para ajustar; el resto es OOS puro
    min_windows=30,
)


def cargar(symbol: str) -> np.ndarray:
    """Log-retornos desde la API pública de Binance (cacheado localmente)."""
    df = fetch_klines_with_flow(symbol, interval="4h", years=4.0).dropna()
    c = df["Close"].to_numpy(float)
    return np.diff(np.log(c))          # log-retornos


def alinear(feature_vals, t_w, n):
    """Lleva una característica definida en los t_w a la rejilla completa.

    Se rellena hacia adelante: en un instante t, el último valor DISPONIBLE es
    el de la última ventana terminada en <= t. Nunca interpola con futuro.
    """
    out = np.full(n, np.nan)
    out[t_w] = feature_vals
    return pd.Series(out).ffill().to_numpy()


def evaluar_activo(nombre: str, symbol: str, field: str = "abs") -> dict:
    r = cargar(symbol)
    n = len(r)
    print(f"\n  {'='*70}\n  {nombre}  ({n} barras de 4h, campo='{field}')\n  {'='*70}")

    # --- características de no-equilibrio (causales) ---
    surf = two_time_surface(r, window=CONF["window"], max_lag=CONF["max_lag"],
                            step=CONF["step"], field=field)
    D = tti_breaking(surf, reference="expanding", min_windows=CONF["min_windows"])
    TC = correlation_time(surf, method="integral")
    D_full = alinear(D, surf.t_w, n)
    TC_full = alinear(TC, surf.t_w, n)

    # --- líneas base ---
    sd_ewma = ewma_vol(r, lam=0.94)
    split = int(n * CONF["train_frac"])
    par = fit_garch11(r[:split])
    sd_garch = garch_forecast_path(r, par)
    print(f"  GARCH(1,1) ajustado en train: alpha={par['alpha']:.3f} "
          f"beta={par['beta']:.3f} persistencia={par['persistence']:.3f}")

    # --- objetivo ---
    rv = realized_vol_forward(r, CONF["horizon"])

    # --- modelos ajustados ---
    # CRÍTICO: hay que separar dos efectos que la primera versión de este
    # experimento confundía. Un modelo `log(rv) ~ a + b·log(σ) + c·D` gana a
    # la σ cruda por DOS motivos distintos: (i) recalibración —reajusta nivel
    # y pendiente, o sea corrige el sesgo de la línea base— y (ii) la
    # información que aporte D. Comparar "σ cruda" contra "σ recalibrada + D"
    # atribuye a D una mejora que puede ser toda de (i).
    # El control negativo lo destapó: con field='raw' (sin estructura) la
    # "mejora" era MAYOR que con el campo real. Por eso se añade el modelo
    # RECALIBRADO SIN D, que es la línea base honesta contra la que medir D.
    idx = np.arange(n)
    oos = idx >= split
    aug = {}

    def _ajusta(sd_base, usar_D: bool):
        m_tr = (~oos) & np.isfinite(rv) & np.isfinite(sd_base) & (rv > 0) & (sd_base > 0)
        if usar_D:
            m_tr &= np.isfinite(D_full)
        if m_tr.sum() < 200:
            return None
        cols = [np.ones(m_tr.sum()), np.log(sd_base[m_tr])]
        if usar_D:
            cols.append(D_full[m_tr])
        X = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(X, np.log(rv[m_tr]), rcond=None)
        m_all = np.isfinite(sd_base) & (sd_base > 0)
        if usar_D:
            m_all &= np.isfinite(D_full)
        pred = np.full(n, np.nan)
        lin = coef[0] + coef[1] * np.log(sd_base[m_all])
        if usar_D:
            lin = lin + coef[2] * D_full[m_all]
        pred[m_all] = np.exp(lin)
        return {"pred": pred, "coef": coef.tolist()}

    for base_nombre, sd_base in (("EWMA", sd_ewma), ("GARCH", sd_garch)):
        recal = _ajusta(sd_base, usar_D=False)
        conD = _ajusta(sd_base, usar_D=True)
        aug[base_nombre] = {"recal": recal, "conD": conD}
        if conD is not None:
            print(f"  {base_nombre}: coef_D = {conD['coef'][2]:+.4f}")

    # --- evaluación SOLO fuera de muestra ---
    def corte(x):
        y = np.full(n, np.nan); y[oos] = np.asarray(x)[oos]; return y

    res = {"activo": nombre, "field": field, "n": int(n), "split": int(split),
           "garch_params": par, "modelos": {}, "dm": {}}

    print(f"\n  {'modelo':<22}{'QLIKE':>10}{'RMSE':>12}")
    modelos = {"EWMA": sd_ewma, "GARCH": sd_garch}
    for k, v in aug.items():
        if v["recal"] is not None:
            modelos[f"{k}_recal"] = v["recal"]["pred"]
        if v["conD"] is not None:
            modelos[f"{k}_recal+D"] = v["conD"]["pred"]
    for k, sd in modelos.items():
        q, rm = qlike(corte(rv), corte(sd)), rmse(corte(rv), corte(sd))
        res["modelos"][k] = {"qlike": q, "rmse": rm}
        print(f"  {k:<22}{q:>10.4f}{rm:>12.6f}")

    # --- contrastes ---
    # El que decide es SIEMPRE recal vs recal+D: aísla la aportación de D.
    # recal vs cruda se reporta sólo para cuantificar cuánto de la mejora
    # aparente venía de la simple corrección de sesgo.
    print(f"\n  Diebold-Mariano (QLIKE, H0: igual capacidad predictiva)")
    for base in ("EWMA", "GARCH"):
        r_key, d_key = f"{base}_recal", f"{base}_recal+D"
        if r_key in modelos:
            dm = diebold_mariano(corte(rv), corte(modelos[base]), corte(modelos[r_key]),
                                 horizon=CONF["horizon"])
            res["dm"][f"{base} vs {r_key} (solo recalibracion)"] = dm
            print(f"    [contexto] {base:>5} vs {r_key:<14} DM={dm['dm_stat']:+7.3f} "
                  f"p={dm['p_value']:.4f}   ← mejora por recalibrar, sin D")
        if r_key in modelos and d_key in modelos:
            dm = diebold_mariano(corte(rv), corte(modelos[r_key]), corte(modelos[d_key]),
                                 horizon=CONF["horizon"])
            res["dm_decisivo"] = res.get("dm_decisivo", {})
            res["dm_decisivo"][f"{r_key} vs {d_key}"] = dm
            veredicto = ("D APORTA" if dm["p_value"] < 0.05 and dm["mean_diff"] > 0
                         else "D PERJUDICA" if dm["p_value"] < 0.05
                         else "D no aporta")
            print(f"    [DECIDE ] {r_key:>11} vs {d_key:<14} DM={dm['dm_stat']:+7.3f} "
                  f"p={dm['p_value']:.4f}  n={dm['n']:<5} → {veredicto}")
    return res


def main() -> None:
    print("\n  EXPERIMENTO 01 — ruptura de TTI como predictor de volatilidad")
    print(f"  Config: {CONF}")
    print("\n  CRITERIO EX-ANTE: éxito sólo si bate a AMBAS líneas base con")
    print("  p<0.05 en AMBOS activos. Cualquier otra cosa = no concluyente.")

    resultados = {"conf": CONF, "real": [], "control": []}
    for nombre, sym in ACTIVOS:
        resultados["real"].append(evaluar_activo(nombre, sym, field="abs"))

    print(f"\n\n  {'#'*70}\n  CONTROL NEGATIVO (campo 'raw': retornos, casi incorrelados)")
    print(f"  Si aquí también 'mejora', el método ajusta ruido.\n  {'#'*70}")
    for nombre, sym in ACTIVOS:
        resultados["control"].append(evaluar_activo(nombre, sym, field="raw"))

    # --- veredicto según el criterio ex-ante ---
    def cuenta_exitos(bloque):
        """Sólo cuentan los contrastes DECISIVOS (recal vs recal+D)."""
        ok = 0
        for r in bloque:
            for _, dm in r.get("dm_decisivo", {}).items():
                if np.isfinite(dm.get("p_value", np.nan)) and \
                   dm["p_value"] < 0.05 and dm["mean_diff"] > 0:
                    ok += 1
        return ok

    ex_real, ex_ctrl = cuenta_exitos(resultados["real"]), cuenta_exitos(resultados["control"])
    total = 2 * len(ACTIVOS)
    print(f"\n\n  {'='*70}\n  VEREDICTO\n  {'='*70}")
    print(f"  Campo real ('abs')   : {ex_real}/{total} comparaciones con mejora significativa")
    print(f"  Control neg. ('raw') : {ex_ctrl}/{total}")
    if ex_real == total and ex_ctrl == 0:
        veredicto = "H1 SOBREVIVE — merece continuar"
    elif ex_ctrl > 0:
        veredicto = "SOSPECHOSO — el control tambien 'mejora': el metodo ajusta ruido"
    elif ex_real == 0:
        veredicto = "H0 NO SE RECHAZA — la caracteristica no aporta"
    else:
        veredicto = "NO CONCLUYENTE — mejora parcial; no basta con el criterio ex-ante"
    print(f"  → {veredicto}")
    resultados["veredicto"] = veredicto

    OUT.mkdir(exist_ok=True)
    path = OUT / "exp01_tti_vs_garch.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
