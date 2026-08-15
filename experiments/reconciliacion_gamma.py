# -*- coding: utf-8 -*-
"""Reconciliacion del exponente de memoria gamma: de donde sale -0.586/-0.559.

MOTIVO. El README y AGENTS.md citan `gamma = -0.586 (BTC) / -0.559 (ETH)` como
la validacion frente a Lillo-Farmer, pero ninguna cifra igual aparece en los
logs depositados: `exp06`/`exp09` dan -0.380 y -0.474 con 60 desfases. Por la
regla del repo (ninguna cifra sin log), o se reproduce y se deposita, o se
corrige la bitacora. Este script decide cual de las dos cosas toca.

METODO. Se barre la matriz de variantes plausibles de la medicion sobre los
MISMOS datos del cache y se busca cual reproduce la cifra citada:

  * normalizacion del flujo:  'volume' (la de exp06)  vs  'none' (crudo)
  * ventana de datos:         4.0 anios (la de exp06) vs 2.0 anios (la inicial)
  * desfases del ajuste:      [1,60] (la de exp06)    vs [1,30]

El estimador es el mismo en todos los casos -OLS log-log de C_eps(tau) sobre
los desfases con C_eps > 0- para que la unica diferencia sea la variante.

    py experiments/reconciliacion_gamma.py

Deposita `output/reconciliacion_gamma.json`. Corre en LOCAL (segundos).
"""
import sys, os, json, itertools

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from keldysh_finance.flow import order_flow_imbalance, flow_autocorrelation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "output", "cache")
OUT = os.path.join(REPO, "output", "reconciliacion_gamma.json")

CITADO = {"BTC": -0.586, "ETH": -0.559}   # lo que dicen README y AGENTS.md
TOL = 0.02                                # margen para declarar reproduccion

SIMBOLOS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
NORMS = ["volume", "none"]
ANIOS = [4.0, 2.0]
LAGS = [60, 30]


def gamma_ols(eps: np.ndarray, max_lag: int) -> dict:
    """Exponente por OLS log-log de C_eps(tau) sobre tau=1..max_lag, C>0.

    Identico al estimador de exp06; se devuelve ademas el error tipico para
    poder decir si dos variantes son distinguibles.
    """
    C = flow_autocorrelation(eps, max_lag=max_lag)
    tau = np.arange(1, max_lag + 1)
    m = np.isfinite(C) & (C > 0)
    if m.sum() < 10:
        return {"gamma": float("nan"), "se": float("nan"), "n_puntos": int(m.sum())}
    x, y = np.log(tau[m]), np.log(C[m])
    xc, yc = x - x.mean(), y - y.mean()
    b = float(np.dot(xc, yc) / np.dot(xc, xc))
    resid = yc - b * xc
    dof = max(m.sum() - 2, 1)
    se = float(np.sqrt((resid @ resid) / dof / (xc @ xc)))
    return {"gamma": b, "se": se, "n_puntos": int(m.sum())}


def main():
    filas, faltan = [], []
    for activo, sym in SIMBOLOS.items():
        for anios in ANIOS:
            p = os.path.join(CACHE, f"flow_{sym}_1h_{anios}y.csv")
            if not os.path.exists(p):
                faltan.append(os.path.basename(p))
                continue
            df = pd.read_csv(p).dropna()
            for norm, max_lag in itertools.product(NORMS, LAGS):
                eps = order_flow_imbalance(df, normalize=norm)
                r = gamma_ols(np.asarray(eps, float), max_lag)
                objetivo = CITADO[activo]
                filas.append({
                    "activo": activo, "anios": anios, "normalizacion": norm,
                    "max_lag": max_lag, "n_velas": int(len(df)),
                    **r,
                    "desviacion_vs_citado": r["gamma"] - objetivo,
                    "reproduce": bool(abs(r["gamma"] - objetivo) <= TOL),
                })

    print(f"{'activo':7s}{'anios':>6s}{'norm':>9s}{'lags':>6s}{'n':>7s}"
          f"{'gamma':>9s}{'se':>7s}{'vs citado':>11s}  repro")
    print("-" * 68)
    for f in filas:
        print(f"{f['activo']:7s}{f['anios']:6.1f}{f['normalizacion']:>9s}"
              f"{f['max_lag']:6d}{f['n_velas']:7d}{f['gamma']:+9.3f}"
              f"{f['se']:7.3f}{f['desviacion_vs_citado']:+11.3f}"
              f"  {'SI' if f['reproduce'] else '.'}")

    ok = [f for f in filas if f["reproduce"]]
    if ok:
        veredicto = ("REPRODUCIDA: la cifra citada corresponde a la variante "
                     + "; ".join(sorted({f"{f['normalizacion']}/{f['anios']}y/"
                                         f"{f['max_lag']}lags" for f in ok})))
    else:
        veredicto = ("NO REPRODUCIDA con ninguna variante del barrido: la cifra "
                     "citada en README/AGENTS.md no tiene log y debe corregirse "
                     "por la que si lo tiene")

    print("\n" + veredicto)
    if faltan:
        print("Ficheros de cache ausentes:", ", ".join(faltan))

    payload = {
        "motivo": "localizar el origen de gamma=-0.586/-0.559 citado sin log",
        "citado": CITADO, "tolerancia": TOL,
        "variantes": {"normalizacion": NORMS, "anios": ANIOS, "max_lag": LAGS},
        "estimador": "OLS log-log de C_eps(tau) sobre tau=1..max_lag con C>0",
        "filas": filas, "cache_ausente": faltan, "veredicto": veredicto,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"\nlog -> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
