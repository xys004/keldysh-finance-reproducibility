r"""
EXPERIMENTO 09 — el sector de SEGUNDO ORDEN del modelo MSRJD, consolidado.

QUÉ ES (y qué no es — honestidad primero)
-----------------------------------------
Consolidación de la vía A del programa de modelado: comprobar que TODO el
sector gaussiano/de segundo orden de la medición es descriptible por el
modelo mínimo MSRJD (flujo con kernel de memoria + precio integrador de
innovaciones + sesgo). Dos piezas:

1. LA FORMA DE T_eff — consistencia compuesta, NO predicción independiente.
   T_eff = −[dC/dτ]/R se CONSTRUYE con C y R, así que su pendiente log-log
   b_T tiene que valer (γ−1) − b_R si y sólo si C es una ley de potencias
   limpia y R una ley de potencias (plana ⇒ b_R≈0) sobre el rango ajustado.
   El contenido del test es la CALIDAD DE LA DESCRIPCIÓN de un solo
   exponente, no una sobreidentificación — eso queda dicho aquí para que
   nadie lo venda como lo que no es. La forma cerrada del modelo:

       T_eff(τ) = (c·|γ|/R₀) · τ^(γ−1)        (R plana, C = c·τ^γ)

   El control positivo del estimador está en tests/test_teff_forma.py: un
   modelo exactamente resoluble (AR(1) + integrador de innovaciones, que da
   R PLANA por el mecanismo del modelo: sólo la innovación instantánea
   correlaciona con el flujo) donde la cadena recupera ln φ.

2. EL TRANSFER GAUSSIANO EN FASE — este SÍ es un cierre no trivial.
   La afinidad por fase A1(φ) se ajusta con la función de simetría
   (log-ratio de colas ±), y la predicción gaussiana A_g(φ) = 2μ(φ)/σ²(φ)
   sale de los MOMENTOS. Son estadísticos distintos de la misma muestra:
   coinciden si y sólo si el flujo condicionado a la fase es
   gaussiano-compatible en su sector impar. Si |z| < 2 en los 32 bins
   (4 activos × 8 fases, 1h), la modulación Floquet de la afinidad queda
   ENTERAMENTE explicada por la modulación de (μ, σ²) — el drive del modelo
   entra por los dos primeros momentos y nada más.

CRITERIO EX-ANTE
----------------
El sector de segundo orden se declara CERRADO si:
  (i)  transfer gaussiano: |z| < 2 en ≥ 30 de los 32 bins de fase, y
  (ii) forma compuesta de T_eff: |z| < 2 en ≥ 6 de las 8 series
       (z con los errores de b_T, γ y b_R en cuadratura).

Ejecutar:  py experiments/exp09_msrjd_orden2.py     (LOCAL, ~10 s)
La nota del modelo vive en notes/modelo_msrjd.md; sus cifras salen de ESTE
log y de los de 06/07/08, nunca de la nota.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance.counting import _ols_pendiente, fit_affinity, symmetry_function
from keldysh_finance.flow import (fdt_diagnostic, fetch_klines_with_flow,
                                  flow_autocorrelation, order_flow_imbalance)

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["1h", "4h"]
CONF = dict(years=4.0, gamma_lags=60, teff_tau=(2, 50), n_bins_fase=8,
            criterio="(i) |z|<2 en >=30/32 bins de fase; "
                     "(ii) |z|<2 en >=6/8 series para b_T vs (gamma-1)-b_R")


def datos(symbol: str, interval: str) -> dict:
    df = fetch_klines_with_flow(symbol, interval=interval,
                                years=CONF["years"]).dropna()
    return {"eps": order_flow_imbalance(df, normalize="volume"),
            "log_open": np.log(df["Open"].to_numpy(float)),
            "horas": df.index.hour.to_numpy()}


def forma_teff(nombre: str, interval: str, d: dict) -> dict:
    """γ de C_ε, b_R de R, b_T de |T_eff| y la consistencia compuesta."""
    C = flow_autocorrelation(d["eps"], max_lag=CONF["gamma_lags"])
    tau_c = np.arange(1, CONF["gamma_lags"] + 1, dtype=float)
    m_c = np.isfinite(C) & (C > 0)
    gamma, se_g = _ols_pendiente(np.log(tau_c[m_c]), np.log(C[m_c]))

    res = fdt_diagnostic(d["log_open"], d["eps"], max_lag=CONF["gamma_lags"])
    tau = res.lags.astype(float)
    lo, hi = CONF["teff_tau"]
    m = (tau >= lo) & (tau <= hi)

    r = res.response
    m_r = m & np.isfinite(r) & (r > 0)
    b_r, se_r = _ols_pendiente(np.log(tau[m_r]), np.log(r[m_r]))

    te = np.abs(res.t_eff)
    m_t = m & np.isfinite(te) & (te > 0)
    b_t, se_t = _ols_pendiente(np.log(tau[m_t]), np.log(te[m_t]))

    pred = (gamma - 1.0) - b_r
    se_z = float(np.sqrt(se_t ** 2 + se_g ** 2 + se_r ** 2))
    return {"clave": f"{nombre}|{interval}",
            "gamma": gamma, "se_gamma": se_g,
            "b_R": b_r, "se_b_R": se_r,
            "b_T": b_t, "se_b_T": se_t,
            "prediccion_compuesta": pred,
            "z": (b_t - pred) / se_z if se_z > 0 else float("nan"),
            "n_tau_teff": int(m_t.sum()),
            "teff_curva": {"tau": tau[m_t].tolist(),
                           "teff": te[m_t].tolist()},
            "R_curva": {"tau": tau[m_r].tolist(), "R": r[m_r].tolist()}}


def dos_regimenes(d: dict) -> dict:
    r"""Pendientes locales de C_ε en [2,10] y [10,50], crudo y
    DESESTACIONALIZADO (media por hora restada).

    Motivación (hallazgo de este experimento): la forma compuesta de T_eff
    falla 0/8 porque C_ε NO es una ley de potencias única — tiene un régimen
    rápido (τ≲10, pendiente ~−0.9) y uno lento (τ≳10, ~−0.2). La hipótesis
    obvia era que el régimen lento fuese el suelo determinista del drive
    Floquet (⟨μ(φ)μ(φ+τ)⟩ es periódico y no decae): FALSADA — quitar la
    media por fase apenas mueve las pendientes. El régimen lento es memoria
    estocástica genuina multi-día (splitting de metaórdenes), y el kernel
    del modelo MSRJD necesita DOS componentes.
    """
    e, h = d["eps"], d["horas"]
    e_d = e.copy()
    for hh in range(24):
        m = h == hh
        if m.any():
            e_d[m] -= e[m].mean()
    tau = np.arange(1, CONF["gamma_lags"] + 1, dtype=float)
    out = {}
    for etiqueta, serie in (("crudo", e), ("desestacionalizado", e_d)):
        C = flow_autocorrelation(serie, max_lag=CONF["gamma_lags"])
        tramo = {}
        for lo, hi in ((2, 10), (10, 50)):
            m = (tau >= lo) & (tau <= hi) & np.isfinite(C) & (C > 0)
            if m.sum() > 4:
                b, se = _ols_pendiente(np.log(tau[m]), np.log(C[m]))
                tramo[f"[{lo},{hi}]"] = {"pendiente": b, "se": se}
        out[etiqueta] = tramo
    return out


def transfer_gaussiano(nombre: str, d: dict) -> list[dict]:
    """A1(φ) del ajuste de simetría vs A_g(φ)=2μ/σ² de los momentos (1h)."""
    e, h = d["eps"], d["horas"]
    ancho = 24 // CONF["n_bins_fase"]
    filas = []
    for b in range(CONF["n_bins_fase"]):
        m = ((h // ancho) == b) & np.isfinite(e)
        eb = e[m]
        fit = fit_affinity(*symmetry_function(eb))
        var = float(np.var(eb, ddof=1))
        a_g = 2.0 * float(np.mean(eb)) / var if var > 0 else float("nan")
        z = ((fit["A"] - a_g) / fit["se_A"]
             if np.isfinite(fit["A"]) and fit["se_A"] else float("nan"))
        filas.append({"bin": b, "horas": f"{b*ancho:02d}-{(b+1)*ancho:02d}",
                      "n": int(m.sum()), "A1": fit["A"], "se_A1": fit["se_A"],
                      "A_gauss": a_g, "z": z})
    return filas


def main() -> None:
    print("\n  EXPERIMENTO 09 — sector de segundo orden del modelo MSRJD")
    print(f"  {CONF['criterio']}\n", flush=True)

    t0 = time.time()
    formas, fases, regimenes = [], {}, {}
    for nombre, symbol in ACTIVOS:
        for interval in INTERVALOS:
            d = datos(symbol, interval)
            formas.append(forma_teff(nombre, interval, d))
            if interval == "1h":
                fases[nombre] = transfer_gaussiano(nombre, d)
                regimenes[nombre] = dos_regimenes(d)
    print(f"  computo: {time.time()-t0:.0f} s\n")

    print(f"  {'serie':<9}{'gamma':>8}{'b_R':>8}{'b_T':>9}{'(g-1)-b_R':>10}{'z':>7}")
    ok_forma = 0
    for f in formas:
        marca = ""
        if np.isfinite(f["z"]) and abs(f["z"]) < 2:
            ok_forma += 1
        else:
            marca = "  <<<"
        print(f"  {f['clave']:<9}{f['gamma']:>8.3f}{f['b_R']:>8.3f}"
              f"{f['b_T']:>9.3f}{f['prediccion_compuesta']:>10.3f}"
              f"{f['z']:>7.2f}{marca}")

    print(f"\n  transfer gaussiano en fase (1h): |z| por activo")
    ok_fase, total_bins = 0, 0
    for nombre, filas in fases.items():
        zs = [f["z"] for f in filas if np.isfinite(f["z"])]
        ok_fase += sum(1 for z in zs if abs(z) < 2)
        total_bins += len(zs)
        print(f"    {nombre}: max|z| = {max(abs(z) for z in zs):.2f}  "
              f"({sum(1 for z in zs if abs(z) < 2)}/{len(zs)} bins con |z|<2)")

    cerrado = (ok_fase >= 30) and (ok_forma >= 6)
    veredicto = (f"SECTOR DE SEGUNDO ORDEN CERRADO: transfer gaussiano "
                 f"{ok_fase}/{total_bins} bins, forma T_eff {ok_forma}/8 series"
                 if cerrado else
                 f"sector NO cerrado: transfer {ok_fase}/{total_bins}, "
                 f"forma {ok_forma}/8 — mirar dónde falla antes de modelar encima")
    print(f"\n  → {veredicto}")

    print(f"\n  dos regimenes de C_eps (1h; crudo | desestacionalizado):")
    for nombre, reg in regimenes.items():
        c, dsz = reg["crudo"], reg["desestacionalizado"]
        print(f"    {nombre}: [2,10] {c['[2,10]']['pendiente']:+.2f} | "
              f"{dsz['[2,10]']['pendiente']:+.2f}    "
              f"[10,50] {c['[10,50]']['pendiente']:+.2f} | "
              f"{dsz['[10,50]']['pendiente']:+.2f}")

    salida = {"conf": {**CONF, "activos": [a for a, _ in ACTIVOS],
                       "intervalos": INTERVALOS},
              "forma_teff": formas, "transfer_fase": fases,
              "dos_regimenes": regimenes,
              "veredicto": veredicto}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp09_msrjd_orden2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
