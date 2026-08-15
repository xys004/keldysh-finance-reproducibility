r"""
EXPERIMENTO 04 — ¿anticipa el diagnóstico de no-equilibrio un SALTO de régimen?

POR QUÉ ESTO NO ES EL EXPERIMENTO 02 CON OTRO NOMBRE
-----------------------------------------------------
Es la primera pregunta que hay que contestar, porque la respuesta ingenua
—"probemos a predecir el cambio de volatilidad en vez del nivel"— sería un
re-etiquetado. Basta el álgebra:

    log(rv) ~ a + b·log(σ) + c·X    ES IDÉNTICO A
    log(rv/σ) ~ a + (b−1)·log(σ) + c·X

O sea que "predecir el cambio de nivel" YA se probó, disfrazado, y salió
negativo tres veces. Cambiar el objetivo a ese cociente no probaría nada nuevo.

Lo que los experimentos 01-03 probaron, siempre, es **la media condicional** de
un modelo log-lineal. Lo que NO probaron es otro FUNCIONAL de la distribución
condicional: la **probabilidad de cola**. Una característica puede dejar la
media intacta y aun así ensanchar la distribución —hacer más probable un
reajuste grande— y eso habría sido invisible para todo lo anterior.

Y es exactamente lo que la teoría pide de este diagnóstico. Un sistema lejos
del equilibrio no dice hacia dónde va; dice que está **tenso**, que es propenso
a reorganizarse. Eso es una probabilidad de salto, no una media.

HIPÓTESIS (declarada antes de mirar resultados)
-----------------------------------------------
H1: la violación del FDT medida en ventanas trailing informa sobre la
    PROBABILIDAD de que la volatilidad salte por encima de su régimen actual,
    más allá de lo que ya predice un modelo que sólo conoce σ.

H0: no informa. La diferencia de log-loss frente al modelo con sólo log(σ) no
    es distinguible de cero.

EL OBJETIVO (binario, con umbral sin parámetro libre)
------------------------------------------------------
    S_t = 1  si  rv_t / σ_t^EWMA  >  θ

con rv_t la volatilidad realizada de las 30 barras siguientes y θ el cuantil
0.80 del cociente medido SÓLO en entrenamiento. Definir θ como cuantil del
train y no como una constante (1.5, 2.0…) elimina un parámetro que habría que
elegir: la tasa base queda fijada por construcción para cualquier activo e
intervalo. Medido antes de declarar nada: tasa base OOS 0.15-0.20, sin deriva.

σ del objetivo es SIEMPRE el de EWMA, también cuando la línea base contrastada
es GARCH. Si el evento cambiase de definición entre celdas, las celdas no
serían comparables y el maxT no significaría nada.

LÍNEA BASE HONESTA
------------------
Regresión logística  P(S=1) = logit⁻¹(a + b·log σ)  contra la aumentada
                     P(S=1) = logit⁻¹(a + b·log σ + c·X).
No es un detalle: el cociente rv/σ correlaciona con log σ a −0.25/−0.37 en los
8 series (reversión a la media de la volatilidad — con σ baja, un salto es más
probable), y el signo es el MISMO en todas. Comparar contra una probabilidad
constante regalaría a X un mérito que es de la reversión a la media.

Pérdida: log-loss, que es una regla de puntuación PROPIA — se minimiza al
declarar la probabilidad verdadera, así que no se gana exagerando la
confianza. El contraste es el mismo DM con Newey-West de siempre, vía
`evaluation.diebold_mariano_losses`.

REJILLA (declarada ex-ante, deliberadamente PEQUEÑA)
-----------------------------------------------------
    activos     BTC, ETH, BNB, SOL        (4)
    intervalo   4h, 1h                    (2)
    ventana     500        (la del exp. 02, fija)
    horizonte   30         (la del exp. 02, fija)
    cuantil     0.80       (fijo)
    baseline    EWMA, GARCH               (2)
                                          --------
                                          16 comparaciones

Pequeña a propósito. El exp. 03 barrió 144 celdas y el umbral de maxT quedó
altísimo (el máximo del nulo tenía media 3.796). Aquí se contrasta UNA
hipótesis pre-registrada, no se pesca: con 16 celdas el listón es mucho más
bajo y un efecto real tiene posibilidades de verse.

CRITERIO DE DECISIÓN (ex-ante, no renegociable)
------------------------------------------------
ÉXITO si y sólo si p_global (maxT, K=50 permutaciones del flujo) < 0.05.
Ninguna celda suelta cuenta, por bajo que salga su p nominal.

CONTROL POSITIVO
----------------
`tests/test_transiciones.py::test_control_positivo_*` construye datos donde la
característica SÍ anticipa el salto y exige que este contraste lo detecte
(DM > 3). Sin eso, un negativo aquí sería ambiguo: podría significar "no hay
efecto" o "la maquinaria está rota". Con eso, un negativo significa algo.

Ejecutar:  py experiments/exp04_transiciones.py [K] [n_procesos]
Pensado para Astrum.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Hilos BLAS a 1 ANTES de importar numpy — ver la nota del exp. 03: con Pool(32)
# cada worker abre los suyos y el load average se va a ~500 sin avanzar.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance import fit_garch11
from keldysh_finance.flow import (fetch_klines_with_flow, response_from_frame,
                                  rolling_fdt_violation, align_to_returns)
from keldysh_finance.pipeline import contraste_transicion, westfall_young

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["4h", "1h"]
BASELINES = ["EWMA", "GARCH"]

CONF = dict(years=4.0, ventana=500, max_lag=30, step=1, horizon=30,
            cuantil=0.80, train_frac=0.60, winsor=(0.01, 0.99), seed=410_000)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)

_CACHE: dict = {}


def datos(symbol: str, interval: str) -> dict:
    clave = (symbol, interval)
    if clave not in _CACHE:
        df = fetch_klines_with_flow(symbol, interval=interval,
                                    years=CONF["years"]).dropna()
        _, _, eps, log_open = response_from_frame(df, max_lag=CONF["max_lag"])
        r = np.diff(np.log(df["Close"].to_numpy(float)))
        _CACHE[clave] = {"r": r, "eps": eps, "log_open": log_open,
                         "n_klines": len(log_open),
                         "split": int(len(r) * CONF["train_frac"])}
    return _CACHE[clave]


def ajustar_garch_todos() -> dict:
    par = {}
    for nombre, symbol in ACTIVOS:
        for interval in INTERVALOS:
            d = datos(symbol, interval)
            par[f"{symbol}|{interval}"] = fit_garch11(d["r"][:d["split"]])
            print(f"    GARCH {nombre} {interval}: "
                  f"persistencia={par[f'{symbol}|{interval}']['persistence']:.3f}",
                  flush=True)
    return par


def una_realizacion(tarea: tuple[int, dict]) -> dict:
    """Recorre las 16 celdas para un draw. draw<0 = flujo real (observado)."""
    draw, garch = tarea
    fuera = {}
    for nombre, symbol in ACTIVOS:
        for interval in INTERVALOS:
            d = datos(symbol, interval)
            if draw < 0:
                flujo = d["eps"]
            else:
                rng = np.random.default_rng(
                    CONF["seed"] + 1_000 * draw
                    + 37 * ACTIVOS.index((nombre, symbol))
                    + 7 * INTERVALOS.index(interval))
                flujo = rng.permutation(d["eps"])

            t_w, V = rolling_fdt_violation(d["log_open"], flujo,
                                           window=CONF["ventana"],
                                           max_lag=CONF["max_lag"],
                                           step=CONF["step"])
            V_r = align_to_returns(t_w, V, n_klines=d["n_klines"])
            res = contraste_transicion(d["r"], V_r, horizon=CONF["horizon"],
                                       cuantil=CONF["cuantil"],
                                       train_frac=CONF["train_frac"],
                                       winsor=CONF["winsor"],
                                       garch_params=garch[f"{symbol}|{interval}"])
            for base in BASELINES:
                clave = f"{nombre}|{interval}|{base}"
                dm = res.get("dm_decisivo", {}).get(f"{base} vs {base}+X", {})
                fuera[clave] = float(dm.get("dm_stat", np.nan))
                if draw < 0:
                    fuera[f"_meta|{clave}"] = {
                        "logloss_base": res.get("logloss", {}).get(base),
                        "logloss_conX": res.get("logloss", {}).get(f"{base}+X"),
                        "coef_X": res.get("coef_X", {}).get(base),
                        "tasa_base_oos": res.get("tasa_base_oos"),
                        "theta": res.get("theta"),
                        "p_nominal": dm.get("p_value")}
    return fuera


def main() -> None:
    from multiprocessing import Pool

    print(f"\n  EXPERIMENTO 04 — transiciones de regimen ({len(ACTIVOS)*len(INTERVALOS)*len(BASELINES)} comparaciones)")
    print(f"  Objetivo: P(rv/sigma > q{CONF['cuantil']:.2f} del train) a {CONF['horizon']} barras")
    print(f"  K={K} permutaciones, {NPROC} procesos")
    print(f"  CRITERIO EX-ANTE: exito si y solo si p_global (maxT) < 0.05\n")

    print("  Ajustando GARCH...", flush=True)
    garch = ajustar_garch_todos()

    t0 = time.time()
    print("\n  Observado (flujo real)...", flush=True)
    obs_full = una_realizacion((-1, garch))
    meta = {k[6:]: v for k, v in obs_full.items() if k.startswith("_meta|")}
    obs = {k: v for k, v in obs_full.items() if not k.startswith("_meta|")}
    print(f"    {len(obs)} celdas en {time.time()-t0:.0f} s", flush=True)

    print(f"\n  Nulo por permutacion: {K} draws...", flush=True)
    t0 = time.time()
    with Pool(processes=NPROC) as pool:
        crudos = pool.map(una_realizacion, [(i, garch) for i in range(K)])
    draws = [{k: v for k, v in d.items() if not k.startswith("_meta|")} for d in crudos]
    print(f"    hecho en {time.time()-t0:.0f} s", flush=True)

    wy = westfall_young(obs, draws)

    print(f"\n  {'='*74}\n  RESULTADO\n  {'='*74}")
    print(f"  {'celda':<20}{'logloss base':>14}{'con X':>10}{'DM':>8}{'p_nom':>9}{'p_adj':>8}")
    for k in sorted(obs, key=lambda k: -(obs[k] if np.isfinite(obs[k]) else -np.inf)):
        m = meta.get(k, {})
        lb, lx = m.get("logloss_base"), m.get("logloss_conX")
        print(f"  {k:<20}{(lb if lb is not None else float('nan')):>14.5f}"
              f"{(lx if lx is not None else float('nan')):>10.5f}"
              f"{obs[k]:>8.3f}{(m.get('p_nominal') or float('nan')):>9.4f}"
              f"{wy['p_ajustado'].get(k, float('nan')):>8.3f}")

    print(f"\n  Mejor celda    : {wy['mejor_celda']}   DM = {wy['dm_mejor']:+.3f}")
    print(f"  Nulo del maximo: media={wy['maxT_media']:+.3f} sd={wy['maxT_sd']:.3f} "
          f"q95={wy['maxT_q95']:+.3f}  (K={wy['K']})")
    print(f"  p_global (maxT): {wy['p_global']:.4f}")

    exito = wy["p_global"] < 0.05
    veredicto = ("H1 SOBREVIVE — la violacion del FDT anticipa saltos de regimen"
                 if exito else
                 "H0 NO SE RECHAZA — tampoco anticipa transiciones; el diagnostico "
                 "no informa ni la media ni la cola de la distribucion condicional")
    print(f"\n  → {veredicto}")

    salida = {"conf": CONF, "K": K,
              "grid": {"activos": [a for a, _ in ACTIVOS], "intervalos": INTERVALOS,
                       "baselines": BASELINES},
              "dm_observado": obs, "meta": meta, "westfall_young": wy,
              "veredicto": veredicto}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp04_transiciones.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
