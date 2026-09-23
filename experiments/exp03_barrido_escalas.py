r"""
EXPERIMENTO 03 — barrido de escalas con corrección de multiplicidad honesta.

QUÉ PREGUNTA RESPONDE
---------------------
Los experimentos 01 y 02 fijaron UNA configuración cada uno (una ventana, un
horizonte, dos activos, un intervalo) y dieron negativo. Queda la posibilidad
de que exista una ESCALA a la que el diagnóstico de no-equilibrio sí informe y
que simplemente no se probó. Este experimento la busca.

Y busca con red, porque barrer es la forma más fácil de engañarse: con 144
comparaciones, encontrar una con p<0.05 no es un hallazgo, es aritmética.

EL BARRIDO (declarado ex-ante)
-------------------------------
    activos     BTC, ETH, BNB, SOL        (4)
    intervalo   4h, 1h                    (2)
    ventana     250, 500, 1000            (3)
    horizonte   10, 30, 90 barras         (3)
    baseline    EWMA(0.94), GARCH(1,1)    (2)
                                          --------
                                          144 comparaciones

Característica: `rolling_fdt_violation` con max_lag=30, igual que el exp. 02.
Modelo, transformación y contraste: los del módulo `pipeline`, literalmente el
mismo código que el 02, para que las cifras sean comparables entre experimentos.

CÓMO SE CORRIGE LA MULTIPLICIDAD: WESTFALL-YOUNG maxT
------------------------------------------------------
Corregir 144 p-valores por Bonferroni sería válido pero doblemente malo: muy
conservador, y basado en un supuesto FALSO de independencia (las celdas
comparten activo, datos y característica; las de EWMA y GARCH sobre la misma
celda son casi la misma prueba).

En su lugar se colapsa el barrido a UNA decisión:

  1. Se calcula el estadístico observado  T_obs = max sobre las 144 celdas
     del DM decisivo (`recal` vs `recal+X`).
  2. Para cada permutación i = 1..K se recalcula EL BARRIDO ENTERO con el
     flujo permutado, y se toma  T_i = max sobre las 144 celdas.
  3. p_global = (1 + #{T_i >= T_obs}) / (K + 1)

Ése es el p-valor de la pregunta que de verdad importa: **¿es la mejor de 144
celdas mejor de lo que sale por casualidad cuando se prueban 144 celdas?** La
dependencia entre celdas se contabiliza sola, porque cada permutación las
recorre todas con la misma realización.

De propina, cada celda recibe su p ajustado
`p_adj = (1 + #{T_i >= DM_obs(celda)}) / (K+1)`, que es el maxT de un paso.

CRITERIO DE DECISIÓN (ex-ante, no renegociable)
------------------------------------------------
ÉXITO si y sólo si p_global < 0.05. Nada más cuenta. En particular, NO cuenta
que alguna celda tenga p nominal < 0.05: el experimento reporta a propósito
cuántas lo tienen, para dejar constancia de lo engañoso que sería mirarlas.

POR QUÉ NO HAY CONTROLES `signflip` NI `ruido` AQUÍ
----------------------------------------------------
El nulo por permutación con K draws ES el control, y uno mucho más fuerte que
los del 02: allí un surrogado se corría UNA vez y su 2/4 no se sabía leer.
Aquí el surrogado se corre K veces y define la distribución de referencia.

Ejecutar:  py experiments/exp03_barrido_escalas.py [K] [n_procesos]
Pensado para Astrum: ~110 min en un core, ~5 min en 32.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# HILOS BLAS A 1, ANTES DE IMPORTAR NUMPY. No es cosmético: el paralelismo lo
# pone el Pool, y si además cada uno de los 32 procesos abre sus propios hilos
# de OpenBLAS para el `lstsq` salen ~1000 hilos peleándose por 32 cores. La
# primera ejecución en Astrum subió el load average a 493 y no avanzaba.
# Tiene que ir aquí arriba porque OpenBLAS lee estas variables al importarse,
# no al usarse: ponerlas después de `import numpy` no sirve de nada.
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
from keldysh_finance.pipeline import contraste_caracteristica, westfall_young

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["4h", "1h"]
VENTANAS = [250, 500, 1000]
HORIZONTES = [10, 30, 90]
BASELINES = ["EWMA", "GARCH"]

CONF = dict(years=4.0, max_lag=30, step=1, train_frac=0.60,
            winsor=(0.01, 0.99), seed=310_000)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)

_CACHE: dict = {}


def datos(symbol: str, interval: str) -> dict:
    """Series y parámetros GARCH de un (activo, intervalo). Cacheado por proceso."""
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
    """GARCH una sola vez por (activo, intervalo), en el proceso padre.

    No es sólo ahorro. Ajustar dentro de los workers costaría ~27 s por serie
    de 1h —`_garch_nll` es un bucle Python sobre 21.000 puntos y Nelder-Mead lo
    llama miles de veces— y, peor, dos workers podrían caer en óptimos locales
    distintos y producir líneas base que no son la misma. La línea base tiene
    que ser IDÉNTICA en todas las celdas y en todos los draws.
    """
    par = {}
    for nombre, symbol in ACTIVOS:
        for interval in INTERVALOS:
            d = datos(symbol, interval)
            t0 = time.time()
            par[f"{symbol}|{interval}"] = fit_garch11(d["r"][:d["split"]])
            print(f"    GARCH {nombre} {interval}: "
                  f"persistencia={par[f'{symbol}|{interval}']['persistence']:.3f} "
                  f"({time.time()-t0:.0f} s)", flush=True)
    return par


def una_realizacion(tarea: tuple[int, dict]) -> dict:
    """Recorre el barrido ENTERO para un draw. draw<0 = flujo real (observado).

    Devuelve {clave_celda: dm_stat}. Cada draw usa una permutación distinta por
    (activo, intervalo), pero LA MISMA para todas las ventanas y horizontes de
    ese draw: el máximo tiene que tomarse sobre una realización coherente del
    nulo, no sobre un revoltijo de realizaciones independientes.
    """
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

            for W in VENTANAS:
                t_w, V = rolling_fdt_violation(d["log_open"], flujo, window=W,
                                               max_lag=CONF["max_lag"],
                                               step=CONF["step"])
                V_r = align_to_returns(t_w, V, n_klines=d["n_klines"])
                for H in HORIZONTES:
                    res = contraste_caracteristica(
                        d["r"], V_r, horizon=H, train_frac=CONF["train_frac"],
                        winsor=CONF["winsor"],
                        garch_params=garch[f"{symbol}|{interval}"])
                    for base in BASELINES:
                        clave = f"{nombre}|{interval}|W{W}|H{H}|{base}"
                        dm = res.get("dm_decisivo", {}).get(
                            f"{base}_recal vs {base}_recal+X", {})
                        fuera[clave] = float(dm.get("dm_stat", np.nan))
    return fuera


def main() -> None:
    from multiprocessing import Pool

    n_celdas = len(ACTIVOS) * len(INTERVALOS) * len(VENTANAS) * len(HORIZONTES) * len(BASELINES)
    print(f"\n  EXPERIMENTO 03 — barrido de escalas ({n_celdas} comparaciones)")
    print(f"  K={K} permutaciones, {NPROC} procesos")
    print(f"  CRITERIO EX-ANTE: exito si y solo si p_global (maxT) < 0.05.")
    print(f"  Ninguna celda individual cuenta, por muy bajo que salga su p nominal.\n")

    print("  Ajustando GARCH (una vez por activo/intervalo)...", flush=True)
    garch = ajustar_garch_todos()

    t0 = time.time()
    print(f"\n  Observado (flujo real)...", flush=True)
    obs = una_realizacion((-1, garch))
    print(f"    {len(obs)} celdas en {time.time()-t0:.0f} s", flush=True)

    print(f"\n  Nulo por permutacion: {K} draws x {n_celdas} celdas...", flush=True)
    t0 = time.time()
    with Pool(processes=NPROC) as pool:
        draws = pool.map(una_realizacion, [(i, garch) for i in range(K)])
    print(f"    hecho en {time.time()-t0:.0f} s", flush=True)

    # La corrección vive en la librería para que el 03 y el 04 usen exactamente
    # la misma; sus semánticas están fijadas en tests/test_maxt.py.
    wy = westfall_young(obs, draws)
    claves = sorted(obs)
    maxT = np.array(wy["maxT_nulo"])
    t_obs, mejor, p_global, p_adj = (wy["dm_mejor"], wy["mejor_celda"],
                                     wy["p_global"], wy["p_ajustado"])

    # p nominal por celda, sin corregir — se reporta SOLO para mostrar el engaño
    from math import erfc, sqrt
    p_nom = {k: float(erfc(abs(obs[k]) / sqrt(2.0))) for k in claves if np.isfinite(obs[k])}
    n_nom = sum(1 for k, p in p_nom.items() if p < 0.05 and obs[k] > 0)

    print(f"\n  {'='*74}\n  RESULTADO\n  {'='*74}")
    print(f"  Mejor celda observada : {mejor}   DM = {t_obs:+.3f}")
    print(f"  Nulo del maximo       : media={maxT.mean():+.3f} sd={maxT.std():.3f} "
          f"q95={np.quantile(maxT, 0.95):+.3f}  (K={len(maxT)})")
    print(f"  p_global (maxT)       : {p_global:.4f}")
    print(f"\n  Celdas con p NOMINAL < 0.05 y signo favorable: {n_nom}/{len(claves)}")
    print(f"  (se listan solo para dejar constancia: sin corregir no significan nada)")

    print(f"\n  Top 8 celdas por DM observado:")
    orden = sorted(claves, key=lambda k: -(obs[k] if np.isfinite(obs[k]) else -np.inf))
    print(f"    {'celda':<34}{'DM':>8}{'p_nom':>9}{'p_adj':>9}")
    for k in orden[:8]:
        print(f"    {k:<34}{obs[k]:>8.3f}{p_nom.get(k, float('nan')):>9.4f}{p_adj[k]:>9.3f}")

    exito = p_global < 0.05
    veredicto = ("H1 SOBREVIVE AL BARRIDO — existe una escala con senal real"
                 if exito else
                 "H0 NO SE RECHAZA — ninguna escala aporta; lo mejor del barrido "
                 "esta dentro de lo que produce el azar al probar %d celdas" % len(claves))
    print(f"\n  → {veredicto}")

    salida = {"conf": CONF, "K": K, "n_celdas": len(claves),
              "grid": {"activos": [a for a, _ in ACTIVOS], "intervalos": INTERVALOS,
                       "ventanas": VENTANAS, "horizontes": HORIZONTES,
                       "baselines": BASELINES},
              "dm_observado": {k: obs[k] for k in claves},
              "p_nominal": p_nom, "p_ajustado_maxT": p_adj, "westfall_young": {k: v for k, v in wy.items() if k != "maxT_nulo"},
              "maxT_nulo": maxT.tolist(),
              "mejor_celda": mejor, "dm_mejor": t_obs, "p_global": p_global,
              "n_celdas_p_nominal_significativo": int(n_nom),
              "veredicto": veredicto}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp03_barrido_escalas.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
