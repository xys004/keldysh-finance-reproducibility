r"""
EXPERIMENTO 08 — Floquet: ¿modulan los coeficientes de TRANSPORTE con la
fase del ciclo diario?

QUÉ DECIDE
----------
Cripto opera 24/7 pero sus agentes viven en husos horarios: el ciclo diario
es un FORZADO PERIÓDICO real. Si los coeficientes de transporte (sesgo ⟨ε⟩,
afinidad A, impacto R, Fano F) dependen de la fase del ciclo, el modelo
cuasi-estático de dos reservorios con sesgo constante está incompleto y el
programa de modelado (vías A/C) necesita drive Floquet. Si no modulan, el
modelo estacionario basta y el ciclo sólo vive en la ACTIVIDAD.

Que la actividad (⟨|ε|⟩) modula con la hora es sabido (estacionalidad de
volumen, de libro): funciona de CONTROL POSITIVO — si el test no la ve con
p mínimo, el test está roto y lo demás no se puede leer.

EL NULO (y por qué no vale un desfase rígido)
---------------------------------------------
Rotación de fase INDEPENDIENTE POR DÍA (δ_j ~ U{0..23}): destruye sólo la
coherencia serie↔fase entre días, conservando memoria, colas y
no-estacionariedad. Un desfase circular rígido NO sirve: con etiquetas
periódicas es una permutación cíclica de los bins y la varianza entre bins
es invariante (medido en los tests: nulo idéntico al observado). El p
empírico usa el mismo convenio (1+#{≥obs})/(K+1) que el resto del repo.

CRITERIO EX-ANTE (regla 5, con la multiplicidad a la vista)
-----------------------------------------------------------
Con 4 observables de transporte × 4 activos hay 16 comparaciones a 1h. El
criterio NO es "algún p<0.05": HAY ESTRUCTURA FLOQUET en un observable si
modula con p_empírico < 0.05 en AL MENOS 3 DE LOS 4 activos a 1h — la
réplica entre activos es la guardia de multiplicidad, igual que la
incoherencia entre activos fue la navaja del exp. 04. El 4h (6 fases) es
robustez, no criterio.

HISTORIA DE ITERACIONES DEL TEST (transparencia: el control positivo mandó)
---------------------------------------------------------------------------
v1: control = ⟨|ε|⟩ con ε normalizado por volumen → 2/4. Mal especificado:
    normalizar por volumen DIVIDE FUERA la estacionalidad; medía fracción de
    desbalance, no actividad.
v2: control = ⟨trades⟩ → 1/4, con el patrón de sesiones VISIBLE en las
    tablas (×1.9 entre noche asiática y solape EU/US, coherente en los 4
    activos). Diagnóstico: el nulo lo dominan ráfagas de cola pesada — un
    bloque de 3h de un día de crash cae entero en un bin aleatorio.
v3 (definitiva): winsor declarado sobre trades/dp/ε_raw en el módulo, con
    umbrales de la serie completa (idénticos para observado y nulo: test
    emparejado). Las iteraciones cambiaron el ESTIMADOR, nunca el criterio.

Ejecutar:  py experiments/exp08_floquet.py [K]
Diseñado para ejecución reproducible por lotes (K=50 rotaciones × 8 series).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance.floquet import (modulacion, observables_por_fase,
                                     p_empirico, rotar_horas_por_dia)
from keldysh_finance.flow import fetch_klines_with_flow, order_flow_imbalance

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["1h", "4h"]
N_BINS = {"1h": 8, "4h": 6}          # 3 h por bin a 1h; 4 h por bin a 4h
OBSERVABLES = ["sesgo", "actividad", "A1", "R1", "F1"]
CONF = dict(years=4.0, n_bins=N_BINS, observables=OBSERVABLES, seed=808_000,
            criterio="p<0.05 en >=3 de 4 activos a 1h por observable; "
                     "actividad = control positivo")

K = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def datos(symbol: str, interval: str) -> dict:
    df = fetch_klines_with_flow(symbol, interval=interval,
                                years=CONF["years"]).dropna()
    vol = df["Volume"].to_numpy(float)
    tr = df["trades"].to_numpy(float)
    con = tr > 0
    # floor("D") y no aritmética sobre asi8: pandas puede entregar el índice
    # en us o ns según cómo parsee el CSV, y dividir por la unidad equivocada
    # degenera el código de día EN SILENCIO (pasó: 2 "días" en 4 años, y el
    # nulo por rotación quedó reducido al desfase rígido que no vale).
    dias = np.asarray(df.index.floor("D").asi8)
    n_dias = len(np.unique(dias))
    if n_dias < len(df) / 48:
        raise RuntimeError(f"reloj de días degenerado: {n_dias} días únicos "
                           f"para {len(df)} velas")
    return {"eps": order_flow_imbalance(df, normalize="volume"),
            "eps_raw": order_flow_imbalance(df, normalize="none"),
            "log_open": np.log(df["Open"].to_numpy(float)),
            "trades": tr,
            "horas": df.index.hour.to_numpy(),
            "dias": dias,
            "q_barra": float(np.median(vol[con] / tr[con]))}


def una_serie(nombre: str, symbol: str, interval: str) -> dict:
    d = datos(symbol, interval)
    n_bins = N_BINS[interval]
    t0 = time.time()

    filas = observables_por_fase(d["eps"], d["log_open"], d["eps_raw"],
                                 d["trades"], d["horas"], n_bins=n_bins,
                                 q_barra=d["q_barra"])
    v_obs = {o: modulacion(filas, o) for o in OBSERVABLES}

    rng = np.random.default_rng(CONF["seed"] + 100 * INTERVALOS.index(interval)
                                + ACTIVOS.index((nombre, symbol)))
    v_nulo = {o: [] for o in OBSERVABLES}
    for _ in range(K):
        h_rot = rotar_horas_por_dia(d["horas"], d["dias"], rng)
        filas_n = observables_por_fase(d["eps"], d["log_open"], d["eps_raw"],
                                       d["trades"], h_rot, n_bins=n_bins,
                                       q_barra=d["q_barra"])
        for o in OBSERVABLES:
            v_nulo[o].append(modulacion(filas_n, o))

    ps = {o: p_empirico(v_obs[o], np.asarray(v_nulo[o])) for o in OBSERVABLES}
    return {"clave": f"{nombre}|{interval}", "n_bins": n_bins,
            "q_barra": d["q_barra"], "fases": filas,
            "modulacion_obs": v_obs, "p_empirico": ps,
            "segundos": round(time.time() - t0, 1)}


def main() -> None:
    from multiprocessing import Pool

    tareas = [(n, s, i) for n, s in ACTIVOS for i in INTERVALOS]
    print(f"\n  EXPERIMENTO 08 — transporte por fase del ciclo diario "
          f"(K={K} rotaciones por dia)")
    print(f"  CRITERIO EX-ANTE: estructura Floquet en un observable si "
          f"p<0.05 en >=3/4 activos a 1h\n", flush=True)

    t0 = time.time()
    with Pool(processes=min(os.cpu_count() or 4, len(tareas))) as pool:
        resultados = pool.starmap(una_serie, tareas)
    print(f"  computo total: {time.time()-t0:.0f} s\n", flush=True)

    print(f"  p empirico por observable (nulo: rotacion de fase por dia)")
    print(f"  {'serie':<10}" + "".join(f"{o:>11}" for o in OBSERVABLES))
    for r in resultados:
        print(f"  {r['clave']:<10}"
              + "".join(f"{r['p_empirico'][o]:>11.3f}" for o in OBSERVABLES))

    print(f"\n  VEREDICTO (solo 1h, replica entre activos):")
    veredicto = {}
    for o in OBSERVABLES:
        n_sig = sum(1 for r in resultados
                    if r["clave"].endswith("|1h") and r["p_empirico"][o] < 0.05)
        veredicto[o] = {"activos_significativos_1h": n_sig,
                        "estructura": bool(n_sig >= 3)}
        etiqueta = " (control positivo)" if o == "actividad" else ""
        print(f"    {o:<10} {n_sig}/4 activos con p<0.05 → "
              f"{'MODULA' if n_sig >= 3 else 'no modula'}{etiqueta}")

    salida = {"conf": {**CONF, "K": K,
                       "activos": [a for a, _ in ACTIVOS],
                       "intervalos": INTERVALOS},
              "resultados": resultados, "veredicto": veredicto}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp08_floquet.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
