r"""
EXPERIMENTO 07 — el reloj del quench: ¿envejece el mercado tras un shock?

QUÉ RESPONDE
------------
El aging del 05a salió nulo con t_w medido desde el arranque de la muestra —
un reloj ARBITRARIO, como su propio docstring avisaba ("sin quench no hay
reloj"). Aquí el reloj se pone donde la física lo pide: t_w = tiempo desde el
último shock de volatilidad. Cada shock es un temple; la colección de shocks
es el ENSEMBLE que una trayectoria única no daba, y el promedio sobre shocks
es el ⟨·⟩ de Kadanoff-Baym tras el quench. Es la formulación correcta de los
"estados transitorios" del programa: relajación post-temple, medida.

DOS OBSERVABLES, DOS PAPELES (declarados antes de mirar)
--------------------------------------------------------
1. m(t_w) = ⟨|r(t_s+t_w)|/σ(t_s)⟩ — la relajación de la MEDIA. Su decaimiento
   en ley de potencias es el Omori financiero, que ESTÁ en la literatura
   (Lillo-Farmer 2003, exponentes ~0.2-0.4): es la validación externa del
   pipeline, el papel que el exponente γ jugó para flow.py.
2. τ_c(t_w) — la memoria de la CORRELACIÓN en función de la edad. Que crezca
   con t_w (el sistema olvida más despacio cuanto más viejo) es
   envejecimiento genuino. Esto es lo NUEVO: el 05a no podía verlo sin reloj.

CRITERIO EX-ANTE (regla 5)
--------------------------
HAY RELOJ si y sólo si el Spearman(bin de t_w, τ_c) del POOL de cada
intervalo supera el q97.5 de su nulo (K=50 ensembles de shocks COLOCADOS AL
AZAR con la misma cardinalidad y separación). El p de Omori se reporta con IC
bootstrap [2.5, 97.5]% sobre shocks como anclaje de literatura, no como
criterio — y con bootstrap porque el ruido del ensemble está correlacionado
entre edades y el se del jacobiano lo subestima (medido en los tests).

El POOL (shocks de los 4 activos juntos, campos ya normalizados por la σ
pre-shock de cada uno) es el resultado primario: ~4× más temples que
cualquier activo suelto. Los activos sueltos son robustez.

CONTROL POSITIVO: tests/test_quench_floquet.py — un ensemble sintético con
τ(edad) creciente que el pipeline tiene que ver (Spearman>0.7), y otro
estacionario donde no puede ver nada.

Ejecutar:  py experiments/exp07_reloj_quench.py [B_boot] [K_nulo] [n_proc]
Pensado para Astrum (bootstrap sobre shocks × 10 tareas).
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

from keldysh_finance import ewma_vol
from keldysh_finance.flow import fetch_klines_with_flow
from keldysh_finance.quench import (campo_post_shock, correlacion_por_edad,
                                    detectar_shocks, estadistico_envejecimiento,
                                    exponente_omori, tau_c_filas)

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
CONF = {
    "years": 4.0, "umbral_q": 0.995, "margen_pre": 200, "seed": 707_000,
    "1h": dict(separacion=384, u_max=300, max_lag=40,
               bordes_tw=[8, 16, 32, 64, 128, 256]),
    "4h": dict(separacion=192, u_max=160, max_lag=30,
               bordes_tw=[4, 8, 16, 32, 64, 128]),
}
B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
K_NULO = int(sys.argv[2]) if len(sys.argv) > 2 else 50
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 4)


def campo_de(symbol: str, interval: str) -> tuple[np.ndarray, dict]:
    """W (shocks × edades) de una serie, más metadatos."""
    c = CONF[interval]
    df = fetch_klines_with_flow(symbol, interval=interval,
                                years=CONF["years"]).dropna()
    r = np.diff(np.log(df["Close"].to_numpy(float)))
    sigma = ewma_vol(r)
    shocks, thr = detectar_shocks(r, sigma, umbral_q=CONF["umbral_q"],
                                  separacion=c["separacion"],
                                  margen_post=c["u_max"] + 5,
                                  margen_pre=CONF["margen_pre"])
    W = campo_post_shock(r, sigma, shocks, u_max=c["u_max"])
    W = W[np.isfinite(W).all(axis=1)]
    return W, {"n_shocks": int(W.shape[0]), "umbral": thr, "n_barras": len(r),
               "r": r, "sigma": sigma}


def analizar_ensemble(W: np.ndarray, interval: str, seed: int) -> dict:
    """m(u)+Omori con IC bootstrap, τ_c por bin con IC, Spearman con IC."""
    c = CONF[interval]
    rng = np.random.default_rng(seed)
    m = np.nanmean(W, axis=0)
    om = exponente_omori(m)
    # Robustez declarada ANTES de la corrida definitiva: la humo mostró un p
    # pooled ~1.9, señal de una componente rápida en las primeras edades que
    # un solo power-law promedia mal; el ajuste desde u>=5 aísla el régimen
    # lento, que es el comparable con el Omori de literatura.
    om_lento = exponente_omori(m, u_min=5)

    rho, n_pares = correlacion_por_edad(W, c["bordes_tw"], c["max_lag"])
    tau_c = tau_c_filas(rho)
    sp = estadistico_envejecimiento(tau_c)

    boots_p, boots_sp = [], []
    boots_tc = np.full((B_BOOT, len(tau_c)), np.nan)
    n = W.shape[0]
    for b in range(B_BOOT):
        Wb = W[rng.integers(0, n, n)]
        boots_p.append(exponente_omori(np.nanmean(Wb, axis=0))["p"])
        rho_b, _ = correlacion_por_edad(Wb, c["bordes_tw"], c["max_lag"])
        tc_b = tau_c_filas(rho_b)
        boots_tc[b] = tc_b
        boots_sp.append(estadistico_envejecimiento(tc_b))

    def ic(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return ([float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                if len(v) > 10 else [float("nan")] * 2)

    dec = max(1, len(m) // 300)
    return {"omori": {**om, "ic_boot": ic(boots_p)},
            "omori_umin5": om_lento,
            "tau_c": [float(t) for t in tau_c],
            "tau_c_ic": [ic(boots_tc[:, k]) for k in range(len(tau_c))],
            "spearman": sp, "spearman_ic": ic(boots_sp),
            "bordes_tw": c["bordes_tw"], "n_pares": n_pares.tolist(),
            "rho": [[float(x) for x in fila] for fila in rho],
            "m": m[::dec].tolist(), "m_dec": dec}


def nulo_spearman(series: list[tuple[np.ndarray, np.ndarray]], interval: str,
                  cuentas: list[int], seed: int) -> list[float]:
    """K ensembles de shocks al azar (misma cardinalidad y separación por
    activo), combinados igual que el pool: distribución nula del Spearman."""
    c = CONF[interval]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K_NULO):
        Ws = []
        for (r, sigma), n_s in zip(series, cuentas):
            n = len(r)
            falsos: list[int] = []
            candidatos = rng.permutation(
                np.arange(CONF["margen_pre"], n - c["u_max"] - 5))
            for t in candidatos:            # rechazo: misma separación mínima
                if all(abs(int(t) - f) >= c["separacion"] for f in falsos):
                    falsos.append(int(t))
                    if len(falsos) >= n_s:
                        break
            Wf = campo_post_shock(r, sigma, np.asarray(sorted(falsos)),
                                  u_max=c["u_max"])
            Wf = Wf[np.isfinite(Wf).all(axis=1)]
            if len(Wf):
                Ws.append(Wf)
        if not Ws:
            out.append(float("nan"))
            continue
        rho_f, _ = correlacion_por_edad(np.vstack(Ws), c["bordes_tw"],
                                        c["max_lag"])
        out.append(estadistico_envejecimiento(tau_c_filas(rho_f)))
    return out


def tarea(arg: tuple[str, str]) -> dict:
    """Una tarea: un activo ('BTC','1h') o el pool ('POOL','1h')."""
    nombre, interval = arg
    t0 = time.time()
    if nombre == "POOL":
        Ws, series, cuentas, umbrales = [], [], [], []
        for _, symbol in ACTIVOS:
            W, meta = campo_de(symbol, interval)
            if len(W):
                Ws.append(W)
                series.append((meta["r"], meta["sigma"]))
                cuentas.append(len(W))
                umbrales.append(meta["umbral"])
        W = np.vstack(Ws)
        sem_int = {"1h": 1, "4h": 2}[interval]     # hash() no es determinista
        res = analizar_ensemble(W, interval, CONF["seed"] + sem_int)
        res["nulo_spearman"] = nulo_spearman(series, interval, cuentas,
                                             CONF["seed"] + 77 + sem_int)
        res["umbrales"] = umbrales
    else:
        symbol = dict(ACTIVOS)[nombre]
        W, meta = campo_de(symbol, interval)
        res = analizar_ensemble(W, interval,
                                CONF["seed"] + 31 * len(nombre))
        res["umbral"] = meta["umbral"]
    res.update({"clave": f"{nombre}|{interval}", "n_shocks": int(W.shape[0]),
                "segundos": round(time.time() - t0, 1)})
    return res


def main() -> None:
    from multiprocessing import Pool

    tareas = ([("POOL", i) for i in ("1h", "4h")]
              + [(n, i) for n, _ in ACTIVOS for i in ("1h", "4h")])
    print(f"\n  EXPERIMENTO 07 — el reloj del quench "
          f"(B={B_BOOT} bootstrap, K={K_NULO} nulos, {NPROC} procesos)")
    print(f"  CRITERIO EX-ANTE: hay reloj si Spearman(POOL) > q97.5 del nulo\n",
          flush=True)

    t0 = time.time()
    with Pool(processes=min(NPROC, len(tareas))) as pool:
        resultados = pool.map(tarea, tareas)
    print(f"  computo total: {time.time()-t0:.0f} s\n", flush=True)

    print(f"  {'serie':<10}{'shocks':>7}{'omori p [IC]':>22}"
          f"{'spearman [IC]':>22}{'q97.5 nulo':>11}")
    veredictos = {}
    for r in resultados:
        om = r["omori"]
        icp = om.get("ic_boot", [float("nan")] * 2)
        ics = r["spearman_ic"]
        nul = r.get("nulo_spearman")
        q975 = (float(np.nanquantile(np.asarray(nul, dtype=float), 0.975))
                if nul else float("nan"))
        if nul:
            veredictos[r["clave"]] = bool(r["spearman"] > q975)
        print(f"  {r['clave']:<10}{r['n_shocks']:>7}"
              f"{f'{om['p']:.3f} [{icp[0]:.3f},{icp[1]:.3f}]':>22}"
              f"{f'{r['spearman']:+.3f} [{ics[0]:+.2f},{ics[1]:+.2f}]':>22}"
              f"{q975:>11.3f}")

    print()
    for clave, hay in veredictos.items():
        tc = next(r["tau_c"] for r in resultados if r["clave"] == clave)
        print(f"  {clave}: tau_c por bin = "
              f"{[round(t, 1) if np.isfinite(t) else None for t in tc]}")
        print(f"    → {'HAY RELOJ: tau_c crece con la edad' if hay else 'SIN RELOJ: la correlacion no envejece tras el shock'}")

    salida = {"conf": {k: v for k, v in CONF.items()},
              "B_boot": B_BOOT, "K_nulo": K_NULO,
              "resultados": resultados,
              "veredicto_pools": veredictos}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp07_reloj_quench.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
