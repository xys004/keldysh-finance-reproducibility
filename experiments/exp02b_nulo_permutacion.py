r"""
EXPERIMENTO 02b — distribución NULA por permutación del procedimiento de decisión.

POR QUÉ EXISTE ESTE SCRIPT
--------------------------
El experimento 02 dejó una anomalía: el control `perm` (una única permutación
del flujo) marcó 2/4 comparaciones "significativas" mientras la característica
real marcaba 0/4. El experimento 01 dejó una anomalía del mismo tipo. Con UNA
sola realización del surrogado no hay forma de distinguir dos explicaciones
muy distintas:

  (a) casualidad — un surrogado que salió afortunado, y el procedimiento está
      bien calibrado; o
  (b) sesgo del procedimiento — el contraste DM con Newey-West rechaza mucho
      más del 5% que declara, porque el horizonte de 30 barras solapa y la
      muestra efectiva es ~3476/30 ≈ 116 observaciones independientes, no 3476.

La diferencia importa para TODO el repo: si es (b), ningún veredicto de este
proyecto —incluido el "SOSPECHOSO" del 01— significa lo que dice.

Se resuelve repitiendo el surrogado K veces y mirando la tasa de rechazo
empírica. Bajo un nulo verdadero, un contraste calibrado al 5% debe rechazar
en ~5% de los draws. Si sale ~5%, la explicación es (a). Si sale 20-30%, es (b).

SEGUNDO PRODUCTO: un p-valor empírico para H1
----------------------------------------------
La misma distribución nula permite situar el estadístico DM de la
característica REAL dentro de ella:

    p_emp = (1 + #{DM_nulo >= DM_real}) / (K + 1)

Es el p-valor más defendible que este proyecto puede producir para H1, porque
no depende de la aproximación normal ni de la corrección de Newey-West: la
distribución de referencia se genera con el mismo pipeline, los mismos precios
y el mismo solape.

NOTA — el surrogado es el DECLARADO en el 02
--------------------------------------------
`perm` = permutación global de ε, exactamente como se declaró ex-ante en el
experimento 02. Un surrogado más fino (desplazamiento circular, que preserva
C_ε(τ) intacta y destruye sólo el acoplamiento con el precio) sería un nulo
más quirúrgico, pero cambiar el control DESPUÉS de ver que dio 2/4 sería
renegociar el diseño. Queda anotado como trabajo futuro, no como sustituto.

Ejecutar:  py experiments/exp02b_nulo_permutacion.py     (~15 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance.flow import (response_from_frame, rolling_fdt_violation,
                                  align_to_returns)
from exp02_fdt_vs_garch import (ACTIVOS, CONF, OUT, _datos, contrastar,
                                caracteristica_cruda)

# nº de permutaciones por activo. Se puede bajar por línea de órdenes para una
# prueba de humo (`py experiments/exp02b_nulo_permutacion.py 3`), pero la cifra
# que se reporta es la de K=50: con menos draws la tasa de rechazo empírica
# tiene un error de muestreo mayor que el efecto que se quiere medir.
K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEMILLA_NULO = 777_000      # distinta de la del 02 para no reusar su draw

CLAVES = ("EWMA_recal vs EWMA_recal+X", "GARCH_recal vs GARCH_recal+X")


def dm_decisivos(res: dict) -> dict:
    """Extrae los dos estadísticos DM que deciden, por nombre de contraste."""
    return {k: res.get("dm_decisivo", {}).get(k, {}) for k in CLAVES}


def main() -> None:
    print(f"\n  EXPERIMENTO 02b — nulo por permutacion (K={K} por activo)")
    print(f"  Config heredada del exp02: window={CONF['window']} max_lag={CONF['max_lag']} "
          f"horizon={CONF['horizon']}")

    salida = {"K": K, "conf": CONF, "semilla": SEMILLA_NULO, "activos": {}}

    for nombre, symbol in ACTIVOS:
        df = _datos(symbol)
        n_r = len(df) - 1
        _, _, eps, log_open = response_from_frame(df, max_lag=CONF["max_lag"])

        # --- estadístico observado (característica real) ---
        V_real = caracteristica_cruda(symbol, "real", n_r=n_r)
        res_real = contrastar(nombre, symbol, "real", V_real, verbose=False)
        dm_real = {k: v.get("dm_stat", float("nan")) for k, v in dm_decisivos(res_real).items()}
        print(f"\n  {'='*70}\n  {nombre}: DM observado (real) -> " +
              "  ".join(f"{k.split()[0]}={v:+.3f}" for k, v in dm_real.items()))

        # --- distribución nula ---
        nulo = {k: [] for k in CLAVES}
        t0 = time.time()
        for i in range(K):
            rng = np.random.default_rng(SEMILLA_NULO + 1000 * ACTIVOS.index((nombre, symbol)) + i)
            flujo = rng.permutation(eps)
            t_w, V = rolling_fdt_violation(log_open, flujo, window=CONF["window"],
                                           max_lag=CONF["max_lag"], step=CONF["step"])
            V_r = align_to_returns(t_w, V, n_klines=len(log_open))
            res = contrastar(nombre, symbol, f"perm{i}", V_r, verbose=False)
            for k, v in dm_decisivos(res).items():
                nulo[k].append(v.get("dm_stat", float("nan")))
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{K} draws  ({time.time()-t0:.0f} s)")

        salida["activos"][nombre] = {}
        for k in CLAVES:
            d = np.array(nulo[k], dtype=float)
            fin = d[np.isfinite(d)]
            # tasa de rechazo empírica a una cola (mejora significativa), que es
            # la dirección que el criterio del 02 cuenta como éxito
            tasa = float(np.mean(fin > 1.959964)) if len(fin) else float("nan")
            obs = dm_real[k]
            p_emp = (float((1 + np.sum(fin >= obs)) / (len(fin) + 1))
                     if len(fin) and np.isfinite(obs) else float("nan"))
            salida["activos"][nombre][k] = {
                "dm_observado": obs, "n_draws": int(len(fin)),
                "tasa_rechazo_empirica_5pct": tasa,
                "p_empirico_una_cola": p_emp,
                "nulo_media": float(np.mean(fin)) if len(fin) else float("nan"),
                "nulo_sd": float(np.std(fin)) if len(fin) else float("nan"),
                "nulo_q95": float(np.quantile(fin, 0.95)) if len(fin) else float("nan"),
                "dm_nulo": fin.tolist(),
            }
            print(f"    {k:<34} nulo: media={np.mean(fin):+.3f} sd={np.std(fin):.3f} "
                  f"q95={np.quantile(fin, 0.95):+.3f}")
            print(f"    {'':34} rechazo empirico al 5% = {tasa:.1%}   "
                  f"p_emp(real) = {p_emp:.3f}")

    # --- lectura ---
    tasas = [c["tasa_rechazo_empirica_5pct"]
             for a in salida["activos"].values() for c in a.values()]
    sds = [c["nulo_sd"] for a in salida["activos"].values() for c in a.values()]
    tasa_media = float(np.nanmean(tasas))
    sd_media = float(np.nanmean(sds))
    # Diagnóstico independiente de la tasa: un estadístico DM bien calibrado
    # tiene sd≈1 bajo el nulo por construcción. Usa los K draws enteros, no
    # sólo los que cruzan el umbral, así que es mucho menos ruidoso.
    print(f"\n  sd media del DM nulo = {sd_media:.2f}  (calibrado seria ~1.0)")

    if K < 30:
        lectura = (f"PRUEBA DE HUMO (K={K}) — sin valor estadistico. Con tan pocos draws "
                   f"la tasa de rechazo tiene un error de muestreo mayor que el efecto. "
                   f"Ejecutar con K=50 antes de citar ninguna cifra.")
    elif tasa_media > 0.15:
        lectura = (f"PROCEDIMIENTO MAL CALIBRADO — rechazo empirico medio {tasa_media:.1%} "
                   f"frente al 5% nominal. El DM con Newey-West no corrige lo bastante "
                   f"el solape del horizonte; los umbrales p<0.05 del repo son optimistas.")
    elif tasa_media > 0.08:
        lectura = (f"CALIBRACION FLOJA — rechazo empirico medio {tasa_media:.1%} frente al 5% "
                   f"nominal. Hay algo de exceso, pero no explica por si solo un 2/4.")
    else:
        lectura = (f"PROCEDIMIENTO CALIBRADO — rechazo empirico medio {tasa_media:.1%}, "
                   f"compatible con el 5% nominal. El 2/4 del control 'perm' en el exp02 "
                   f"fue un draw afortunado, no un sesgo del metodo.")
    print(f"\n  {'='*70}\n  LECTURA\n  {'='*70}\n  {lectura}")
    salida["tasa_rechazo_media"] = tasa_media
    salida["sd_dm_nulo_media"] = sd_media
    salida["lectura"] = lectura

    p_emps = [c["p_empirico_una_cola"]
              for a in salida["activos"].values() for c in a.values()]
    print(f"\n  p empiricos de la caracteristica REAL: "
          + ", ".join(f"{p:.3f}" for p in p_emps))
    if all(np.isfinite(p) and p >= 0.05 for p in p_emps):
        print("  -> ninguno por debajo de 0.05: H0 se mantiene tambien con el nulo empirico")

    OUT.mkdir(exist_ok=True)
    path = OUT / "exp02b_nulo_permutacion.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
