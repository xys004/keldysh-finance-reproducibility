r"""Experimento 27 — bandas de confianza para los cuatro estimadores de nu del exp. 11.

QUE FALTABA
-----------
La tabla `model:tab:haar` del paper combinado reporta nu_global (bloques,
db1, db2, db3) por activo como puntos desnudos. Sin incertidumbre no se
puede juzgar si la dispersion entre estimadores (columna "max desviacion"
del exp. 11) es ruido de muestreo o una discrepancia real, y un referee lo
va a preguntar igual que pregunto por la deriva de la media.

EL BOOTSTRAP
------------
Bootstrap de bloques moviles CIRCULAR sobre la serie CRUDA `eps` (no sobre
la secuencia ya reducida de Q_T, que es lo que remuestrea
`block_bootstrap_standardized_cumulants` en `counting.py`): aqui hace falta
preservar la memoria de `eps` en las escalas de octava examinadas (T hasta
2^8=256 velas, ~10.7 dias) para que cada replica siga siendo una serie
valida sobre la que recalcular Var(Q_T) y el diagrama log-escala de
ondicula. La longitud de bloque sigue la convencion ya establecida en el
repo ("la publicacion usa cuatro semanas de tiempo de reloj"): 672 horas,
que es mayor que la escala mas gruesa examinada (256 h) y por tanto no
corta la correlacion que se quiere medir.

Misma construccion vectorizada que `counting.py` (offsets circulares +
modulo), pero aqui cada replica hay que pasarla por
`nu_por_bloques`/`nu_por_ondicula` del exp. 11 -- eso no se vectoriza sobre
filas porque la DWT no es lineal en bloques de muestras --, asi que el
costo es un bucle de `replicates` iteraciones por activo. Con n~35000 y una
DWT de O(n) por replica esto corre en decenas de segundos en LOCAL; no hace
falta Astrum.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

from keldysh_finance.flow import fetch_klines_with_flow, order_flow_imbalance
from exp11_haar_crossover import ACTIVOS, OCTAVAS, nu_por_bloques, nu_por_ondicula

BLOCK_LENGTH_HORAS = 672   # 4 semanas de reloj; > 256 h, la octava mas gruesa
REPLICATES = 999
SEED = 0


def bootstrap_indices(n: int, block_length: int, replicates: int,
                       seed: int) -> np.ndarray:
    """Indices (replicates, n) de un bootstrap de bloques moviles circular."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    starts = rng.integers(0, n, size=(replicates, n_blocks, 1))
    idx = (starts + offsets[None, None, :]) % n
    return idx.reshape(replicates, -1)[:, :n]


def nu_replica(x: np.ndarray) -> dict:
    fila = {"bloques": nu_por_bloques(x, OCTAVAS)["nu_global"]}
    for m in (1, 2, 3):
        fila[f"db{m}"] = nu_por_ondicula(x, OCTAVAS, m)["nu_global"]
    return fila


def main() -> None:
    print("=" * 78)
    print("  EXP 27 -- bandas de confianza (bootstrap de bloques) para nu")
    print("=" * 78)
    print(f"  bloque = {BLOCK_LENGTH_HORAS} h, replicas = {REPLICATES}\n")

    resultados = []
    for sym in ACTIVOS:
        t0 = time.time()
        df = fetch_klines_with_flow(sym, interval="1h", years=4.0).dropna()
        eps = order_flow_imbalance(df, normalize="volume")
        n = eps.size

        puntual = nu_replica(eps)
        idx = bootstrap_indices(n, BLOCK_LENGTH_HORAS, REPLICATES, SEED)
        boot = {k: np.empty(REPLICATES, dtype=float) for k in puntual}
        for r in range(REPLICATES):
            fila = nu_replica(eps[idx[r]])
            for k, v in fila.items():
                boot[k][r] = v

        fila_out = {"activo": sym[:-4], "n": int(n), "puntual": puntual}
        for k in puntual:
            b = boot[k]
            b = b[np.isfinite(b)]
            fila_out[k] = {
                "media_boot": float(np.mean(b)),
                "sd_boot": float(np.std(b, ddof=1)),
                "ci95": [float(v) for v in np.quantile(b, [0.025, 0.975])],
            }
        resultados.append(fila_out)
        dt = time.time() - t0
        print(f"  {sym[:-4]:<6} ({dt:5.1f} s)  "
              + "  ".join(f"{k}={puntual[k]:.2f}±{fila_out[k]['sd_boot']:.2f}"
                          for k in ("bloques", "db1", "db2", "db3")))

    out = {"block_length_horas": BLOCK_LENGTH_HORAS,
           "replicates": REPLICATES, "seed": SEED,
           "octavas": OCTAVAS, "resultados": resultados}
    path = RAIZ / "output" / "exp27_bootstrap_nu.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  log: {path}")


if __name__ == "__main__":
    main()
