r"""Experimento 25 - el cruce de C_sm, es artefacto de tener solo 8 criptos?

CONTEXTO
--------
El exp. 23 ya probo que un sintetico SIN reversion poblacional, con el mismo
n y el mismo pipeline de desmediado, cruza en la misma escala que los 8
activos reales (tau*=23-166). Esa comparacion usaba un solo n fijo (la
mediana de los 8) para todo el barrido sintetico. Pregunta nueva de Nelson:
ese resultado, es fragil porque el panel real es chico (8) y muy
correlacionado (todo cripto, un solo exchange)? O aparece igual con un panel
mas grande y mas diverso?

DISENO
------
Se amplia el panel real con 12 activos NUEVOS (no en el panel original),
elegidos del censo de liquidez (experiments/../ohlcv, output/censo_binance_
pares.json) para maximizar diversidad estructural, no solo volumen:

  - L1 grandes no correlacionados 1:1 con BTC/ETH: DOT, ATOM, NEAR, ETC
  - DeFi (mecanismo de mercado distinto, mas OTC/AMM): UNI, AAVE
  - "legacy"/pago: LTC, BCH, TRX
  - oraculo/infra: LINK
  - **PAXG** (token respaldado por oro -- el activo con MENOS motivo
    estructural para compartir el mecanismo de cripto especulativa)
  - **SHIB** (meme coin, flujo dominado por retail -- microestructura
    distinta a los L1 establecidos)

Para cada uno de los 12 nuevos se calcula (a) el cruce REAL de C_sm con el
mismo pipeline que exp23, y (b) su propio control sintetico SIN reversion
poblacional, calibrado a SU PROPIO n (no al n mediano del panel viejo). Esto
es mas estricto que exp23: cada activo nuevo se compara contra un nulo hecho
a su propia medida, no contra un nulo generico.

VEREDICTO
---------
Si el panel ampliado (20 activos: 8 viejos + 12 nuevos, con mucha mas
diversidad de mecanismo) sigue cruzando en un rango indistinguible del que
predice el nulo sin reversion, activo por activo -> el cruce NO es un
artefacto de tener pocos criptos: es una propiedad del pipeline de
desmediado que aparece para CUALQUIER activo con esa longitud de muestra,
independientemente de su mecanismo de mercado. Si en cambio los activos
nuevos (sobre todo PAXG o SHIB, los mas distintos) dejan de cruzar o cruzan
muy fuera del rango del nulo -> la aparente universalidad del hallazgo en
los 8 originales si dependia de que eran pocos y muy parecidos entre si.
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
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from keldysh_finance.fano_validation import prepare_complete_weeks
from keldysh_finance.flow import fetch_klines_with_flow

import exp23_control_cruce_forzado as exp23

ACTIVOS_VIEJOS = exp23.ACTIVOS
ACTIVOS_NUEVOS = ["DOT", "ATOM", "NEAR", "ETC", "UNI", "AAVE",
                  "LTC", "BCH", "TRX", "LINK", "PAXG", "SHIB"]
TAU_MAX = exp23.TAU_MAX
N_REPS_NULO_POR_ACTIVO = 5
KAPPA_NULO = 3.0


def cargar_o_descargar(activo: str) -> pd.DataFrame:
    symbol = f"{activo}USDT"
    df = fetch_klines_with_flow(symbol=symbol, interval="1h", years=4.0,
                                cache_dir=str(RAIZ / "output" / "cache"))
    return df


def cruce_real(activo: str, df: pd.DataFrame) -> dict:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    prep = prepare_complete_weeks(df.dropna(), flow_mode="raw")
    eps = prep.flow_demeaned
    c = exp23.c_sm(eps, TAU_MAX)
    t = exp23.primer_cruce_sostenido(c)
    return {"n_candles": int(eps.size), "n_semanas": int(prep.n_complete_weeks),
           "tau_cruce": t}


def nulo_para_n(n_candles: int, rng: np.random.Generator, n_reps: int) -> list:
    cruces = []
    for _ in range(n_reps):
        df = exp23.sintetico_sin_reversion(n_candles, kappa=KAPPA_NULO, rng=rng)
        prep = prepare_complete_weeks(df, flow_mode="raw")
        eps = prep.flow_demeaned
        c = exp23.c_sm(eps, TAU_MAX)
        cruces.append(exp23.primer_cruce_sostenido(c))
    return cruces


def main() -> None:
    print("=" * 84)
    print("  EXP 25 - el cruce de C_sm, artefacto de tener pocos criptos?")
    print("=" * 84)

    print("\n  [1] PANEL VIEJO (8 activos, ya en el paper) -- referencia")
    print("  " + "-" * 78)
    resultados = {}
    for a in ACTIVOS_VIEJOS:
        p = RAIZ / "output" / "cache" / f"flow_{a}USDT_1h_4.0y.csv"
        df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
        r = cruce_real(a, df)
        r["grupo"] = "viejo"
        resultados[a] = r
        print("    %-6s: %5d velas, %3d semanas -> %s" %
              (a, r["n_candles"], r["n_semanas"],
               ("cruce en tau=%d" % r["tau_cruce"]) if r["tau_cruce"] else "sin cruce"))

    print("\n  [2] PANEL NUEVO (12 activos, fuera del paper) -- descarga si hace falta")
    print("  " + "-" * 78)
    for a in ACTIVOS_NUEVOS:
        t0 = time.time()
        df = cargar_o_descargar(a)
        r = cruce_real(a, df)
        r["grupo"] = "nuevo"
        resultados[a] = r
        print("    %-6s: %5d velas, %3d semanas -> %s  (%.1fs)" %
              (a, r["n_candles"], r["n_semanas"],
               ("cruce en tau=%d" % r["tau_cruce"]) if r["tau_cruce"] else "sin cruce",
               time.time() - t0))

    print("\n  [3] NULO SIN REVERSION, calibrado al n PROPIO de cada activo nuevo")
    print("  " + "-" * 78)
    rng = np.random.default_rng(25)
    nulos = {}
    for a in ACTIVOS_NUEVOS:
        n = resultados[a]["n_candles"]
        cruces = nulo_para_n(n, rng, N_REPS_NULO_POR_ACTIVO)
        nulos[a] = cruces
        con = [t for t in cruces if t is not None]
        if con:
            print("    %-6s: nulo cruza %d/%d, tau %d-%d" %
                  (a, len(con), len(cruces), min(con), max(con)))
        else:
            print("    %-6s: nulo NO cruza en [1,%d]" % (a, TAU_MAX))

    print("\n  " + "=" * 78)
    print("  VEREDICTO")
    print("  " + "-" * 78)

    todos_tau = [r["tau_cruce"] for r in resultados.values() if r["tau_cruce"]]
    viejos_tau = [resultados[a]["tau_cruce"] for a in ACTIVOS_VIEJOS if resultados[a]["tau_cruce"]]
    nuevos_tau = [resultados[a]["tau_cruce"] for a in ACTIVOS_NUEVOS if resultados[a]["tau_cruce"]]
    n_total = len(resultados)
    print("  panel total: %d/%d activos cruzan (antes: %d/%d viejos)" %
          (len(todos_tau), n_total, len(viejos_tau), len(ACTIVOS_VIEJOS)))
    if nuevos_tau:
        print("  panel NUEVO: %d/%d cruzan, tau %d-%d  (viejo: tau %d-%d)" %
              (len(nuevos_tau), len(ACTIVOS_NUEVOS), min(nuevos_tau), max(nuevos_tau),
               min(viejos_tau), max(viejos_tau)))

    dentro_de_su_nulo = []
    fuera_de_su_nulo = []
    for a in ACTIVOS_NUEVOS:
        t_real = resultados[a]["tau_cruce"]
        con_nulo = [t for t in nulos[a] if t is not None]
        if t_real is None:
            continue
        if con_nulo and min(con_nulo) - 20 <= t_real <= max(con_nulo) + 20:
            dentro_de_su_nulo.append(a)
        else:
            fuera_de_su_nulo.append(a)

    print("\n  de los activos nuevos que cruzan, cuantos caen dentro del rango")
    print("  de SU PROPIO nulo sin reversion (+-20 velas de margen):")
    print("    dentro: %s" % (", ".join(dentro_de_su_nulo) or "(ninguno)"))
    print("    fuera:  %s" % (", ".join(fuera_de_su_nulo) or "(ninguno)"))

    n_nuevos_con_cruce = len(nuevos_tau)
    frac_dentro = len(dentro_de_su_nulo) / n_nuevos_con_cruce if n_nuevos_con_cruce else 0.0

    print()
    if n_nuevos_con_cruce >= len(ACTIVOS_NUEVOS) * 0.7 and frac_dentro >= 0.7:
        print("  EL PANEL AMPLIADO REPRODUCE EL PATRON, INCLUSO EN ACTIVOS SIN")
        print("  RAZON ESTRUCTURAL PARA COMPARTIR EL MECANISMO (oro, meme coin).")
        print("  Esto NO rescata el cruce como hallazgo real -- lo contrario:")
        print("  confirma que es una propiedad del PIPELINE (n, desmediado por")
        print("  estrato), no de que el panel original fuera chico y correlacionado.")
        print("  Con 8 criptos o con 20 mas diversos, el cruce aparece igual porque")
        print("  lo produce la aritmetica del centrado, no el mecanismo de mercado.")
    else:
        print("  EL PANEL AMPLIADO NO REPRODUCE EL PATRON DE FORMA UNIFORME.")
        print("  Algunos activos (revisar 'fuera' arriba) no cruzan donde su propio")
        print("  nulo sin reversion predice. La aparente universalidad en los 8")
        print("  originales SI podria depender de que eran pocos y muy parecidos --")
        print("  hace falta revisar cuales activos rompen el patron y por que.")

    out = RAIZ / "output" / "exp25_expansion_panel_cruce.json"

    def _l(o):
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return str(o)

    out.write_text(json.dumps(
        {"resultados": resultados, "nulos_por_activo_nuevo": nulos,
         "dentro_de_su_nulo": dentro_de_su_nulo, "fuera_de_su_nulo": fuera_de_su_nulo},
        indent=2, default=_l), encoding="utf-8")
    print("\n  log: %s" % out)


if __name__ == "__main__":
    main()
