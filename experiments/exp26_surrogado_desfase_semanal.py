r"""Experimento 26 - zanjar el exp. 25: senal real o nulo mal calibrado?

CONTEXTO
--------
El exp. 25 amplio el panel a 20 activos y encontro un patron que no se podia
leer limpio: 6 de los 12 activos nuevos (NEAR, AAVE, LTC, TRX, PAXG, SHIB --
justo los mas distintos en mecanismo, elegidos a proposito por diversidad)
cruzan SU C_sm MUCHO ANTES de lo que predice el control sintetico sin
reversion de exp23/25. Ese control es PARAMETRICO: genera signo con memoria
larga fija (fGn, H=0.75) y magnitud acoplada con kappa fijo -- calibrado, sin
decirlo, al comportamiento de BTC/ETH-tipo. No se puede distinguir con el
si el cruce temprano es (a) senal real de esos 6 activos, o (b) el nulo
parametrico no representa su memoria/colas reales y por eso cruza tarde por
sobra de persistencia que esos activos no tienen.

EL CONTROL QUE ZANJA
---------------------
Un surrogado NO parametrico, construido sobre el DATO REAL de cada activo,
no sobre un generador ajeno:

    eps'_t = s_t * m_{t+Delta}

donde s_t = signo(eps_t) y m_t = |eps_t| son la serie REAL (memoria, colas,
estacionalidad -- todo real, nada supuesto), y Delta es un desfase circular
de un numero ENTERO y grande de semanas (asi el desfase no rompe la fase
semana-del-mismo-activo: hour_of_week(t) es periodico en 168, y ademas el
pipeline de demediado usa la media POR ESTRATO de la propia serie surrogada,
asi que cualquier nivel que arrastre el desfase se absorbe solo). Con Delta
grande (>= 26 semanas, es decir medio ano), la magnitud que se empareja con
cada signo viene de un tramo de mercado esencialmente no relacionado -- se
destruye la sincronia signo-tamano real a los desfases que importan (tau<=200
horas) sin tocar la memoria de signo real, ni la memoria de magnitud real, ni
las colas reales, ni la estacionalidad semanal real. Es justo el tipo de
surrogado que pide la regla 4 del proyecto: mata SOLO la estructura buscada.

Se corre en los 20 activos (8 del paper + 12 del exp. 25), no solo en los 6
sospechosos, para tener el panel completo bajo el MISMO control mejor
calibrado.
"""
from __future__ import annotations
import json
import os
import sys
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

import exp23_control_cruce_forzado as exp23
import exp25_expansion_panel_cruce as exp25

ACTIVOS = exp23.ACTIVOS + exp25.ACTIVOS_NUEVOS
TAU_MAX = exp23.TAU_MAX
HOURS_PER_WEEK = 168
N_REPS = 8
DELTA_MIN_SEMANAS = 26


def cargar(activo: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{activo}USDT_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def cruce_real(df: pd.DataFrame) -> dict:
    prep = prepare_complete_weeks(df, flow_mode="raw")
    eps = prep.flow_demeaned
    c = exp23.c_sm(eps, TAU_MAX)
    t = exp23.primer_cruce_sostenido(c)
    return {"n_candles": int(eps.size), "n_semanas": int(prep.n_complete_weeks),
           "tau_cruce": t}


def surrogado_desfase(df: pd.DataFrame, delta_semanas: int) -> pd.DataFrame:
    """eps'_t = s_t * m_{t+Delta}, Delta = delta_semanas*168 horas, circular.

    Reconstruye Volume/tbBase consistentes con eps' para que
    `prepare_complete_weeks` recalcule el demediado por estrato sobre la
    serie surrogada (no sobre las medias del dato real).
    """
    vol = df["Volume"].to_numpy(float)
    tb = df["tbBase"].to_numpy(float)
    eps = 2.0 * tb - vol
    s = np.sign(eps)
    s = np.where(s == 0, 1.0, s)
    m = np.abs(eps)

    n = len(df)
    n_semanas = n // HOURS_PER_WEEK
    n_uso = n_semanas * HOURS_PER_WEEK
    s = s[:n_uso]
    m = m[:n_uso]

    corrimiento = (delta_semanas % n_semanas) * HOURS_PER_WEEK
    m_desfasada = np.roll(m, -corrimiento)

    eps_sur = s * m_desfasada
    volume_sur = m_desfasada
    tbbase_sur = (volume_sur + eps_sur) / 2.0
    trades_sur = df["trades"].to_numpy(float)[:n_uso]
    trades_sur = np.where(trades_sur > 0, trades_sur, 1.0)

    out = pd.DataFrame({"Volume": volume_sur, "tbBase": tbbase_sur,
                        "trades": trades_sur}, index=df.index[:n_uso])
    return out


def main() -> None:
    print("=" * 84)
    print("  EXP 26 - surrogado por desfase semanal (control no parametrico)")
    print("=" * 84)

    rng = np.random.default_rng(26)
    resultados = {}

    print("\n  activo   n_sem  tau_real   surrogado (8 reps, Delta>=26 sem)      veredicto")
    print("  " + "-" * 82)

    for a in ACTIVOS:
        df = cargar(a)
        r_real = cruce_real(df)
        n_semanas = r_real["n_semanas"]

        cruces_sur = []
        max_delta = max(DELTA_MIN_SEMANAS, n_semanas - DELTA_MIN_SEMANAS)
        for _ in range(N_REPS):
            delta = int(rng.integers(DELTA_MIN_SEMANAS, max(max_delta, DELTA_MIN_SEMANAS + 1)))
            df_sur = surrogado_desfase(df, delta)
            r_sur = cruce_real(df_sur)
            cruces_sur.append(r_sur["tau_cruce"])

        con = [t for t in cruces_sur if t is not None]
        t_real = r_real["tau_cruce"]

        if con:
            lo, hi = min(con), max(con)
            rango_txt = "tau %d-%d (%d/%d)" % (lo, hi, len(con), N_REPS)
            if t_real is not None and (lo - 20) <= t_real <= (hi + 20):
                veredicto = "DENTRO -> artefacto"
            elif t_real is not None:
                veredicto = "FUERA -> senal real"
            else:
                veredicto = "sin cruce real"
        else:
            rango_txt = "sin cruce (%d/%d)" % (len(con), N_REPS)
            veredicto = "FUERA -> senal real" if t_real is not None else "sin cruce real"

        t_real_txt = ("tau=%d" % t_real) if t_real is not None else "sin cruce"
        print("  %-6s  %5d  %-9s  %-32s  %s" %
              (a, n_semanas, t_real_txt, rango_txt, veredicto))

        resultados[a] = {"real": r_real, "surrogado_cruces": cruces_sur,
                         "veredicto": veredicto}

    print("\n  " + "=" * 78)
    print("  RESUMEN")
    print("  " + "-" * 78)
    n_dentro = sum(1 for r in resultados.values() if r["veredicto"] == "DENTRO -> artefacto")
    n_fuera = sum(1 for r in resultados.values() if r["veredicto"] == "FUERA -> senal real")
    print("  dentro del surrogado (artefacto): %d/%d" % (n_dentro, len(ACTIVOS)))
    print("  fuera del surrogado (senal real):  %d/%d" % (n_fuera, len(ACTIVOS)))
    fuera = [a for a, r in resultados.items() if r["veredicto"] == "FUERA -> senal real"]
    dentro = [a for a, r in resultados.items() if r["veredicto"] == "DENTRO -> artefacto"]
    print("    fuera:  %s" % (", ".join(fuera) or "(ninguno)"))
    print("    dentro: %s" % (", ".join(dentro) or "(ninguno)"))

    print()
    if n_fuera == 0:
        print("  CON EL CONTROL BUENO (surrogado real, no parametrico), TODOS los")
        print("  activos caen dentro de lo que el artefacto solo predice. El patron")
        print("  temprano del exp. 25 en NEAR/AAVE/LTC/TRX/PAXG/SHIB era un efecto")
        print("  del nulo parametrico mal calibrado, no senal real. La objecion de")
        print("  David queda confirmada en el panel completo de 20 activos.")
    elif n_fuera == len(ACTIVOS):
        print("  NINGUN activo cae dentro del surrogado real -- el cruce muestral")
        print("  sobrevive incluso al control mas estricto disponible. Hay que")
        print("  revisar si el pipeline de demediado en si mismo fabrica esto para")
        print("  CUALQUIER dato real (lo cual apuntaria de nuevo a artefacto, por")
        print("  una via distinta) o si es senal genuina.")
    else:
        print("  RESULTADO MIXTO incluso con el control correcto: %d activos siguen" % n_fuera)
        print("  cruzando antes de lo que su propio surrogado real predice. Esto ya")
        print("  no se puede atribuir a mala calibracion del nulo -- el surrogado usa")
        print("  la memoria y las colas REALES de cada activo. Es la evidencia mas")
        print("  fuerte hasta ahora de que en un subconjunto puede haber estructura")
        print("  genuina, heterogenea entre activos -- no universal, no artefacto puro.")

    out = RAIZ / "output" / "exp26_surrogado_desfase_semanal.json"

    def _l(o):
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return str(o)

    out.write_text(json.dumps(resultados, indent=2, default=_l), encoding="utf-8")
    print("\n  log: %s" % out)


if __name__ == "__main__":
    main()
