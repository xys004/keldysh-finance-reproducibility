r"""Experimento 23 - el control que David propuso: el cruce de signo de
C_sm, es artefacto del desmediado, o es real?

LA OBJECION
-----------
Para cualquier serie finita demediada -global o por estrato, verificado
simbolicamente, PASS-, la autocovarianza muestral satisface exactamente

    Sum_{tau=1}^{n-1} gamma_hat(tau) = -gamma_hat(0)/2 < 0.

Y C_sm(tau) = gamma_eps(tau) - mbar^2 <s_t s_{t+tau}>, asi que hereda esa
restriccion: el AREA TOTAL de C_sm esta obligada a ser negativa, con
magnitud ~ (E[m^2]-mbar^2)/2, grande con colas pesadas. El cruce a negativo
que reportamos en el exp. 22 (tau* = 23-166 velas) podria no ser hallazgo:
podria ser aritmetica del centrado, mas apretada cuanto mas fino el
desmediado (estrato trimestre x hora-de-semana en vez de uno global).

EL CONTROL
----------
Un proceso sintetico cuya C_sm POBLACIONAL es positiva a TODO desfase (sin
reversion alguna por construccion), pasado por el MISMO pipeline de
desmediado estratificado y el MISMO estimador. Si el cruce MUESTRAL cae en
la misma escala que el dato real (decenas de velas), es el centrado. Si cae
mucho mas tarde -o no aparece en el rango accesible-, sobrevive como
resultado real.

Generador: s_t con memoria larga (signo de fGn) y m_t acoplada a un promedio
movil de |s| con ventana LARGA (>> tau_max), de modo que la covarianza
poblacional signo-tamano decae suavemente sin nunca cambiar de signo dentro
del rango de tau explorado.
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

from keldysh_finance.fano_validation import prepare_complete_weeks
from keldysh_finance.wavelets import fgn

ACTIVOS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX"]
TAU_MAX = 200


def c_sm(eps: np.ndarray, tau_max: int) -> np.ndarray:
    s, m = np.sign(eps), np.abs(eps)
    mb2 = float(m.mean()) ** 2
    out = np.empty(tau_max)
    for t in range(1, tau_max + 1):
        out[t - 1] = np.mean(s[:-t] * s[t:] * (m[:-t] * m[t:] - mb2))
    return out


def primer_cruce_sostenido(x: np.ndarray, k: int = 3):
    s0 = np.sign(x[0])
    for i in range(len(x) - k):
        if all(np.sign(x[i + j]) == -s0 and x[i + j] != 0 for j in range(k)):
            return int(i + 1)
    return None


def sintetico_sin_reversion(n: int, kappa: float,
                            rng: np.random.Generator) -> pd.DataFrame:
    """s con memoria larga (H=0.75); m acoplada a una media movil ANCHA de
    |s|, ventana >> TAU_MAX, de modo que la cross-cov poblacional decae
    monótonamente y no tiene motivo estructural para cambiar de signo dentro
    del rango explorado.
    """
    z = fgn(n, H=0.75, seed=int(rng.integers(0, 2**31)))
    s = np.sign(z)
    s = np.where(s == 0, 1.0, s)
    ventana = 2000
    nucleo = np.ones(ventana) / ventana
    envolvente = np.convolve(np.abs(s), nucleo, mode="same")
    rng2 = np.random.default_rng(int(rng.integers(0, 2**31)))
    ruido = np.abs(rng2.standard_t(4, n))
    base = ruido / ruido.mean()
    m = base * (1.0 + kappa * envolvente)
    eps = s * m
    n_uso = (n // 168) * 168
    idx = pd.date_range("2022-01-03", periods=n_uso, freq="h", tz="UTC")
    trades = np.full(n_uso, 100.0)
    volume = np.abs(eps[:n_uso]) + 1.0
    tbbase = (volume + eps[:n_uso]) / 2.0
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                       "Volume": volume, "tbBase": tbbase, "trades": trades},
                      index=idx)
    return df


def main() -> None:
    print("=" * 84)
    print("  EXP 23 - control del cruce forzado (propuesto por David)")
    print("=" * 84)

    print("\n  [1] EL DATO REAL: donde cruza C_sm, en el mismo pipeline")
    print("  " + "-" * 78)
    reales = {}
    for a in ACTIVOS:
        p = RAIZ / "output" / "cache" / f"flow_{a}USDT_1h_4.0y.csv"
        df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        prep = prepare_complete_weeks(df.dropna(), flow_mode="raw")
        eps = prep.flow_demeaned
        c = c_sm(eps, TAU_MAX)
        t = primer_cruce_sostenido(c)
        n_semanas = prep.n_complete_weeks
        reales[a] = {"n_candles": int(eps.size), "n_semanas": int(n_semanas),
                    "tau_cruce": t}
        if t:
            print("    %s: %d velas, %d semanas -> cruce en tau=%d" %
                  (a, eps.size, n_semanas, t))
        else:
            print("    %s: sin cruce" % a)

    print("\n  [2] EL CONTROL: sintetico SIN reversion poblacional,")
    print("      mismo pipeline de desmediado estratificado")
    print("  " + "-" * 78)
    rng = np.random.default_rng(23)
    n_candles = int(np.median([r["n_candles"] for r in reales.values()]))
    print("    n = %d velas (mediana de los 8 activos)" % n_candles)

    cruces_ctrl = []
    for kappa in (1.0, 3.0, 8.0):
        print("\n    kappa=%.1f:" % kappa)
        for rep in range(8):
            df = sintetico_sin_reversion(n_candles, kappa=kappa, rng=rng)
            prep = prepare_complete_weeks(df, flow_mode="raw")
            eps = prep.flow_demeaned
            c = c_sm(eps, TAU_MAX)
            t = primer_cruce_sostenido(c)
            cruces_ctrl.append(t)
            if t:
                print("      rep %d: cruce en tau=%d" % (rep, t))
            else:
                print("      rep %d: SIN cruce en [1,%d] -- sobrevive" % (rep, TAU_MAX))

    print("\n  " + "=" * 78)
    print("  VEREDICTO")
    print("  " + "-" * 78)
    con_cruce = [t for t in cruces_ctrl if t is not None]
    tau_real = [r["tau_cruce"] for r in reales.values() if r["tau_cruce"]]
    n_ctrl = len(cruces_ctrl)
    print("  dato real:     tau* en %d-%d velas (%d/%d cruzan, 8 activos)" %
          (min(tau_real), max(tau_real), len(tau_real), len(reales)))
    if con_cruce:
        print("  control (sin reversion poblacional): %d/%d repeticiones cruzan, en tau %d-%d" %
              (len(con_cruce), n_ctrl, min(con_cruce), max(con_cruce)))
    else:
        print("  control (sin reversion poblacional): 0/%d repeticiones cruzan" % n_ctrl)

    if len(con_cruce) >= n_ctrl // 2:
        solapa = any(min(tau_real) <= t <= max(tau_real) for t in con_cruce)
        print()
        if solapa:
            print("  EL CONTROL CRUZA EN LA MISMA ESCALA QUE EL DATO REAL.")
            print("  La objecion de David SOBREVIVE: el cruce de tau*=23-166 no")
            print("  se puede distinguir de un artefacto del desmediado por")
            print("  estrato. La seccion del mecanismo necesita esta salvedad")
            print("  como caveat central, no como nota al pie.")
        else:
            print("  El control SI cruza, pero en otra escala que el dato real.")
            print("  Hay artefacto de desmediado, pero no explica el tau*")
            print("  observado por si solo.")
    else:
        print()
        print("  El control NO cruza (o cruza raramente) en el rango accesible")
        print("  pese a tener kappa alto y el MISMO desmediado estratificado.")
        print("  El cruce real SOBREVIVE al control: no es artefacto puro del")
        print("  centrado. Puede reportarse, con esta comprobacion citada.")

    out = RAIZ / "output" / "exp23_control_cruce_forzado.json"

    def _l(o):
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return str(o)

    out.write_text(json.dumps(
        {"reales": reales, "control_kappa": 3.0, "control_cruces": cruces_ctrl},
        indent=2, default=_l), encoding="utf-8")
    print("\n  log: %s" % out)


if __name__ == "__main__":
    main()
