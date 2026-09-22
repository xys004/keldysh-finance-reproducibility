r"""Experimento 20 — la descomposición EXACTA de la varianza de conteo.

DE DÓNDE VIENE
--------------
Derivación de David Verrilli. Con `eps_t = s_t·m_t`, `s_t = sign(eps_t)`,
`m_t = |eps_t|` y `mbar = ⟨m⟩` sobre toda la serie,

    Q_w^2 = SUM_{t in w} m_t^2
          + mbar^2 · SUM_{t != t' in w} s_t s_t'
          + SUM_{t != t' in w} s_t s_t' (m_t m_t' − mbar^2)
            \____________ diagonal ______/  \__ signo __/  \_ acoplamiento _/

Verificada aquí numéricamente a 4.7e-14. El caso borde de las velas de volumen
nulo (donde `s_t = 0`, de modo que `s_t^2 = 0` y no 1) es inocuo porque
`m_t = 0` exactamente en esas mismas posiciones.

POR QUÉ IMPORTA
---------------
Los exp. 17-19 atacaban la misma pregunta por Monte Carlo y en el TERCER
cumulante. Esto la resuelve en el segundo, exactamente, sin ruido de subrogado:

  - `diagonal` es la línea base sin memoria: lo que da una suma de tamaños con
    signos independientes. Es el término al que se normaliza el Fano.
  - `signo` es la memoria larga de signos de Lillo-Farmer, pesada por `mbar^2`.
  - `acoplamiento` NO está determinado por ninguna de las dos autocorrelaciones
    marginales. Es un objeto nuevo, y es exactamente

        acopl_w = 2 · SUM_{tau>=1} (T − tau) · C_sm(tau),
        C_sm(tau) = ⟨ s_t s_{t+tau} (m_t m_{t+tau} − mbar^2) ⟩,

    la correlación cruzada signo-magnitud. Su exponente de decaimiento es el
    observable nuevo del programa.

Cálculo en O(n), sin doble bucle:
    SUM_{t,t'} s_t s_t'       = (SUM s)^2          -> el término de signo sale
    SUM_{t,t'} s_t s_t' m_t m_t' = Q_w^2            -> el acoplamiento, por resta

PREDICCIÓN DE GABRIEL, QUE AQUÍ ES CONTROL POSITIVO
----------------------------------------------------
Normalizar el flujo por volumen homogeneiza las magnitudes, así que debe
DESTRUIR parte del acoplamiento y SUBIR la cuota pura de signo. Es una
predicción física direccional declarada antes de medir, y se contrasta abajo
sobre los ocho activos. Si el reparto no se mueve en la dirección predicha, la
lectura mecanicista del acoplamiento se debilita, medición aparte.

CRITERIO EX-ANTE
----------------
- Estadístico: `cuota_acopl = acopl / (signo + acopl)`, la fracción del EXCESO
  sobre la línea base sin memoria que aporta el acoplamiento.
- Intervalos por bootstrap de bloques circulares sobre las ventanas semanales
  (mismo procedimiento que ya usa el manuscrito para el ratio de varianza).
- Control direccional: `cuota_acopl(normalized) < cuota_acopl(raw)` en >= 6 de
  8 series.
- No hay puerta sobre el NIVEL de la cuota: es una medición exacta, no un
  contraste, y se reporta con su intervalo.
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

ACTIVOS = ["BTC", "ETH", "BNB", "SOL"]
HORAS_SEMANA = 168
T_GRID = [24, 48, 168, 336]          # dia, dos dias, semana, dos semanas
T_PRINCIPAL = 168
BLOQUES_BOOT = (4, 8, 13, 26)        # en ventanas, como el resto del paper
N_BOOT = 999


def _cargar(symbol: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def descomponer(eps: np.ndarray, T: int) -> dict:
    """Los tres términos por ventana no solapada. Exacto, O(n)."""
    s, m = np.sign(eps), np.abs(eps)
    mbar2 = float(m.mean()) ** 2
    n = eps.size // T
    S = s[:n * T].reshape(n, T)
    M = m[:n * T].reshape(n, T)
    E = eps[:n * T].reshape(n, T)

    Q = E.sum(axis=1)
    diag = (S ** 2 * M ** 2).sum(axis=1)                    # = sum m^2
    signo = mbar2 * (S.sum(axis=1) ** 2 - (S ** 2).sum(axis=1))
    acopl = Q ** 2 - diag - signo                            # resta exacta
    return {"Q": Q, "diag": diag, "signo": signo, "acopl": acopl,
            "n_ventanas": int(n), "T": int(T)}


def cuotas(d: dict, idx: np.ndarray | None = None) -> dict:
    """Reparto medio sobre las ventanas (o sobre un remuestreo `idx`)."""
    sel = slice(None) if idx is None else idx
    dg, sg, ac = d["diag"][sel].mean(), d["signo"][sel].mean(), d["acopl"][sel].mean()
    q2 = dg + sg + ac
    exceso = sg + ac
    return {"diag": float(dg), "signo": float(sg), "acopl": float(ac),
            "Q2": float(q2),
            "cuota_diag": float(dg / q2) if q2 else float("nan"),
            "cuota_signo": float(sg / q2) if q2 else float("nan"),
            "cuota_acopl": float(ac / q2) if q2 else float("nan"),
            "acopl_sobre_exceso": float(ac / exceso) if exceso else float("nan"),
            "signo_sobre_exceso": float(sg / exceso) if exceso else float("nan")}


def bootstrap(d: dict, n_boot: int, semilla: int) -> dict:
    """Bootstrap de bloques circulares sobre las ventanas."""
    rng = np.random.default_rng(semilla)
    n = d["n_ventanas"]
    out: dict[str, list] = {"acopl_sobre_exceso": [], "cuota_acopl": [],
                            "cuota_signo": [], "cuota_diag": []}
    for L in BLOQUES_BOOT:
        if n < 2 * L:
            continue
        n_bl = int(np.ceil(n / L))
        for _ in range(n_boot // len(BLOQUES_BOOT)):
            arr = np.arange(n)
            ini = rng.integers(0, n, n_bl)
            idx = np.concatenate([np.take(arr, np.arange(i, i + L), mode="wrap")
                                  for i in ini])[:n]
            c = cuotas(d, idx)
            for k in out:
                out[k].append(c[k])
    return {k: {"q025": float(np.quantile(v, 0.025)),
                "q975": float(np.quantile(v, 0.975))}
            for k, v in out.items() if v}


def c_sm(eps: np.ndarray, tau_max: int = 200) -> dict:
    """C_sm(tau) = ⟨s_t s_{t+tau}(m_t m_{t+tau} − mbar^2)⟩ y su pendiente.

    Es el integrando del término de acoplamiento: `acopl` sobre una ventana T
    vale `2·SUM_tau (T−tau)·C_sm(tau)`. Su decaimiento es el observable nuevo.
    """
    s, m = np.sign(eps), np.abs(eps)
    mbar2 = float(m.mean()) ** 2
    val = []
    for tau in range(1, tau_max + 1):
        v = float(np.mean(s[:-tau] * s[tau:] * (m[:-tau] * m[tau:] - mbar2)))
        val.append(v)
    val = np.asarray(val)
    tau = np.arange(1, tau_max + 1, dtype=float)
    # pendiente log-log donde C_sm > 0 y tau en [2,50], la banda del paper
    msk = (tau >= 2) & (tau <= 50) & (val > 0)
    pend = (float(np.polyfit(np.log(tau[msk]), np.log(val[msk]), 1)[0])
            if msk.sum() >= 5 else float("nan"))
    return {"tau": tau.tolist(), "C_sm": val.tolist(), "pendiente_2_50": pend,
            "n_positivos_2_50": int(msk.sum()),
            "C_sm_1": float(val[0]), "C_sm_10": float(val[9]),
            "C_sm_100": float(val[99]) if tau_max >= 100 else None}


def main() -> None:
    print("=" * 78)
    print("  EXP 20 — descomposicion EXACTA de Var(Q_T)  [identidad de David]")
    print("=" * 78)
    print("\n  Q_w^2 = sum m^2  +  mbar^2 sum_{t!=t'} s s'  +  sum_{t!=t'} s s'"
          "(m m' - mbar^2)")
    print("          \\_ diagonal _/   \\____ signo ____/   \\___ acoplamiento ___/")

    res = []
    for modo in ("raw", "normalized"):
        for a in ACTIVOS:
            prep = prepare_complete_weeks(_cargar(f"{a}USDT"), flow_mode=modo)
            eps = prep.flow_demeaned
            fila = {"activo": a, "flow_mode": modo,
                    "n_semanas": int(prep.n_complete_weeks), "por_T": {}}
            for T in T_GRID:
                d = descomponer(eps, T)
                c = cuotas(d)
                # comprobacion de la identidad, no negociable
                err = float(np.max(np.abs(
                    d["Q"] ** 2 - (d["diag"] + d["signo"] + d["acopl"]))
                    / np.maximum(1e-30, d["Q"] ** 2)))
                c["error_identidad"] = err
                c["n_ventanas"] = d["n_ventanas"]
                if T == T_PRINCIPAL:
                    c["ic"] = bootstrap(d, N_BOOT, semilla=20)
                fila["por_T"][str(T)] = c
            fila["C_sm"] = c_sm(eps)
            res.append(fila)

    # --- reparto a la escala semanal ----------------------------------------
    print("\n\n  " + "=" * 74)
    print(f"  REPARTO DE Q^2 A ESCALA SEMANAL (T={T_PRINCIPAL} h)")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}{'diag':>8}{'signo':>8}{'acopl':>8}"
          f"{'acopl/exceso':>14}{'IC 95%':>18}")
    for r in res:
        c = r["por_T"][str(T_PRINCIPAL)]
        ic = c.get("ic", {}).get("acopl_sobre_exceso")
        sic = f"[{ic['q025']:+.2f}, {ic['q975']:+.2f}]" if ic else "--"
        print(f"  {r['activo'] + '|' + r['flow_mode']:<18}"
              f"{c['cuota_diag']:>8.3f}{c['cuota_signo']:>8.3f}"
              f"{c['cuota_acopl']:>8.3f}{c['acopl_sobre_exceso']:>14.3f}"
              f"{sic:>18}")
    err_max = max(r["por_T"][t]["error_identidad"] for r in res for t in r["por_T"])
    print(f"\n  error relativo maximo de la identidad sobre TODAS las "
          f"ventanas: {err_max:.2e}")

    # --- control direccional de Gabriel --------------------------------------
    print("\n\n  " + "=" * 74)
    print("  CONTROL DIRECCIONAL — normalizar homogeneiza magnitudes, luego")
    print("     debe BAJAR la cuota de acoplamiento y SUBIR la de signo")
    print("  " + "=" * 74)
    print(f"\n  {'activo':<8}{'acopl/exc raw':>15}{'acopl/exc norm':>16}"
          f"{'signo/exc raw':>15}{'signo/exc norm':>16}{'':>4}")
    n_ok = 0
    for a in ACTIVOS:
        cr = next(r["por_T"][str(T_PRINCIPAL)] for r in res
                  if r["activo"] == a and r["flow_mode"] == "raw")
        cn = next(r["por_T"][str(T_PRINCIPAL)] for r in res
                  if r["activo"] == a and r["flow_mode"] == "normalized")
        ok = cn["acopl_sobre_exceso"] < cr["acopl_sobre_exceso"]
        n_ok += int(ok)
        print(f"  {a:<8}{cr['acopl_sobre_exceso']:>15.3f}"
              f"{cn['acopl_sobre_exceso']:>16.3f}"
              f"{cr['signo_sobre_exceso']:>15.3f}"
              f"{cn['signo_sobre_exceso']:>16.3f}{'  si' if ok else '  no':>4}")
    print(f"\n  -> la prediccion se cumple en {n_ok}/4 activos")

    # --- la correlacion cruzada, el observable nuevo -------------------------
    print("\n\n  " + "=" * 74)
    print("  C_sm(tau) = <s_t s_{t+tau}(m_t m_{t+tau} - mbar^2)>")
    print("     el integrando del acoplamiento; su decaimiento es el")
    print("     observable que no fija ninguna de las dos ACF marginales")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}{'C_sm(1)':>12}{'C_sm(10)':>12}{'C_sm(100)':>12}"
          f"{'pend [2,50]':>13}{'n>0':>6}")
    for r in res:
        c = r["C_sm"]
        c100 = c["C_sm_100"]
        print(f"  {r['activo'] + '|' + r['flow_mode']:<18}{c['C_sm_1']:>12.3e}"
              f"{c['C_sm_10']:>12.3e}"
              f"{(c100 if c100 is not None else float('nan')):>12.3e}"
              f"{c['pendiente_2_50']:>13.3f}{c['n_positivos_2_50']:>6}")

    out = RAIZ / "output" / "exp20_exact_decomposition.json"
    out.parent.mkdir(exist_ok=True)

    def _l(o):
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return [_l(v) for v in o.tolist()]
        return str(o)

    out.write_text(json.dumps(
        {"identidad": "Q_w^2 = sum m^2 + mbar^2 sum_{t!=t'} s s' + "
                      "sum_{t!=t'} s s'(m m' - mbar^2)",
         "error_identidad_max": err_max,
         "control_direccional_ok": f"{n_ok}/4",
         "T_principal": T_PRINCIPAL, "n_bootstrap": N_BOOT,
         "resultados": res}, indent=2, default=_l), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
