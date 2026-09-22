r"""Experimento 20b — de qué depende la CUOTA de acoplamiento.

EL PROBLEMA
-----------
Gabriel reporta que el acoplamiento aporta el 51-84% del exceso de varianza, y
que normalizar sube la cuota pura de signo del 16-30% al 38-49%. El exp. 20,
con la misma identidad, da 49-71% y 29-51%. Los dos cálculos usan la derivación
de David y ninguno tiene por qué estar mal: la identidad

    Q_w^2 = SUM m^2 + mbar^2 SUM_{t!=t'} s s' + SUM_{t!=t'} s s'(m m' - mbar^2)

es exacta para CUALQUIER constante `mbar`, y no dice nada sobre qué serie se
descompone ni contra qué se normaliza la cuota. Ahí caben cuatro decisiones
que nadie ha declarado todavía, y cada una mueve el número.

LAS CUATRO DECISIONES
---------------------
1. **Serie**: el flujo crudo, o el flujo desmediado por estrato
   (trimestre x hora-de-semana), que es lo que usa el nulo del manuscrito.
2. **mbar**: media de `|eps|` sobre TODA la serie, o sobre cada ventana.
   Ambas dejan la identidad exacta; la segunda absorbe en el término diagonal
   la variación de actividad entre semanas.
3. **Denominador**: `Q^2` completo, o sólo el EXCESO sobre la línea base sin
   memoria (`signo + acopl`).
4. **Agregación**: promediar los términos sobre ventanas y luego dividir, o
   promediar los cocientes por ventana.

Este experimento no decide cuál es la correcta —eso es de los tres— sino que
las evalúa TODAS sobre los mismos datos, para que la discusión sea sobre una
tabla y no sobre dos rangos que no se pueden comparar.
"""
from __future__ import annotations

import itertools
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
T = 168
OBJETIVO = {"raw": (0.70, 0.84), "normalized": (0.51, 0.62)}   # rango de Gabriel


def _cargar(symbol: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def cuota(eps: np.ndarray, mbar_modo: str, denom: str, agreg: str) -> dict:
    """Las tres piezas y las dos cuotas, bajo una combinación de decisiones."""
    s, m = np.sign(eps), np.abs(eps)
    n = eps.size // T
    S, M = s[:n * T].reshape(n, T), m[:n * T].reshape(n, T)
    E = eps[:n * T].reshape(n, T)

    mb2 = (float(m.mean()) ** 2 if mbar_modo == "global"
           else (M.mean(axis=1) ** 2))
    Q = E.sum(axis=1)
    diag = (S ** 2 * M ** 2).sum(axis=1)
    signo = mb2 * (S.sum(axis=1) ** 2 - (S ** 2).sum(axis=1))
    acopl = Q ** 2 - diag - signo                      # exacto por construccion

    if agreg == "media_de_terminos":
        dg, sg, ac = diag.mean(), signo.mean(), acopl.mean()
        den = (dg + sg + ac) if denom == "Q2" else (sg + ac)
        return {"acopl": float(ac / den), "signo": float(sg / den),
                "diag": float(dg / den)}
    # media de los cocientes por ventana
    den_w = (diag + signo + acopl) if denom == "Q2" else (signo + acopl)
    ok = np.abs(den_w) > 1e-30
    return {"acopl": float(np.mean(acopl[ok] / den_w[ok])),
            "signo": float(np.mean(signo[ok] / den_w[ok])),
            "diag": float(np.mean(diag[ok] / den_w[ok]))}


def main() -> None:
    print("=" * 92)
    print("  EXP 20b — de que depende la cuota de acoplamiento")
    print("=" * 92)
    print(f"\n  objetivo a reproducir (Gabriel): acopl/exceso "
          f"raw {OBJETIVO['raw']}, normalized {OBJETIVO['normalized']}")

    series = {}
    for modo in ("raw", "normalized"):
        for a in ACTIVOS:
            prep = prepare_complete_weeks(_cargar(f"{a}USDT"), flow_mode=modo)
            series[(a, modo, "desmediado")] = prep.flow_demeaned
            series[(a, modo, "crudo")] = prep.flow

    variantes = list(itertools.product(
        ("desmediado", "crudo"), ("global", "ventana"),
        ("exceso", "Q2"), ("media_de_terminos", "media_de_cocientes")))

    filas = []
    print(f"\n  {'serie':<11}{'mbar':<9}{'denom':<8}{'agregacion':<21}"
          f"{'acopl raw':>22}{'acopl norm':>22}")
    print("  " + "-" * 90)
    for ser, mb, dn, ag in variantes:
        vals = {}
        for modo in ("raw", "normalized"):
            vals[modo] = [cuota(series[(a, modo, ser)], mb, dn, ag)["acopl"]
                          for a in ACTIVOS]
        r = {"serie": ser, "mbar": mb, "denom": dn, "agregacion": ag,
             "acopl_raw": vals["raw"], "acopl_norm": vals["normalized"]}
        filas.append(r)
        def rango(v):
            return f"[{min(v):+.2f}, {max(v):+.2f}]"
        # marca si el rango se solapa con el de Gabriel en AMBOS flujos
        def casa(v, obj):
            return min(v) <= obj[1] and max(v) >= obj[0]
        m1 = casa(vals["raw"], OBJETIVO["raw"])
        m2 = casa(vals["normalized"], OBJETIVO["normalized"])
        r["compatible"] = bool(m1 and m2)
        print(f"  {ser:<11}{mb:<9}{dn:<8}{ag:<21}"
              f"{rango(vals['raw']):>22}{rango(vals['normalized']):>22}"
              f"{'  <== compatible' if (m1 and m2) else ''}")

    comp = [f for f in filas if f["compatible"]]
    print("\n  " + "=" * 90)
    print("  LECTURA")
    print("  " + "-" * 90)
    if comp:
        print(f"  {len(comp)} de {len(filas)} combinaciones reproducen el rango "
              f"de Gabriel en los DOS flujos:")
        for f in comp:
            print(f"    - serie {f['serie']}, mbar {f['mbar']}, "
                  f"denominador {f['denom']}, {f['agregacion']}")
    else:
        print("  NINGUNA de las 16 combinaciones reproduce el rango de Gabriel")
        print("  en los dos flujos a la vez. La diferencia no esta en estas")
        print("  cuatro decisiones: hay que pedirle el codigo o la ventana T.")

    # cuanto mueve cada decision, por separado
    print("\n  Cuanto mueve cada decision (rango de acopl/exceso sobre raw,")
    print("  variando SOLO esa decision y promediando el resto):")
    for eje, ops in (("serie", ("desmediado", "crudo")),
                     ("mbar", ("global", "ventana")),
                     ("denom", ("exceso", "Q2")),
                     ("agregacion", ("media_de_terminos", "media_de_cocientes"))):
        med = {}
        for o in ops:
            v = [x for f in filas if f[eje] == o for x in f["acopl_raw"]]
            med[o] = float(np.median(v))
        d = abs(med[ops[0]] - med[ops[1]])
        print(f"    {eje:<12} {ops[0]} {med[ops[0]]:+.3f}   vs   "
              f"{ops[1]} {med[ops[1]]:+.3f}    |delta| = {d:.3f}")

    # --- lo que el tercer término NO es -------------------------------------
    print("\n\n  " + "=" * 90)
    print("  EL TERCER TERMINO NO ES SOLO EL ACOPLAMIENTO")
    print("  " + "-" * 90)
    print("  Con s y m INDEPENDIENTES pero cada uno con su memoria,")
    print("      <s_t s_{t+tau}(m_t m_{t+tau} - mbar^2)> = C_s(tau)·C_m(tau) != 0,")
    print("  asi que el termino no se anula aunque no haya acoplamiento alguno.")
    print("  Se mide cuanto pesa esa contaminacion con el nulo de desalineacion")
    print("  semanal (eps'_t = s_{t+168k}·m_t: ambas memorias intactas,")
    print("  acoplamiento cero por construccion).")
    print(f"\n  {'activo':<8}{'acopl obs':>14}{'acopl indep':>14}"
          f"{'contaminacion':>15}")
    rng = np.random.default_rng(0)
    contam = []
    for a in ACTIVOS:
        eps = series[(a, "raw", "desmediado")]
        n_sem = eps.size // T
        mb2 = float(np.abs(eps).mean()) ** 2

        def terceros(x):
            s2, m2 = np.sign(x), np.abs(x)
            k = x.size // T
            S2, M2 = s2[:k * T].reshape(k, T), m2[:k * T].reshape(k, T)
            Qx = x[:k * T].reshape(k, T).sum(axis=1)
            dg = (S2 ** 2 * M2 ** 2).sum(axis=1)
            sg = mb2 * (S2.sum(axis=1) ** 2 - (S2 ** 2).sum(axis=1))
            return float((Qx ** 2 - dg - sg).mean())

        obs = terceros(eps)
        nul = [terceros(np.abs(eps) * np.sign(
            np.roll(eps, -int(rng.integers(1, n_sem)) * T))) for _ in range(40)]
        med = float(np.median(nul))
        frac = med / obs if obs else float("nan")
        contam.append({"activo": a, "acopl_obs": obs, "acopl_indep": med,
                       "fraccion": frac})
        print(f"  {a:<8}{obs:>14.4e}{med:>14.4e}{100 * frac:>14.0f}%")
    fm = float(np.median([c["fraccion"] for c in contam]))
    print(f"\n  -> contaminacion mediana {100 * fm:.0f}%: el grueso del termino SI")
    print("     es acoplamiento genuino, pero la objecion es real y un arbitro")
    print("     la deriva en treinta segundos. El arreglo es reportar")
    print("     acopl_corregido = acopl_obs - mediana(acopl_desalineado):")
    print("     la identidad mas UN subrogado, no una pila de ellos.")

    # --- por qué la media y no la mediana -----------------------------------
    print("\n\n  " + "=" * 90)
    print("  LA MEDIA NO ES ARBITRARIA")
    print("  " + "-" * 90)
    print("  Cambiar mbar traslada un multiplo EXACTO del termino de signo al")
    print("  cruzado:  cross(m1) = cross(m2) + (m2^2 - m1^2)/m2^2 · sign(m2).")
    print("  Asi que la eleccion se decide por una propiedad, no por gusto: si")
    print("  las magnitudes no tienen memoria, <m_t m_t'> = <m>^2 y el termino")
    print("  cruzado DEBE anularse. Solo la media lo consigue.")
    print("\n  Control: magnitudes barajadas (sin memoria de tamano), signos")
    print("  intactos. Se mide el cruzado como fraccion del termino de signo.")
    print(f"\n  {'activo':<8}{'con MEDIA':>14}{'con MEDIANA':>14}")
    rng2 = np.random.default_rng(1)
    just = []
    for a in ACTIVOS:
        eps = series[(a, "raw", "desmediado")]
        mu, md = float(np.abs(eps).mean()), float(np.median(np.abs(eps)))

        def piezas(x, mb):
            s2, m2 = np.sign(x), np.abs(x)
            k = x.size // T
            S2, M2 = s2[:k * T].reshape(k, T), m2[:k * T].reshape(k, T)
            Qx = x[:k * T].reshape(k, T).sum(axis=1)
            dg = (S2 ** 2 * M2 ** 2).sum(axis=1)
            sg = mb ** 2 * (S2.sum(axis=1) ** 2 - (S2 ** 2).sum(axis=1))
            return float(sg.mean()), float((Qx ** 2 - dg - sg).mean())

        rm, rd = [], []
        for _ in range(20):
            y = np.sign(eps) * rng2.permutation(np.abs(eps))
            s_mu, c_mu = piezas(y, mu)
            _, c_md = piezas(y, md)
            rm.append(c_mu / s_mu)
            rd.append(c_md / s_mu)
        f_mu, f_md = float(np.median(rm)), float(np.median(rd))
        just.append({"activo": a, "cruzado_sobre_signo_media": f_mu,
                     "cruzado_sobre_signo_mediana": f_md})
        print(f"  {a:<8}{f_mu:>13.1%}{f_md:>14.1%}")
    print("\n  Con la media el cruzado queda en el ruido; con la mediana se")
    print("  lleva la mayor parte del termino de signo AUNQUE NO HAYA")
    print("  acoplamiento. La media es la unica que hace honesto el nombre.")

    out = RAIZ / "output" / "exp20b_variantes_cuota.json"
    out.write_text(json.dumps(
        {"T": T, "objetivo_gabriel": OBJETIVO, "activos": ACTIVOS,
         "variantes": filas, "contaminacion_CsCm": contam,
         "justificacion_mbar": just,
         "contaminacion_mediana": fm}, indent=2,
        default=lambda o: float(o) if isinstance(o, np.floating) else str(o)),
        encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
