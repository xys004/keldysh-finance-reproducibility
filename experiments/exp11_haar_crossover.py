r"""Experimento 11 — control de deriva sobre el cruce del exponente de Var(Q_T).

QUÉ ESTÁ EN JUEGO
-----------------
La derivación MSRJD produce una predicción falsable y que cruza familias de
observables: para `C_eps(tau) ~ tau^-a` se tiene `Var(Q_T) ~ T^(2-a)`, de modo
que el kernel bicomponente OBLIGA a que la pendiente LOCAL derive de `2-a_f` a
`2-a_s`. El exp. 06 la verifica: sube monótona de ~1.22 a ~1.5 en los cuatro
activos, dentro de la banda que fijan los exponentes medidos en el sector de
correlación. Es el resultado predictivo del Paper 2.

Y tiene un flanco. `Var(Q_T)` se estima sobre BLOQUES no solapados de longitud
T. Si la media por bloque `⟨Q_T⟩` deriva, esa deriva entra en la varianza
muestral. Y sabemos que deriva: el exp. 06 mide sesgo vendedor persistente en
BTC/ETH/SOL a lo largo de cuatro años, y el exp. 08 mide modulación con la
fase diaria en los cuatro activos. Una media que se mueve de forma
determinista aporta a la varianza un término que escala como `T²` — pendiente
2, justo por encima del 1.5 observado y en la misma dirección. Un árbitro lo
preguntará, y con razón.

EL CONTROL
----------
La diferencia de Haar entre dos bloques adyacentes,

    d_T(k) = Q_T(bloque 2) − Q_T(bloque 1),

cancela EXACTAMENTE cualquier media constante, y satisface la identidad

    Var(d_T) = 4·Var(Q_T) − Var(Q_2T),

que para una potencia pura `Var(Q_T) = A·T^nu` da `Var(d_T) = A(4−2^nu)T^nu`:
el MISMO exponente. Es decir, `d_T` mide `nu` sin tocar la predicción, y es
ciego a la parte de la señal que preocupa.

Subiendo el número de momentos nulos se sube el grado del polinomio anulado:
`db1` (Haar) mata una media constante, `db2` una deriva lineal, `db3` una
cuadrática. Si los cuatro estimadores dan el mismo `nu_loc`, ninguna deriva
polinómica de la media está inflando el cruce.

Relación de exponentes: el diagrama log-escala ortonormal tiene pendiente
`alpha = nu − 1`, porque el detalle ortonormal lleva el factor `1/sqrt(2T)`.
Se convierte con `nu = alpha + 1` y se compara con `2 − a` directamente.

CONTROL POSITIVO (sin él esto no vale)
--------------------------------------
Un acuerdo entre estimadores no demuestra nada si el estimador de Haar tampoco
sabría VER la contaminación. Así que se inyecta la contaminación a mano: fGn
con `nu` conocido más una rampa determinista en el flujo, y se exige que
`Var(Q_T)` se infle y que Haar NO. Es el criterio que decide si el control
tiene sensibilidad, y se declara antes de mirar los datos reales.
"""
from __future__ import annotations

import json
import os
import sys
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

from keldysh_finance.flow import fetch_klines_with_flow, order_flow_imbalance
from keldysh_finance.wavelets import diagrama_log_escala, fgn

ACTIVOS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
OCTAVAS = [3, 4, 5, 6, 7, 8]          # T de 8 a 256 velas, la banda del cruce
LONGITUDES = {1: 2, 2: 4, 3: 6}       # momentos nulos -> longitud del filtro


def nu_por_bloques(x: np.ndarray, octavas: list[int]) -> dict:
    """`nu_loc` por la vía del exp. 06: varianza de Q_T sobre bloques."""
    T_list = [2 ** j for j in octavas]
    var = []
    for T in T_list:
        n = x.size // T
        q = x[:n * T].reshape(n, T).sum(axis=1)
        var.append(float(q.var(ddof=1)))
    lv, lT = np.log(np.array(var)), np.log(np.array(T_list, dtype=float))
    local = (np.diff(lv) / np.diff(lT)).tolist()
    return {"T": T_list, "var": var, "nu_local": local,
            "nu_global": float(np.polyfit(lT, lv, 1)[0])}


def nu_por_ondicula(x: np.ndarray, octavas: list[int],
                    momentos: int) -> dict:
    """`nu_loc` por la vía de ondícula: nu = alpha + 1, con alpha la pendiente
    local del diagrama log-escala. `momentos` momentos nulos."""
    d = diagrama_log_escala(x, longitud=LONGITUDES[momentos])
    j = np.asarray(d["j"], dtype=float)
    sel = np.isin(j, octavas)
    y, jj = np.asarray(d["y"])[sel], j[sel]
    if jj.size < 2:
        return {"error": "octavas insuficientes"}
    local = (np.diff(y) / np.diff(jj) + 1.0).tolist()
    return {"j": jj.astype(int).tolist(), "nu_local": local,
            "nu_global": float(np.polyfit(jj, y, 1)[0] + 1.0)}


def control_positivo() -> dict:
    """fGn de nu conocido, contaminado con una deriva determinista.

    Criterio ex-ante: el estimador por bloques debe INFLARSE al menos 0.20 y el
    de Haar debe moverse menos de 0.10. Si Haar también se infla, no sirve como
    control y el experimento no concluye.

    Nota sobre la amplitud. La primera versión inyectaba una rampa de amplitud
    0.60 sigma y movía el estimador por bloques 0.165 — por debajo del 0.20
    declarado, de modo que el control no llegaba a probarse EN la barra fijada.
    Lo arbitrario ahí no era el umbral sino la amplitud, que es un parámetro
    libre del control y no una propiedad del dato. Se barre la amplitud hasta
    cruzar la barra declarada y se lee Haar allí; el umbral no se toca.
    """
    nu_true = 1.40                          # nu = 2 - a  =>  a = 0.60
    n = 200_000
    x = fgn(n, H=1.0 - (2.0 - nu_true) / 2.0, seed=7)
    x = x / x.std()
    t = np.arange(n, dtype=float) / n

    def medir(y: np.ndarray) -> dict:
        fila = {"bloques": nu_por_bloques(y, OCTAVAS)["nu_global"]}
        for m in (1, 2, 3):
            fila[f"db{m}"] = nu_por_ondicula(y, OCTAVAS, m)["nu_global"]
        return fila

    limpio = medir(x)
    out = {"nu_true": nu_true, "casos": [dict(caso="limpio", amplitud=0.0,
                                              **limpio)]}
    # barrido de amplitud hasta cruzar el 0.20 declarado
    for amp in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0):
        fila = medir(x + amp * t)
        fila.update(caso="rampa lineal", amplitud=amp)
        out["casos"].append(fila)
        if abs(fila["bloques"] - limpio["bloques"]) >= 0.20:
            break
    # y una cuadrática a la misma amplitud, que db1 no anula por momentos
    amp = out["casos"][-1]["amplitud"]
    fila = medir(x + amp * t ** 2)
    fila.update(caso="rampa cuadratica", amplitud=amp)
    out["casos"].append(fila)
    return out


def deriva_lenta(x: np.ndarray, octava: int = 10) -> np.ndarray:
    """La componente lenta de `x`: media móvil centrada de `2^octava` velas.

    Es exactamente la magnitud que preocupa — la media local de `eps`, cuya
    integral sobre un bloque es `⟨Q_T⟩` — y sirve para hacer la pregunta
    cuantitativa en vez de la cualitativa: no "¿podría una deriva inflar el
    cruce?" sino "¿lo infla LA QUE HAY?".
    """
    w = 2 ** octava
    nucleo = np.ones(w) / w
    return np.convolve(x, nucleo, mode="same")


def main() -> None:
    print("=" * 78)
    print("  EXP 11 — control de deriva sobre el cruce del exponente")
    print("=" * 78)

    # --- control positivo, ANTES de los datos reales ------------------------
    cp = control_positivo()
    print(f"\n  CONTROL POSITIVO — fGn con nu = {cp['nu_true']:.2f} mas deriva "
          f"inyectada")
    print("  " + "-" * 74)
    print(f"  {'contaminacion':<22}{'amp':>6}{'bloques':>10}{'db1(Haar)':>11}"
          f"{'db2':>9}{'db3':>9}")
    for f in cp["casos"]:
        print(f"  {f['caso']:<22}{f['amplitud']:>6.1f}{f['bloques']:>10.3f}"
              f"{f['db1']:>11.3f}{f['db2']:>9.3f}{f['db3']:>9.3f}")

    limpio = next(f for f in cp["casos"] if f["caso"] == "limpio")
    peor_b = max(abs(f["bloques"] - limpio["bloques"]) for f in cp["casos"])
    peor_h = max(abs(f["db1"] - limpio["db1"]) for f in cp["casos"])
    sensible = peor_b >= 0.20 and peor_h < 0.10
    print(f"\n    la deriva mueve el estimador por bloques hasta {peor_b:.3f}")
    print(f"    y el de Haar hasta {peor_h:.3f}")
    print(f"    -> el control TIENE sensibilidad: {'SI' if sensible else 'NO'}")
    if not sensible:
        print("\n    Sin sensibilidad demostrada no se puede leer nada de los")
        print("    datos reales. El experimento NO concluye.")
        return

    # --- datos reales --------------------------------------------------------
    print("\n\n  DATOS REALES (1h, 4 anos) — nu_loc por octava")
    print("  " + "-" * 74)
    reales = []
    for sym in ACTIVOS:
        df = fetch_klines_with_flow(sym, interval="1h", years=4.0).dropna()
        eps = order_flow_imbalance(df, normalize="volume")
        fila = {"activo": sym[:-4], "n": int(eps.size),
                "bloques": nu_por_bloques(eps, OCTAVAS)}
        for m in (1, 2, 3):
            fila[f"db{m}"] = nu_por_ondicula(eps, OCTAVAS, m)

        # la pregunta cuantitativa: la deriva que ESTE activo tiene, inyectada
        # sobre fGn de nu conocido, ¿cuanto infla el estimador por bloques?
        der = deriva_lenta(eps)
        base = fgn(eps.size, H=1.0 - (2.0 - 1.40) / 2.0, seed=11)
        base = base / base.std() * float(np.std(eps - der))
        fila["deriva_real"] = {
            "amplitud_rel": float(np.std(der) / np.std(eps)),
            "nu_limpio": nu_por_bloques(base, OCTAVAS)["nu_global"],
            "nu_con_deriva": nu_por_bloques(base + der, OCTAVAS)["nu_global"],
            "haar_con_deriva": nu_por_ondicula(base + der, OCTAVAS,
                                               1)["nu_global"]}
        reales.append(fila)

    Ts = [2 ** j for j in OCTAVAS]
    medios = [f"{int(np.sqrt(a * b))}" for a, b in zip(Ts[:-1], Ts[1:])]
    for fila in reales:
        print(f"\n  {fila['activo']}   (T en velas)")
        print("    " + "estimador".ljust(14)
              + "".join(f"{t:>9}" for t in medios) + f"{'global':>10}")
        for clave, etiq in (("bloques", "Var(Q_T)"), ("db1", "Haar (m=1)"),
                            ("db2", "db2 (m=2)"), ("db3", "db3 (m=3)")):
            d = fila[clave]
            linea = "    " + etiq.ljust(14)
            linea += "".join(f"{v:>9.2f}" for v in d["nu_local"])
            linea += f"{d['nu_global']:>10.2f}"
            print(linea)

    # --- veredicto -----------------------------------------------------------
    print("\n\n  " + "=" * 74)
    print("  VEREDICTO")
    print("  " + "-" * 74)
    print(f"  {'activo':<8}{'Var(Q_T)':>10}{'Haar':>9}{'db2':>9}{'db3':>9}"
          f"{'max desv.':>12}")
    veredicto = []
    for fila in reales:
        vals = [fila[k]["nu_global"] for k in ("bloques", "db1", "db2", "db3")]
        desv = max(abs(v - vals[0]) for v in vals[1:])
        veredicto.append({"activo": fila["activo"], "nu": vals,
                          "max_desviacion": float(desv)})
        print(f"  {fila['activo']:<8}" + "".join(f"{v:>9.2f}" for v in vals)
              + f"{desv:>12.3f}")

    print(f"\n  La deriva REAL de cada activo, inyectada sobre fGn de nu=1.40:")
    print(f"  {'activo':<8}{'amp/sigma':>11}{'nu limpio':>11}"
          f"{'nu+deriva':>11}{'Haar+deriva':>13}{'inflado':>10}")
    for fila in reales:
        dd = fila["deriva_real"]
        infl = dd["nu_con_deriva"] - dd["nu_limpio"]
        print(f"  {fila['activo']:<8}{dd['amplitud_rel']:>11.3f}"
              f"{dd['nu_limpio']:>11.3f}{dd['nu_con_deriva']:>11.3f}"
              f"{dd['haar_con_deriva']:>13.3f}{infl:>+10.3f}")
    infl_max = max(f["deriva_real"]["nu_con_deriva"]
                   - f["deriva_real"]["nu_limpio"] for f in reales)
    print(f"\n  Inflado maximo por la deriva realmente presente: "
          f"{infl_max:+.3f}")

    peor = max(v["max_desviacion"] for v in veredicto)
    print(f"  Maxima discrepancia entre los cuatro estimadores: {peor:.3f}")
    if peor < 0.20:
        print("  Por debajo de los 0.20 que la deriva inyectada produce en el")
        print("  control, asi que NINGUNA deriva polinomica de la media hasta")
        print("  grado 2 esta inflando el cruce. La prediccion 2-a_f -> 2-a_s")
        print("  se sostiene sobre un estimador ciego al sesgo.")
    else:
        print("  POR ENCIMA del efecto que la deriva produce en el control:")
        print("  hay que mirar activo por activo antes de sostener el cruce.")

    out = RAIZ / "output" / "exp11_haar_crossover.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"octavas": OCTAVAS, "control_positivo": cp, "sensible": bool(sensible),
         "reales": reales, "veredicto": veredicto,
         "max_discrepancia": float(peor), "inflado_deriva_real": float(infl_max)},
        indent=2, default=lambda o: (float(o) if isinstance(o, np.floating)
                                     else int(o) if isinstance(o, np.integer)
                                     else o.tolist() if isinstance(o, np.ndarray)
                                     else str(o))), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
