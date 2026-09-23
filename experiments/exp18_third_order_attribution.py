r"""Experimento 18 — atribución del tercer orden, y un exponente nuevo.

DE DÓNDE VIENE
--------------
El exp. 17 encuentra estructura por encima del segundo orden: 7 de 8 series
con `p_global < 0.05` tras Westfall-Young, con el nulo estratificado que
conserva marginal, estacionalidad y deriva de calendario. Pero el reparto de
esa señal importa más que el total:

  - `gamma_2` son 30 de las 38 celdas significativas. Curtosis que sobrevive a
    la agregación es la firma de libro del CLUSTERING DE VOLATILIDAD. Real,
    pero conocido desde Cont y no atribuible a este trabajo.
  - `gamma_1` es la parte que no puede ser eso: un clustering simétrico da
    asimetría CERO a todo T. ETH|normalized va de −0.151 a −1.426 monótono en
    seis octavas con `p_maxT = 0.000`. Pero es UNA serie.

Este experimento hace dos cosas que el 17 no puede.

(1) ATRIBUCIÓN
--------------
Se usan los tres nulos del exp. 15, nombrados aquí por lo que CONSERVAN, no
por lo que permutan — la nomenclatura del Letter («the sign null») invita a
leerlo del revés:

  - `joint`             permuta `eps` entero: no conserva nada temporal.
  - `preserva_magnitud` permuta los signos y deja `|eps_t|` en su sitio: el
                        clustering de volatilidad queda INTACTO.
  - `preserva_signo`    permuta las magnitudes y deja `sign(eps_t)` en su
                        sitio: la persistencia de signo queda INTACTA.

Ninguno de los dos últimos conserva el emparejamiento contemporáneo
signo–magnitud; son diagnósticos de atribución, no modelos generativos.

Se mide qué fracción del exceso reproduce cada camino por separado:

    atribucion_X = [mediana(S | X) − mediana(S | joint)]
                   / [S_observado − mediana(S | joint)]

Si `atribucion_magnitud ≈ 1` para `gamma_2`, el exceso de curtosis ES el
clustering de volatilidad y no hay que venderlo como otra cosa. Si para
`gamma_1` los dos caminos por separado dan ≈ 0, la asimetría vive en el
acoplamiento y no en ninguno de los dos caminos aislados.

(2) UN EXPONENTE QUE NO FIJA NINGUNA IDENTIDAD DE SEGUNDO ORDEN
----------------------------------------------------------------
`Var(Q_T) = Σ C_eps` obliga a `nu = 2 − a`: por eso el sector de segundo orden
no aporta información nueva. El tercer cumulante no tiene esa atadura,

    kappa_3(Q_T) = Σ_{t,t',t''} ⟨eps eps eps⟩_c  ~  T^{nu_3},

y `nu_3` NO está determinado por `C_eps`. Para incrementos independientes
`nu_3 = 1`. Es el análogo de tercer orden de `nu`, y es la primera cantidad de
todo el programa que no se reduce a una identidad ni a la ACF de signos.

En el lenguaje de transporte: los cumulantes impares de una corriente se
anulan en equilibrio por balance detallado, de modo que `kappa_3` es el
observable de no-equilibrio propiamente dicho. Aquí `⟨eps⟩ ≠ 0` ya de entrada
(sesgo vendedor persistente), así que `kappa_3 ≠ 0` a `T=1` no dice nada; lo
que dice algo es su ESCALADO.

CRITERIO EX-ANTE (declarado antes de mirar)
-------------------------------------------
Se corrige el defecto de diseño del exp. 17, que evaluaba en el T de menos
potencia del barrido. El estadístico primario es ahora **un solo número por
serie** —el exponente ajustado sobre todo el rango admisible—, así que no hay
punto de evaluación que elegir mal:

- Primario: `nu_3` ajustado por MCO sobre `log|kappa_3(T)|` vs `log T`, en los
  T con ≥ 200 ventanas no solapadas.
- POSITIVO si `nu_3` observado supera el cuantil 0.975 del nulo `joint` en ≥ 6
  de 8 series, tras corrección maxT sobre la familia de las 8.
- La atribución es DESCRIPTIVA y se reporta pase lo que pase; no es una puerta.
- Si `nu_3` no supera el nulo, la lectura es que el tercer orden tiene nivel
  pero no escalado anómalo, y el resultado del exp. 17 se reporta como
  clustering de volatilidad más una asimetría en una sola serie.
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

from keldysh_finance.fano_validation import (_stratum_period,
                                             prepare_complete_weeks)

ACTIVOS = ["BTC", "ETH", "BNB", "SOL"]
SIMBOLOS = {a: f"{a}USDT" for a in ACTIVOS}
T_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
MIN_VENTANAS = 200
ESTRATIFICACION = "quarterly"
NULOS = ("joint", "preserva_magnitud", "preserva_signo")


def _cargar(symbol: str) -> pd.DataFrame:
    """Lee el caché depositado; sin red, para que corra en el cluster."""
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def _momentos(Q: np.ndarray) -> tuple[float, float, float]:
    """(kappa_3, gamma_1, gamma_2) — kappa_3 sin estandarizar."""
    n = Q.size
    if n < 4:
        return (np.nan,) * 3
    d = Q - Q.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 0:
        return (np.nan,) * 3
    m3, m4 = float((d ** 3).mean()), float((d ** 4).mean())
    k3 = m3 * n * n / ((n - 1) * (n - 2))              # k-estadistico insesgado
    g1 = np.sqrt(n * (n - 1)) / (n - 2) * (m3 / m2 ** 1.5)
    g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (m4 / m2 ** 2 - 3.0) + 6.0)
    return float(k3), float(g1), float(g2)


def _perfil(x: np.ndarray, t_grid) -> dict:
    k3, g1, g2, nv = [], [], [], []
    for T in t_grid:
        n = x.size // T
        Q = x[:n * T].reshape(n, T).sum(axis=1)
        a, b, c = _momentos(Q)
        k3.append(a); g1.append(b); g2.append(c); nv.append(n)
    return {"kappa3": k3, "gamma1": g1, "gamma2": g2, "n_ventanas": nv}


def _nu3(k3, t_grid) -> float:
    """Pendiente MCO de log|kappa_3| vs log T. i.i.d. da 1."""
    k = np.abs(np.asarray(k3, dtype=float))
    m = np.isfinite(k) & (k > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(t_grid, float)[m]),
                            np.log(k[m]), 1)[0])


def _surrogado(x: np.ndarray, estrato: np.ndarray, modo: str,
               rng: np.random.Generator) -> np.ndarray:
    """Permutación dentro de estrato, según lo que el nulo CONSERVA.

    `base` ordena posiciones por estrato en orden original y `orden` por
    estrato con clave aleatoria; `out[base] = v[orden]` coloca en las plazas de
    cada estrato una permutación aleatoria de los valores de ESE estrato. Los
    estratos unitarios quedan fijos, como en `fano_validation`.
    """
    n = x.size
    base = np.lexsort((np.arange(n), estrato))
    orden = np.lexsort((rng.random(n), estrato))
    if modo == "joint":
        out = np.empty_like(x)
        out[base] = x[orden]
        return out
    if modo == "preserva_magnitud":          # permuta signos, deja |eps|
        s = np.sign(x)
        sp = np.empty_like(s)
        sp[base] = s[orden]
        return np.abs(x) * sp
    if modo == "preserva_signo":             # permuta magnitudes, deja signo
        m = np.abs(x)
        mp = np.empty_like(m)
        mp[base] = m[orden]
        return np.sign(x) * mp
    raise ValueError(modo)


def tarea(args) -> dict:
    activo, modo, flow_mode, n_perm, semilla = args
    df = _cargar(SIMBOLOS[activo])
    prep = prepare_complete_weeks(df, flow_mode=flow_mode)
    x = prep.flow_demeaned
    periodo = _stratum_period(prep, ESTRATIFICACION)
    estrato = periodo.astype(np.int64) * 168 + prep.hour_of_week.astype(np.int64)
    t_grid = [T for T in T_GRID if x.size // T >= MIN_VENTANAS]

    obs = _perfil(x, t_grid)
    rng = np.random.default_rng(semilla)
    nul = {k: np.empty((n_perm, len(t_grid))) for k in
           ("kappa3", "gamma1", "gamma2")}
    nu3_nul = np.empty(n_perm)
    for i in range(n_perm):
        p = _perfil(_surrogado(x, estrato, modo, rng), t_grid)
        for k in nul:
            nul[k][i] = p[k]
        nu3_nul[i] = _nu3(p["kappa3"], t_grid)

    return {"activo": activo, "flow_mode": flow_mode, "nulo": modo,
            "T_grid": t_grid, "n_velas": int(x.size),
            "observado": obs, "nu3_obs": _nu3(obs["kappa3"], t_grid),
            "nu3_nulo": nu3_nul.tolist(),
            "nulo_medianas": {k: np.median(v, axis=0).tolist()
                              for k, v in nul.items()}}


def main() -> None:
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 199
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    from multiprocessing import Pool

    trabajos = [(a, m, f, n_perm, 18)
                for f in ("raw", "normalized")
                for a in ACTIVOS for m in NULOS]
    print("=" * 78)
    print("  EXP 18 — atribucion del tercer orden y exponente nu_3")
    print("=" * 78)
    print(f"\n  {len(trabajos)} tareas ({len(ACTIVOS)} activos x {len(NULOS)} "
          f"nulos x 2 flujos), {n_perm} permutaciones, {n_proc} procesos")

    with Pool(n_proc) as pool:
        res = pool.map(tarea, trabajos)

    idx = {(r["activo"], r["flow_mode"], r["nulo"]): r for r in res}

    # --- (1) nu_3 contra el nulo joint, con maxT sobre las 8 series ----------
    print("\n\n  " + "=" * 74)
    print("  (1) EXPONENTE nu_3 DE kappa_3   (i.i.d. da 1; nu_2 medido = 1.1-1.4)")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}{'nu3_obs':>9}{'nulo med':>10}{'nulo q975':>11}"
          f"{'p_nom':>8}{'p_maxT':>8}")

    familia = [(a, f) for f in ("raw", "normalized") for a in ACTIVOS]
    z_obs, z_nul = [], []
    for a, f in familia:
        r = idx[(a, f, "joint")]
        nn = np.asarray(r["nu3_nulo"], float)
        nn = nn[np.isfinite(nn)]
        med = np.median(nn)
        mad = 1.4826 * np.median(np.abs(nn - med)) or np.nan
        z_obs.append((r["nu3_obs"] - med) / mad)
        z_nul.append((nn - med) / mad)
    n_min = min(len(z) for z in z_nul)
    max_nulo = np.nanmax(np.vstack([z[:n_min] for z in z_nul]), axis=0)

    filas1, n_sup = [], 0
    for (a, f), zo in zip(familia, z_obs):
        r = idx[(a, f, "joint")]
        nn = np.asarray(r["nu3_nulo"], float); nn = nn[np.isfinite(nn)]
        q975 = float(np.quantile(nn, 0.975))
        p_nom = float((nn >= r["nu3_obs"]).mean())
        p_max = float((max_nulo >= zo).mean())
        sup = r["nu3_obs"] > q975
        n_sup += int(sup and p_max < 0.05)
        filas1.append({"activo": a, "flow_mode": f, "nu3_obs": r["nu3_obs"],
                       "nulo_mediana": float(np.median(nn)), "nulo_q975": q975,
                       "p_nominal": p_nom, "p_maxT": p_max, "supera": bool(sup)})
        print(f"  {a + '|' + f:<18}{r['nu3_obs']:>9.3f}{np.median(nn):>10.3f}"
              f"{q975:>11.3f}{p_nom:>8.3f}{p_max:>8.3f}"
              f"{'  *' if sup and p_max < 0.05 else ''}")

    positivo = n_sup >= 6
    print(f"\n  -> {n_sup}/8 superan el nulo joint tras maxT. Criterio "
          f"declarado (>=6): {'PASA' if positivo else 'FALLA'}")

    # --- (2) atribucion: que fraccion reproduce cada camino ------------------
    print("\n\n  " + "=" * 74)
    print("  (2) ATRIBUCION — fraccion del exceso que reproduce cada camino")
    print("      (~1 = ese camino solo basta;  ~0 = no lo explica)")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}{'cumulante':<11}{'T':>5}{'obs':>9}{'joint':>9}"
          f"{'|eps| solo':>11}{'signo solo':>11}")

    filas2 = []
    for f in ("raw", "normalized"):
        for a in ACTIVOS:
            rj = idx[(a, f, "joint")]
            rm = idx[(a, f, "preserva_magnitud")]
            rs = idx[(a, f, "preserva_signo")]
            t_grid = rj["T_grid"]
            i = min(range(len(t_grid)), key=lambda j: abs(t_grid[j] - 16))
            for k in ("gamma1", "gamma2"):
                o = rj["observado"][k][i]
                mj = rj["nulo_medianas"][k][i]
                mm = rm["nulo_medianas"][k][i]
                ms = rs["nulo_medianas"][k][i]
                den = o - mj
                am = (mm - mj) / den if abs(den) > 1e-12 else np.nan
                asg = (ms - mj) / den if abs(den) > 1e-12 else np.nan
                filas2.append({"activo": a, "flow_mode": f, "cumulante": k,
                               "T": t_grid[i], "obs": o, "joint": mj,
                               "atribucion_magnitud": float(am),
                               "atribucion_signo": float(asg)})
                print(f"  {a + '|' + f:<18}{k:<11}{t_grid[i]:>5}{o:>9.3f}"
                      f"{mj:>9.3f}{am:>11.2f}{asg:>11.2f}")

    print("\n  " + "-" * 74)
    for k in ("gamma1", "gamma2"):
        sub = [r for r in filas2 if r["cumulante"] == k
               and np.isfinite(r["atribucion_magnitud"])]
        if not sub:
            continue
        am = float(np.median([r["atribucion_magnitud"] for r in sub]))
        asg = float(np.median([r["atribucion_signo"] for r in sub]))
        print(f"  {k}: mediana sobre 8 series — |eps| solo {am:+.2f}, "
              f"signo solo {asg:+.2f}")
        if k == "gamma2" and am > 0.5:
            print("     -> el exceso de curtosis ES el camino de magnitudes:")
            print("        clustering de volatilidad. Conocido; no venderlo.")
        if k == "gamma1" and abs(am) < 0.5 and abs(asg) < 0.5:
            print("     -> ningun camino aislado reproduce la asimetria: vive")
            print("        en el acoplamiento signo-magnitud.")

    out = RAIZ / "output" / "exp18_third_order_attribution.json"
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
        {"criterio": "nu_3 observado sobre el q975 del nulo joint en >=6 de 8 "
                     "series tras maxT",
         "criterio_pasa": bool(positivo), "n_permutaciones": n_perm,
         "estratificacion": ESTRATIFICACION, "nulos": list(NULOS),
         "nu3": filas1, "atribucion": filas2, "crudo": res},
        indent=2, default=_l), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
