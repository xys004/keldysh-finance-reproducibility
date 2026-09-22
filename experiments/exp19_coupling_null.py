r"""Experimento 19 — el nulo que conserva el emparejamiento signo–magnitud.

EL PROBLEMA QUE CIERRA
----------------------
El exp. 18 concluye que la asimetría `gamma_1` del flujo acumulado no la
reproduce ni el camino de signos ni el de magnitudes por separado, y de ahí
que «viva en el acoplamiento». Pero eso es un argumento POR RESTA: los dos
nulos de atribución permutan `s` y `m` de forma independiente, así que rompen
el emparejamiento contemporáneo de paso. El residuo no es una identificación.

Un árbitro lo dirá, y con razón. Hace falta un nulo que destruya el
acoplamiento Y NADA MÁS.

EL NULO
-------
Desalineación circular por semanas enteras:

    eps'_t = s_{t + 168k} · m_t ,      k entero, 1 <= k <= n_semanas − 1

Lo que conserva, exactamente y no aproximadamente:

  - la marginal de `|eps|`, punto por punto y en su sitio temporal, de modo que
    el CLUSTERING DE VOLATILIDAD queda intacto;
  - la secuencia completa de signos, de modo que la MEMORIA LARGA DE SIGNOS
    (Lillo-Farmer) queda intacta, sólo desplazada en bloque;
  - la hora-de-semana de ambos, porque el desplazamiento es múltiplo de 168:
    `s` y `m` siguen viniendo del mismo día de la semana y la misma hora, así
    que la estacionalidad diaria y semanal no se toca.

Lo único que rompe es QUÉ signo le toca a QUÉ magnitud. Es el nulo quirúrgico
que el `AGENTS.md` tenía anotado como pendiente.

EL CONTROL POSITIVO VIENE DENTRO
--------------------------------
`gamma_2` DEBE sobrevivir a este nulo: el camino de magnitudes no se toca y el
exp. 18 atribuye a él el 62% de su exceso. Si `gamma_2` también se hunde, el
desalineamiento está destruyendo más de lo que dice y el experimento NO
CONCLUYE — es el mismo requisito de sensibilidad que se le exige a cualquier
ensayo, y aquí sale gratis porque el diseño ya lo contiene.

LA ESCALERA DE BLOQUES
----------------------
Complemento con signo opuesto: permutar BLOQUES de `L` semanas consecutivas,
`L = 1, 2, 4, 8, 16`. Cada bloque conserva íntegro todo lo de dentro
—emparejamiento, orden local, acoplamiento— y se destruye sólo la ordenación
entre bloques. La `L` a la que el estadístico se recupera es la ESCALA sobre
la que el acoplamiento tiene que estar intacto. Eso es identificación
positiva: no «no es ninguno de los otros dos» sino «hace falta que esté
emparejado durante tantas semanas».

CRITERIO EX-ANTE (declarado antes de mirar)
-------------------------------------------
Recuperación de un estadístico `S` bajo el nulo `X`:

    rec_X = [mediana(S | X) − mediana(S | joint)] / [S_obs − mediana(S | joint)]

con `joint` = permutación estratificada del exp. 17-18 (destruye todo el orden
conservando el emparejamiento). `rec = 1` significa que ese nulo reproduce el
fenómeno entero; `rec = 0`, que no lo reproduce en absoluto.

- **Control (se evalúa primero, y manda):** `rec` de `gamma_2` bajo
  desalineación >= 0.5 en >= 6 de 8 series. Si falla, el experimento no
  concluye y no se lee nada más.
- **Primario:** `rec` de `gamma_1` bajo desalineación <= 0.5 en >= 6 de 8
  series, y el signo de la conclusión estable en T = 8, 16 y 32.
- T = 16 se HEREDA de la tabla de atribución del exp. 18; no se elige mirando
  este resultado. El perfil completo en T se reporta igual.
- La escalera de bloques es DESCRIPTIVA: da la escala, no decide nada.
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
HORAS_SEMANA = 168
T_EVAL = [8, 16, 32]          # T=16 es el primario, heredado del exp. 18
T_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
MIN_VENTANAS = 200
ESTRATIFICACION = "quarterly"
BLOQUES = [1, 2, 4, 8, 16]    # en SEMANAS


def _cargar(symbol: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def _cumulantes(Q: np.ndarray) -> tuple[float, float]:
    n = Q.size
    if n < 4:
        return float("nan"), float("nan")
    d = Q - Q.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 0:
        return float("nan"), float("nan")
    m3, m4 = float((d ** 3).mean()), float((d ** 4).mean())
    g1 = np.sqrt(n * (n - 1)) / (n - 2) * (m3 / m2 ** 1.5)
    g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (m4 / m2 ** 2 - 3.0) + 6.0)
    return float(g1), float(g2)


def _perfil(x: np.ndarray, t_grid) -> dict:
    g1, g2 = [], []
    for T in t_grid:
        n = x.size // T
        a, b = _cumulantes(x[:n * T].reshape(n, T).sum(axis=1))
        g1.append(a); g2.append(b)
    return {"gamma1": g1, "gamma2": g2}


def _joint(x: np.ndarray, estrato: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """Permutación estratificada — el nulo de referencia del exp. 17-18."""
    n = x.size
    base = np.lexsort((np.arange(n), estrato))
    orden = np.lexsort((rng.random(n), estrato))
    out = np.empty_like(x)
    out[base] = x[orden]
    return out


def _desalinear(x: np.ndarray, n_semanas: int,
                rng: np.random.Generator) -> np.ndarray:
    """`eps'_t = s_{t+168k}·m_t`: rompe SOLO el emparejamiento.

    El desplazamiento es múltiplo de una semana entera, así que el signo que
    llega a la posición `t` viene del mismo día de la semana y la misma hora:
    la estacionalidad queda alineada y lo único que cambia es de QUÉ semana
    procede el signo.
    """
    k = int(rng.integers(1, n_semanas))
    return np.abs(x) * np.sign(np.roll(x, -k * HORAS_SEMANA))


def _bloques(x: np.ndarray, n_semanas: int, L: int,
             rng: np.random.Generator) -> np.ndarray:
    """Permuta bloques de `L` semanas consecutivas; dentro, todo intacto."""
    n_bloques = n_semanas // L
    usable = n_bloques * L * HORAS_SEMANA
    bl = x[:usable].reshape(n_bloques, L * HORAS_SEMANA)
    out = bl[rng.permutation(n_bloques)].reshape(-1)
    if usable < x.size:                      # cola no divisible, se deja igual
        out = np.concatenate([out, x[usable:]])
    return out


def tarea(args) -> dict:
    activo, flow_mode, n_rep, semilla = args
    prep = prepare_complete_weeks(_cargar(f"{activo}USDT"), flow_mode=flow_mode)
    x = prep.flow_demeaned
    periodo = _stratum_period(prep, ESTRATIFICACION)
    estrato = periodo.astype(np.int64) * 168 + prep.hour_of_week.astype(np.int64)
    n_sem = int(prep.n_complete_weeks)
    t_grid = [T for T in T_GRID if x.size // T >= MIN_VENTANAS]

    obs = _perfil(x, t_grid)
    rng = np.random.default_rng(semilla)

    def medianas(gen) -> dict:
        ac = {"gamma1": [], "gamma2": []}
        for _ in range(n_rep):
            p = _perfil(gen(), t_grid)
            ac["gamma1"].append(p["gamma1"]); ac["gamma2"].append(p["gamma2"])
        return {k: np.median(np.asarray(v), axis=0).tolist()
                for k, v in ac.items()}

    med = {"joint": medianas(lambda: _joint(x, estrato, rng)),
           "desalineado": medianas(lambda: _desalinear(x, n_sem, rng))}
    for L in BLOQUES:
        if n_sem // L >= 8:
            med[f"bloques{L}"] = medianas(lambda L=L: _bloques(x, n_sem, L, rng))

    return {"activo": activo, "flow_mode": flow_mode, "T_grid": t_grid,
            "n_semanas": n_sem, "n_repeticiones": n_rep,
            "observado": obs, "medianas": med}


def _rec(o: float, mj: float, mx: float) -> float:
    den = o - mj
    return float((mx - mj) / den) if abs(den) > 1e-12 else float("nan")


def main() -> None:
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 199
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    from multiprocessing import Pool

    trabajos = [(a, f, n_rep, 19) for f in ("raw", "normalized")
                for a in ACTIVOS]
    print("=" * 78)
    print("  EXP 19 - el nulo que conserva el emparejamiento signo-magnitud")
    print("=" * 78)
    print(f"\n  desalineado: eps'_t = s_(t+168k)·m_t   ->  conserva AMBOS "
          f"caminos y la\n  hora-de-semana; destruye SOLO que signo le toca a "
          f"que magnitud.")
    print(f"  {len(trabajos)} series x {n_rep} repeticiones, {n_proc} procesos")

    with Pool(n_proc) as pool:
        res = pool.map(tarea, trabajos)

    # --- control primero: gamma_2 DEBE sobrevivir ---------------------------
    print("\n\n  " + "=" * 74)
    print("  CONTROL (manda) — gamma_2 debe SOBREVIVIR al desalineado")
    print("     el camino de magnitudes no se toca; si se hunde, el nulo")
    print("     destruye de mas y no se lee nada mas")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}" + "".join(f"{'T=' + str(t):>10}" for t in T_EVAL))
    ctrl = []
    for r in res:
        fila, tg = [], r["T_grid"]
        for T in T_EVAL:
            i = tg.index(T)
            fila.append(_rec(r["observado"]["gamma2"][i],
                             r["medianas"]["joint"]["gamma2"][i],
                             r["medianas"]["desalineado"]["gamma2"][i]))
        ctrl.append({"serie": f"{r['activo']}|{r['flow_mode']}", "rec": fila})
        print(f"  {r['activo'] + '|' + r['flow_mode']:<18}"
              + "".join(f"{v:>10.2f}" for v in fila))
    i16 = T_EVAL.index(16)
    n_ctrl = sum(1 for c in ctrl if np.isfinite(c["rec"][i16])
                 and c["rec"][i16] >= 0.5)
    print(f"\n  -> gamma_2 sobrevive (rec >= 0.5) en {n_ctrl}/8 series a T=16")
    control_ok = n_ctrl >= 6
    print(f"     CONTROL: {'PASA' if control_ok else 'FALLA'} (exigia >=6/8)")
    if not control_ok:
        print("\n     El desalineado destruye mas que el acoplamiento. NO se")
        print("     puede leer nada del contraste primario. Fin.")

    # --- primario: gamma_1 debe MORIR ---------------------------------------
    print("\n\n  " + "=" * 74)
    print("  PRIMARIO — gamma_1 debe MORIR al desalineado si vive en el")
    print("             acoplamiento (rec <= 0.5)")
    print("  " + "=" * 74)
    print(f"\n  {'serie':<18}{'obs(16)':>9}{'joint':>9}"
          + "".join(f"{'rec T=' + str(t):>10}" for t in T_EVAL))
    prim = []
    for r in res:
        fila, tg = [], r["T_grid"]
        for T in T_EVAL:
            i = tg.index(T)
            fila.append(_rec(r["observado"]["gamma1"][i],
                             r["medianas"]["joint"]["gamma1"][i],
                             r["medianas"]["desalineado"]["gamma1"][i]))
        j = tg.index(16)
        prim.append({"serie": f"{r['activo']}|{r['flow_mode']}", "rec": fila,
                     "obs": r["observado"]["gamma1"][j],
                     "joint": r["medianas"]["joint"]["gamma1"][j]})
        print(f"  {r['activo'] + '|' + r['flow_mode']:<18}"
              f"{r['observado']['gamma1'][j]:>9.3f}"
              f"{r['medianas']['joint']['gamma1'][j]:>9.3f}"
              + "".join(f"{v:>10.2f}" for v in fila))
    n_prim = sum(1 for p in prim if np.isfinite(p["rec"][i16])
                 and p["rec"][i16] <= 0.5)
    estable = all(
        sum(1 for p in prim if np.isfinite(p["rec"][k]) and p["rec"][k] <= 0.5) >= 6
        for k in range(len(T_EVAL)))
    print(f"\n  -> gamma_1 muere (rec <= 0.5) en {n_prim}/8 series a T=16")
    print(f"     estable en T = {T_EVAL}: {'SI' if estable else 'NO'}")
    primario_ok = control_ok and n_prim >= 6 and estable

    # --- escalera de bloques (descriptiva) ----------------------------------
    print("\n\n  " + "=" * 74)
    print("  ESCALERA DE BLOQUES (descriptiva) — recuperacion de gamma_1")
    print("     permuta bloques de L semanas; dentro, todo intacto")
    print("  " + "=" * 74)
    disp = [L for L in BLOQUES if f"bloques{L}" in res[0]["medianas"]]
    print(f"\n  {'serie':<18}" + "".join(f"{'L=' + str(L):>9}" for L in disp))
    escalera = []
    for r in res:
        tg = r["T_grid"]; i = tg.index(16)
        fila = [_rec(r["observado"]["gamma1"][i],
                     r["medianas"]["joint"]["gamma1"][i],
                     r["medianas"][f"bloques{L}"]["gamma1"][i]) for L in disp]
        escalera.append({"serie": f"{r['activo']}|{r['flow_mode']}",
                         "L": disp, "rec": fila})
        print(f"  {r['activo'] + '|' + r['flow_mode']:<18}"
              + "".join(f"{v:>9.2f}" for v in fila))
    med_esc = [float(np.nanmedian([e["rec"][j] for e in escalera]))
               for j in range(len(disp))]
    print(f"  {'mediana':<18}" + "".join(f"{v:>9.2f}" for v in med_esc))
    recup = [disp[j] for j, v in enumerate(med_esc) if v >= 0.5]
    if recup:
        print(f"\n     gamma_1 se recupera a la mitad con bloques de "
              f"{min(recup)} semana(s):")
        print(f"     esa es la escala sobre la que el emparejamiento tiene que")
        print(f"     estar intacto.")
    else:
        print("\n     ningun tamano de bloque recupera la mitad: el "
              "acoplamiento necesita\n     mas de "
              f"{max(disp)} semanas seguidas, o el estadistico es inestable.")

    # --- veredicto -----------------------------------------------------------
    print("\n\n  " + "=" * 74)
    print("  VEREDICTO")
    print("  " + "-" * 74)
    if not control_ok:
        print("  NO CONCLUYE: el control falla.")
    elif primario_ok:
        print("  POSITIVO. Conservando INTACTOS el camino de signos y el de")
        print("  magnitudes, y rompiendo solo su emparejamiento, la asimetria")
        print("  se cae y la curtosis no. La asimetria del flujo acumulado NO")
        print("  es la memoria de signos ni el clustering de volatilidad:")
        print("  esta en el acoplamiento entre ambos, por identificacion")
        print("  positiva y no por resta.")
    else:
        print("  NEGATIVO en el primario: el desalineado no mata gamma_1, asi")
        print("  que el acoplamiento contemporaneo NO es el mecanismo. Hay que")
        print("  volver al exp. 18 y buscar de otra forma.")

    out = RAIZ / "output" / "exp19_coupling_null.json"
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
        {"nulo": "eps'_t = s_(t+168k)*m_t, k semanas enteras",
         "criterio": "control: rec(gamma2) >= 0.5 en >=6/8; primario: "
                     "rec(gamma1) <= 0.5 en >=6/8 y estable en T=8,16,32",
         "control_pasa": bool(control_ok), "primario_pasa": bool(primario_ok),
         "T_eval": T_EVAL, "n_repeticiones": n_rep,
         "control": ctrl, "primario": prim, "escalera": escalera,
         "escalera_mediana": med_esc, "crudo": res},
        indent=2, default=_l), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
