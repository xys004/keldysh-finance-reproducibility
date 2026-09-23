r"""Experimento 17 — ¿hay memoria por encima del segundo orden?

POR QUÉ ESTE EXPERIMENTO
------------------------
Todo lo medido hasta ahora vive en el sector de SEGUNDO ORDEN, y ese sector
está lleno de identidades:

  - `Var(Q_T) = Σ_{t,t'} C_eps(t−t')` es una identidad exacta de muestra finita,
    no una corroboración cruzada (exp. 16);
  - `s(Q) = [2⟨Q⟩/Var(Q)]·Q` es una identidad algebraica de cualquier gaussiana
    (derivación MSRJD, Observación 1);
  - `A_{T2}/A_{T1} = (T2/T1)(V_{T1}/V_{T2})` se sigue de las dos anteriores;
  - y la dominancia del camino de signos sobre el de magnitudes es, a segundo
    orden, la afirmación de que `C_s` decae más despacio que `C_m` — que es el
    resultado fundacional de Lillo-Farmer.

Por eso las cuatro degradaciones de las últimas revisiones no fueron cuatro
errores independientes: fueron el mismo hecho estructural saliendo cuatro
veces. Es un hallazgo sobre el observable, pero deja el formalismo sin nada
que predecir que la estadística de segundo orden no dé ya.

El tercer y el cuarto cumulante son el sitio donde eso deja de ser cierto.
`kappa_3(Q_T) = Σ_{t,t',t''} ⟨eps·eps·eps⟩_c` NO está determinado por `C_eps`:
ninguna identidad de segundo orden lo alcanza, y un gaussiano lo pone a cero.

LA PREGUNTA, Y POR QUÉ EL NIVEL NO SIRVE
-----------------------------------------
El nulo estratificado conserva la distribución marginal, así que **a T=1 el
observado y el nulo coinciden por construcción**. Cualquier no-gaussianidad a
T=1 es la cola de `eps_t` y no es evidencia de nada temporal.

Lo que sí discrimina es la SUPERVIVENCIA a la agregación. Para incrementos
independientes con marginal fija,

    gamma_1(T)/gamma_1(1) = T^(−1/2),      gamma_2(T)/gamma_2(1) = T^(−1),

o sea que la no-gaussianidad se disuelve al sumar. Si sobrevive más de lo que
el nulo permite, hay organización temporal de orden ≥3. El cociente normaliza
la marginal exactamente, que es justo el confusor.

Ventaja lateral del nulo de permutación: la curtosis muestral es un estimador
inestable con colas pesadas y pocas ventanas. El nulo tiene el MISMO n y la
MISMA marginal en cada T, así que absorbe esa inestabilidad — el contraste no
depende de que el estimador sea bueno, sólo de que sea el mismo a ambos lados.

CRITERIO EX-ANTE (declarado antes de mirar)
-------------------------------------------
- Estadístico primario: `R_k(T) = gamma_k(T)/gamma_k(1)`, con `k=1,2`.
- T se limita a que queden ≥ 200 ventanas no solapadas; con 34.608 velas eso
  llega a T=128 (270 ventanas).
- POSITIVO si en ≥3 de 4 activos, y en las DOS definiciones de flujo, el
  observado supera el cuantil 0.975 del nulo en el mayor T admisible.
- El nulo es el mismo que sostiene el resultado principal del Letter
  (permutación dentro de estratos trimestre × hora-de-semana, semanas UTC
  completas). No se introduce un nulo nuevo para este contraste.
- Si el contraste sale NEGATIVO, la lectura es que el sector de conteo de este
  observable está cerrado al segundo orden, y ese es el resultado.

QUÉ PASÓ CON ESE CRITERIO, Y QUÉ SE CORRIGE
--------------------------------------------
El criterio declarado **FALLA**: en el mayor T admisible pasa 1 de 4 activos en
cada celda. Se reporta como fallado y no se renegocia.

Pero deja dos defectos de DISEÑO que sí hay que corregir, porque no son el
umbral sino el instrumento:

1. **El estadístico de razón está roto para gamma_1.** Divide por `gamma_1(1)`,
   que en SOL vale 0.018, y produce razones de +680 que no significan nada. Y
   es innecesario: la razón existía para cancelar la marginal, pero **el nulo
   ya conserva la marginal**, así que comparar el nivel absoluto `gamma_k(T)`
   contra su propio nulo hace el mismo trabajo sin el denominador inestable.
2. **El punto de evaluación era el de menos potencia del barrido.** A T=128
   quedan 270 ventanas y el cuantil 0.975 del nulo se dispara (1.5-2.0 frente a
   1.1-1.2 en T=16). Fijar ahí el veredicto era apuntar el contraste al sitio
   donde menos puede ver.

La corrección NO es mirar otro T y llamarlo positivo: es leer el barrido
ENTERO con corrección de Westfall-Young (regla 9 del proyecto — un barrido sin
maxT fabrica hallazgos por construcción, medido en el exp. 03). El estadístico
por celda es `z = (|gamma_obs| − mediana|gamma_nulo|)/(1.4826·MAD)`, y cada
permutación aporta su MÁXIMO sobre todas las celdas.
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

from keldysh_finance.fano_validation import (_stratum_period,
                                             prepare_complete_weeks)
from keldysh_finance.flow import fetch_klines_with_flow

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
T_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
MIN_VENTANAS = 200
ESTRATIFICACION = "quarterly"


def _cumulantes(Q: np.ndarray) -> tuple[float, float]:
    """gamma_1 y gamma_2 insesgados (mismas fórmulas que `counting`)."""
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


def _perfil(x: np.ndarray, t_grid: list[int]) -> dict:
    """gamma_1(T), gamma_2(T) sobre ventanas NO solapadas."""
    g1, g2, nv = [], [], []
    for T in t_grid:
        n = x.size // T
        Q = x[:n * T].reshape(n, T).sum(axis=1)
        a, b = _cumulantes(Q)
        g1.append(a); g2.append(b); nv.append(n)
    return {"T": list(t_grid), "gamma1": g1, "gamma2": g2, "n_ventanas": nv}


def _permutar_en_estratos(x: np.ndarray, estrato: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Permutación dentro de estrato, vectorizada.

    `base` ordena las posiciones por estrato conservando el orden original;
    `orden` las ordena por estrato con clave aleatoria. Asignar
    `out[base] = x[orden]` coloca en las plazas de cada estrato una permutación
    aleatoria de los valores de ESE estrato. Los estratos de tamaño 1 quedan
    fijos por construcción, como documenta `fano_validation`.
    """
    n = x.size
    base = np.lexsort((np.arange(n), estrato))
    orden = np.lexsort((rng.random(n), estrato))
    out = np.empty_like(x)
    out[base] = x[orden]
    return out


def analizar(nombre: str, symbol: str, flow_mode: str,
             n_perm: int, semilla: int) -> dict:
    df = fetch_klines_with_flow(symbol, interval="1h", years=4.0).dropna()
    prep = prepare_complete_weeks(df, flow_mode=flow_mode)
    x = prep.flow_demeaned
    periodo = _stratum_period(prep, ESTRATIFICACION)
    estrato = periodo.astype(np.int64) * 168 + prep.hour_of_week.astype(np.int64)

    t_grid = [T for T in T_GRID if x.size // T >= MIN_VENTANAS]
    obs = _perfil(x, t_grid)

    rng = np.random.default_rng(semilla)
    nulos_g1 = np.empty((n_perm, len(t_grid)))
    nulos_g2 = np.empty((n_perm, len(t_grid)))
    for k in range(n_perm):
        p = _perfil(_permutar_en_estratos(x, estrato, rng), t_grid)
        nulos_g1[k] = p["gamma1"]
        nulos_g2[k] = p["gamma2"]

    # (a) criterio DECLARADO: razon R_k(T) = gamma_k(T)/gamma_k(1)
    def razones(g):
        g = np.asarray(g, dtype=float)
        return g / g[0] if np.isfinite(g[0]) and g[0] != 0 else np.full_like(g, np.nan)

    declarado = []
    for k, nul in (("gamma1", nulos_g1), ("gamma2", nulos_g2)):
        ro = razones(obs[k])
        rn = np.array([razones(r) for r in nul])
        i = len(t_grid) - 1
        col = rn[:, i][np.isfinite(rn[:, i])]
        declarado.append({
            "cumulante": k, "T": t_grid[i],
            "R_obs": float(ro[i]) if np.isfinite(ro[i]) else None,
            "R_nulo_q975": float(np.quantile(np.abs(col), 0.975)) if col.size else None,
            "supera": bool(np.isfinite(ro[i]) and col.size
                           and abs(ro[i]) > np.quantile(np.abs(col), 0.975))})

    # (b) analisis corregido: nivel absoluto + maxT sobre TODO el barrido
    filas, z_obs_todas, z_nul_todas = [], [], []
    for k, nul in (("gamma1", nulos_g1), ("gamma2", nulos_g2)):
        A = np.abs(nul)                                  # (n_perm, n_T)
        med = np.median(A, axis=0)
        mad = 1.4826 * np.median(np.abs(A - med), axis=0)
        mad = np.where(mad > 0, mad, np.nan)
        z_o = (np.abs(obs[k]) - med) / mad
        z_n = (A - med) / mad
        z_obs_todas.append(z_o)
        z_nul_todas.append(z_n)
        for i, T in enumerate(t_grid):
            col = A[:, i]
            filas.append({"cumulante": k, "T": T,
                          "n_ventanas": obs["n_ventanas"][i],
                          "gamma_obs": obs[k][i],
                          "nulo_mediana": float(med[i]),
                          "nulo_q975": float(np.quantile(col, 0.975)),
                          "z": float(z_o[i]),
                          "p_nominal": float((col >= abs(obs[k][i])).mean())})

    # Westfall-Young: cada permutacion aporta su maximo sobre el barrido entero
    z_o_all = np.concatenate(z_obs_todas)
    z_n_all = np.concatenate(z_nul_todas, axis=1)
    max_nulo = np.nanmax(z_n_all, axis=1)
    z_max_obs = float(np.nanmax(z_o_all))
    p_global = float((max_nulo >= z_max_obs).mean())
    for f in filas:
        f["p_maxT"] = float((max_nulo >= f["z"]).mean())

    return {"activo": nombre, "flow_mode": flow_mode, "n_velas": int(x.size),
            "n_semanas": int(prep.n_complete_weeks), "n_permutaciones": n_perm,
            "estratificacion": ESTRATIFICACION, "T_grid": t_grid,
            "observado": obs, "criterio_declarado": declarado, "filas": filas,
            "z_max_obs": z_max_obs, "p_global_maxT": p_global,
            "nulo_max_media": float(np.nanmean(max_nulo))}

def main() -> None:
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    print("=" * 78)
    print("  EXP 17 - memoria por encima del segundo orden")
    print("=" * 78)
    print(f"\n  nulo: permutacion dentro de estratos {ESTRATIFICACION} x "
          f"hora-de-semana, {n_perm} permutaciones")
    print("  el nulo conserva marginal, estacionalidad y deriva de calendario;")
    print("  destruye SOLO el orden temporal. A T=1 coinciden por construccion.")

    todo = []
    for modo in ("raw", "normalized"):
        for nombre, sym in ACTIVOS:
            todo.append(analizar(nombre, sym, modo, n_perm, semilla=17))

    # --- (a) el criterio declarado -------------------------------------------
    print("\n\n  " + "=" * 74)
    print("  (a) CRITERIO DECLARADO EX-ANTE  --  razon R_k en el mayor T")
    print("  " + "-" * 74)
    decl = {}
    for modo in ("raw", "normalized"):
        for k in ("gamma1", "gamma2"):
            ok = [r["activo"] for r in todo if r["flow_mode"] == modo
                  and any(d["cumulante"] == k and d["supera"]
                          for d in r["criterio_declarado"])]
            decl[f"{modo}|{k}"] = ok
            print(f"  {modo:<11}{k}: {len(ok)}/4  {ok if ok else ''}")
    paso = all(len(v) >= 3 for v in decl.values())
    print(f"\n  -> criterio declarado: {'PASA' if paso else 'FALLA'}"
          f"  (exigia >=3/4 en las cuatro celdas)")
    if not paso:
        print("     Se reporta fallado y no se renegocia. Los dos defectos de")
        print("     diseno que deja estan en el docstring; abajo va el barrido")
        print("     entero con el estadistico corregido y maxT.")

    # --- (b) barrido corregido con maxT --------------------------------------
    print("\n\n  " + "=" * 74)
    print("  (b) BARRIDO CORREGIDO  --  |gamma_k(T)| vs su nulo, maxT global")
    print("  " + "=" * 74)
    for modo in ("raw", "normalized"):
        print(f"\n  FLUJO {modo}")
        print("    " + "activo".ljust(7) + "cum".ljust(8) + "T".rjust(5)
              + "vent.".rjust(7) + "gamma".rjust(9) + "nulo".rjust(9)
              + "z".rjust(8) + "p_nom".rjust(8) + "p_maxT".rjust(8))
        for r in todo:
            if r["flow_mode"] != modo:
                continue
            for f in r["filas"]:
                if f["T"] == 1:
                    continue           # identico por construccion
                marca = "*" if f["p_maxT"] < 0.05 else " "
                print(f"    {r['activo']:<7}{f['cumulante']:<8}{f['T']:>5}"
                      f"{f['n_ventanas']:>7}{f['gamma_obs']:>9.3f}"
                      f"{f['nulo_mediana']:>9.3f}{f['z']:>8.1f}"
                      f"{f['p_nominal']:>8.3f}{f['p_maxT']:>8.3f}{marca}")

    # --- veredicto -----------------------------------------------------------
    print("\n\n  " + "=" * 74)
    print("  VEREDICTO")
    print("  " + "-" * 74)
    print(f"  {'serie':<20}{'z_max':>9}{'max nulo':>11}{'p_global':>11}")
    n_sig = 0
    for r in todo:
        sig = r["p_global_maxT"] < 0.05
        n_sig += int(sig)
        print(f"  {r['activo'] + '|' + r['flow_mode']:<20}"
              f"{r['z_max_obs']:>9.1f}{r['nulo_max_media']:>11.1f}"
              f"{r['p_global_maxT']:>11.4f}{'  *' if sig else ''}")
    print(f"\n  {n_sig}/8 series con p_global < 0.05 tras maxT.")
    if n_sig >= 6:
        print("  Hay organizacion temporal de orden >=3: ninguna identidad de")
        print("  segundo orden la alcanza, y el nulo conserva la marginal.")
    elif n_sig >= 1:
        print("  Senal en algunas series y no en otras. Reportable con nombre")
        print("  y apellido, sin promocionar a resultado general.")
    else:
        print("  El sector de conteo esta cerrado al segundo orden. Ese es el")
        print("  resultado, y encuadra el paper.")

    out = RAIZ / "output" / "exp17_cumulant_memory_null.json"
    out.parent.mkdir(exist_ok=True)

    def _limpiar(o):
        if isinstance(o, np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return [_limpiar(v) for v in o.tolist()]
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        return str(o)

    out.write_text(json.dumps(
        {"criterio_declarado": "R_k(T_max) sobre el cuantil 0.975 del nulo en "
                               ">=3 de 4 activos y las dos definiciones",
         "criterio_declarado_pasa": bool(paso), "detalle_declarado": decl,
         "correccion": "nivel absoluto |gamma_k(T)| vs nulo + Westfall-Young "
                       "maxT sobre el barrido entero",
         "estratificacion": ESTRATIFICACION, "n_permutaciones": n_perm,
         "min_ventanas": MIN_VENTANAS, "resultados": todo,
         "n_series_significativas": n_sig},
        indent=2, default=_limpiar), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
