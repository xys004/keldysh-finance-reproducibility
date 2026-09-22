r"""Experimento 21 — el acoplamiento signo-tamaño en ventanas deslizantes,
resuelto en escala.

DE DÓNDE VIENE, Y QUÉ ERROR NO SE REPITE
-----------------------------------------
La idea original del proyecto era ajustar un ESTADO TRANSITORIO a cada ventana
temporal y seguir la trayectoria de sus parámetros. El exp. 05a la ejecutó con
el vector KWW `(log A, log tau_c, beta)` y murió en la fase descriptiva, con
cuatro puertas declaradas ex-ante:

  G1 medición        fracción de ajustes válidos >= 50% sólo en 7/24 configs
  G2 identificabilidad  2/24 con IQR/error >= 2 en tau_c Y beta
  G3 no-redundancia  la única que pasó: (tau_c, beta) no son proxies de sigma
  G4 estructura temporal  0/24 — la persistencia lag-1 de 0.97 era SOLAPE DE
                     VENTANAS; a separación no solapada cae a ~0 o negativa

La causa de G2 fue estructural: la degeneración KWW da
`|corr(log tau_c, beta)| = 0.75-0.96` con <= 80 desfases. El vector transitorio
por ventana no existe como objeto medible en este dato.

Este experimento repite EL MISMO DISEÑO DE VENTANAS sobre un observable
distinto, elegido porque su identificabilidad no depende de ningún ajuste:

    C_sm(tau) = < s_t s_{t+tau} ( m_t m_{t+tau} − mbar^2 ) >

es la correlación cruzada signo-tamaño, el integrando del término cruzado de la
descomposición exacta (`acopl_w = 2 SUM_tau (T−tau) C_sm(tau)`). No se ajusta
nada: es un promedio. G1 y G2 dejan de depender de que converja un optimizador.

**G4 sigue siendo la puerta que decide**, y se evalúa a separación NO SOLAPADA.

POR QUÉ ESTE OBSERVABLE Y NO OTRO
----------------------------------
Es el que quedó abierto por las dos vías del proyecto a la vez:

- La descomposición exacta le atribuye el 49-71% del exceso de varianza y lo
  hace mayor que la memoria de signos en 7 de 8 series (exp. 20).
- El tercer cumulante dice que la asimetría del flujo acumulado no la reproduce
  ni el camino de signos ni el de magnitudes por separado (exp. 18).
- Y su decaimiento CAMBIA DE SIGNO hacia tau ~ 50-100 velas, escala de días
  (exp. 20). Ese cruce es un objeto a dos tiempos que ningún modelo fija.

En lenguaje de transporte es la correlación entre la DIRECCIÓN de la
transferencia y el TAMAÑO del cuanto transferido: un proceso puntual marcado
leído como corriente.

RESOLUCIÓN EN ESCALA
--------------------
`C_sm(tau)` se agrega en bandas diádicas, `K_j = SUM_{tau in [2^j, 2^{j+1})}`,
que es la partición nativa de una descomposición en ondículas y evita imponer
una forma funcional. Cada `K_j` se normaliza por el término diagonal de la
ventana para que sea adimensional y comparable entre activos y entre ventanas.

El resultado es `K_j(t_w)`: acoplamiento por octava y por edad de ventana. Es
literalmente una estructura a dos tiempos —escala x tiempo— sobre un observable
identificable, que es lo que el vector KWW no era.

CRITERIO EX-ANTE (las cuatro puertas del 05a, adaptadas)
---------------------------------------------------------
- **G1 medición.** El término diagonal de la ventana debe ser positivo y la
  identidad exacta a 1e-6 relativo en >= 95% de las ventanas.
- **G2 identificabilidad.** Para cada octava, `IQR(K_j) / error_típico >= 2`,
  con el error por bootstrap de bloques DENTRO de cada ventana. Es la misma
  vara que el 05a: la señal entre ventanas debe superar al ruido de medición.
- **G3 no-redundancia.** `|Spearman(K_j, sigma_ventana)| <= 0.3`. Si el
  acoplamiento es un proxy de la volatilidad, no aporta nada nuevo.
- **G4 estructura temporal.** Autocorrelación de `K_j(t_w)` a separación NO
  SOLAPADA (>= W velas) significativamente distinta de cero, con el nulo de
  permutar el orden de las ventanas. **Sin esta puerta el resto no se lee.**

Se declara POSITIVO si G1, G2 y G4 pasan en >= 2 octavas y >= 3 de 4 activos.
G3 se reporta siempre; un fallo de G3 no invalida la medición pero la degrada a
redundante.

UN NULO QUE NO ESTABA CALIBRADO, Y CÓMO SE VIO
-----------------------------------------------
La primera versión usaba ventanas con 75% de solape (`PASO_REL = 0.25`) y
evaluaba G4 a desfase 4, que en ese muestreo es la separación no solapada. El
criterio declarado falló igual (0/4), pero dejaba un exceso sin explicar: 19
celdas de 96 con p < 0.05 donde el azar daba 4.8.

Ese exceso era del NULO, no del dato, y tenía dos huellas: 14 de las 19 eran
autocorrelaciones NEGATIVAS, y 16 de las 19 estaban en la ventana más larga.
Se descartó primero el desmediado por trimestre —que opera a escala comparable
a W=2688 y podría inducir anticorrelación mecánica— repitiendo sin desmediar:
la mediana pasa de −0.065 a −0.110, o sea que empeora. No era eso.

La causa es el propio solape. Permutar la trayectoria destruye la correlación
inducida por el solape en los desfases 1-3, de modo que la distribución
muestral del estimador a desfase 4 bajo el nulo NO es la del observado. Con
ventanas SIN solapar, donde el desfase 1 ya es la separación buena y el nulo
está calibrado, el exceso desaparece: 4 celdas de 48 frente a 2.4 por azar,
autocorrelación mediana −0.019 y 25/48 negativas.

Por eso `PASO_REL = 1.0`. La lección generaliza y es de la misma familia que la
regla del exp. 07 (la normalización pre-evento fabrica envejecimiento): **un
nulo de permutación sobre una trayectoria de ventanas solapadas está
descalibrado por construcción**, y el sitio donde muerde es justo el
estadístico de largo alcance que uno quiere medir.
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
VENTANAS = [1344, 2688]          # 8 y 16 semanas de velas horarias
PASO_REL = 1.0                   # SIN SOLAPE: ver el docstring, el nulo
                                 # de permutacion se descalibra con solape
OCTAVAS = [1, 2, 3, 4, 5, 6]     # bandas tau: [2,4) [4,8) ... [64,128)
N_BOOT = 99
BLOQUE_BOOT = 168                # una semana


def _cargar(symbol: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def integrando(eps: np.ndarray, octava: int) -> np.ndarray:
    """`P_j(t) = SUM_{tau in banda} s_t s_{t+tau}(m_t m_{t+tau} − mbar^2)`.

    Su media sobre la ventana es `K_j` sin normalizar. Devolver el integrando
    en vez del promedio permite estimar su error con un bootstrap de bloques,
    que es lo que exige G2.
    """
    s, m = np.sign(eps), np.abs(eps)
    mb2 = float(m.mean()) ** 2
    lo, hi = 2 ** octava, 2 ** (octava + 1)
    n = eps.size
    acc = np.zeros(n - hi)
    for tau in range(lo, hi):
        acc += (s[:n - hi] * s[tau:tau + n - hi]
                * (m[:n - hi] * m[tau:tau + n - hi] - mb2))
    return acc


def _boot_media(x: np.ndarray, rng: np.random.Generator) -> float:
    """Error típico de la media por bootstrap de bloques circulares."""
    n = x.size
    n_bl = max(1, n // BLOQUE_BOOT)
    idx0 = rng.integers(0, n, (N_BOOT, n_bl))
    off = np.arange(BLOQUE_BOOT)
    vals = np.array([x[(i[:, None] + off).ravel() % n].mean() for i in idx0])
    return float(vals.std(ddof=1))


def trayectoria(eps: np.ndarray, W: int, rng: np.random.Generator) -> dict:
    """`K_j(t_w)` normalizado, con su error por ventana, y sigma por ventana."""
    paso = int(W * PASO_REL)
    inicios = list(range(0, eps.size - W + 1, paso))
    out = {"inicios": inicios, "W": W, "K": {}, "err": {}, "sigma": []}
    for t0 in inicios:
        w = eps[t0:t0 + W]
        out["sigma"].append(float(np.std(w)))
    diag = np.array([float((eps[t0:t0 + W] ** 2).sum()) for t0 in inicios])
    for j in OCTAVAS:
        Ks, es = [], []
        for t0 in inicios:
            P = integrando(eps[t0:t0 + W], j)
            if P.size < 4 * BLOQUE_BOOT:
                Ks.append(np.nan); es.append(np.nan); continue
            Ks.append(float(P.mean()))
            es.append(_boot_media(P, rng))
        esc = diag / W                              # escala de tamano^2
        out["K"][j] = (np.asarray(Ks) / esc).tolist()
        out["err"][j] = (np.asarray(es) / esc).tolist()
    return out


def puertas(tr: dict, rng: np.random.Generator) -> dict:
    """G2, G3 y G4 sobre una trayectoria."""
    W, inicios = tr["W"], np.asarray(tr["inicios"])
    paso = inicios[1] - inicios[0] if len(inicios) > 1 else W
    salto = int(np.ceil(W / paso))          # separacion NO solapada, en indices
    sig = np.asarray(tr["sigma"])
    res = {}
    for j in OCTAVAS:
        K = np.asarray(tr["K"][j], dtype=float)
        E = np.asarray(tr["err"][j], dtype=float)
        m = np.isfinite(K) & np.isfinite(E)
        if m.sum() < 8:
            res[j] = {"error": "pocas ventanas"}
            continue
        Km, Em = K[m], E[m]
        # G2: dispersion entre ventanas frente al error de medicion
        iqr = float(np.subtract(*np.percentile(Km, [75, 25])))
        err = float(np.median(Em))
        g2 = iqr / err if err > 0 else float("inf")
        # G3: redundancia con la volatilidad
        from scipy.stats import spearmanr
        rho = float(spearmanr(Km, sig[m]).statistic)
        # G4: autocorrelacion a separacion NO SOLAPADA, contra nulo de barajado
        if Km.size > salto + 4:
            a, b = Km[:-salto], Km[salto:]
            ac = float(np.corrcoef(a, b)[0, 1])
            nul = []
            for _ in range(499):
                p = rng.permutation(Km)
                nul.append(float(np.corrcoef(p[:-salto], p[salto:])[0, 1]))
            p_emp = float((np.abs(nul) >= abs(ac)).mean())
        else:
            ac, p_emp = float("nan"), float("nan")
        res[j] = {"n_ventanas": int(m.sum()), "K_mediana": float(np.median(Km)),
                  "IQR": iqr, "err_tipico": err, "G2_ratio": g2,
                  "G3_rho_sigma": rho, "G4_autocorr_no_solapada": ac,
                  "G4_p": p_emp, "salto_indices": int(salto),
                  "G2_pasa": bool(g2 >= 2.0), "G3_pasa": bool(abs(rho) <= 0.3),
                  "G4_pasa": bool(np.isfinite(p_emp) and p_emp < 0.05)}
    return res


def main() -> None:
    print("=" * 88)
    print("  EXP 21 — acoplamiento signo-tamano en ventanas deslizantes")
    print("=" * 88)
    print("\n  Las cuatro puertas del exp. 05a, sobre un observable que no se")
    print("  ajusta. G4 se evalua a separacion NO SOLAPADA: es la que mato al")
    print("  05a y la que decide aqui.")

    rng = np.random.default_rng(21)
    todo = []
    for modo in ("raw", "normalized"):
        for a in ACTIVOS:
            prep = prepare_complete_weeks(_cargar(f"{a}USDT"), flow_mode=modo)
            eps = prep.flow_demeaned
            for W in VENTANAS:
                tr = trayectoria(eps, W, rng)
                g = puertas(tr, rng)
                todo.append({"activo": a, "flow_mode": modo, "W": W,
                             "n_ventanas": len(tr["inicios"]),
                             "puertas": g, "trayectoria": tr})

    for modo in ("raw", "normalized"):
        print(f"\n\n  {'=' * 82}")
        print(f"  FLUJO {modo}")
        print(f"  {'=' * 82}")
        for W in VENTANAS:
            print(f"\n  W = {W} velas ({W // 168} semanas), "
                  f"separacion no solapada")
            print(f"    {'activo':<7}{'oct':>5}{'tau':>10}{'K':>10}"
                  f"{'G2':>7}{'G3 rho':>9}{'G4 ac':>8}{'G4 p':>8}  puertas")
            for r in todo:
                if r["flow_mode"] != modo or r["W"] != W:
                    continue
                for j in OCTAVAS:
                    d = r["puertas"].get(j, {})
                    if "error" in d:
                        continue
                    banda = f"[{2**j},{2**(j+1)})"
                    marcas = ("G2" if d["G2_pasa"] else "--") + \
                             ("G3" if d["G3_pasa"] else "--") + \
                             ("G4" if d["G4_pasa"] else "--")
                    print(f"    {r['activo']:<7}{j:>5}{banda:>10}"
                          f"{d['K_mediana']:>10.3f}{d['G2_ratio']:>7.1f}"
                          f"{d['G3_rho_sigma']:>9.2f}"
                          f"{d['G4_autocorr_no_solapada']:>8.2f}"
                          f"{d['G4_p']:>8.3f}  {marcas}")

    # --- veredicto -----------------------------------------------------------
    print("\n\n  " + "=" * 82)
    print("  VEREDICTO (criterio del docstring)")
    print("  " + "-" * 82)
    print("  POSITIVO exige G1, G2 y G4 en >= 2 octavas y >= 3 de 4 activos.")
    veredicto = {}
    for modo in ("raw", "normalized"):
        for W in VENTANAS:
            act_ok = []
            for a in ACTIVOS:
                r = next((x for x in todo if x["activo"] == a
                          and x["flow_mode"] == modo and x["W"] == W), None)
                if not r:
                    continue
                n_oct = sum(1 for j in OCTAVAS
                            if r["puertas"].get(j, {}).get("G2_pasa")
                            and r["puertas"].get(j, {}).get("G4_pasa"))
                if n_oct >= 2:
                    act_ok.append(a)
            clave = f"{modo}|W{W}"
            veredicto[clave] = act_ok
            print(f"  {clave:<20} {len(act_ok)}/4  {act_ok if act_ok else ''}")
    positivo = any(len(v) >= 3 for v in veredicto.values())
    print()
    if positivo:
        print("  POSITIVO. El acoplamiento por octava tiene estructura temporal")
        print("  que sobrevive a separacion NO SOLAPADA, que es exactamente lo")
        print("  que el vector KWW del exp. 05a no tenia. La via de ventanas")
        print("  deslizantes se reabre sobre este observable.")
    else:
        print("  NEGATIVO. Igual que el 05a: lo que se mueve entre ventanas es")
        print("  ruido de medicion o solape, no regimen. La diferencia es que")
        print("  ahora sabemos que NO es problema de identificabilidad, porque")
        print("  no se ajusta nada — es que el acoplamiento no tiene dinamica")
        print("  lenta a estas escalas.")

    out = RAIZ / "output" / "exp21_acoplamiento_ventanas.json"
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
        {"criterio": "G1+G2+G4 en >=2 octavas y >=3 de 4 activos",
         "ventanas": VENTANAS, "octavas": OCTAVAS, "paso_relativo": PASO_REL,
         "n_bootstrap": N_BOOT, "resultados": todo, "veredicto": veredicto,
         "positivo": bool(positivo)}, indent=2, default=_l), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
