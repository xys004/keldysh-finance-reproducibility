r"""
EXPERIMENTO 02 — ¿La violación del FDT mejora la predicción de volatilidad?

QUÉ AÑADE SOBRE EL 01
---------------------
El experimento 01 midió D: la ruptura de TTI del campo de volatilidad. Eso es
sólo la mitad del diagnóstico — usa la CORRELACIÓN y nada más. El diagnóstico
que la teoría pide de verdad es la relación entre RESPUESTA y CORRELACIÓN:

    T_eff(τ) = −[dC_ε/dτ](τ) / R(τ)

constante en equilibrio, dependiente de τ fuera de él. `rolling_fdt_violation`
resume esa NO constancia en un escalar por ventana trailing. Es la primera
característica del repo que usa el flujo de órdenes firmado, es decir, la
variable conjugada al precio — y no sólo el precio consigo mismo.

HIPÓTESIS (declarada antes de mirar resultados)
-----------------------------------------------
H1: la violación del FDT V(t_w), medida en ventanas trailing sobre el flujo
    firmado, aporta información sobre la volatilidad realizada futura que
    EWMA(0.94) y GARCH(1,1) recalibrados no capturan.

H0: no aporta nada. La diferencia de pérdida QLIKE frente al modelo
    recalibrado SIN la característica no es distinguible de cero
    (Diebold-Mariano, p >= 0.05).

CRITERIO DE DECISIÓN (fijado ex-ante, no renegociable después)
--------------------------------------------------------------
ÉXITO sólo si el modelo con V bate al modelo recalibrado sin V con p < 0.05
en el test DM sobre QLIKE, contra AMBAS líneas base y en AMBOS activos (4/4),
Y ADEMÁS los tres controles negativos dan 0/4. Cualquier otra combinación se
reporta como NO CONCLUYENTE. Con 2 activos × 2 baselines son 4 comparaciones y
el máximo de 4 tests sube el falso positivo a ~19% sin corregir.

CONTROLES NEGATIVOS — POR QUÉ TRES, Y POR QUÉ NO EL DEL 01
-----------------------------------------------------------
El control del experimento 01 (campo 'raw') resultó ser DÉBIL: D sobre
retornos crudos se sigue calculando a partir de la misma serie de precios, así
que puede colar información de volatilidad por la puerta de atrás a través del
ruido muestral. No era un nulo de verdad, y por eso el veredicto del 01 salió
"SOSPECHOSO" sin que quedara claro de qué. Aquí los controles son surrogados
que destruyen la estructura buscada dejando el resto intacto:

  perm     ε permutado (semilla fija). Conserva la distribución marginal del
           flujo y destruye su memoria y su alineación con el precio. Es el
           nulo natural de "la estructura a dos tiempos no informa".

  signflip |ε_t| intacto en el tiempo, signos aleatorios. Es el control
           EXIGENTE: la magnitud del desequilibrio (que va con la actividad, y
           la actividad va con la volatilidad) sobrevive; lo único que muere es
           la estructura FIRMADA, que es justo lo que el FDT mide. Si la
           característica real no bate a este control, lo que se estaba
           midiendo era actividad, no no-equilibrio.

  ruido    una característica AR(1) sin ninguna relación con los datos, con la
           misma persistencia aproximada que una ventana rodante de `window`
           barras. No prueba la física: prueba la MAQUINARIA. Si esto "mejora"
           la predicción, el problema está en el procedimiento de regresión y
           ningún resultado del repo es interpretable.

TRANSFORMACIÓN DE LA CARACTERÍSTICA (declarada ANTES de evaluar)
----------------------------------------------------------------
V es un cociente cuyo denominador R(τ) puede pasar cerca de cero, así que su
cola es salvaje por construcción (en BTC 4h: mediana ~3, máximo ~8·10³). Sin
tratarla, una sola ventana con R≈0 fija el coeficiente de la regresión. Se usa
por tanto, con estadísticos calculados SÓLO en el tramo de entrenamiento:

    X = zscore( winsorizar( log V , q01_train, q99_train ) )

log porque V es positiva y de tipo cociente; winsorizar por la división por
casi-cero; entrenamiento sólo porque cualquier estadístico global sería
look-ahead. La MISMA transformación se aplica a los controles.

DIFERENCIA METODOLÓGICA CON EL 01
----------------------------------
Aquí `recal` y `recal+X` se ajustan sobre EXACTAMENTE el mismo soporte (el de
X finita). En el 01 el modelo sin D se ajustaba con ~500 puntos iniciales de
más, lo que introducía una diferencia entre los dos modelos que no era D.

Ejecutar:  py experiments/exp02_fdt_vs_garch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance import fit_garch11
from keldysh_finance.pipeline import contraste_caracteristica
from keldysh_finance.flow import (fetch_klines_with_flow, response_from_frame,
                                  rolling_fdt_violation, align_to_returns)

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]

CONF = dict(
    interval="4h",
    years=4.0,
    window=500,        # ventana trailing del diagnóstico FDT (~83 días de 4h)
    max_lag=30,        # desfases sobre los que se mide la no constancia de T_eff
    step=1,
    horizon=30,        # vol realizada de las próximas 30 barras (~5 días)
    train_frac=0.60,
    winsor=(0.01, 0.99),
    seed=20260808,
)

VARIANTES = [
    ("real",     "flujo firmado real"),
    ("perm",     "CONTROL: flujo permutado (marginal intacta, memoria destruida)"),
    ("signflip", "CONTROL: |eps| intacto, signos aleatorios (muere solo lo firmado)"),
    ("ruido",    "CONTROL: caracteristica AR(1) ajena a los datos (prueba la maquinaria)"),
]

# Semilla por variante y activo, EXPLÍCITA. `hash()` sobre str está salado por
# proceso en Python (PYTHONHASHSEED), así que usarlo aquí haría el experimento
# irreproducible entre ejecuciones sin que nada lo delatara.
_OFFSET_VARIANTE = {v: k for k, (v, _) in enumerate(VARIANTES)}
_OFFSET_ACTIVO = {sym: k for k, (_, sym) in enumerate(ACTIVOS)}

_CACHE_DF: dict[str, object] = {}
# El GARCH depende sólo del activo, no de la variante. Cachearlo no es sólo
# ahorro: garantiza que las cuatro variantes se comparan contra EXACTAMENTE la
# misma línea base, sin que un óptimo local distinto del ajuste introduzca
# diferencias que luego se atribuirían a la característica.
_CACHE_GARCH: dict[str, dict] = {}


def _datos(symbol: str):
    if symbol not in _CACHE_DF:
        _CACHE_DF[symbol] = fetch_klines_with_flow(
            symbol, interval=CONF["interval"], years=CONF["years"]).dropna()
    return _CACHE_DF[symbol]


def caracteristica_cruda(symbol: str, variante: str, n_r: int) -> np.ndarray:
    """V(t) en la rejilla de RETORNOS (longitud n_r), sin transformar todavía.

    Para las variantes de flujo, los PRECIOS son siempre los reales: lo único
    que se altera es la estructura del flujo. Así el control responde a la
    pregunta correcta —dados estos precios, ¿informa la estructura firmada del
    flujo?— y no a "¿informa una serie de precios distinta?".
    """
    rng = np.random.default_rng(CONF["seed"]
                                + 100 * _OFFSET_VARIANTE[variante]
                                + _OFFSET_ACTIVO[symbol])

    if variante == "ruido":
        # AR(1) con tiempo de correlación ~ window, para que la persistencia se
        # parezca a la de un estadístico calculado en ventanas solapadas. Se
        # exponencia para que sea positiva y el log de la transformación común
        # tenga sentido: log(V) es entonces exactamente el AR(1).
        phi = float(np.exp(-1.0 / CONF["window"]))
        x = np.empty(n_r)
        x[0] = 0.0
        for t in range(1, n_r):
            x[t] = phi * x[t - 1] + rng.normal(0.0, 1.0)
        V = np.exp(x)
        V[:CONF["window"] - 1] = np.nan      # misma disponibilidad que la real
        return V

    df = _datos(symbol)
    _, _, eps, log_open = response_from_frame(df, max_lag=CONF["max_lag"])
    if variante == "real":
        flujo = eps
    elif variante == "perm":
        flujo = rng.permutation(eps)
    elif variante == "signflip":
        flujo = np.abs(eps) * rng.choice([-1.0, 1.0], size=len(eps))
    else:
        raise ValueError(f"variante desconocida: {variante!r}")

    t_w, V = rolling_fdt_violation(log_open, flujo, window=CONF["window"],
                                   max_lag=CONF["max_lag"], step=CONF["step"])
    V_r = align_to_returns(t_w, V, n_klines=len(log_open))
    assert len(V_r) == n_r, f"desalineacion: {len(V_r)} != {n_r}"
    return V_r


def contrastar(nombre: str, symbol: str, etiqueta: str, V: np.ndarray,
               verbose: bool = True) -> dict:
    """Pipeline completo a partir de la caracteristica CRUDA V.

    El trabajo lo hace `pipeline.contraste_caracteristica`, que es codigo de
    LIBRERIA y no de este script. Importa que sea asi por dos razones: el nulo
    por permutacion (`exp02b`) tiene que pasar por el mismo camino que el
    resultado que calibra, y el barrido de escalas (`exp03`) tiene que medir
    con la misma regla o sus cifras no serian comparables con estas.
    """
    df = _datos(symbol)
    r = np.diff(np.log(df["Close"].to_numpy(float)))
    split = int(len(r) * CONF["train_frac"])
    if symbol not in _CACHE_GARCH:
        _CACHE_GARCH[symbol] = fit_garch11(r[:split])

    res = contraste_caracteristica(r, V, horizon=CONF["horizon"],
                                   train_frac=CONF["train_frac"],
                                   winsor=CONF["winsor"],
                                   garch_params=_CACHE_GARCH[symbol])
    res["activo"], res["variante"] = nombre, etiqueta
    if "error" in res:
        if verbose:
            print(f"  {nombre}/{etiqueta}: caracteristica degenerada, se omite")
        return res

    if verbose:
        print(f"\n  {'-'*74}\n  {nombre} / {etiqueta}   "
              f"(n={res['n']}, split={res['split']})")
        if res["coef_X"]:
            print(f"  coef_X: EWMA {res['coef_X']['EWMA']:+.4f}   "
                  f"GARCH {res['coef_X']['GARCH']:+.4f}   "
                  f"corr_oos(X, log rv) = {res['corr_oos_X_logrv']:+.3f}")
        print(f"  {'modelo':<20}{'QLIKE':>10}{'RMSE':>12}")
        for k, m in res["modelos"].items():
            print(f"  {k:<20}{m['qlike']:>10.4f}{m['rmse']:>12.6f}")
        for k, dm in res["dm_contexto"].items():
            print(f"    [contexto] {k:<26} DM={dm['dm_stat']:+7.3f} "
                  f"p={dm['p_value']:.4f}   <- solo recalibrar")
        for k, dm in res["dm_decisivo"].items():
            ver = ("X APORTA" if dm["p_value"] < 0.05 and dm["mean_diff"] > 0
                   else "X PERJUDICA" if dm["p_value"] < 0.05 else "X no aporta")
            print(f"    [DECIDE ] {k:<26} DM={dm['dm_stat']:+7.3f} "
                  f"p={dm['p_value']:.4f}  n={dm['n']:<5} -> {ver}")
    return res


def evaluar(nombre: str, symbol: str, variante: str) -> dict:
    n = len(_datos(symbol)) - 1                 # rejilla de log-retornos
    V = caracteristica_cruda(symbol, variante, n_r=n)
    return contrastar(nombre, symbol, variante, V, verbose=True)


def cuenta_exitos(bloque: list[dict]) -> int:
    ok = 0
    for r in bloque:
        for dm in r.get("dm_decisivo", {}).values():
            if np.isfinite(dm.get("p_value", np.nan)) and \
               dm["p_value"] < 0.05 and dm["mean_diff"] > 0:
                ok += 1
    return ok


def main() -> None:
    print("\n  EXPERIMENTO 02 — violacion del FDT como predictor de volatilidad")
    print(f"  Config: {CONF}")
    print("\n  CRITERIO EX-ANTE: exito solo si X bate al modelo recalibrado sin X")
    print("  con p<0.05 en AMBAS baselines y AMBOS activos (4/4), y los tres")
    print("  controles dan 0/4. Cualquier otra cosa = NO CONCLUYENTE.")

    resultados = {"conf": CONF, "variantes": {}}
    for variante, desc in VARIANTES:
        print(f"\n\n  {'#'*74}\n  VARIANTE '{variante}' — {desc}\n  {'#'*74}")
        resultados["variantes"][variante] = [evaluar(nombre, sym, variante)
                                             for nombre, sym in ACTIVOS]

    total = 2 * len(ACTIVOS)
    exitos = {v: cuenta_exitos(resultados["variantes"][v]) for v, _ in VARIANTES}
    print(f"\n\n  {'='*74}\n  VEREDICTO\n  {'='*74}")
    for v, desc in VARIANTES:
        etiqueta = "REAL   " if v == "real" else "control"
        print(f"  {etiqueta} {v:<9}: {exitos[v]}/{total} comparaciones con mejora significativa")

    # El criterio ex-ante de ÉXITO no se toca: 4/4 en real Y 0/4 en los tres
    # controles. Lo que sigue sólo ETIQUETA los modos de no-éxito, y lo hace en
    # el orden correcto: si la variante real da 0/4, H0 se mantiene por sí sola
    # y lo que hagan los controles es diagnóstico del procedimiento, no una
    # conclusión distinta sobre H1.
    ctrl = {v: exitos[v] for v, _ in VARIANTES if v != "real"}
    if exitos["real"] == total and all(c == 0 for c in ctrl.values()):
        veredicto = "H1 SOBREVIVE — merece continuar"
    elif exitos["real"] == 0:
        veredicto = "H0 NO SE RECHAZA — la violacion del FDT no aporta"
    elif exitos.get("ruido", 0) > 0:
        veredicto = ("MAQUINARIA COMPROMETIDA — una caracteristica ajena tambien 'mejora'; "
                     "el procedimiento de regresion fabrica senal")
    elif any(c > 0 for c in ctrl.values()):
        veredicto = ("SOSPECHOSO — algun control tambien 'mejora': lo medido no es "
                     "la estructura firmada")
    else:
        veredicto = "NO CONCLUYENTE — mejora parcial; no basta con el criterio ex-ante"
    print(f"  → {veredicto}")

    aviso = None
    if any(c > 0 for c in ctrl.values()):
        peor = max(ctrl, key=lambda v: ctrl[v])
        aviso = (f"AVISO: el control '{peor}' marca {ctrl[peor]}/{total}. Con una sola "
                 f"realizacion del surrogado eso no distingue casualidad de sesgo del "
                 f"procedimiento; lo calibra exp02b (nulo por permutacion, K draws).")
        print(f"  {aviso}")
    resultados["exitos"] = exitos
    resultados["veredicto"] = veredicto
    resultados["aviso_control"] = aviso

    OUT.mkdir(exist_ok=True)
    path = OUT / "exp02_fdt_vs_garch.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
