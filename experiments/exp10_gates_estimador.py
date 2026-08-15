r"""
EXPERIMENTO 10 — los tres gates defensivos que el barrido de literatura exigió.

Ninguno de los tres busca un hallazgo. Los tres existen para cerrar objeciones
concretas que la revisión de 2026-08-13 identificó como las más probables de
un árbitro, y cada uno lleva su propio control.

GATE A — ¿depende nuestro γ del estimador?
    Lillo & Farmer obtienen 0.39 con OLS log-log sobre la ACF y 0.61 con
    periodograma SOBRE LOS MISMOS DATOS, y dicen explícitamente que la
    autocorrelación muestral "is a poor method for estimating γ". Nosotros
    usamos OLS log-log. Aquí se calcula γ por TRES vías —OLS sobre C_ε,
    log-periodograma (GPH) y DFA— sobre las mismas series, más la dependencia
    con el rango de ajuste. Control positivo: las tres se validan primero
    contra series sintéticas de γ CONOCIDO (`counting.synthetic_flow`), así
    que un desacuerdo en datos reales es del dato y no del estimador.

    Conversiones usadas: C(τ) ~ τ^(−γ) ⟺ H = 1 − γ/2 ⟺ d = H − 1/2, y el
    espectro f(ω) ~ ω^(−2d) cuando ω → 0.

GATE B — ¿sobrevive la modulación de fase en tiempo de TRADE?
    Barardehi & Bernhardt (J. Financial Markets 74, 2025) demuestran que
    medir en tiempo de calendario cuando la intensidad de trading modula con
    la fase produce sesgo de sobre-agregación, y que ese sesgo FABRICA las
    U-shapes en la lambda de Kyle. Nuestro exp. 08 midió R(1) por fase en
    tiempo de reloj, y la actividad modula ×1.9. Dos réplicas independientes:
      B1 — actividad EMPAREJADA: dentro de cada fase, usar solo velas cuyo
           número de trades cae en una banda común a todas las fases. Si la
           modulación sobrevive con actividad igualada, no es agregación.
      B2 — reloj de TRADE grueso: agregar velas consecutivas hasta alcanzar
           un número objetivo de trades, formando barras de actividad ~
           constante y duración variable.
    No tenemos trades individuales (las klines solo traen el conteo), así que
    B2 es un reloj de trade GRUESO y se declara como tal.

GATE C — ¿puede el sesgo de Kendall explicar el aging aparente del exp. 07?
    Kendall (Biometrika 41, 403, 1954): la autocorrelación muestral está
    sesgada a la baja en muestras finitas, y el sesgo del tiempo de
    correlación ajustado crece con el número de puntos. Si nuestra longitud
    efectiva de muestra co-variase con la edad t_w, tendríamos un segundo
    artefacto además del de normalización. Dos partes:
      C1 — cuantificar el sesgo sobre AR(1) sintético de ρ conocido, para
           tener la magnitud.
      C2 — medir el tamaño de muestra efectivo por bin de edad en la
           geometría real del exp. 07. Si es constante en t_w, Kendall no
           puede producir un perfil en edad y queda descartado.

Ejecutar:  py experiments/exp10_gates_estimador.py [n_procesos]
Corre en LOCAL (~1 min).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance import ewma_vol
from keldysh_finance.counting import _ols_pendiente
from keldysh_finance.wavelets import fgn, gamma_abry_veitch, varianza_haar
from keldysh_finance.flow import (fetch_klines_with_flow, flow_autocorrelation,
                                  order_flow_imbalance, response_from_frame)
from keldysh_finance.quench import campo_post_shock, detectar_shocks

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
CONF = dict(years=4.0, acf_lags=60, rangos_ols=[(1, 60), (2, 30), (5, 60)],
            gph_potencia=0.5, dfa_orden=1, n_bins_fase=8,
            banda_trades=(0.35, 0.65), trades_por_barra_frac=1.5,
            umbral_q=0.995, separacion=384, u_max=300, margen_pre=200)


# ---------------------------------------------------------------- GATE A ----

def gamma_ols(eps: np.ndarray, lo: int, hi: int) -> dict:
    """γ por OLS log-log sobre C_ε(τ) — el estimador que usamos hasta ahora."""
    C = flow_autocorrelation(eps, max_lag=CONF["acf_lags"])
    tau = np.arange(1, CONF["acf_lags"] + 1, dtype=float)
    m = (tau >= lo) & (tau <= hi) & np.isfinite(C) & (C > 0)
    if m.sum() < 8:
        return {"gamma": np.nan, "se": np.nan, "n": int(m.sum())}
    b, se = _ols_pendiente(np.log(tau[m]), np.log(C[m]))
    return {"gamma": -b, "se": se, "n": int(m.sum())}


def gamma_gph(eps: np.ndarray, potencia: float = 0.5) -> dict:
    r"""γ por regresión log-periodograma (Geweke--Porter-Hudak).

    Para memoria larga, f(ω) ~ ω^(−2d) cuando ω → 0, luego
    log I(ω_j) = c − d·log[4 sin²(ω_j/2)] + ruido.
    La pendiente de esa regresión es −d, y γ = 1 − 2d.
    Se usan las m = N^`potencia` frecuencias de Fourier más bajas (m ≈ √N es
    la elección estándar).
    """
    x = np.asarray(eps, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 500:
        return {"gamma": np.nan, "se": np.nan, "m": 0}
    xc = x - x.mean()
    per = np.abs(np.fft.rfft(xc)) ** 2 / (2.0 * np.pi * n)
    m = int(n ** potencia)
    j = np.arange(1, m + 1)
    omega = 2.0 * np.pi * j / n
    reg = np.log(4.0 * np.sin(omega / 2.0) ** 2)
    y = np.log(per[j])
    fin = np.isfinite(y) & np.isfinite(reg)
    if fin.sum() < 10:
        return {"gamma": np.nan, "se": np.nan, "m": 0}
    b, se = _ols_pendiente(reg[fin], y[fin])
    d = -b
    return {"gamma": 1.0 - 2.0 * d, "se": 2.0 * se, "d": d, "m": int(fin.sum())}


def gamma_dfa(eps: np.ndarray, orden: int = 1, n_min: int = 16,
              n_max_frac: float = 0.1, n_escalas: int = 20) -> dict:
    r"""γ por Detrended Fluctuation Analysis: F(n) ~ n^H, γ = 2 − 2H.

    Es el estimador que usan Gould, Porter & Howison en FX, y por eso el
    comparable directo de nuestras cifras con las suyas.
    """
    x = np.asarray(eps, dtype=float)
    x = x[np.isfinite(x)]
    n_tot = len(x)
    if n_tot < 500:
        return {"gamma": np.nan, "se": np.nan, "H": np.nan}
    y = np.cumsum(x - x.mean())
    escalas = np.unique(np.geomspace(n_min, max(n_min * 2, int(n_tot * n_max_frac)),
                                     n_escalas).astype(int))
    f_n = []
    for n in escalas:
        n_seg = n_tot // n
        if n_seg < 4:
            continue
        seg = y[:n_seg * n].reshape(n_seg, n)
        t = np.arange(n, dtype=float)
        # ajuste polinómico por segmento, vectorizado
        V = np.vander(t, orden + 1)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        resid = seg.T - V @ coef
        f_n.append((n, float(np.sqrt(np.mean(resid ** 2)))))
    if len(f_n) < 5:
        return {"gamma": np.nan, "se": np.nan, "H": np.nan}
    ns = np.array([a for a, _ in f_n], dtype=float)
    fs = np.array([b for _, b in f_n], dtype=float)
    ok = fs > 0
    H, se = _ols_pendiente(np.log(ns[ok]), np.log(fs[ok]))
    return {"gamma": 2.0 - 2.0 * H, "se": 2.0 * se, "H": H,
            "n_escalas": int(ok.sum())}


def gate_a_control() -> list[dict]:
    r"""Control positivo sobre γ CONOCIDO, con ruido gaussiano fraccionario.

    La serie de prueba es fGn EXACTO (Davies-Harte), no el generador de
    `counting.synthetic_flow`. Es una corrección con consecuencias: aquel
    empotra una ACF (1+τ)^γ en un círculo y recorta autovalores negativos, lo
    que reproduce bien los desfases cortos pero distorsiona las frecuencias
    BAJAS —justo las que leen el log-periodograma, el DFA y la ondícula—. La
    versión anterior de este control, hecha sobre él, atribuía a DFA un sesgo
    sistemático de 0.15-0.19 que en realidad era del generador: sobre fGn
    exacto DFA es de los estimadores más fieles.
    """
    filas = []
    for gamma_true in (0.4, 0.5, 0.6, 0.7, 0.8):
        x = fgn(400_000, H=1.0 - gamma_true / 2.0, seed=int(gamma_true * 100))
        filas.append({
            "gamma_true": gamma_true,
            "ols": gamma_ols(x, 1, CONF["acf_lags"])["gamma"],
            "gph": gamma_gph(x)["gamma"],
            "dfa": gamma_dfa(x)["gamma"],
            "wav": gamma_abry_veitch(x, longitud=6, j_min=3)["gamma"]})
    return filas


def gate_a(nombre: str, eps: np.ndarray) -> dict:
    res = {"activo": nombre,
           "gph": gamma_gph(eps, CONF["gph_potencia"]),
           "dfa": gamma_dfa(eps, CONF["dfa_orden"]),
           "wav": gamma_abry_veitch(eps, longitud=6, j_min=3),
           "ols_por_rango": {}}
    for lo, hi in CONF["rangos_ols"]:
        res["ols_por_rango"][f"[{lo},{hi}]"] = gamma_ols(eps, lo, hi)
    vals = [res["gph"]["gamma"], res["dfa"]["gamma"]] + \
           [v["gamma"] for v in res["ols_por_rango"].values()]
    vals = [v for v in vals if np.isfinite(v)]
    res["dispersion_estimadores"] = (float(np.max(vals) - np.min(vals))
                                     if len(vals) > 1 else float("nan"))
    return res


# ---------------------------------------------------------------- GATE B ----

def impacto_por_fase(eps: np.ndarray, dp: np.ndarray, horas: np.ndarray,
                     mascara: np.ndarray, n_bins: int) -> list[float]:
    """R(1) por bin de fase, con ε centrado DENTRO de cada fase."""
    ancho = 24 // n_bins
    out = []
    for b in range(n_bins):
        m = mascara & ((horas // ancho) == b)
        if m.sum() < 200:
            out.append(np.nan)
            continue
        e = eps[m] - eps[m].mean()
        var = float(np.mean(e ** 2))
        out.append(float(np.mean(dp[m] * e) / var) if var > 0 else np.nan)
    return out


def gate_b(nombre: str, d: dict) -> dict:
    """B1 actividad emparejada, B2 reloj de trade grueso."""
    n_bins = CONF["n_bins_fase"]
    eps, tr, horas = d["eps"], d["trades"], d["horas"]
    n = min(len(eps), len(d["log_open"]) - 1, len(tr), len(horas))
    eps, tr, horas = eps[:n], tr[:n], horas[:n]
    dp = d["log_open"][1:n + 1] - d["log_open"][:n]
    fin = np.isfinite(eps) & np.isfinite(dp) & np.isfinite(tr) & (tr > 0)

    r_todo = impacto_por_fase(eps, dp, horas, fin, n_bins)

    # B1: banda común de actividad (cuantiles GLOBALES del conteo de trades)
    lo, hi = np.quantile(tr[fin], CONF["banda_trades"])
    m_banda = fin & (tr >= lo) & (tr <= hi)
    r_banda = impacto_por_fase(eps, dp, horas, m_banda, n_bins)

    # B2: reloj de trade grueso — agregar velas hasta un objetivo de trades
    objetivo = float(np.median(tr[fin])) * CONF["trades_por_barra_frac"]
    idx, acum, barras = [], 0.0, []
    for i in np.flatnonzero(fin):
        idx.append(i)
        acum += tr[i]
        if acum >= objetivo:
            barras.append(np.array(idx))
            idx, acum = [], 0.0
    r_trade = [np.nan] * n_bins
    if len(barras) > 8 * 200:
        ancho = 24 // n_bins
        e_b = np.array([eps[b].mean() for b in barras])
        dp_b = np.array([dp[b].sum() for b in barras])
        h_b = np.array([horas[b[len(b) // 2]] for b in barras])
        n_b = np.array([len(b) for b in barras])
        fin_b = np.isfinite(e_b) & np.isfinite(dp_b)
        r_trade = impacto_por_fase(e_b, dp_b, h_b, fin_b, n_bins)

    def modul(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return (float(np.max(v) / np.min(v)) if len(v) >= 3 and np.min(v) > 0
                else float("nan"))

    return {"activo": nombre, "n_bins": n_bins,
            "R1_calendario": r_todo, "R1_banda_actividad": r_banda,
            "R1_reloj_trade": r_trade,
            "razon_max_min": {"calendario": modul(r_todo),
                              "banda_actividad": modul(r_banda),
                              "reloj_trade": modul(r_trade)},
            "banda_trades": [float(lo), float(hi)],
            "n_barras_trade": len(barras),
            "velas_por_barra_media": (float(np.mean([len(b) for b in barras]))
                                      if barras else float("nan"))}


# ---------------------------------------------------------------- GATE C ----

def gate_c1_sintetico() -> list[dict]:
    """Magnitud del sesgo de Kendall: τ̂ vs longitud de muestra, ρ conocido."""
    rng = np.random.default_rng(1010)
    filas = []
    for rho in (0.80, 0.90, 0.95):
        tau_true = -1.0 / np.log(rho)
        for n in (50, 100, 200, 400, 1600):
            taus = []
            for _ in range(400):
                x = np.empty(n)
                x[0] = rng.normal(0, 1 / np.sqrt(1 - rho ** 2))
                for t in range(1, n):
                    x[t] = rho * x[t - 1] + rng.normal()
                xc = x - x.mean()
                den = float(np.dot(xc, xc))
                r1 = float(np.dot(xc[:-1], xc[1:]) / den) if den > 0 else np.nan
                if np.isfinite(r1) and 0 < r1 < 1:
                    taus.append(-1.0 / np.log(r1))
            filas.append({"rho": rho, "n": n, "tau_true": tau_true,
                          "tau_medido": float(np.median(taus)) if taus else np.nan,
                          "fraccion": (float(np.median(taus)) / tau_true
                                       if taus else np.nan)})
    return filas


def gate_c2_geometria(nombre: str, d: dict, bordes: list[int]) -> dict:
    """¿Varía el tamaño de muestra efectivo con la EDAD en el exp. 07?

    Si el número de pares finitos que entra en cada estimación de correlación
    es constante en t_w, el sesgo de Kendall no puede generar un perfil en
    edad, y queda descartado como explicación del aging aparente.
    """
    sigma = ewma_vol(d["r"])
    shocks, _ = detectar_shocks(d["r"], sigma, umbral_q=CONF["umbral_q"],
                                separacion=CONF["separacion"],
                                margen_post=CONF["u_max"] + 5,
                                margen_pre=CONF["margen_pre"])
    W = campo_post_shock(d["r"], sigma, shocks, u_max=CONF["u_max"])
    W = W[np.isfinite(W).all(axis=1)]
    if len(W) < 20:
        return {"activo": nombre, "error": "ensemble insuficiente"}

    filas = []
    for b in range(len(bordes) - 1):
        a, z = bordes[b], bordes[b + 1]
        cuentas = []
        for t_w in range(a, min(z, W.shape[1] - 1)):
            prod = W[:, t_w] * W[:, min(t_w + 1, W.shape[1] - 1)]
            cuentas.append(int(np.isfinite(prod).sum()))
        filas.append({"bin": f"[{a},{z})",
                      "n_efectivo_medio": (float(np.mean(cuentas))
                                           if cuentas else float("nan"))})
    ns = [f["n_efectivo_medio"] for f in filas if np.isfinite(f["n_efectivo_medio"])]
    return {"activo": nombre, "n_shocks": int(W.shape[0]), "por_bin": filas,
            "n_efectivo_constante": bool(len(set(np.round(ns, 6))) == 1),
            "rango_relativo": (float((max(ns) - min(ns)) / max(ns))
                               if ns else float("nan"))}


# ------------------------------------------------------------------ main ----

def datos(symbol: str) -> dict:
    df = fetch_klines_with_flow(symbol, interval="1h",
                                years=CONF["years"]).dropna()
    _, _, eps, log_open = response_from_frame(df, max_lag=30)
    return {"eps": eps, "log_open": log_open,
            "r": np.diff(np.log(df["Close"].to_numpy(float))),
            "trades": df["trades"].to_numpy(float),
            "horas": df.index.hour.to_numpy()}


def main() -> None:
    print("\n  EXPERIMENTO 10 — gates defensivos (estimador, reloj, Kendall)\n",
          flush=True)
    t0 = time.time()

    print("  GATE A — control positivo sobre gamma conocido:")
    ctrl = gate_a_control()
    print(f"    {'gamma_true':>11}{'OLS':>9}{'GPH':>9}{'DFA':>9}{'WAV':>9}")
    for f in ctrl:
        print(f"    {f['gamma_true']:>11.2f}{f['ols']:>9.3f}"
              f"{f['gph']:>9.3f}{f['dfa']:>9.3f}{f['wav']:>9.3f}")
    err = {k: float(np.mean([abs(f[k] - f["gamma_true"]) for f in ctrl]))
           for k in ("ols", "gph", "dfa", "wav")}
    print(f"    {'|error| medio':>11}{err['ols']:>9.3f}{err['gph']:>9.3f}"
          f"{err['dfa']:>9.3f}{err['wav']:>9.3f}")

    cache = {n: datos(s) for n, s in ACTIVOS}

    print("\n  GATE A — datos reales (1h):")
    print(f"    {'activo':<7}{'OLS[1,60]':>11}{'OLS[2,30]':>11}{'OLS[5,60]':>11}"
          f"{'GPH':>9}{'DFA':>9}{'WAV':>9}{'se_W':>7}{'chi2':>8}{'disp':>8}")
    a_res = []
    for nombre, _ in ACTIVOS:
        r = gate_a(nombre, cache[nombre]["eps"])
        a_res.append(r)
        o = r["ols_por_rango"]
        print(f"    {nombre:<7}{o['[1,60]']['gamma']:>11.3f}"
              f"{o['[2,30]']['gamma']:>11.3f}{o['[5,60]']['gamma']:>11.3f}"
              f"{r['gph']['gamma']:>9.3f}{r['dfa']['gamma']:>9.3f}"
              f"{r['wav']['gamma']:>9.3f}{r['wav']['se']:>7.3f}"
              f"{r['wav']['chi2_dof']:>8.1f}"
              f"{r['dispersion_estimadores']:>8.3f}")

    print("\n  GATE B — impacto R(1) por fase, tres relojes:")
    print(f"    {'activo':<7}{'calendario':>12}{'banda act.':>12}{'reloj trade':>13}"
          f"   (razon max/min)")
    b_res = []
    for nombre, _ in ACTIVOS:
        r = gate_b(nombre, cache[nombre])
        b_res.append(r)
        z = r["razon_max_min"]
        print(f"    {nombre:<7}{z['calendario']:>12.2f}"
              f"{z['banda_actividad']:>12.2f}{z['reloj_trade']:>13.2f}")

    print("\n  GATE C1 — sesgo de Kendall sobre AR(1) sintetico "
          "(tau medido / tau real):")
    c1 = gate_c1_sintetico()
    for rho in (0.80, 0.90, 0.95):
        fila = [f for f in c1 if f["rho"] == rho]
        s = "  ".join(f"n={f['n']}:{f['fraccion']:.2f}" for f in fila)
        print(f"    rho={rho}: {s}")

    print("\n  GATE C2 — tamano de muestra efectivo por edad (geometria exp.07):")
    bordes = [8, 16, 32, 64, 128, 256]
    c2 = []
    for nombre, _ in ACTIVOS:
        r = gate_c2_geometria(nombre, cache[nombre], bordes)
        c2.append(r)
        if "error" in r:
            print(f"    {nombre}: {r['error']}")
            continue
        print(f"    {nombre:<7} n_shocks={r['n_shocks']:<5} "
              f"n_efectivo constante en la edad: "
              f"{'SI' if r['n_efectivo_constante'] else 'NO'} "
              f"(rango relativo {r['rango_relativo']:.1e})")

    todos_ctes = all(r.get("n_efectivo_constante", False) for r in c2)
    print(f"\n  → GATE C: {'Kendall DESCARTADO' if todos_ctes else 'revisar'} "
          f"— el tamano de muestra no varia con t_w, luego el sesgo de "
          f"Kendall\n    no puede generar un perfil en edad.")

    salida = {"conf": CONF, "gate_a_control": ctrl, "gate_a_error_medio": err, "gate_a": a_res, "gate_a_diagrama_lineal": None,
              "gate_b": b_res, "gate_c1": c1, "gate_c2": c2,
              "gate_c_veredicto": bool(todos_ctes)}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp10_gates_estimador.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  computo {time.time()-t0:.0f} s. Log depositado en {path}\n")


if __name__ == "__main__":
    main()
