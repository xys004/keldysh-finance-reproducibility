r"""
EXPERIMENTO 05a — ¿son MEDIBLES los parámetros de la función de Green local?

QUÉ ES ESTO (y qué no es)
-------------------------
Fase A, DESCRIPTIVA, del programa "ajustar formas funcionales de relajación a
ventanas y estudiar la trayectoria de sus parámetros". Aquí NO hay contraste
predictivo, y por diseño: los experimentos 01-04 cerraron la vía de los
ESCALARES (D, violación del FDT) como predictores. Lo único que las reglas
del proyecto dejan abierto es una característica NUEVA, y el vector de
parámetros KWW por ventana lo es — pero antes de gastar una fase predictiva
pre-registrada hay que responder la pregunta previa:

    con UNA realización por ventana, ¿se pueden siquiera MEDIR τ_c y β?

Si la respuesta es no, la fase B no tiene sentido y el programa muere aquí,
que es más barato que morir tras otro barrido con maxT.

QUÉ SE AJUSTA
-------------
A cada perfil local ρ(t_w, τ) de la superficie a dos tiempos (ventanas
TRAILING, causalidad heredada) se le ajusta la exponencial estirada

    ρ(t_w, τ) ≈ A · exp[ −(τ/τ_c)^β ]

y se obtiene la trayectoria { (log A, log τ_c, β)(t_w) } con sus errores.
A es pariente de σ y por el argumento algebraico del exp. 04 su contenido ya
está probado (negativo). Los candidatos con contenido potencialmente nuevo
son τ_c (escala de memoria) y β (forma de la relajación: β<1 = estirada).
Aparte, sobre el flujo: T_eff mediana por ventana (nivel del cociente FDT).

CRITERIOS GO/NO-GO PARA LA FASE B (declarados ANTES de mirar resultados)
------------------------------------------------------------------------
Son puertas descriptivas, no p-valores; se declaran ex-ante para no
renegociarlas mirando la tabla (regla 5). La fase B sólo se diseña si:

  G1 MEDICIÓN       fracción de ajustes válidos (convergió, sin tocar cotas,
                    R² ≥ 0.2) ≥ 0.50 en al menos la mitad de las 24 configs.
  G2 IDENTIFICABILIDAD  ratio = IQR(trayectoria) / (1.349·mediana(SE)) ≥ 2
                    para τ_c Y β en alguna config: el movimiento del
                    parámetro debe superar claramente su ruido de ajuste.
                    (1.349·SE es el IQR de una normal: numerador y
                    denominador quedan en la misma escala.)
  G3 NO-REDUNDANCIA en las configs que pasan G2, |Spearman| con log σ_EWMA
                    ≤ 0.8 para τ_c o β. Si todo es un proxy de σ, es una
                    reparametrización y YA está probada.
  G4 ESTRUCTURA     autocorrelación de la trayectoria a separación NO
                    solapada (Δt_w ≥ window) > 0.2 en esas configs. El
                    solape de ventanas fabrica suavidad espuria a lags
                    cortos; sólo la persistencia más allá del solape es
                    evidencia de régimen y no artefacto.

El control positivo de la MEDICIÓN (regla 11) está en
`tests/test_transient_fit.py`: sobre datos sintéticos con (τ_c, β) conocidos
el ajuste los recupera. Con eso, un NO aquí significa "el dato real no
restringe la forma", no "el ajuste está roto".

REJILLA (descriptiva, no de contraste)
--------------------------------------
    activos     BTC, ETH, BNB, SOL           (4)
    intervalo   4h, 1h                       (2)
    ventana W   250, 500, 1000  retornos     (3)   → 24 configs de ajuste
    max_lag     40 / 60 / 80  según W        (τ_max << W, regla de two_time)
    T_eff       ventana 500 velas, max_lag 30 (la del exp. 02/04) → 8 configs

Ejecutar:  py experiments/exp05a_ajuste_transiente.py [n_procesos]
Pensado para Astrum (~24+8 tareas, una por config).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Hilos BLAS a 1 ANTES de importar numpy — ver la nota del exp. 03.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance import correlation_time, ewma_vol
from keldysh_finance.flow import (align_to_returns, fetch_klines_with_flow,
                                  response_from_frame)
from keldysh_finance.stationarity import aging_collapse
from keldysh_finance.transient import rolling_teff, rolling_transient_fit
from keldysh_finance.two_time import two_time_surface

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["4h", "1h"]
VENTANAS = [250, 500, 1000]
MAX_LAG = {250: 40, 500: 60, 1000: 80}
PASO = {"4h": 1, "1h": 4}          # ~8-9k ventanas por config en ambos casos

CONF = dict(years=4.0, field="abs", r2_min=0.2,
            ventanas=VENTANAS, max_lag=MAX_LAG, paso=PASO,
            teff_ventana=500, teff_max_lag=30,
            gates=dict(g1_frac_valida=0.50, g1_configs=0.50,
                       g2_ratio_ident=2.0, g3_spearman_max=0.8,
                       g4_persistencia=0.2))

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 4)
N_TRAZA = 400              # puntos por trayectoria depositada en el JSON


def datos(symbol: str, interval: str) -> dict:
    df = fetch_klines_with_flow(symbol, interval=interval,
                                years=CONF["years"]).dropna()
    _, _, eps, log_open = response_from_frame(df, max_lag=CONF["teff_max_lag"])
    r = np.diff(np.log(df["Close"].to_numpy(float)))
    return {"r": r, "eps": eps, "log_open": log_open, "n_klines": len(log_open)}


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    q75, q25 = np.quantile(x, [0.75, 0.25])
    return float(q75 - q25)


def _mediana(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) >= 20 else float("nan")


def _persistencia(x: np.ndarray, ok: np.ndarray, L: int) -> float:
    """Correlación de la trayectoria consigo misma L índices más tarde."""
    if L < 1 or L >= len(x):
        return float("nan")
    a, b = x[:-L], x[L:]
    m = ok[:-L] & ok[L:] & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray, ok: np.ndarray) -> float:
    from scipy.stats import spearmanr

    m = ok & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)


def _traza(t_w, ok, **series) -> dict:
    """Trayectoria decimada a <= N_TRAZA puntos para el log."""
    dec = max(1, int(np.ceil(len(t_w) / N_TRAZA)))
    out = {"t_w": t_w[::dec].tolist(),
           "valido": ok[::dec].astype(int).tolist()}
    for k, v in series.items():
        out[k] = np.asarray(v)[::dec].tolist()
    return out


def tarea_ajuste(arg: tuple[str, str, str, int]) -> dict:
    """Una config (activo, intervalo, W): superficie -> ajuste -> resúmenes."""
    nombre, symbol, interval, W = arg
    d = datos(symbol, interval)
    step = PASO[interval]
    t0 = time.time()

    s = two_time_surface(d["r"], window=W, max_lag=MAX_LAG[W], step=step,
                         field=CONF["field"])
    traj = rolling_transient_fit(s, r2_min=CONF["r2_min"])
    ok = traj.valido
    sigma = ewma_vol(d["r"])
    log_sigma_tw = np.log(sigma[traj.t_w])
    L_no_solape = int(np.ceil(W / step))
    tau_integral = correlation_time(s, method="integral")
    aging = aging_collapse(s)

    res = {"clave": f"{nombre}|{interval}|W{W}",
           "n_ventanas": int(len(ok)),
           "frac_valida": float(ok.mean()),
           "frac_en_borde": float(np.mean(~ok & np.isfinite(traj.beta))),
           "r2_mediana": _mediana(traj.r2[ok]),
           "degeneracion_tau_beta": _mediana(np.abs(traj.corr_tau_beta[ok])),
           "segundos": round(time.time() - t0, 1),
           "parametros": {}, "aging": {
               "mu_best": aging["mu_best"],
               "residual_best": aging["residual_best"],
               "residual_tti": aging["residual_tti"],
               "mejora_vs_tti": (aging["residual_best"] / aging["residual_tti"]
                                 if aging["residual_tti"] else float("nan"))}}

    for par, serie, se in (("log_tau_c", traj.log_tau_c, traj.se_log_tau_c),
                           ("beta", traj.beta, traj.se_beta),
                           ("log_A", traj.log_A, None)):
        v = np.where(ok, serie, np.nan)
        iqr_tray, se_med = _iqr(v), (_mediana(se[ok]) if se is not None else None)
        res["parametros"][par] = {
            "mediana": _mediana(v),
            "iqr_trayectoria": iqr_tray,
            "se_mediana": se_med,
            "ratio_ident": (iqr_tray / (1.349 * se_med)
                            if se_med and np.isfinite(iqr_tray) else None),
            "persistencia_no_solapada": _persistencia(v, ok, L_no_solape),
            "persistencia_lag1": _persistencia(v, ok, 1),
            "spearman_log_sigma": _spearman(v, log_sigma_tw, ok)}

    # cruce de estimadores: el τ_c ajustado contra el integral no paramétrico
    res["spearman_tau_fit_vs_integral"] = _spearman(
        np.where(ok, traj.log_tau_c, np.nan), np.log(np.maximum(tau_integral, 1e-9)), ok)

    res["traza"] = _traza(traj.t_w, ok, log_tau_c=traj.log_tau_c,
                          beta=traj.beta, log_A=traj.log_A, r2=traj.r2)
    return res


def tarea_teff(arg: tuple[str, str, str]) -> dict:
    """T_eff mediana por ventana trailing sobre el flujo, por (activo, intervalo)."""
    nombre, symbol, interval = arg
    d = datos(symbol, interval)
    step = PASO[interval]
    t0 = time.time()
    t_w, teff, viol = rolling_teff(d["log_open"], d["eps"],
                                   window=CONF["teff_ventana"],
                                   max_lag=CONF["teff_max_lag"], step=step)
    ok = np.isfinite(teff)
    # a la rejilla de retornos para correlar con sigma (misma vía que exp. 02/04)
    lt_r = align_to_returns(t_w, np.log(np.maximum(teff, 1e-12)),
                            n_klines=d["n_klines"])
    sigma = ewma_vol(d["r"])
    L_no_solape = int(np.ceil(CONF["teff_ventana"] / step))
    lt = np.log(np.maximum(teff, 1e-12))
    return {"clave": f"{nombre}|{interval}",
            "n_ventanas": int(len(ok)), "frac_finita": float(ok.mean()),
            "log_teff_mediana": _mediana(lt[ok]),
            "log_teff_iqr": _iqr(lt[ok]),
            "persistencia_no_solapada": _persistencia(lt, ok, L_no_solape),
            "spearman_log_sigma": _spearman(lt_r, np.log(sigma),
                                            np.isfinite(lt_r)),
            "violacion_mediana": _mediana(viol[ok]),
            "segundos": round(time.time() - t0, 1),
            "traza": _traza(t_w, ok, log_teff=lt, violacion=viol)}


def evaluar_gates(ajustes: list[dict]) -> dict:
    g = CONF["gates"]
    g1_pasa = [a["clave"] for a in ajustes
               if a["frac_valida"] >= g["g1_frac_valida"]]
    g1 = len(g1_pasa) >= g["g1_configs"] * len(ajustes)

    def ident(a, par):
        r = a["parametros"][par]["ratio_ident"]
        return r is not None and np.isfinite(r) and r >= g["g2_ratio_ident"]

    g2_pasa = [a["clave"] for a in ajustes
               if ident(a, "log_tau_c") and ident(a, "beta")]

    def finito(v):
        return v is not None and np.isfinite(v)

    g3_pasa, g4_pasa = [], []
    for a in ajustes:
        if a["clave"] not in g2_pasa:
            continue
        sp = [abs(a["parametros"][p]["spearman_log_sigma"])
              for p in ("log_tau_c", "beta")
              if finito(a["parametros"][p]["spearman_log_sigma"])]
        if sp and min(sp) <= g["g3_spearman_max"]:
            g3_pasa.append(a["clave"])
            pe = [a["parametros"][p]["persistencia_no_solapada"]
                  for p in ("log_tau_c", "beta")]
            if any(finito(x) and x > g["g4_persistencia"] for x in pe):
                g4_pasa.append(a["clave"])

    return {"G1_medicion": {"pasa": bool(g1), "configs": g1_pasa},
            "G2_identificabilidad": {"pasa": bool(g2_pasa), "configs": g2_pasa},
            "G3_no_redundancia": {"pasa": bool(g3_pasa), "configs": g3_pasa},
            "G4_estructura": {"pasa": bool(g4_pasa), "configs": g4_pasa},
            "fase_b_justificada": bool(g1 and g4_pasa),
            "configs_candidatas": g4_pasa}


def main() -> None:
    from multiprocessing import Pool

    tareas_fit = [(n, s, i, w) for n, s in ACTIVOS for i in INTERVALOS
                  for w in VENTANAS]
    tareas_te = [(n, s, i) for n, s in ACTIVOS for i in INTERVALOS]

    print(f"\n  EXPERIMENTO 05a — ajuste transiente KWW por ventana (descriptivo)")
    print(f"  {len(tareas_fit)} configs de ajuste + {len(tareas_te)} de T_eff, "
          f"{NPROC} procesos")
    print(f"  GATES EX-ANTE: G1 frac>=0.50 en mitad de configs; G2 ratio>=2 "
          f"en tau_c Y beta;\n                 G3 |rho_s(sigma)|<=0.8; "
          f"G4 persistencia no solapada >0.2\n", flush=True)

    t0 = time.time()
    with Pool(processes=NPROC) as pool:
        ajustes = pool.map(tarea_ajuste, tareas_fit)
        teffs = pool.map(tarea_teff, tareas_te)
    print(f"  computo total: {time.time()-t0:.0f} s\n", flush=True)

    print(f"  {'config':<16}{'%val':>6}{'R2':>6}{'tau_c':>7}{'beta':>6}"
          f"{'id_tau':>7}{'id_b':>6}{'|deg|':>6}{'sp_tau':>7}{'sp_b':>6}"
          f"{'pers_tau':>9}{'pers_b':>7}")
    for a in sorted(ajustes, key=lambda x: x["clave"]):
        p = a["parametros"]
        fmt = lambda v, w=6, d=2: (f"{v:>{w}.{d}f}"
                                   if v is not None and np.isfinite(v)
                                   else " " * (w - 3) + "---")
        print(f"  {a['clave']:<16}{a['frac_valida']:>6.0%}{fmt(a['r2_mediana'])}"
              f"{fmt(np.exp(p['log_tau_c']['mediana']), 7, 1)}"
              f"{fmt(p['beta']['mediana'])}"
              f"{fmt(p['log_tau_c']['ratio_ident'], 7)}"
              f"{fmt(p['beta']['ratio_ident'])}"
              f"{fmt(a['degeneracion_tau_beta'])}"
              f"{fmt(p['log_tau_c']['spearman_log_sigma'], 7)}"
              f"{fmt(p['beta']['spearman_log_sigma'])}"
              f"{fmt(p['log_tau_c']['persistencia_no_solapada'], 9)}"
              f"{fmt(p['beta']['persistencia_no_solapada'], 7)}")

    print(f"\n  {'T_eff':<12}{'%fin':>6}{'log_med':>9}{'IQR':>6}{'pers':>7}"
          f"{'sp_sigma':>9}{'viol_med':>9}")
    for a in sorted(teffs, key=lambda x: x["clave"]):
        fmt = lambda v, w=7, d=2: (f"{v:>{w}.{d}f}"
                                   if v is not None and np.isfinite(v)
                                   else " " * (w - 3) + "---")
        print(f"  {a['clave']:<12}{a['frac_finita']:>6.0%}"
              f"{fmt(a['log_teff_mediana'], 9)}{fmt(a['log_teff_iqr'], 6)}"
              f"{fmt(a['persistencia_no_solapada'])}"
              f"{fmt(a['spearman_log_sigma'], 9)}"
              f"{fmt(a['violacion_mediana'], 9)}")

    gates = evaluar_gates(ajustes)
    print(f"\n  {'='*74}\n  GATES\n  {'='*74}")
    for k, v in gates.items():
        if isinstance(v, dict):
            print(f"  {k:<22}: {'PASA' if v['pasa'] else 'NO'}   "
                  f"({len(v['configs'])} configs)")
    veredicto = ("FASE B JUSTIFICADA en: " + ", ".join(gates["configs_candidatas"])
                 if gates["fase_b_justificada"] else
                 "FASE B NO JUSTIFICADA — los parametros KWW no son a la vez "
                 "medibles, no redundantes con sigma y persistentes; el programa "
                 "de ajuste por ventanas se queda en descripcion")
    print(f"\n  → {veredicto}")

    salida = {"conf": {**CONF, "activos": [a for a, _ in ACTIVOS],
                       "intervalos": INTERVALOS},
              "ajustes": ajustes, "teff": teffs, "gates": gates,
              "veredicto": veredicto}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp05a_ajuste_transiente.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
