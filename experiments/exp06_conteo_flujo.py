r"""
EXPERIMENTO 06 — estadística de conteo del flujo firmado: carga, sesgo y
ruido de disparo del mercado.

QUÉ ES ESTO
-----------
MEDICIÓN (vía "publicar la medición", opción 2 del repo), no contraste
predictivo. Completa el diccionario de transporte que el repo ya tenía a
medias: ε_t es la corriente, R(τ) la conductancia, C_ε(τ) el ruido, T_eff el
cociente Johnson-Nyquist — y aquí se añade la capa de CONTEO (Levitov-
Lesovik): la estadística de la carga transferida Q_T = Σ ε_t por ventana.

Verificación de literatura (2026-08-09, búsqueda web): nada de FCS ni de la
simetría de intercambio sobre el flujo FIRMADO. Lo más cercano: el estimador
log-ratio aplicado a RETORNOS (arXiv:2509.23692, "market temperature"), FT
integral sobre cascadas de volatilidad, y termodinámica estocástica del
impacto (arXiv:2512.03123 — teórico, P&L de round-trips, sin FCS). El hueco
es la CORRIENTE como observable + el marco de transporte completo sobre el
mismo dataset. OJO antes de reclamar novedad en un manuscrito: barrido serio
(Scholar/SSRN/microestructura), no sólo arXiv — lección de superenergía.

TRES OBSERVABLES (e interpretación declarada ANTES de mirar)
------------------------------------------------------------
1. SIMETRÍA DE FLUCTUACIÓN  s(Q) = ln[P(Q)/P(−Q)].
   El teorema de intercambio (Gallavotti-Cohen) predice s = A·Q, lineal, con
   A la afinidad (el sesgo, análogo de eV/kT). Para Q gaussiana es exacta con
   A = 2μ/σ². Lectura: b₃ compatible con 0 y χ²/dof ~ 1 ⇒ la afinidad existe
   como número; b₃ ≠ 0 ⇒ el "sesgo" depende de la escala de Q y no hay un
   solo parámetro termodinámico. AVISO FÍSICO: GC exige microreversibilidad
   y estacionariedad, que un mercado no garantiza — la simetría es una
   predicción rígida que se CONTRASTA, y su ruptura es información.
2. ESCALADO DE CUMULANTES  κ_n(T).
   i.i.d. ⇒ pendiente 1 en todos. Memoria C_ε ~ τ^γ ⇒ κ₂ ~ T^(2+γ)
   asintóticamente. El contraste FUERTE no es el exponente (el crossover lo
   sesga) sino la predicción discreta EXACTA de segundo orden construida con
   la C_ε medida:  κ₂_pred(T) = Var(ε)·[T + 2·Σ_{k<T}(T−k)·ρ(k)].
   Coincidir ⇒ el conteo es consistente con la ACF por una vía independiente
   (varianzas de sumas, no correlaciones). Desviarse ⇒ contenido más allá
   del segundo orden.
3. FACTOR DE FANO  F(T) = Var(Q_T)/(q̄²·⟨N_T⟩), con q̄ = volumen mediano por
   trade (el cuanto de carga; el crudo, porque el cuanto vive en esas
   unidades). Nulo: trades independientes ±q̄ ⇒ F=1. F≫1 = bunching = order
   splitting, el mecanismo documentado a nivel de cuenta (arXiv:2308.01112)
   detrás de la memoria Lillo-Farmer. Con memoria, F crece ~ T^(1+γ).

DECISIONES EX-ANTE (para no renegociar mirando la tabla — regla 5)
------------------------------------------------------------------
- Flujo para simetría y cumulantes: ε normalizado por volumen de vela
  (adimensional, comparable entre regímenes; el crudo cambia órdenes de
  magnitud en 4 años). El crudo SÓLO para Fano.
- Ventanas NO solapadas (los errores binomiales exigen independencia de
  cuentas; lección G4 del 05a aplicada en diseño).
- T_GRID: 1h → {1,2,4,8,16,32,64,128,256}; 4h → {1,2,4,8,16,32,64}.
- A(T) sólo donde hay ≥ 400 ventanas (un log-ratio con pocas cuentas es
  anécdota); s(Q) se deposita en T ∈ {1,16,64}.
- La pendiente de κ₂ QUE CUENTA es sobre T ≥ 8 (el tramo bajo mezcla el
  crossover); la del grid completo se reporta como contexto.
- γ_ACF por OLS log-log de C_ε en τ ∈ [1,60].
- NULOS, K=20 cada uno:
  · signflip (|ε| intacto, signos i.i.d.): mata signo Y memoria firmada ⇒
    A ≈ 0 y pendiente κ₂ ≈ 1.
  · permutación: mata la memoria, conserva la marginal ⇒ A(1) idéntico por
    construcción y A(T) ≈ A(1) constante; la FIRMA de la memoria real es
    A(T) decayendo respecto a A(1).
- Robustez: 4 tramos temporales consecutivos iguales; A(16) y pendiente por
  tramo. Un signo de A que cambia entre tramos no es un sesgo, es ruido.

Ejecutar:  py experiments/exp06_conteo_flujo.py [K]
Corre en LOCAL (~medio minuto: sumas vectorizadas, sin ajustes caros).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from keldysh_finance.counting import (cumulant_scaling, fano_factor,
                                      fit_affinity, net_charge,
                                      symmetry_function)
from keldysh_finance.flow import (fetch_klines_with_flow, flow_autocorrelation,
                                  order_flow_imbalance)

OUT = Path(__file__).resolve().parents[1] / "output"

ACTIVOS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
           ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
INTERVALOS = ["1h", "4h"]                     # 1h primario, 4h robustez
T_GRID = {"1h": [1, 2, 4, 8, 16, 32, 64, 128, 256],
          "4h": [1, 2, 4, 8, 16, 32, 64]}
T_SIMETRIA = [1, 16, 64]                      # dónde se deposita s(Q)
MIN_VENTANAS_A = 400                          # mínimo para ajustar A(T)
T_PENDIENTE_MIN = 8                           # la pendiente que cuenta: T >= 8
GAMMA_LAGS = 60                               # OLS log-log de C_eps en [1,60]
CONF = dict(years=4.0, normalizacion="volume", t_grid=T_GRID,
            t_simetria=T_SIMETRIA, min_ventanas_A=MIN_VENTANAS_A,
            t_pendiente_min=T_PENDIENTE_MIN, gamma_lags=GAMMA_LAGS,
            n_tramos=4, seed=606_000)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def datos(symbol: str, interval: str) -> dict:
    df = fetch_klines_with_flow(symbol, interval=interval,
                                years=CONF["years"]).dropna()
    return {"eps": order_flow_imbalance(df, normalize="volume"),
            "eps_raw": order_flow_imbalance(df, normalize="none"),
            "volumen": df["Volume"].to_numpy(float),
            "trades": df["trades"].to_numpy(float),
            "n": len(df)}


def gamma_acf(eps: np.ndarray) -> float:
    """Exponente de memoria por OLS log-log sobre C_eps(tau), tau=[1,60]."""
    C = flow_autocorrelation(eps, max_lag=GAMMA_LAGS)
    tau = np.arange(1, GAMMA_LAGS + 1)
    m = np.isfinite(C) & (C > 0)
    if m.sum() < 10:
        return float("nan")
    x, y = np.log(tau[m]), np.log(C[m])
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def k2_prediccion_acf(eps: np.ndarray, T_grid) -> dict:
    """kappa_2 predicha EXACTA a segundo orden desde la C_eps medida."""
    t_max = max(T_grid)
    rho = flow_autocorrelation(eps, max_lag=t_max - 1) if t_max > 1 else np.empty(0)
    var_e = float(np.var(eps[np.isfinite(eps)], ddof=1))
    out = {}
    for T in T_grid:
        k = np.arange(1, T)
        r = rho[:T - 1] if T > 1 else np.empty(0)
        fin = np.isfinite(r)
        out[int(T)] = var_e * (T + 2.0 * float(np.sum((T - k)[fin] * r[fin])))
    return out


def afinidades(eps: np.ndarray, T_grid) -> dict:
    """A(T) con su error donde hay ventanas suficientes; s(Q) en T_SIMETRIA."""
    res = {"por_T": {}, "simetria": {}}
    for T in T_grid:
        Q = net_charge(eps, T)
        if len(Q) < MIN_VENTANAS_A:
            continue
        q, s, se = symmetry_function(Q)
        fit = fit_affinity(q, s, se)
        var_q = float(np.var(Q, ddof=1))
        fit["A_gauss"] = 2.0 * float(np.mean(Q)) / var_q if var_q > 0 else float("nan")
        fit["n_ventanas"] = int(len(Q))
        res["por_T"][int(T)] = fit
        if T in T_SIMETRIA:
            res["simetria"][int(T)] = {"q": q.tolist(), "s": s.tolist(),
                                       "se": se.tolist()}
    return res


def pendiente_k2(eps: np.ndarray, T_grid, t_min: int = 1) -> dict:
    esc = cumulant_scaling(eps, [t for t in T_grid if t >= t_min])
    return {"pendiente": esc["pendiente_k2"], "se": esc["se_pendiente_k2"],
            "tabla": esc["tabla"]}


def nulos(eps: np.ndarray, T_grid, seed0: int) -> dict:
    """K draws de signflip y permutación: A(16) y pendiente k2 (T>=8)."""
    grid_p = [t for t in T_grid if t >= T_PENDIENTE_MIN]
    out = {"signflip": {"A16": [], "pend": []},
           "perm": {"A16": [], "pend": []}}
    for i in range(K):
        rng = np.random.default_rng(seed0 + i)
        variantes = {
            "signflip": np.abs(eps) * rng.choice([-1.0, 1.0], size=len(eps)),
            "perm": rng.permutation(eps)}
        for nombre, y in variantes.items():
            fit = fit_affinity(*symmetry_function(net_charge(y, 16)))
            out[nombre]["A16"].append(fit["A"])
            esc = cumulant_scaling(y, grid_p)
            out[nombre]["pend"].append(esc["pendiente_k2"])
    resumen = {}
    for nombre, d in out.items():
        resumen[nombre] = {}
        for k_, v in d.items():
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            resumen[nombre][k_] = {
                "q025": float(np.quantile(v, 0.025)) if len(v) else float("nan"),
                "mediana": float(np.median(v)) if len(v) else float("nan"),
                "q975": float(np.quantile(v, 0.975)) if len(v) else float("nan")}
    return resumen


def por_tramos(eps: np.ndarray, T_grid) -> list[dict]:
    """A(16) y pendiente k2 (T>=8) en 4 tramos temporales consecutivos."""
    grid_p = [t for t in T_grid if T_PENDIENTE_MIN <= t <= 64]
    filas = []
    for i, tramo in enumerate(np.array_split(np.asarray(eps, float),
                                             CONF["n_tramos"])):
        fit = fit_affinity(*symmetry_function(net_charge(tramo, 16)))
        esc = cumulant_scaling(tramo, grid_p)
        filas.append({"tramo": i, "n": int(len(tramo)), "A16": fit["A"],
                      "se_A16": fit["se_A"], "pend_k2": esc["pendiente_k2"]})
    return filas


def una_serie(nombre: str, symbol: str, interval: str) -> dict:
    d = datos(symbol, interval)
    grid = T_GRID[interval]
    t0 = time.time()

    g = gamma_acf(d["eps"])
    afin = afinidades(d["eps"], grid)
    esc_full = pendiente_k2(d["eps"], grid, t_min=1)
    esc_alta = pendiente_k2(d["eps"], grid, t_min=T_PENDIENTE_MIN)
    k2_pred = k2_prediccion_acf(d["eps"], grid)
    ratio_pred = {T: (next((f["k2"] for f in esc_full["tabla"] if f["T"] == T),
                           float("nan")) / k2_pred[T])
                  for T in k2_pred if k2_pred[T] > 0}
    fano = fano_factor(d["eps_raw"], d["volumen"], d["trades"], grid)
    nul = nulos(d["eps"], grid,
                CONF["seed"] + 10_000 * ACTIVOS.index((nombre, symbol))
                + 1_000 * INTERVALOS.index(interval))
    tramos = por_tramos(d["eps"], grid)

    return {"clave": f"{nombre}|{interval}", "n_velas": d["n"],
            "media_eps": float(np.mean(d["eps"])),
            "gamma_acf": g, "prediccion_pendiente_k2": 2.0 + g,
            "afinidad": afin,
            "pendiente_k2_full": esc_full, "pendiente_k2_alta": esc_alta,
            "k2_pred_acf": {str(k_): v for k_, v in k2_pred.items()},
            "ratio_k2_obs_pred": {str(k_): float(v)
                                  for k_, v in ratio_pred.items()},
            "fano": fano, "nulos": nul, "tramos": tramos,
            "segundos": round(time.time() - t0, 1)}


def main() -> None:
    print(f"\n  EXPERIMENTO 06 — estadistica de conteo del flujo firmado")
    print(f"  {len(ACTIVOS)} activos x {INTERVALOS}, K={K} nulos "
          f"(signflip, permutacion) por serie\n", flush=True)

    resultados = []
    for nombre, symbol in ACTIVOS:
        for interval in INTERVALOS:
            r = una_serie(nombre, symbol, interval)
            resultados.append(r)
            print(f"    {r['clave']:<10} hecho en {r['segundos']} s", flush=True)

    print(f"\n  {'serie':<10}{'gamma':>7}{'A(1)':>9}{'A(16)':>9}{'A(64)':>9}"
          f"{'b3(16)':>9}{'pend_k2':>9}{'2+gamma':>8}{'F(1)':>7}{'F(64)':>9}")
    for r in resultados:
        pt = r["afinidad"]["por_T"]
        fan = {f["T"]: f["fano"] for f in r["fano"]["tabla"]}
        A = lambda T: (f"{pt[T]['A']:>9.4f}" if T in pt else "      ---")
        b3 = (f"{pt[16]['b3']:>9.4f}" if 16 in pt else "      ---")
        print(f"  {r['clave']:<10}{r['gamma_acf']:>7.3f}{A(1)}{A(16)}{A(64)}"
              f"{b3}{r['pendiente_k2_alta']['pendiente']:>9.3f}"
              f"{r['prediccion_pendiente_k2']:>8.3f}"
              f"{fan.get(1, float('nan')):>7.1f}{fan.get(64, float('nan')):>9.0f}")

    print(f"\n  {'serie':<10}{'A16 signflip [2.5-97.5%]':>26}"
          f"{'pend perm [2.5-97.5%]':>24}{'A16/A1 obs':>12}")
    for r in resultados:
        sf = r["nulos"]["signflip"]["A16"]
        pm = r["nulos"]["perm"]["pend"]
        pt = r["afinidad"]["por_T"]
        ratio = (pt[16]["A"] / pt[1]["A"]
                 if 1 in pt and 16 in pt and pt[1]["A"] not in (0.0,) else float("nan"))
        print(f"  {r['clave']:<10}"
              f"{f'[{sf['q025']:+.4f}, {sf['q975']:+.4f}]':>26}"
              f"{f'[{pm['q025']:.3f}, {pm['q975']:.3f}]':>24}{ratio:>12.3f}")

    salida = {"conf": {**CONF, "K": K,
                       "activos": [a for a, _ in ACTIVOS],
                       "intervalos": INTERVALOS},
              "resultados": resultados}
    OUT.mkdir(exist_ok=True)
    path = OUT / "exp06_conteo_flujo.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Log depositado en {path}\n")


if __name__ == "__main__":
    main()
