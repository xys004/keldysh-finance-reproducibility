r"""
Estadística de conteo completa (FCS) del flujo de órdenes firmado.

LA HERENCIA DE TRANSPORTE QUE FALTABA
-------------------------------------
Con el diccionario transporte↔mercado, el repo ya tenía medida la mitad de la
analogía: ε_t es la CORRIENTE (flujo neto de inventario entre los reservorios
comprador y vendedor), R(τ) es la respuesta/conductancia, C_ε(τ) el ruido de
corriente, y el cociente T_eff la comprobación Johnson-Nyquist. Lo que no
estaba es la capa de arriba de la teoría de transporte: la estadística de la
CARGA TRANSFERIDA

    Q_T = Σ_{t∈ventana de T velas} ε_t

que en mesoscópica (Levitov-Lesovik) es donde vive la física que la
conductancia media no ve.

PRECISIÓN QUE HAY QUE MANTENER (corregida 2026-08-13)
-----------------------------------------------------
Este módulo calcula la estadística de conteo CLÁSICA, y eso **no necesita el
contorno de Keldysh**. Para una corriente clásica, ln⟨e^{χQ_T}⟩ es una función
generatriz de cumulantes ordinaria: una sola rama, sin ambigüedad de orden.
Lo que SÍ exige las dos ramas —con χ entrando como ±χ/2 en la rama de ida y en
la de vuelta— es el campo de conteo CUÁNTICO de Levitov-Lesovik, donde el
orden de operadores que no conmutan hace que la generatriz ingenua esté mal
definida. Esa construcción vive en el dispositivo (repo QTEOM,
`current_noise.py`), no aquí.

Una versión anterior de este docstring decía que esta capa era "la pieza más
genuinamente Keldysh del programa". Es falso y conviene que quede escrito: era
autoridad prestada, del tipo exacto que el resto del proyecto se esfuerza en
no usar.

TRES OBSERVABLES
----------------
1. La SIMETRÍA DE FLUCTUACIÓN. Para un conductor entre dos reservorios, el
   teorema de fluctuación de intercambio (Gallavotti-Cohen) predice

       s(Q) := ln[ P(Q_T=Q) / P(Q_T=−Q) ] = A·Q      (lineal, pendiente A)

   con A la afinidad — el sesgo termodinámico, el análogo de eV/kT. Para Q
   gaussiana la relación es EXACTA con A = 2μ/σ², así que la curvatura c₃ de
   s(Q) mide directamente cuánto se aparta la cola del régimen gaussiano.
   HONESTIDAD FÍSICA: GC se deriva de microreversibilidad + estado
   estacionario y un mercado no garantiza ninguna; aquí la simetría es una
   PREDICCIÓN RÍGIDA DE UN PARÁMETRO que se contrasta, y medir cómo se rompe
   es información, igual que lo fue la violación del FDT.

2. El ESCALADO DE CUMULANTES κ_n(T). Con incrementos i.i.d., κ_n ∝ T para
   todo n (difusivo). Con la memoria larga del flujo, C_ε(τ) ~ τ^γ, la
   varianza pasa a Var(Q_T) ~ T^(2+γ): para el γ≈−0.5 de Lillo-Farmer, T^1.5.
   Esto convierte la FCS en un control de coherencia interna del repo: el
   exponente de κ₂ tiene que REENCONTRAR el γ ya medido por la ACF, por una
   vía independiente (varianzas de sumas, no correlaciones).

3. El FACTOR DE FANO. F(T) = Var(Q_T) / (q̄²·⟨N_T⟩), con q̄ el "cuanto de
   carga" (volumen mediano por trade, columna `trades` de las klines) y N_T
   los trades por ventana. El nulo son N trades independientes de tamaño ±q̄:
   F=1. F≫1 es bunching de portadores — que en mercados ES el order
   splitting, el mecanismo microscópico documentado a nivel de cuenta
   (arXiv:2308.01112) detrás de la memoria de Lillo-Farmer. Con memoria
   τ^γ, F(T) crece como T^(1+γ).

QUÉ NO ES ESTE MÓDULO
---------------------
No produce características predictivas por ventana. Es MEDICIÓN sobre la
muestra completa (la vía viva del repo: publicar la medición). Si algún día
A o F quisieran ser características por ventana, pasan antes por las cuatro
puertas de identificabilidad del exp. 05a.
"""
from __future__ import annotations

import numpy as np

_MIN_VENTANAS = 30       # ventanas no solapadas mínimas para estimar cumulantes


def net_charge(flow: np.ndarray, T: int) -> np.ndarray:
    """Q_T: carga transferida en ventanas NO solapadas de T velas.

    No solapadas a propósito: el solape fabricaría dependencia entre ventanas
    y los errores binomiales de la función de simetría dejarían de valer
    (la lección del G4 del exp. 05a, aplicada en diseño y no a posteriori).
    """
    e = np.asarray(flow, dtype=float)
    e = e[np.isfinite(e)]
    n = (len(e) // int(T)) * int(T)
    if n == 0:
        return np.empty(0)
    return e[:n].reshape(-1, int(T)).sum(axis=1)


def symmetry_function(Q: np.ndarray, n_bins: int = 15,
                      q_max_quantile: float = 0.99,
                      min_count: int = 10):
    r"""s(q) = ln[ #{Q∈+bin} / #{Q∈−bin} ] con su error binomial.

    Bins simétricos en ±|q| hasta el cuantil `q_max_quantile` de |Q| (la cola
    extrema no tiene cuentas suficientes en ambos signos). El error estándar
    de ln(n₊/n₋) es √(1/n₊ + 1/n₋) (Poisson en cada cuenta). Los bins con
    menos de `min_count` cuentas en CUALQUIERA de los dos signos se omiten:
    un log-ratio con 3 cuentas no es un dato, es una anécdota.

    Returns
    -------
    (q_centro, s, se) — arrays de los bins que sobreviven.
    """
    Q = np.asarray(Q, dtype=float)
    Q = Q[np.isfinite(Q)]
    if len(Q) < 4 * min_count:
        return (np.empty(0),) * 3
    qmax = float(np.quantile(np.abs(Q), q_max_quantile))
    if qmax <= 0:
        return (np.empty(0),) * 3
    edges = np.linspace(0.0, qmax, n_bins + 1)
    centros, s, se = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        n_pos = int(np.sum((Q > lo) & (Q <= hi)))
        n_neg = int(np.sum((Q < -lo) & (Q >= -hi)))
        if n_pos >= min_count and n_neg >= min_count:
            centros.append(0.5 * (lo + hi))
            s.append(np.log(n_pos / n_neg))
            se.append(np.sqrt(1.0 / n_pos + 1.0 / n_neg))
    return np.asarray(centros), np.asarray(s), np.asarray(se)


def fit_affinity(q: np.ndarray, s: np.ndarray, se: np.ndarray) -> dict:
    r"""Ajusta la simetría: s = A·q (GC) y s = A·q + c₃·q³ (curvatura).

    s(q) es impar por construcción, así que el desarrollo sólo tiene términos
    impares y el primer test de no-linealidad es c₃. La simetría de
    intercambio exige c₃ = 0; un c₃ incompatible con cero significa que la
    "afinidad" depende de la escala de Q — la simetría se rompe y no hay un
    solo sesgo termodinámico que resuma el transporte.

    Returns
    -------
    dict con A, se_A, chi2_dof (del ajuste lineal), c3, se_c3, n_bins.
    """
    out = {"A": np.nan, "se_A": np.nan, "chi2_dof": np.nan,
           "c3": np.nan, "se_c3": np.nan, "n_bins": int(len(q))}
    if len(q) < 3 or np.any(se <= 0):
        return out
    w = 1.0 / se ** 2

    swq2 = float(np.sum(w * q * q))
    if swq2 <= 0:
        return out
    A = float(np.sum(w * q * s) / swq2)
    out["A"], out["se_A"] = A, float(1.0 / np.sqrt(swq2))
    dof = len(q) - 1
    out["chi2_dof"] = float(np.sum(w * (s - A * q) ** 2) / dof) if dof > 0 else np.nan

    if len(q) >= 4:
        X = np.column_stack([q, q ** 3])
        XtW = X.T * w
        try:
            cov = np.linalg.inv(XtW @ X)
            beta = cov @ (XtW @ s)
            out["c3"] = float(beta[1])
            out["se_c3"] = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return out


def cumulant_scaling(flow: np.ndarray, T_grid) -> dict:
    r"""κ₁..κ₄ de Q_T por T, y pendientes log-log de |κ₁| y κ₂.

    La predicción de contraste: incrementos i.i.d. dan pendiente 1 en TODOS
    los cumulantes; memoria C_ε ~ τ^γ da pendiente 2+γ en κ₂ (y deja κ₁ en 1,
    porque la media no ve la correlación). k-estadísticos insesgados de scipy.
    """
    from scipy.stats import kstat

    filas = []
    for T in T_grid:
        Q = net_charge(flow, int(T))
        if len(Q) < _MIN_VENTANAS:
            continue
        filas.append({"T": int(T), "n_ventanas": int(len(Q)),
                      "k1": float(kstat(Q, 1)), "k2": float(kstat(Q, 2)),
                      "k3": float(kstat(Q, 3)), "k4": float(kstat(Q, 4))})

    out = {"tabla": filas, "pendiente_k2": np.nan, "se_pendiente_k2": np.nan,
           "pendiente_k1": np.nan}
    if len(filas) >= 3:
        lt = np.log([f["T"] for f in filas])
        lk2 = np.log([f["k2"] for f in filas])
        out["pendiente_k2"], out["se_pendiente_k2"] = _ols_pendiente(lt, lk2)
        k1 = np.array([f["k1"] for f in filas])
        if np.all(np.abs(k1) > 0) and (np.all(k1 > 0) or np.all(k1 < 0)):
            out["pendiente_k1"], _ = _ols_pendiente(lt, np.log(np.abs(k1)))
    return out


def _ols_pendiente(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pendiente OLS con su error estándar (para los log-log de escalado)."""
    n = len(x)
    xc = x - x.mean()
    sxx = float(np.dot(xc, xc))
    b = float(np.dot(xc, y - y.mean()) / sxx)
    resid = y - (y.mean() + b * xc)
    se = float(np.sqrt(np.dot(resid, resid) / max(n - 2, 1) / sxx))
    return b, se


def fano_factor(flow_raw: np.ndarray, volume: np.ndarray, trades: np.ndarray,
                T_grid) -> dict:
    r"""F(T) = Var(Q_T) / (q̄² · ⟨N_T⟩), con Q en unidades de volumen CRUDO.

    q̄ es el volumen MEDIANO por trade de toda la muestra (el cuanto de carga;
    la mediana y no la media porque la distribución de tamaños tiene cola).
    ⟨N_T⟩ es el número medio de trades por ventana. El nulo es N_T trades
    independientes de tamaño ±q̄, que da exactamente F = 1 (Var = N·q̄²).

    F se calcula sobre el flujo CRUDO (unidades de volumen) porque el cuanto
    vive en esas unidades; la no-estacionariedad del volumen a 4 años se
    trata en el experimento con el desglose por año, no aquí.
    """
    e = np.asarray(flow_raw, dtype=float)
    v = np.asarray(volume, dtype=float)
    tr = np.asarray(trades, dtype=float)
    m = np.isfinite(e) & np.isfinite(v) & np.isfinite(tr) & (tr > 0)
    e, v, tr = e[m], v[m], tr[m]
    if len(e) < 4 * _MIN_VENTANAS:
        return {"q_barra": np.nan, "tabla": []}
    q_barra = float(np.median(v / tr))

    filas = []
    for T in T_grid:
        Q = net_charge(e, int(T))
        if len(Q) < _MIN_VENTANAS:
            continue
        n_t = net_charge(tr, int(T))          # trades por ventana (misma rejilla)
        filas.append({"T": int(T), "n_ventanas": int(len(Q)),
                      "fano": float(np.var(Q, ddof=1)
                                    / (q_barra ** 2 * float(np.mean(n_t)))),
                      "trades_por_ventana": float(np.mean(n_t))})
    out = {"q_barra": q_barra, "tabla": filas, "pendiente_log": np.nan}
    if len(filas) >= 3:
        lt = np.log([f["T"] for f in filas])
        lf = np.log([f["fano"] for f in filas])
        out["pendiente_log"], _ = _ols_pendiente(lt, lf)
    return out


def synthetic_flow(n: int, gamma: float, mu: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    r"""Flujo gaussiano con ACF ~ (1+τ)^γ, sd 1 y media `mu`.

    CONTROL POSITIVO de la FCS (regla 11): un proceso donde el escalado es
    verdad por construcción — Var(Q_T) ~ T^(2+γ) y, por gaussianidad, la
    simetría s(Q)=A·Q exacta con A(T) = 2·μT/Var(Q_T). El pipeline tiene que
    recuperar ambas cosas o los negativos/positivos sobre datos reales no
    significan nada. Mismo embedding circulante que
    `transient.synthetic_series_with_acf`, con cola en ley de potencias (la
    memoria tipo Lillo-Farmer) en vez de KWW.
    """
    from scipy.fft import next_fast_len

    rng = np.random.default_rng(seed)
    m = next_fast_len(4 * int(n))
    d = np.minimum(np.arange(m), m - np.arange(m)).astype(float)
    c = (1.0 + d) ** float(gamma)
    lam = np.clip(np.fft.rfft(c).real, 0.0, None)
    x = np.fft.irfft(np.fft.rfft(rng.normal(size=m)) * np.sqrt(lam), m)[:int(n)]
    sd = x.std()
    if sd > 0:
        x = x / sd
    return x + float(mu)
