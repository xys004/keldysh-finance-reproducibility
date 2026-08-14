r"""
Transporte resuelto en FASE: el mercado como sistema abierto con forzado
periódico (Floquet).

LA PREGUNTA
-----------
Cripto opera 24/7 pero sus agentes no: el ciclo día/noche y las sesiones
(Asia/Europa/América) son un FORZADO PERIÓDICO real sobre el sistema. Un
conductor forzado periódicamente no tiene por qué tener transporte
estacionario: sus coeficientes (sesgo, impacto, ruido) pueden depender de la
fase del ciclo. Esto decide un ingrediente del modelo: si A, R o Fano modulan
con la hora del día, el modelo cuasi-estático de dos reservorios con sesgo
constante está incompleto y hace falta drive Floquet.

Que la ACTIVIDAD (volumen, |ε|) modula con la hora es sabido y aquí funciona
de CONTROL POSITIVO del pipeline: si el test no la ve, el test está roto. Lo
no trivial es si modulan los coeficientes de TRANSPORTE: ⟨ε⟩ (el sesgo),
A(1) (la afinidad), R(1) (el impacto) y F(1) (el Fano).

EL NULO
-------
Desfase circular: se ruedan las etiquetas de fase un desplazamiento aleatorio
(≥ 1 período) manteniendo intactas TODAS las series. Eso destruye únicamente
el anclaje serie↔fase y conserva memoria, colas y no-estacionariedades — el
nulo quirúrgico para "¿está la estructura EN la fase?".
"""
from __future__ import annotations

import numpy as np

from .counting import fit_affinity, symmetry_function

_MIN_POR_BIN = 500


def observables_por_fase(eps: np.ndarray, log_open: np.ndarray,
                         eps_raw: np.ndarray, trades: np.ndarray,
                         horas: np.ndarray, n_bins: int = 8,
                         q_barra: float | None = None) -> list[dict]:
    r"""Por bin de fase horaria: sesgo, actividad, afinidad, impacto y Fano.

    - `sesgo`      ⟨ε⟩ con su error estándar
    - `actividad`  ⟨trades⟩ por vela — el control positivo. OJO: la primera
                   versión usaba ⟨|ε|⟩ con ε normalizado por volumen, y eso
                   DIVIDE FUERA la estacionalidad del volumen: el control
                   medía la fracción de desbalance (que no tiene por qué
                   modular) y salió 2/4 donde el volumen modula de libro. El
                   control tiene que ser una cantidad de ACTIVIDAD cruda.
    - `A1`         afinidad a T=1 sobre las velas del bin
    - `R1`         impacto a un paso: ⟨(p_{t+1}−p_t)·δε_t⟩/⟨δε²⟩ con δε
                   centrado POR FASE (si no, la estacionalidad del propio
                   sesgo se colaría dentro del impacto)
    - `F1`         Var(ε_raw)/(q̄²·⟨trades⟩) del bin, con q̄ GLOBAL (el cuanto
                   no depende de la hora; usar uno por bin mezclaría la
                   modulación del cuanto con la del ruido)
    """
    e = np.asarray(eps, dtype=float)
    p = np.asarray(log_open, dtype=float)
    er = np.asarray(eps_raw, dtype=float)
    tr = np.asarray(trades, dtype=float)
    h = np.asarray(horas, dtype=int)
    n = min(len(e), len(p) - 1, len(er), len(tr), len(h))
    ancho = 24 // n_bins
    dp = p[1:n + 1] - p[:n]

    # WINSOR DECLARADO (mismo patrón que pipeline.transformar_caracteristica).
    # Sin esto el test no tiene potencia: la varianza entre bins del NULO la
    # dominan las ráfagas de cola pesada (un bloque de 3h de un día de crash
    # cae entero en un bin aleatorio), y una modulación suave real de ×1.9 en
    # la actividad quedaba en p~0.10-0.16 (medido, 2026-08-09). Los umbrales
    # se calculan sobre la serie COMPLETA, así que son idénticos para el
    # observado y para cada rotación nula: el test queda emparejado.
    con_tr = np.isfinite(tr[:n]) & (tr[:n] > 0)
    if con_tr.sum() > 100:
        tr = np.minimum(tr, float(np.quantile(tr[:n][con_tr], 0.99)))
    fin_dp = np.isfinite(dp)
    if fin_dp.sum() > 100:
        lim = float(np.quantile(np.abs(dp[fin_dp]), 0.995))
        dp = np.clip(dp, -lim, lim)
    fin_er = np.isfinite(er[:n])
    if fin_er.sum() > 100:
        lim = float(np.quantile(np.abs(er[:n][fin_er]), 0.99))
        er = np.clip(er, -lim, lim)

    filas = []
    for b in range(n_bins):
        m = (h[:n] // ancho) == b
        m = m & np.isfinite(e[:n]) & np.isfinite(dp)
        nb = int(m.sum())
        fila = {"bin": b, "horas": f"{b*ancho:02d}-{(b+1)*ancho:02d}",
                "n": nb, "sesgo": np.nan, "se_sesgo": np.nan,
                "actividad": np.nan, "A1": np.nan, "se_A1": np.nan,
                "R1": np.nan, "F1": np.nan}
        if nb >= _MIN_POR_BIN:
            eb = e[:n][m]
            fila["sesgo"] = float(eb.mean())
            fila["se_sesgo"] = float(eb.std(ddof=1) / np.sqrt(nb))
            mt_act = m & np.isfinite(tr[:n]) & (tr[:n] > 0)
            if mt_act.sum() >= _MIN_POR_BIN:
                fila["actividad"] = float(np.mean(tr[:n][mt_act]))
            fit = fit_affinity(*symmetry_function(eb))
            fila["A1"], fila["se_A1"] = fit["A"], fit["se_A"]
            dec = eb - eb.mean()
            var = float(np.mean(dec ** 2))
            if var > 0:
                fila["R1"] = float(np.mean(dp[m] * dec) / var)
            mt = m & np.isfinite(er[:n]) & np.isfinite(tr[:n]) & (tr[:n] > 0)
            if q_barra and q_barra > 0 and mt.sum() >= _MIN_POR_BIN:
                fila["F1"] = float(np.var(er[:n][mt], ddof=1)
                                   / (q_barra ** 2 * float(np.mean(tr[:n][mt]))))
        filas.append(fila)
    return filas


def rotar_horas_por_dia(horas: np.ndarray, dias: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    r"""El NULO: rotación de fase independiente por día.

    Un desfase circular rígido NO sirve de nulo aquí — con etiquetas
    periódicas sólo permuta cíclicamente los bins, y la varianza entre bins
    es invariante bajo permutaciones (medido: nulo idéntico al observado).
    Lo que hay que destruir es la COHERENCIA de fase entre días: a cada día j
    se le asigna un desfase δ_j ~ U{0..23} independiente. Las series quedan
    intactas (memoria, colas, no-estacionariedad conservadas); sólo se rompe
    el anclaje día-a-día con el ciclo, que es exactamente lo que el test
    pregunta.
    """
    h = np.asarray(horas, dtype=int).copy()
    d = np.asarray(dias)
    for dia in np.unique(d):
        m = d == dia
        h[m] = (h[m] + int(rng.integers(0, 24))) % 24
    return h


def modulacion(filas: list[dict], clave: str) -> float:
    """Estadístico de modulación: varianza del observable entre bins de fase.

    Sin normalizar a propósito: se compara SIEMPRE contra su propio nulo de
    desfase circular, así que la escala se cancela en el p empírico.
    """
    v = np.array([f[clave] for f in filas], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.var(v)) if len(v) >= 3 else float("nan")


def p_empirico(v_obs: float, v_nulo: np.ndarray) -> float:
    """(1 + #{nulo >= obs}) / (K + 1) — mismo convenio que westfall_young."""
    v_nulo = np.asarray(v_nulo, dtype=float)
    v_nulo = v_nulo[np.isfinite(v_nulo)]
    if not np.isfinite(v_obs) or len(v_nulo) == 0:
        return float("nan")
    return float((1 + np.sum(v_nulo >= v_obs)) / (len(v_nulo) + 1))
