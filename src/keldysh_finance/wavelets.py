r"""
Estimador de ondícula de Abry--Veitch para el exponente de memoria.

POR QUÉ ESTE ESTIMADOR Y NO OTRO
---------------------------------
El exp. 10 midió que nuestro γ depende del estimador: la dispersión entre OLS
log-log sobre C_ε, log-periodograma y DFA va de 0.23 a 0.53 según el activo.
No es sorpresa —Lillo y Farmer obtienen 0.39 y 0.61 sobre los MISMOS datos y
llaman al primero "a poor method"— pero deja el valor sin una banda honesta.

El estimador de ondícula resuelve el problema por construcción, y por dos
razones distintas que conviene no mezclar:

1. **DECORRELACIÓN.** Para un proceso de memoria larga, los coeficientes de
   ondícula dentro de una octava están casi decorrelacionados, mientras que la
   ACF muestral tiene correlaciones fuertes entre desfases. Es lo que hace que
   la regresión log-log del diagrama log-escala tenga intervalos de confianza
   que significan algo, cosa que la regresión sobre la ACF no tiene.

2. **MOMENTOS NULOS.** Una ondícula de Daubechies de orden N aniquila
   exactamente los polinomios de grado N−1. Cualquier tendencia lenta
   —justamente el mecanismo que fabrica el envejecimiento espurio de la
   sección de relojes— desaparece del estimador sin necesidad de detrending
   manual, que es donde se cuelan los artefactos.

LA RELACIÓN DE EXPONENTES
--------------------------
Con C_ε(τ) ~ τ^(−γ) se tiene el índice de Hurst H = 1 − γ/2, y la densidad
espectral f(ω) ~ ω^(−α) con α = 2H − 1 = 1 − γ. En el diagrama log-escala,

    log₂ E[d²_j]  =  α·j + const,

luego la pendiente del diagrama da α y

    γ = 1 − α .

CONVENCIÓN DEL ESTIMADOR
------------------------
Se sigue Abry & Veitch: media de cuadrados por octava, corrección del sesgo
logarítmico (E[log X] ≠ log E[X] para X de tipo χ²) mediante la digamma, y
mínimos cuadrados PONDERADOS con los pesos exactos ζ(2, n_j/2)/ln²2. Sin la
corrección de sesgo el estimador se desvía sistemáticamente en las octavas
altas, donde quedan pocos coeficientes.
"""
from __future__ import annotations

import numpy as np

# Filtros de escala de Daubechies. El nº de momentos nulos es L/2.
DAUBECHIES: dict[int, list[float]] = {
    2: [0.7071067811865476, 0.7071067811865476],                     # Haar, 1 mn
    4: [0.48296291314469025, 0.836516303737469,
        0.22414386804185735, -0.12940952255092145],                  # db2, 2 mn
    6: [0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
        -0.13501102001039084, -0.08544127388224149,
        0.035226291882100656],                                       # db3, 3 mn
    8: [0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
        -0.02798376941698385, -0.18703481171888114,
        0.030841381835986965, 0.032883011666982945,
        -0.010597401784997278],                                      # db4, 4 mn
}


def _filtros(longitud: int) -> tuple[np.ndarray, np.ndarray]:
    """(h, g): escala y ondícula. g_k = (−1)^k h_{L−1−k}."""
    if longitud not in DAUBECHIES:
        raise ValueError(f"longitud {longitud} no disponible; "
                         f"usa {sorted(DAUBECHIES)}")
    h = np.asarray(DAUBECHIES[longitud], dtype=float)
    g = h[::-1].copy()
    g[1::2] *= -1.0
    return h, g


def _filtrar_diezmar(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Convolución circular seguida de diezmado por 2 (pirámide de Mallat)."""
    n = x.size
    L = f.size
    idx = (np.arange(0, n, 2)[:, None] + np.arange(L)[None, :]) % n
    return x[idx] @ f[::-1]


def descomponer(x: np.ndarray, n_octavas: int | None = None,
                longitud: int = 6) -> list[np.ndarray]:
    r"""DWT ortogonal: devuelve los coeficientes de detalle d_j, j = 1..J.

    Se descartan los coeficientes de borde de cada octava, contaminados por la
    extensión circular: en la octava j el filtro abarca (L−1)(2^j−1)+1 muestras
    del original, y esos coeficientes mezclan los dos extremos de la serie.
    Para memoria larga ese contagio no es despreciable.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 64:
        raise ValueError("serie demasiado corta para el diagrama log-escala")
    h, g = _filtros(longitud)
    max_oct = int(np.floor(np.log2(n / (longitud * 2))))
    J = max_oct if n_octavas is None else min(int(n_octavas), max_oct)

    detalles: list[np.ndarray] = []
    a = v - v.mean()
    for j in range(1, J + 1):
        d = _filtrar_diezmar(a, g)
        a = _filtrar_diezmar(a, h)
        borde = min(longitud - 1, max(d.size // 4, 1))
        detalles.append(d[borde:d.size - borde] if d.size > 2 * borde else d)
    return detalles


def diagrama_log_escala(x: np.ndarray, longitud: int = 6,
                        n_octavas: int | None = None) -> dict:
    r"""y_j = log₂⟨d²_j⟩ corregido de sesgo, con su varianza teórica.

    Corrección de Abry--Veitch: si μ̂_j es la media de n_j cuadrados de
    variables gaussianas, entonces
        E[log₂ μ̂_j] = log₂ E[μ̂_j] + g_j ,  g_j = ψ(n_j/2)/ln2 − log₂(n_j/2)
        Var[log₂ μ̂_j] = ζ(2, n_j/2)/ln²2 .
    Se resta g_j para que la regresión sea insesgada y se usa 1/Var como peso.
    """
    from scipy.special import polygamma, psi

    detalles = descomponer(x, n_octavas=n_octavas, longitud=longitud)
    ln2 = np.log(2.0)
    js, ys, vs, ns = [], [], [], []
    for j, d in enumerate(detalles, start=1):
        n_j = d.size
        if n_j < 8:
            continue
        mu = float(np.mean(d ** 2))
        if not np.isfinite(mu) or mu <= 0:
            continue
        g_j = psi(n_j / 2.0) / ln2 - np.log2(n_j / 2.0)
        js.append(j)
        ys.append(np.log2(mu) - g_j)
        vs.append(polygamma(1, n_j / 2.0) / ln2 ** 2)
        ns.append(n_j)
    return {"j": np.asarray(js, dtype=float), "y": np.asarray(ys),
            "var": np.asarray(vs), "n_coef": np.asarray(ns, dtype=int),
            "longitud_filtro": longitud,
            "momentos_nulos": longitud // 2}


def gamma_abry_veitch(x: np.ndarray, longitud: int = 6,
                      j_min: int = 2, j_max: int | None = None) -> dict:
    r"""γ = 1 − α, con α la pendiente ponderada del diagrama log-escala.

    `j_min` descarta las octavas más finas, donde el comportamiento de escala
    aún no se ha establecido (efecto de la discretización de la serie); es el
    único parámetro de decisión del estimador y conviene reportar la
    sensibilidad a él.
    """
    dia = diagrama_log_escala(x, longitud=longitud)
    j, y, v = dia["j"], dia["y"], dia["var"]
    m = j >= j_min
    if j_max is not None:
        m &= j <= j_max
    if m.sum() < 3:
        return {"gamma": np.nan, "se": np.nan, "n_octavas": int(m.sum())}

    w = 1.0 / v[m]
    S0, S1, S2 = w.sum(), (w * j[m]).sum(), (w * j[m] ** 2).sum()
    Sy, Sjy = (w * y[m]).sum(), (w * j[m] * y[m]).sum()
    det = S0 * S2 - S1 ** 2
    if det <= 0:
        return {"gamma": np.nan, "se": np.nan, "n_octavas": int(m.sum())}
    alpha = (S0 * Sjy - S1 * Sy) / det
    se_alpha = np.sqrt(S0 / det)

    resid = y[m] - (alpha * j[m] + (Sy - alpha * S1) / S0)
    chi2 = float(np.sum(w * resid ** 2))
    dof = int(m.sum()) - 2
    return {"gamma": float(1.0 - alpha), "se": float(se_alpha),
            "alpha": float(alpha), "H": float(1.0 - (1.0 - alpha) / 2.0),
            "n_octavas": int(m.sum()), "j_min": j_min,
            "chi2_dof": chi2 / dof if dof > 0 else float("nan"),
            "momentos_nulos": dia["momentos_nulos"],
            "diagrama": {"j": j.tolist(), "y": y.tolist(),
                         "se": np.sqrt(v).tolist()}}


def varianza_haar(x: np.ndarray, escalas) -> dict:
    r"""Varianza de Haar SIN normalizar por escala — control de la deriva ν_loc.

    A escala T se compara la suma del bloque siguiente con la del anterior:

        d_T(k) = Σ_{t∈bloque 2} x_t − Σ_{t∈bloque 1} x_t .

    Deliberadamente **sin** el factor 1/√(2T) de la ondícula ortonormal: con la
    diferencia cruda se cumple

        Var(d_T) = 4·Var(Q_T) − Var(Q_{2T}) ~ T^ν ,

    el MISMO exponente que la varianza de conteo, de modo que su pendiente
    local es directamente comparable con ν_loc. (Normalizada daría T^(ν−1) y la
    comparación exigiría acordarse de sumar uno.)

    La diferencia decisiva no es el exponente sino la inmunidad: restar dos
    bloques aniquila las tendencias lineales, mientras que la suma en bloque de
    Var(Q_T) las acumula. Si la deriva del exponente local sobrevive aquí, no
    es un efecto de tendencia.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    filas = []
    for T in escalas:
        T = int(T)
        n_par = v.size // (2 * T)
        if n_par < 16:
            continue
        bloques = v[:n_par * 2 * T].reshape(n_par, 2, T).sum(axis=2)
        d = bloques[:, 1] - bloques[:, 0]
        filas.append({"T": T, "n_pares": int(n_par),
                      "var_haar": float(np.var(d, ddof=1))})
    out = {"tabla": filas}
    if len(filas) >= 3:
        lt = np.log([f["T"] for f in filas])
        lv = np.log([f["var_haar"] for f in filas])
        out["pendiente_local"] = np.gradient(lv, lt).tolist()
        xc = lt - lt.mean()
        out["pendiente_global"] = float(np.dot(xc, lv - lv.mean())
                                        / np.dot(xc, xc))
    return out


def fgn(n: int, H: float, seed: int = 0) -> np.ndarray:
    r"""Ruido gaussiano fraccionario EXACTO por el método de Davies--Harte.

    Es el patrón de referencia para probar estimadores de memoria larga, y hace
    falta uno propio: `counting.synthetic_flow` construye su serie empotrando
    una ACF (1+τ)^γ en un círculo y recortando los autovalores negativos, lo
    cual reproduce bien los desfases CORTOS —que es para lo que se escribió—
    pero distorsiona justamente las frecuencias BAJAS. Un estimador espectral o
    de ondícula lee esas frecuencias bajas, así que validarlo contra
    `synthetic_flow` mide la distorsión del generador y no el sesgo del
    estimador.

    Aquí la autocovarianza es la exacta del fGn,

        γ(k) = ½[|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H}] ,

    cuyos autovalores circulantes son no negativos por construcción para
    0 < H < 1, de modo que no hay recorte y el espectro de baja frecuencia es
    el correcto. Relación con nuestro exponente: γ_ACF = 2 − 2H.
    """
    from scipy.fft import next_fast_len

    if not 0.0 < H < 1.0:
        raise ValueError("H debe estar en (0,1)")
    m = next_fast_len(2 * int(n))
    k = np.arange(m)
    k = np.minimum(k, m - k).astype(float)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k - 1) ** (2 * H))
    lam = np.fft.rfft(g).real
    if lam.min() < -1e-8 * max(lam.max(), 1.0):
        raise RuntimeError(f"empotramiento no PSD para H={H}")
    lam = np.clip(lam, 0.0, None)
    rng = np.random.default_rng(seed)
    return np.fft.irfft(np.fft.rfft(rng.normal(size=m)) * np.sqrt(lam), m)[:int(n)]
