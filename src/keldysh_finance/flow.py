r"""
Flujo de órdenes y FUNCIÓN DE RESPUESTA — la mitad que faltaba del diagnóstico.

POR QUÉ ESTE MÓDULO ES EL IMPORTANTE
------------------------------------
`stationarity.py` mide sólo la CORRELACIÓN. El diagnóstico de no-equilibrio de
verdad es la violación del teorema fluctuación-disipación (FDT), que relaciona
correlación con RESPUESTA. Sin respuesta, el programa está a medias.

La variable conjugada al precio en un mercado es el FLUJO DE ÓRDENES firmado
(Bouchaud, Farmer, Lillo). La respuesta es el impacto de mercado:

    R(τ) = ⟨ (p_{t+τ} − p_t) · ε_t ⟩ / ⟨ ε_t² ⟩

donde ε_t es el flujo firmado en t. Es literalmente la función de respuesta
retardada G^R del formalismo: cuánto se mueve el observable en t+τ ante una
perturbación aplicada en t.

DE DÓNDE SALEN LOS DATOS (hallazgo 2026-08-08)
-----------------------------------------------
No hacen falta datos de pago ni descargas de gigabytes. Las **klines públicas
de Binance ya traen el flujo firmado**: el campo `taker_buy_base` es el volumen
ejecutado por agresores COMPRADORES, y el resto del volumen es de agresores
vendedores. De ahí:

    ε_t = taker_buy − (volumen_total − taker_buy) = 2·taker_buy − volumen

Sin clave de API, con años de histórico, y en la misma llamada que ya hacía
`data.get_binance` del proyecto de trading — que descargaba esos campos y los
descartaba en `df.set_index(...)[OHLCV]`.

La alternativa (`data.binance.vision`, aggTrades trade a trade, ~11 MB/día
comprimido) da resolución de operación individual y el signo exacto vía el
flag `isBuyerMaker`. Merece la pena sólo si se quiere estudiar la escala
intra-vela; para R(τ) a escalas de minutos-horas, la agregación por vela basta
y es tres órdenes de magnitud más barata.

EL FDT EN MERCADOS
------------------
En equilibrio, respuesta y correlación están ligadas: R(τ) = −β dC(τ)/dτ.
Fuera de equilibrio la relación se rompe, y el cociente

    T_eff(τ) = − [dC(τ)/dτ] / R(τ)

deja de ser constante. Su dependencia con τ es la medida canónica de
violación del FDT (el "cociente fluctuación-disipación" X de los sistemas
vítreos). Constante ⇒ comportamiento tipo equilibrio; variable ⇒ transitorio.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

_KLINE_COLS = ["openTime", "Open", "High", "Low", "Close", "Volume", "closeTime",
               "qVol", "trades", "tbBase", "tbQuote", "ignore"]


@dataclass(frozen=True)
class ResponseResult:
    """R(τ), C_ε(τ) y el diagnóstico de violación del FDT."""
    lags: np.ndarray
    response: np.ndarray          # R(τ): impacto de mercado
    flow_autocorr: np.ndarray     # C_ε(τ): memoria del flujo
    t_eff: np.ndarray             # T_eff(τ): cociente FDT
    fdt_violation: float          # dispersión relativa de T_eff (0 = equilibrio)
    n_obs: int


def fetch_klines_with_flow(symbol: str = "BTCUSDT", interval: str = "1h",
                           years: float = 2.0, cache_dir: str | None = None,
                           refresh: bool = False) -> pd.DataFrame:
    """Klines de Binance CONSERVANDO los campos de flujo.

    Devuelve un DataFrame con Open/High/Low/Close/Volume + `tbBase` (volumen
    de agresores compradores) y `trades`. Público, sin clave.

    `requests` se importa DENTRO de la rama de descarga, no arriba: leer un CSV
    ya cacheado no debe exigir una librería de red. Estaba al principio de la
    función y eso rompía la ejecución en el cluster —donde los datos van
    copiados y no hay `requests` instalado— con un ModuleNotFoundError que no
    tenía nada que ver con lo que el código iba a hacer.
    """
    symbol = symbol.replace("-", "").replace("=", "").replace("USD", "USDT") \
        if symbol.endswith("-USD") else symbol.replace("-", "").replace("=", "")
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__),
                                          "..", "..", "output", "cache")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"flow_{symbol}_{interval}_{years}y.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import requests

    start_ms = int((time.time() - years * 365.25 * 86400) * 1000)
    rows = []
    while True:
        r = requests.get(BINANCE_KLINES,
                         params={"symbol": symbol, "interval": interval,
                                 "startTime": start_ms, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
        time.sleep(0.15)                       # cortesía con el rate limit

    if not rows:
        raise RuntimeError(f"Binance no devolvió datos para {symbol} {interval}")

    df = pd.DataFrame(rows, columns=_KLINE_COLS)
    df["Datetime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    keep = ["Open", "High", "Low", "Close", "Volume", "tbBase", "trades"]
    df = df.set_index("Datetime")[keep].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


def order_flow_imbalance(df: pd.DataFrame, normalize: str = "volume") -> np.ndarray:
    r"""ε_t: flujo de órdenes firmado por vela.

        ε_t = taker_buy − taker_sell = 2·tbBase − Volume

    normalize:
      'volume' → ε/Volumen ∈ [−1,1]. Adimensional; comparable entre regímenes
                 de actividad muy distintos. Es el valor por defecto porque el
                 volumen absoluto de cripto cambia órdenes de magnitud en años
                 y contaminaría cualquier correlación con esa tendencia.
      'none'   → ε en unidades de volumen base.
    """
    vol = df["Volume"].to_numpy(float)
    tb = df["tbBase"].to_numpy(float)
    eps = 2.0 * tb - vol
    if normalize == "none":
        return eps
    if normalize != "volume":
        raise ValueError(f"normalize desconocido: {normalize!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vol > 0, eps / vol, 0.0)
    return np.clip(out, -1.0, 1.0)


_MIN_PARES = 30          # nº mínimo de pares por desfase para no devolver NaN


def _sumas_correlacion(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    r"""S[k] = Σ_{t=0}^{n-1-k} a[t]·b[t+k], para k = 0..max_lag, por FFT.

    Correlación LINEAL (no circular): se rellena con ceros hasta
    `nfft >= n + max_lag`, de modo que ningún término dé la vuelta. Los
    sumandos con t+k >= n valen 0 porque b está rellenado, que es justo el
    rango variable —n−k términos por desfase— que usa la versión de referencia.

    Es exacto, no aproximado: sustituye un bucle de `max_lag` productos por
    tres transformadas. Ver `tests/test_flow_vectorizado.py`, que fija la
    equivalencia con la implementación directa hasta ~1e-12.

    Se usa `numpy.fft` y no `scipy.fft` a propósito: con ventanas de ~500
    puntos la transformada en sí es barata y lo que domina es el coste de
    llamada. El backend `uarray` de scipy añadía ~0.18 s por cada 1000
    ventanas (medido con cProfile el 2026-08-08) sin aportar nada aquí;
    `next_fast_len` sí se toma de scipy porque es cálculo puro y no despacha.
    """
    from scipy.fft import next_fast_len

    n = len(a)
    nfft = next_fast_len(n + max_lag)
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    return np.fft.irfft(np.conj(A) * B, nfft)[:max_lag + 1]


def response_function(log_prices_pre: np.ndarray, flow: np.ndarray,
                      max_lag: int = 60) -> tuple[np.ndarray, np.ndarray]:
    r"""R(τ) = ⟨(p_{t+τ} − p_t)·ε_t⟩ / ⟨ε²⟩ — el impacto de mercado.

    ⚠ `log_prices_pre` DEBE ser el precio ANTERIOR a que actúe el flujo de la
    vela t — es decir, el log(Open), no el log(Close). Usar Close es un error
    de temporización sutil y grave: el impacto de ε_t ya está incorporado en
    Close_t, así que R mediría la RELAJACIÓN posterior en lugar del impacto, y
    sale con el signo cambiado.

    Medido en BTCUSDT 1h, 2 años (2024-08 → 2026-08):
        referencia Close : R(1) = −4.7e−4   (relajación, signo invertido)
        referencia Open  : R(1) = +1.32e−2  (impacto real, persistente)

    Usa `response_from_frame()` para no tener que acordarte de esto.

    Implementación: vía rápida por FFT si no hay no-finitos, y si los hay se
    cae al bucle directo. Las dos dan el mismo número (tests de equivalencia).
    """
    p = np.asarray(log_prices_pre, float)
    e = np.asarray(flow, float)
    n = min(len(p), len(e))
    p, e = p[:n], e[:n]
    # `nanmean` es caro y se llama por cada ventana rodante; cuando no hay
    # no-finitos —el caso normal— `mean` da EXACTAMENTE lo mismo y evita el
    # rodeo de enmascarado. La comprobación se reaprovecha para elegir vía.
    fin_e = bool(np.isfinite(e).all())
    ec = e - (e.mean() if fin_e else np.nanmean(e))
    var_e = float((ec ** 2).mean() if fin_e else np.nanmean(ec ** 2))
    lags = np.arange(1, max_lag + 1)
    R = np.full(len(lags), np.nan)
    if var_e <= 0:
        return lags, R

    if fin_e and np.isfinite(p).all():
        # Σ (p[t+k] − p[t])·ec[t] se parte en dos: la cruzada es una
        # correlación (FFT) y la otra una suma prefija de p·ec. Restar la media
        # de p es EXACTO —la diferencia p[t+k]−p[t] no la ve— y mejora mucho el
        # condicionamiento, porque los log-precios son ~11 y el flujo ~0.2.
        pc = p - p.mean()
        cruz = _sumas_correlacion(ec, pc, max_lag)          # Σ ec[t]·p[t+k]
        prefijo = np.cumsum(pc * ec)                        # Σ_{t<=m} p[t]·ec[t]
        k = np.arange(max_lag + 1)
        pares = n - k
        valido = pares > _MIN_PARES
        num = np.where(valido, cruz - prefijo[np.clip(n - 1 - k, 0, n - 1)], np.nan)
        with np.errstate(invalid="ignore"):
            todo = np.where(valido, num / np.where(valido, pares, 1) / var_e, np.nan)
        return lags, todo[1:]                               # k=0 no se reporta

    for idx, lag in enumerate(lags):
        dp = p[lag:] - p[:-lag]
        prod = dp * ec[:-lag]
        m = np.isfinite(prod)
        if m.sum() > _MIN_PARES:
            R[idx] = float(np.mean(prod[m])) / var_e
    return lags, R


def flow_autocorrelation(flow: np.ndarray, max_lag: int = 60) -> np.ndarray:
    r"""C_ε(τ) = ⟨ε_t ε_{t+τ}⟩ / ⟨ε²⟩ — memoria del flujo de órdenes.

    Resultado conocido (Lillo-Farmer): el flujo tiene memoria LARGA con
    decaimiento en ley de potencias, mientras el precio es casi difusivo. Esa
    tensión es la que la respuesta tiene que compensar — y es el sitio natural
    donde buscar una condición de balance de tipo FDT.

    Igual que `response_function`: vía rápida por FFT, con caída al bucle si
    hay no-finitos.
    """
    e = np.asarray(flow, float)
    fin_e = bool(np.isfinite(e).all())          # ver nota en response_function
    ec = e - (e.mean() if fin_e else np.nanmean(e))
    denom = float((ec ** 2).mean() if fin_e else np.nanmean(ec ** 2))
    n = len(ec)
    lags = np.arange(1, max_lag + 1)
    C = np.full(len(lags), np.nan)
    if denom <= 0:
        return C

    if fin_e:
        auto = _sumas_correlacion(ec, ec, max_lag)
        k = np.arange(max_lag + 1)
        pares = n - k
        valido = pares > _MIN_PARES
        with np.errstate(invalid="ignore"):
            todo = np.where(valido, auto / np.where(valido, pares, 1) / denom, np.nan)
        return todo[1:]

    for idx, lag in enumerate(lags):
        prod = ec[:-lag] * ec[lag:]
        m = np.isfinite(prod)
        if m.sum() > _MIN_PARES:
            C[idx] = float(np.mean(prod[m])) / denom
    return C


def response_from_frame(df: pd.DataFrame, max_lag: int = 60,
                        normalize: str = "volume"):
    """R(τ) y ε desde el DataFrame, con la temporización YA correcta.

    Es la vía recomendada: toma log(Open) como precio de referencia, que es lo
    que evita el error descrito en `response_function`. Devuelve
    (lags, R, flujo, log_open).
    """
    for col in ("Open", "Close", "Volume", "tbBase"):
        if col not in df.columns:
            raise KeyError(f"falta la columna {col!r}: ¿usaste fetch_klines_with_flow?")
    eps = order_flow_imbalance(df, normalize=normalize)
    log_open = np.log(df["Open"].to_numpy(float))
    lags, R = response_function(log_open, eps, max_lag)
    return lags, R, eps, log_open


def fdt_diagnostic(log_prices_pre: np.ndarray, flow: np.ndarray,
                   max_lag: int = 60) -> ResponseResult:
    r"""Cociente T_eff(τ) y su dispersión como medida de violación del FDT.

    En equilibrio R(τ) ∝ −dC(τ)/dτ con constante de proporcionalidad fija (la
    temperatura). Aquí se calcula

        T_eff(τ) = −[dC/dτ](τ) / R(τ)

    y se resume su NO constancia mediante la dispersión relativa
    (desviación típica / |mediana|) sobre los desfases con R bien definida.

    Interpretación: 0 ⇒ compatible con equilibrio; valores grandes ⇒ el
    sistema está lejos del régimen donde respuesta y fluctuación están ligadas.
    """
    lags, R = response_function(log_prices_pre, flow, max_lag)
    C = flow_autocorrelation(flow, max_lag)

    dC = np.full_like(C, np.nan)
    if len(C) >= 3:
        dC[1:-1] = (C[2:] - C[:-2]) / 2.0          # diferencia central
        dC[0] = C[1] - C[0] if len(C) > 1 else np.nan
        dC[-1] = C[-1] - C[-2] if len(C) > 1 else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        t_eff = np.where(np.abs(R) > 1e-12, -dC / R, np.nan)

    fin = t_eff[np.isfinite(t_eff)]
    if len(fin) >= 5:
        med = np.median(np.abs(fin))
        viol = float(np.std(fin) / med) if med > 0 else float("nan")
    else:
        viol = float("nan")

    return ResponseResult(lags=lags, response=R, flow_autocorr=C, t_eff=t_eff,
                          fdt_violation=viol,
                          n_obs=int(min(len(log_prices_pre), len(flow))))


def align_to_returns(t_w: np.ndarray, values: np.ndarray,
                     n_klines: int) -> np.ndarray:
    r"""Pasa una característica indexada por VELA a la rejilla de LOG-RETORNOS.

    El desfase es de UNA posición y no es cosmético: es la diferencia entre un
    experimento causal y uno con look-ahead.

    Las características de este módulo viven en el índice de vela j, y usan
    `Open[<=j]` y `ε[<=j]`. Como ε_j es el flujo ejecutado DURANTE la vela j,
    ese valor no se conoce hasta el cierre de j. Por otro lado, los
    log-retornos de los experimentos son `r = diff(log(Close))`, con

        r[i] = log C[i+1] − log C[i]

    así que en el cierre de la vela j el último retorno conocido es r[j−1].
    Ambas cosas se saben en el MISMO instante de reloj, luego:

        característica en la vela j  ↔  índice de retorno i = j − 1

    o sea, hay que quitar el primer elemento de la rejilla de velas. Usar
    directamente el array de velas contra el de retornos adelanta la
    característica una barra y le regala al modelo información que en ese
    momento no existía.

    El relleno es hacia adelante (último valor DISPONIBLE), nunca interpolando
    con futuro, igual que en el experimento 01.

    Returns
    -------
    Array de longitud `n_klines - 1`, alineado con `np.diff(np.log(Close))`.
    """
    out = np.full(int(n_klines), np.nan, dtype=float)
    out[np.asarray(t_w, dtype=int)] = np.asarray(values, dtype=float)
    out = pd.Series(out).ffill().to_numpy()
    return out[1:]


def rolling_fdt_violation(log_prices_pre: np.ndarray, flow: np.ndarray,
                          window: int = 500, max_lag: int = 30,
                          step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Violación del FDT calculada en ventanas TRAILING.

    Es la versión causal, apta para usarse como característica predictiva:
    el valor en t_w usa sólo datos con índice <= t_w.

    Returns
    -------
    (t_w, violacion) — índices y serie del diagnóstico.
    """
    p = np.asarray(log_prices_pre, float)
    e = np.asarray(flow, float)
    n = min(len(p), len(e))
    ends = np.arange(window - 1, n, step)
    out = np.full(len(ends), np.nan)
    for k, end in enumerate(ends):
        sl = slice(end - window + 1, end + 1)      # TRAILING
        res = fdt_diagnostic(p[sl], e[sl], max_lag=max_lag)
        out[k] = res.fdt_violation
    return ends, out
