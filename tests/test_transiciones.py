r"""
Maquinaria del experimento 04: objetivo binario, logística y log-loss.

EL TEST QUE FALTABA EN TODO EL REPO: UN CONTROL POSITIVO
---------------------------------------------------------
Los experimentos 01-03 tienen controles NEGATIVOS excelentes (surrogados,
ruido, permutaciones) que demuestran que el pipeline no fabrica señal donde no
la hay. Pero ninguno demuestra lo contrario: que el pipeline SABE VER una
señal cuando sí la hay. Sin eso, un negativo es ambiguo — puede significar "no
hay efecto" o "la maquinaria está rota".

`test_control_positivo_*` construye datos donde la característica SÍ predice
el evento, por construcción, y exige que el contraste lo detecte. Si algún día
el experimento 04 da negativo, ese test es lo que permite afirmar que el
negativo es del fenómeno y no del código.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.evaluation import diebold_mariano, diebold_mariano_losses
from keldysh_finance.pipeline import (_ajusta_logistica, _log_loss,
                                      contraste_transicion, objetivo_transicion)
from keldysh_finance.baselines import ewma_vol


# --- el refactor del DM no puede haber movido nada ---------------------------

def test_dm_sobre_perdidas_coincide_con_el_dm_de_siempre():
    """`diebold_mariano` ahora delega en `diebold_mariano_losses`. Este test
    fija que la delegación no cambió el estadístico."""
    rng = np.random.default_rng(0)
    n = 2000
    rv = np.abs(rng.normal(0.01, 0.003, n))
    a = np.abs(rng.normal(0.011, 0.003, n))
    b = np.abs(rng.normal(0.010, 0.003, n))
    directo = diebold_mariano(rv, a, b, horizon=30)
    # misma cuenta, pasando por la ruta de series de pérdida
    from keldysh_finance.evaluation import _qlike_losses
    via_perdidas = diebold_mariano_losses(_qlike_losses(rv, a),
                                          _qlike_losses(rv, b), horizon=30)
    assert directo == via_perdidas


# --- logística ---------------------------------------------------------------

def test_logistica_recupera_coeficientes_conocidos():
    rng = np.random.default_rng(1)
    n = 20000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    beta = np.array([-0.7, 1.3, -0.9])
    X = np.column_stack([np.ones(n), x1, x2])
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = (rng.uniform(size=n) < p).astype(float)
    est = _ajusta_logistica(X, y)
    assert est is not None
    np.testing.assert_allclose(est, beta, atol=0.06)


def test_logistica_sobrevive_a_separacion_casi_perfecta():
    """Con separación, la verosimilitud no tiene máximo finito. El ridge
    minúsculo tiene que impedir que esto devuelva NaN o reviente."""
    n = 500
    x = np.linspace(-3, 3, n)
    y = (x > 0).astype(float)                       # separación perfecta
    est = _ajusta_logistica(np.column_stack([np.ones(n), x]), y)
    assert est is not None and np.all(np.isfinite(est))


def test_logloss_es_regla_propia():
    """Se minimiza al declarar la probabilidad VERDADERA, no otra."""
    rng = np.random.default_rng(2)
    p_true = 0.3
    y = (rng.uniform(size=200000) < p_true).astype(float)
    perdidas = {p: _log_loss(y, np.full_like(y, p)).mean()
                for p in (0.1, 0.2, 0.3, 0.4, 0.6)}
    assert min(perdidas, key=perdidas.get) == 0.3, perdidas


# --- el objetivo -------------------------------------------------------------

def _serie(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.97 * u[t - 1] + rng.normal(0, 0.15)
    return rng.normal(0, 0.01 * np.exp(u)), u


def test_umbral_del_objetivo_sale_solo_del_train():
    """θ es un cuantil del tramo de entrenamiento: cambiar el futuro no puede
    moverlo. Si se calculase sobre toda la muestra sería look-ahead, y del que
    no detecta ningún test de ventana."""
    r, _ = _serie()
    n = len(r)
    entrena = np.arange(n) < int(n * 0.60)
    r2 = r.copy()
    r2[int(n * 0.60):] *= 8.0                        # futuro muy distinto
    _, th1 = objetivo_transicion(r, ewma_vol(r), horizon=30, entrena=entrena)
    _, th2 = objetivo_transicion(r2, ewma_vol(r2), horizon=30, entrena=entrena)
    assert abs(th1 - th2) < 1e-12, f"theta se movio: {th1} vs {th2}"


def test_tasa_base_es_la_declarada_en_train():
    r, _ = _serie()
    n = len(r)
    entrena = np.arange(n) < int(n * 0.60)
    S, _ = objetivo_transicion(r, ewma_vol(r), horizon=30, entrena=entrena,
                               cuantil=0.80)
    tr = S[entrena & np.isfinite(S)]
    assert abs(tr.mean() - 0.20) < 0.02, tr.mean()


# --- controles positivo y negativo del contraste completo --------------------

def _datos_con_senal(n=6000, ruido=1.0, seed=3):
    """Serie donde la característica SÍ anticipa el salto, por construcción.

    V_t se construye a partir del estado de volatilidad FUTURO. Eso es hacer
    trampa a propósito: es lo que convierte esto en un control positivo. Lo que
    se comprueba no es que la trampa funcione, sino que el contraste es capaz
    de verla — y por tanto que un negativo en datos reales significa algo.
    """
    rng = np.random.default_rng(seed)
    r, u = _serie(n, seed)
    H = 30
    fut = np.full(n, np.nan)
    for t in range(n - H - 1):
        fut[t] = u[t + 1:t + 1 + H].mean()
    V = np.exp(fut + rng.normal(0, ruido, n))        # positiva, como las reales
    return r, V


def test_control_positivo_el_contraste_ve_una_senal_real():
    r, V = _datos_con_senal(ruido=0.8)
    res = contraste_transicion(r, V, horizon=30)
    assert "error" not in res, res
    dm = res["dm_decisivo"]["EWMA vs EWMA+X"]
    assert dm["mean_diff"] > 0, "con senal real, el modelo con X debe perder MENOS"
    assert dm["dm_stat"] > 3.0, f"no detecta una senal puesta a mano: DM={dm['dm_stat']:.2f}"
    assert res["logloss"]["EWMA+X"] < res["logloss"]["EWMA"]


def test_control_negativo_una_caracteristica_de_ruido_no_pasa():
    rng = np.random.default_rng(11)
    r, _ = _serie(6000, seed=4)
    V = np.exp(rng.normal(0, 1.0, len(r)))           # sin relacion con nada
    res = contraste_transicion(r, V, horizon=30)
    assert "error" not in res, res
    dm = res["dm_decisivo"]["EWMA vs EWMA+X"]
    assert abs(dm["dm_stat"]) < 3.0, \
        f"ruido puro no deberia dar un DM grande: {dm['dm_stat']:.2f}"


def test_mismo_soporte_en_los_dos_modelos():
    """Los dos modelos tienen que evaluarse sobre exactamente los mismos
    puntos, o la diferencia de log-loss mezcla el efecto de X con el de haber
    visto datos distintos."""
    r, V = _datos_con_senal(ruido=1.0)
    V[:800] = np.nan                                  # la caracteristica falta al principio
    res = contraste_transicion(r, V, horizon=30)
    dm = res["dm_decisivo"]["EWMA vs EWMA+X"]
    assert dm["n"] > 100
    # ambas perdidas se promedian sobre el mismo n: si no, este DM no existiria
    assert np.isfinite(res["logloss"]["EWMA"]) and np.isfinite(res["logloss"]["EWMA+X"])
