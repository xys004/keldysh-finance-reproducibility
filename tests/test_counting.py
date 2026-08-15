"""
Tests de la estadística de conteo (exp. 06): recuperación exacta, nulos y
propiedades estructurales.

El ancla es que para Q gaussiana la simetría de fluctuación es EXACTA:
s(Q) = (2μ/σ²)·Q. Eso convierte el control positivo en una comprobación
contra un valor analítico, no contra una simulación de referencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.counting import (
    block_bootstrap_standardized_cumulants,
    cumulant_scaling,
    fano_factor,
    fit_affinity,
    net_charge,
    standardized_cumulants,
    symmetry_function,
    synthetic_flow,
)


# --- control positivo: valores analíticos conocidos --------------------------

def test_afinidad_gaussiana_iid_exacta():
    """i.i.d. N(mu, sigma²) a T=1: A = 2·mu/sigma², analítico."""
    rng = np.random.default_rng(1)
    mu, sigma = 0.15, 1.0
    x = rng.normal(mu, sigma, 400_000)
    q, s, se = symmetry_function(net_charge(x, 1))
    fit = fit_affinity(q, s, se)
    A_teoria = 2.0 * mu / sigma ** 2
    assert abs(fit["A"] - A_teoria) < 3.5 * fit["se_A"] + 0.02, \
        f"A={fit['A']:.4f} vs teoría {A_teoria:.4f}"
    # gaussiano ⇒ sin curvatura
    assert fit["b3"] == fit["c3"]
    assert fit["se_b3"] == fit["se_c3"]
    assert abs(fit["b3"]) < 3.5 * fit["se_b3"] + 1e-3


def test_escalado_difusivo_iid():
    """Incrementos i.i.d.: pendiente de κ₂ = 1 (difusivo)."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 300_000)
    esc = cumulant_scaling(x, [1, 2, 4, 8, 16, 32, 64])
    assert abs(esc["pendiente_k2"] - 1.0) < 0.05, \
        f"pendiente {esc['pendiente_k2']:.3f} vs 1"


def test_cumulantes_estandarizados_gaussianos_incluyen_cero():
    rng = np.random.default_rng(12)
    x = rng.normal(size=20_000)
    result = block_bootstrap_standardized_cumulants(
        x, block_length=64, replicates=199, seed=13, batch_size=8
    )
    assert abs(result["gamma1"]) < 0.05
    assert abs(result["gamma2"]) < 0.10
    assert result["gamma1_ci95"][0] < 0 < result["gamma1_ci95"][1]
    assert result["gamma2_ci95"][0] < 0 < result["gamma2_ci95"][1]


def test_cumulantes_estandarizados_detectan_asimetria_y_colas():
    rng = np.random.default_rng(14)
    x = rng.lognormal(mean=0.0, sigma=0.8, size=50_000)
    result = standardized_cumulants(x)
    assert result["gamma1"] > 2.0
    assert result["gamma2"] > 5.0


def _var_teorica(T: int, gamma: float) -> float:
    """Var(Q_T) exacta del fGn unitario: T^(2H), H=1+gamma/2."""
    return float(T ** (2.0 + gamma))


def test_escalado_superdifusivo_con_memoria():
    """El fGn con gamma=-0.5 tiene Var(Q_T)=T^1.5 exactamente."""
    x = synthetic_flow(600_000, gamma=-0.5, seed=3)
    esc = cumulant_scaling(x, [16, 32, 64, 128, 256])
    for fila in esc["tabla"]:
        teo = _var_teorica(fila["T"], -0.5)
        assert abs(fila["k2"] / teo - 1.0) < 0.20, \
            f"k2(T={fila['T']}) = {fila['k2']:.1f} vs teórica {teo:.1f}"
    assert esc["pendiente_k2"] > 1.3, \
        f"pendiente {esc['pendiente_k2']:.3f}, esperada superdifusiva (>1.3)"


def test_afinidad_decae_con_memoria():
    """Con memoria y sesgo, A(T) = 2·mu·T/Var(Q_T) ~ T^(-0.5): la correlación
    diluye el sesgo aparente. Comprueba el decaimiento entre T=1 y T=64."""
    x = synthetic_flow(600_000, gamma=-0.5, mu=0.08, seed=4)
    ajustes = {}
    for T in (1, 64):
        Q = net_charge(x, T)
        fit = fit_affinity(*symmetry_function(Q))
        ajustes[T] = fit
        # predicción gaussiana exacta por ventana
        A_teo = 2.0 * np.mean(Q) / np.var(Q, ddof=1)
        assert abs(fit["A"] - A_teo) < 4.0 * fit["se_A"] + 0.05 * abs(A_teo)
    assert ajustes[64]["A"] < 0.5 * ajustes[1]["A"], \
        "A(64) debería ser bastante menor que A(1) con memoria gamma=-0.5"


# --- nulos: lo destruido tiene que salir destruido ---------------------------

def test_signflip_mata_la_afinidad():
    """Signos aleatorios ⇒ P(Q) simétrica ⇒ A compatible con 0."""
    rng = np.random.default_rng(5)
    x = synthetic_flow(400_000, gamma=-0.5, mu=0.1, seed=6)
    y = np.abs(x) * rng.choice([-1.0, 1.0], size=len(x))
    fit = fit_affinity(*symmetry_function(net_charge(y, 16)))
    assert abs(fit["A"]) < 3.5 * fit["se_A"], \
        f"signflip dejó A={fit['A']:.4f} (se={fit['se_A']:.4f})"


def test_permutar_devuelve_el_escalado_difusivo():
    """La permutación conserva la marginal (y el sesgo) pero mata la memoria:
    la pendiente de κ₂ tiene que volver a 1."""
    rng = np.random.default_rng(7)
    x = synthetic_flow(300_000, gamma=-0.5, mu=0.05, seed=8)
    esc = cumulant_scaling(rng.permutation(x), [1, 2, 4, 8, 16, 32, 64])
    assert abs(esc["pendiente_k2"] - 1.0) < 0.06


def test_fano_uno_con_trades_independientes():
    """N trades por vela, cada uno ±q de signo independiente ⇒ F = 1."""
    rng = np.random.default_rng(9)
    n, por_vela, q = 100_000, 20, 2.0
    signos = rng.choice([-1.0, 1.0], size=(n, por_vela))
    flujo = q * signos.sum(axis=1)
    volumen = np.full(n, por_vela * q)
    trades = np.full(n, float(por_vela))
    fano = fano_factor(flujo, volumen, trades, [1, 4, 16, 64])
    for fila in fano["tabla"]:
        assert abs(fila["fano"] - 1.0) < 0.05, \
            f"F(T={fila['T']}) = {fila['fano']:.3f}, esperado 1"
    assert abs(fano["pendiente_log"]) < 0.05


# --- propiedades estructurales ----------------------------------------------

def test_funcion_de_simetria_es_impar():
    """Cambiar Q → −Q invierte s exactamente (misma partición de bins)."""
    rng = np.random.default_rng(10)
    Q = rng.normal(0.3, 1.0, 50_000)
    q1, s1, _ = symmetry_function(Q)
    q2, s2, _ = symmetry_function(-Q)
    np.testing.assert_allclose(q1, q2)
    np.testing.assert_allclose(s1, -s2, rtol=1e-12)


def test_net_charge_no_solapa_y_trunca():
    Q = net_charge(np.arange(10, dtype=float), 3)
    np.testing.assert_allclose(Q, [0 + 1 + 2, 3 + 4 + 5, 6 + 7 + 8])


def test_degenerados_no_revientan():
    assert len(net_charge(np.empty(0), 4)) == 0
    q, s, se = symmetry_function(np.zeros(1000))
    assert len(q) == 0
    fit = fit_affinity(np.empty(0), np.empty(0), np.empty(0))
    assert np.isnan(fit["A"])
    esc = cumulant_scaling(np.random.default_rng(0).normal(size=50), [64, 128])
    assert np.isnan(esc["pendiente_k2"])


def test_sintetico_tiene_media_y_acf_correctas():
    """La media de UNA realización con memoria gamma=-0.5 fluctúa como
    n^(-1/4) (~0.05 con n=2e5) — es el mismo fenómeno que hace decaer A(T),
    así que se comprueba el promedio SOBRE SEMILLAS, no una sola."""
    medias = [synthetic_flow(200_000, gamma=-0.5, mu=0.1, seed=s).mean()
              for s in range(12)]
    assert abs(np.mean(medias) - 0.1) < 0.05, \
        f"media ensemble {np.mean(medias):.3f} vs 0.1"

    x = synthetic_flow(200_000, gamma=-0.5, mu=0.1, seed=11)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    lags = np.array([1, 2, 4, 8, 16, 32])
    acf = np.array([np.dot(xc[:-k], xc[k:]) / denom for k in lags])
    H = 0.75
    objetivo = 0.5 * (
        (lags + 1.0) ** (2.0 * H)
        - 2.0 * lags ** (2.0 * H)
        + (lags - 1.0) ** (2.0 * H)
    )
    assert np.max(np.abs(acf - objetivo)) < 0.06, \
        "Davies--Harte no reproduce la covarianza exacta del fGn"
