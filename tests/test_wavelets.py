"""
Tests del estimador de ondícula: recuperación de gamma conocido, inmunidad a
tendencias, y la ortogonalidad del banco de filtros.

El control positivo es el mismo que usa el exp. 10 para los otros tres
estimadores, de modo que las cuatro vías se juzgan con la misma vara.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.wavelets import (DAUBECHIES, _filtros, descomponer, fgn,
                                      gamma_abry_veitch, varianza_haar)


# --- el banco de filtros es correcto -----------------------------------------

def test_filtros_ortonormales():
    """h debe tener norma 1 y ser ortogonal a sus desplazamientos pares."""
    for L in DAUBECHIES:
        h, g = _filtros(L)
        assert abs(np.dot(h, h) - 1.0) < 1e-10, f"L={L}: ||h|| != 1"
        assert abs(np.dot(g, g) - 1.0) < 1e-10, f"L={L}: ||g|| != 1"
        assert abs(np.dot(h, g)) < 1e-12, f"L={L}: h no ortogonal a g"
        for s in range(2, L, 2):
            assert abs(np.dot(h[s:], h[:-s])) < 1e-10, f"L={L}: shift {s}"


def test_momentos_nulos():
    """Una db de longitud L aniquila polinomios de grado < L/2."""
    for L in (4, 6, 8):
        _, g = _filtros(L)
        k = np.arange(L, dtype=float)
        for potencia in range(L // 2):
            assert abs(np.dot(g, k ** potencia)) < 1e-9, \
                f"L={L} no aniquila k^{potencia}"


# --- control positivo: gamma conocido ----------------------------------------

def _recupera(gamma_true, n=400_000, seed=0, longitud=6, j_min=3):
    """Serie de gamma conocido via fGn EXACTO, donde gamma = 2 - 2H.

    NO se usa `counting.synthetic_flow`: su empotramiento circulante reproduce
    bien los desfases cortos pero distorsiona las frecuencias bajas, que son
    justo las que lee un estimador de ondicula. Validar contra el habria
    medido el sesgo del generador, no el del estimador.
    """
    x = fgn(n, H=1.0 - gamma_true / 2.0, seed=seed)
    return gamma_abry_veitch(x, longitud=longitud, j_min=j_min)


def test_recupera_gamma_conocido():
    """Sobre fGn exacto el estimador debe clavar gamma."""
    for g_true in (0.4, 0.5, 0.6, 0.7, 0.8):
        r = _recupera(g_true)
        assert abs(r["gamma"] - g_true) < 0.03, \
            f"gamma={r['gamma']:.3f} vs {g_true} (se={r['se']:.3f})"


def test_ruido_blanco_da_alpha_cero():
    """Sin memoria, el diagrama log-escala debe ser plano."""
    rng = np.random.default_rng(0)
    r = gamma_abry_veitch(rng.normal(size=200_000), longitud=6, j_min=3)
    assert abs(r["alpha"]) < 0.05, f"alpha={r['alpha']:.3f} sobre ruido blanco"


def test_error_declarado_es_realista():
    """La dispersión entre semillas no debe exceder mucho el se declarado —
    es lo que distingue este estimador de una regresion sobre la ACF, cuyo
    error nominal no significa nada."""
    vals = [_recupera(0.6, seed=s)["gamma"] for s in range(6)]
    se_declarado = _recupera(0.6, seed=0)["se"]
    disp = float(np.std(vals, ddof=1))
    assert disp < 4.0 * se_declarado + 0.03, \
        f"dispersion {disp:.3f} frente a se declarado {se_declarado:.3f}"


# --- inmunidad a tendencias: la propiedad que motiva usarlo ------------------

def test_tendencia_lineal_no_mueve_el_estimador():
    """Sumar una rampa lineal no puede alterar gamma si hay >=2 momentos nulos."""
    x = fgn(200_000, H=0.7, seed=11)
    rampa = np.linspace(0.0, 5.0 * x.std(), x.size)
    g_limpio = gamma_abry_veitch(x, longitud=6, j_min=3)["gamma"]
    g_sucio = gamma_abry_veitch(x + rampa, longitud=6, j_min=3)["gamma"]
    assert abs(g_limpio - g_sucio) < 0.02, \
        f"la rampa movio gamma de {g_limpio:.3f} a {g_sucio:.3f}"


def test_tendencia_cuadratica_necesita_tres_momentos():
    """db3 (3 momentos nulos) aguanta una parabola; Haar (1 momento) no."""
    x = fgn(200_000, H=0.7, seed=12)
    t = np.linspace(0.0, 1.0, x.size)
    par = 8.0 * x.std() * t ** 2
    g6 = gamma_abry_veitch(x, longitud=6, j_min=3)["gamma"]
    g6_sucio = gamma_abry_veitch(x + par, longitud=6, j_min=3)["gamma"]
    g2_sucio = gamma_abry_veitch(x + par, longitud=2, j_min=3)["gamma"]
    assert abs(g6 - g6_sucio) < 0.05, "db3 deberia aguantar la parabola"
    assert abs(g6 - g2_sucio) > abs(g6 - g6_sucio), \
        "Haar deberia degradarse mas que db3 ante una tendencia cuadratica"


# --- varianza de Haar --------------------------------------------------------

def test_varianza_haar_escala_como_toca():
    """Para C(tau) ~ tau^-a, la varianza de Haar escala como T^(2-a)."""
    a = 0.5
    x = fgn(400_000, H=1.0 - a / 2.0, seed=13)
    r = varianza_haar(x, escalas=[8, 16, 32, 64, 128, 256])
    assert abs(r["pendiente_global"] - (2.0 - a)) < 0.15, \
        f"pendiente {r['pendiente_global']:.3f} vs {2.0 - a}"


def test_varianza_haar_ignora_rampa():
    """La diferencia de bloques mata la tendencia lineal; la suma no."""
    x = fgn(200_000, H=0.75, seed=14)
    rampa = np.linspace(0.0, 3.0 * x.std(), x.size)
    escalas = [8, 16, 32, 64, 128]
    p_limpio = varianza_haar(x, escalas)["pendiente_global"]
    p_sucio = varianza_haar(x + rampa, escalas)["pendiente_global"]
    assert abs(p_limpio - p_sucio) < 0.05


# --- degenerados -------------------------------------------------------------

def test_serie_corta_da_error_claro():
    import pytest
    with pytest.raises(ValueError, match="corta"):
        descomponer(np.zeros(20))
