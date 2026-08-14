r"""
Corrección de multiplicidad Westfall-Young (maxT).

Es la pieza de la que depende el veredicto del experimento 03 y del 04, así que
conviene que esté fijada: un error aquí no da un fallo ruidoso, da un p-valor
plausible y equivocado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.pipeline import westfall_young


def _nulo(k=50, n_celdas=10, seed=0, escala=1.0):
    rng = np.random.default_rng(seed)
    return [{f"c{j}": float(rng.normal(0, escala)) for j in range(n_celdas)}
            for _ in range(k)]


def test_efecto_enorme_da_el_p_minimo_posible():
    draws = _nulo()
    obs = {f"c{j}": 0.0 for j in range(10)}
    obs["c3"] = 50.0                                  # muy por encima de cualquier nulo
    res = westfall_young(obs, draws)
    assert res["mejor_celda"] == "c3"
    assert res["p_global"] == 1.0 / (len(draws) + 1)


def test_observado_por_debajo_del_nulo_da_p_alto():
    """El caso del exp. 03: el máximo observado cae por debajo de la media de
    los máximos del nulo, así que no hay nada que reportar."""
    draws = _nulo(escala=1.0)
    obs = {f"c{j}": -1.0 for j in range(10)}
    res = westfall_young(obs, draws)
    assert res["p_global"] > 0.9
    assert res["dm_mejor"] < res["maxT_media"]


def test_maxT_es_mas_exigente_que_el_p_nominal():
    """La razón de ser de la corrección: con muchas celdas, un estadístico que
    sería 'significativo' suelto deja de serlo."""
    draws = _nulo(k=200, n_celdas=50, seed=7)
    obs = {f"c{j}": 0.0 for j in range(50)}
    obs["c10"] = 2.1                                  # p nominal ~0.036
    res = westfall_young(obs, draws)
    assert res["p_global"] > 0.05, \
        f"maxT deberia rechazar un 2.1 con 50 celdas, dio p={res['p_global']:.3f}"


def test_p_ajustado_es_monotono_en_el_estadistico():
    draws = _nulo(k=100, n_celdas=6, seed=3)
    obs = {"c0": -1.0, "c1": 0.0, "c2": 1.0, "c3": 2.0, "c4": 3.0, "c5": 4.0}
    p = westfall_young(obs, draws)["p_ajustado"]
    valores = [p[f"c{j}"] for j in range(6)]
    assert all(a >= b for a, b in zip(valores, valores[1:])), valores


def test_celda_no_finita_no_rompe_ni_gana():
    draws = _nulo(k=30, n_celdas=4, seed=5)
    obs = {"c0": np.nan, "c1": 0.5, "c2": np.nan, "c3": 1.0}
    res = westfall_young(obs, draws)
    assert res["mejor_celda"] == "c3"
    assert set(res["p_ajustado"]) == {"c1", "c3"}      # las NaN no reciben p


def test_nulo_vacio_se_reporta_como_error_y_no_como_significativo():
    res = westfall_young({"c0": 5.0}, [])
    assert "error" in res and "p_global" not in res
