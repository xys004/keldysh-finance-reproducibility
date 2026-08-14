"""
Control positivo del exp. 09: la cadena C_ε → R → T_eff recupera la forma
en un modelo exactamente resoluble.

El modelo sintético es la versión mínima del MSRJD de la nota: flujo AR(1)
(C(τ) = φ^τ conocida) y precio INTEGRADOR DE INNOVACIONES — p sólo acumula
la sorpresa w_t del flujo, no el flujo entero. Eso produce R(τ) PLANA por
construcción (cov(p_{t+τ}−p_t, ε_t) = cov(w_t, ε_t) porque las innovaciones
futuras son ortogonales a ε_t), que es la forma medida en datos ("impacto
~plano") y la microeconomía estándar (el precio eficiente responde a la
información nueva). Con R plana y C exponencial:

    T_eff(τ) = −[dC/dτ]/R ∝ φ^τ   ⇒   pendiente log-lineal = ln φ.

Si el estimador ve esta forma con parámetros conocidos, un desacuerdo en
datos reales es del mercado, no de la cadena de estimación.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.counting import _ols_pendiente
from keldysh_finance.flow import fdt_diagnostic


def _modelo_minimo(n=300_000, phi=0.90, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, n)
    eps = np.empty(n)
    eps[0] = w[0]
    for t in range(1, n):
        eps[t] = phi * eps[t - 1] + w[t]
    # Precio PRE-flujo: p_t acumula innovaciones ANTERIORES a la vela t
    # (cumsum desplazado). Con el cumsum inclusivo, el impacto de w_t queda
    # dentro de p_t — exactamente el error del log(Close) que fija la regla 2
    # del proyecto — y la R verdadera es CERO (este test lo reprodujo antes
    # de corregirse: la "R creciente" era el paseo del ruido del estimador).
    integrado = np.concatenate(([0.0], np.cumsum(w)[:-1]))
    ruido_ind = np.concatenate(([0.0], np.cumsum(rng.normal(0, 1, n))[:-1]))
    log_open = 4.6 + 0.01 * integrado + 0.02 * ruido_ind
    return log_open, eps


def test_recupera_r_plana_y_teff_exponencial():
    phi = 0.90
    log_open, eps = _modelo_minimo(phi=phi)
    res = fdt_diagnostic(log_open, eps, max_lag=40)

    tau = res.lags.astype(float)
    m = (tau >= 2) & (tau <= 30)

    r = res.response[m]
    assert np.all(np.isfinite(r)) and np.all(r > 0)
    b_r, se_r = _ols_pendiente(np.log(tau[m]), np.log(r))
    assert abs(b_r) < 0.05, f"R deberia ser plana; pendiente {b_r:.3f}"

    # C exponencial ⇒ T_eff ∝ φ^τ es log-LINEAL en τ (semilog), no en log τ:
    # la carta del ajuste la dicta la forma del modelo. (El log-log aquí daría
    # ln(φ)·⟨τ⟩ ≈ −1.2, la firma clásica de ajustar en la carta equivocada;
    # en datos reales, con C en ley de potencias, la carta correcta es log-log.)
    te = np.abs(res.t_eff[m])
    fin = np.isfinite(te) & (te > 0)
    b_t, se_t = _ols_pendiente(tau[m][fin], np.log(te[fin]))
    assert abs(b_t - np.log(phi)) < 0.02, \
        f"pendiente semilog T_eff {b_t:.4f} vs ln(phi) = {np.log(phi):.4f}"


def test_la_violacion_fdt_refleja_la_no_constancia():
    """Con T_eff ∝ φ^τ la dispersión relativa es grande y positiva — fija
    que el escalar 'violación' del exp. 02 es la sombra de esta FORMA."""
    log_open, eps = _modelo_minimo(phi=0.90, seed=1)
    res = fdt_diagnostic(log_open, eps, max_lag=40)
    assert np.isfinite(res.fdt_violation) and res.fdt_violation > 0.5
