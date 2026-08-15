"""
keldysh-finance — diagnósticos de no-equilibrio sobre series financieras.

Programa: importar la ESTRUCTURA A DOS TIEMPOS del formalismo de Keldysh
(ruptura de invariancia traslacional temporal, residuo tipo FDT) como
característica de RÉGIMEN, y contrastar si mejora la predicción de
volatilidad frente a EWMA y GARCH(1,1).

Lo que NO es: un generador de señales direccionales. El holdout
pre-registrado del proyecto trading-ma-montecarlo (30 pares vírgenes,
z̄=−0.45, 0/30 con p<0.05) establece que el timing técnico no tiene alfa
detectable sobre la deriva. Este paquete apunta al SEGUNDO momento, que sí
es predecible y donde el sistema existente ya tiene un hueco
(`portfolio.py` contempla vol_method='hmm' como alternativa a EWMA).
"""

__version__ = "0.1.0"

from .two_time import TwoTimeSurface, two_time_surface, volatility_field
from .stationarity import tti_breaking, correlation_time, aging_collapse
from .transient import (TransientTrajectory, fit_stretched_exp,
                        rolling_transient_fit, rolling_teff,
                        synthetic_series_with_acf)
from .counting import (block_bootstrap_standardized_cumulants,
                       cumulant_scaling, fano_factor, fit_affinity,
                       net_charge, standardized_cumulants,
                       symmetry_function, synthetic_flow)
from .fano_validation import (prepare_complete_weeks,
                              stratified_fano_memory_test,
                              validate_hourly_klines)
from .trade_validation import (aggregate_trade_hours, normalization_calibration,
                               read_daily_trades, reconcile_hourly)
from .quench import (campo_post_shock, correlacion_por_edad, detectar_shocks,
                     estadistico_envejecimiento, exponente_omori, tau_c_filas)
from .floquet import (modulacion, observables_por_fase, p_empirico,
                      rotar_horas_por_dia)
from .baselines import (ewma_vol, fit_garch11, garch_forecast_path,
                        realized_vol_forward)
from .evaluation import qlike, rmse, diebold_mariano, walk_forward_split

__all__ = [
    "TwoTimeSurface", "two_time_surface", "volatility_field",
    "tti_breaking", "correlation_time", "aging_collapse",
    "TransientTrajectory", "fit_stretched_exp", "rolling_transient_fit",
    "rolling_teff", "synthetic_series_with_acf",
    "block_bootstrap_standardized_cumulants", "cumulant_scaling",
    "fano_factor", "fit_affinity", "net_charge", "standardized_cumulants",
    "symmetry_function", "synthetic_flow",
    "prepare_complete_weeks", "stratified_fano_memory_test",
    "validate_hourly_klines",
    "aggregate_trade_hours", "normalization_calibration",
    "read_daily_trades", "reconcile_hourly",
    "campo_post_shock", "correlacion_por_edad", "detectar_shocks",
    "estadistico_envejecimiento", "exponente_omori", "tau_c_filas",
    "modulacion", "observables_por_fase", "p_empirico", "rotar_horas_por_dia",
    "ewma_vol", "fit_garch11", "garch_forecast_path", "realized_vol_forward",
    "qlike", "rmse", "diebold_mariano", "walk_forward_split",
]
