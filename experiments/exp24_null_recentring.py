r"""Experiment 24: align the exact decomposition with the operational null.

With mbar per stratum (quarter x hour-of-week), writing
u_t = s_t * mbar_{h(t)}:

    D = < Sum_t m_t^2 >                          diagonal
    S = < (Sum_t u_t)^2 - Sum_t u_t^2 >           signo
    K = < Q_w^2 > - D - S                         acoplamiento

Antes de recentrar cada surrogate, la independencia entre estratos da

    D   = E[nulo conjunto]
    D+S = E[nulo de magnitudes]

El nulo operacional recentra dentro de estrato. En general, el nulo conjunto
adquiere una correccion igual a menos la media semanal ponderada de las medias
estratificadas al cuadrado. En este analisis desaparece por construccion,
porque el residual ya fue centrado en exactamente los mismos estratos. En el
nulo de magnitudes aparece una correccion C_rec que se calcula aqui
exactamente a partir de los momentos de una permutacion finita:

    E[nulo de magnitudes recentrado] = D + S + C_rec.

Este experimento verifica las expectativas analiticas numericamente sobre los
ocho activos, separa el termino
estacional que la mbar GLOBAL le regalaba al acoplamiento, rehace las
razones R con la MEDIA del nulo en vez de la MEDIANA, y da IC bootstrap
PAREADOS (mismo remuestreo de semanas para D, S y K a la vez).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

for _v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from keldysh_finance.fano_validation import (
    _stratum_period,
    expected_recentered_joint_second_moment,
    expected_recentered_magnitude_second_moment,
    prepare_complete_weeks,
)

ACTIVOS = ["BTC", "ETH", "BNB", "SOL"]
T = 168
N_NULO = 999
N_BOOT = 3999
BLOQUES_BOOT = (4, 8, 13, 26)


def _cargar(symbol: str) -> pd.DataFrame:
    p = RAIZ / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    df = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.dropna()


def mbar_por_estrato(m: np.ndarray, estrato: np.ndarray) -> np.ndarray:
    """mbar_{h(t)}: la media de m dentro del estrato de t, para cada t."""
    df = pd.DataFrame({"m": m, "h": estrato})
    return df.groupby("h")["m"].transform("mean").to_numpy()


def dsk_por_ventana(eps: np.ndarray, mbar_h_t: np.ndarray) -> dict:
    """D, S, K por ventana de T velas, con u_t = s_t * mbar_h(t)."""
    s, m = np.sign(eps), np.abs(eps)
    u = s * mbar_h_t
    n = eps.size // T
    M = m[:n * T].reshape(n, T)
    U = u[:n * T].reshape(n, T)
    Q = eps[:n * T].reshape(n, T).sum(axis=1)
    D = (M ** 2).sum(axis=1)
    S = U.sum(axis=1) ** 2 - (U ** 2).sum(axis=1)
    K = Q ** 2 - D - S
    return {"D": D, "S": S, "K": K, "Q2": Q ** 2}


def termino_estacional(eps: np.ndarray, mbar_h_t: np.ndarray,
                       mbar_global: float) -> np.ndarray:
    """<Sum_{t!=t'} s_t s_t' (mbar_h mbar_h' - mbar_global^2)> por ventana.

    Lo que la mbar GLOBAL le regalaba al acoplamiento: persistencia de
    signos multiplicada por la modulacion ESTACIONAL del tamano, que el
    nulo YA conserva (permuta dentro de estrato) y por tanto no deberia
    contar como acoplamiento genuino.
    """
    s = np.sign(eps)
    n = eps.size // T
    S_ = s[:n * T].reshape(n, T)
    MB = mbar_h_t[:n * T].reshape(n, T)
    u_g = S_ * MB  # mismo u_t, pero el termino se aisla por resta:
    signo_estrato = u_g.sum(axis=1) ** 2 - (u_g ** 2).sum(axis=1)
    signo_global = (mbar_global ** 2) * (S_.sum(axis=1) ** 2 - (S_ ** 2).sum(axis=1))
    return signo_global - signo_estrato  # = termino estacional que se resta


def nulo_conjunto_y_magnitud(eps: np.ndarray, estrato: np.ndarray,
                             rng: np.random.Generator, n_perm: int) -> dict:
    """Permutaciones DENTRO de estrato: conjunta (permuta eps) y de
    magnitudes (permuta m, deja s fijo) -- la maquinaria de fano_validation,
    reescrita aqui para devolver Q^2 por ventana en cada permutacion."""
    n_tot = eps.size
    base = np.lexsort((np.arange(n_tot), estrato))
    _, starts, counts = np.unique(
        estrato[base], return_index=True, return_counts=True
    )
    s, m = np.sign(eps), np.abs(eps)
    Q2_conj, Q2_mag = [], []
    for _ in range(n_perm):
        orden = np.lexsort((rng.random(n_tot), estrato))
        eps_c = np.empty_like(eps)
        eps_c[base] = eps[orden]
        m_p = np.empty_like(m)
        m_p[base] = m[orden]
        eps_m = s * m_p
        # Mismo recentrado estratificado que usa el nulo principal.
        for arr in (eps_c, eps_m):
            ordered = arr[base]
            means = np.repeat(np.add.reduceat(ordered, starts) / counts, counts)
            arr[base] = ordered - means
        n = n_tot // T
        Q2_conj.append((eps_c[:n * T].reshape(n, T).sum(axis=1) ** 2).mean())
        Q2_mag.append((eps_m[:n * T].reshape(n, T).sum(axis=1) ** 2).mean())
    return {"conjunto": np.array(Q2_conj), "magnitud": np.array(Q2_mag)}


def boot_pareado(D: np.ndarray, S: np.ndarray, K: np.ndarray,
                 rng: np.random.Generator) -> dict:
    """Bootstrap de bloques circulares PAREADO: el MISMO remuestreo de
    semanas para D, S y K a la vez, así el IC de K/(S+K) es correcto."""
    n = D.size
    por_bloque = {}
    for L in BLOQUES_BOOT:
        if n < 2 * L:
            continue
        valores = {"D": [], "S": [], "K": [], "K_sobre_SK": []}
        n_bl = int(np.ceil(n / L))
        for _ in range(N_BOOT):
            ini = rng.integers(0, n, n_bl)
            off = np.arange(L)
            idx = np.concatenate([(i + off) % n for i in ini])[:n]
            d, s_, k = D[idx].mean(), S[idx].mean(), K[idx].mean()
            valores["D"].append(d); valores["S"].append(s_); valores["K"].append(k)
            exc = s_ + k
            valores["K_sobre_SK"].append(k / exc if abs(exc) > 1e-12 else np.nan)
        resumen = {}
        for k_, v in valores.items():
            v = np.array(v)
            v = v[np.isfinite(v)]
            resumen[k_] = {"media": float(v.mean()),
                           "ic95": [float(np.quantile(v, 0.025)),
                                   float(np.quantile(v, 0.975))]}
        por_bloque[str(L)] = resumen
    return {
        "replicates_per_block_length": N_BOOT,
        "by_block_weeks": por_bloque,
    }


def main() -> None:
    print("=" * 88)
    print("  EXP 24 - null-aligned decomposition and paired intervals")
    print("=" * 88)

    rng = np.random.default_rng(24)
    resultados = []
    for modo in ("raw", "normalized"):
      for a in ACTIVOS:
        prep = prepare_complete_weeks(_cargar(f"{a}USDT"), flow_mode=modo)
        eps = prep.flow_demeaned
        periodo = _stratum_period(prep, "quarterly")
        estrato = periodo.astype(np.int64) * 168 + prep.hour_of_week.astype(np.int64)

        m = np.abs(eps)
        mbar_h_t = mbar_por_estrato(m, estrato)
        mbar_global = float(m.mean())

        dsk = dsk_por_ventana(eps, mbar_h_t)
        D, S, K = dsk["D"], dsk["S"], dsk["K"]

        # --- expectativa exacta del nulo operacional ----------------------
        q2_conj_exacto = expected_recentered_joint_second_moment(
            eps, estrato, window=T
        )
        q2_mag_exacto = expected_recentered_magnitude_second_moment(
            eps, estrato, window=T
        )
        C_conj = q2_conj_exacto - D
        C_rec = q2_mag_exacto - D - S
        S_rec = S + C_rec
        K_rec = K - C_rec

        # --- verificacion Monte Carlo de las expectativas analiticas -------
        nulos = nulo_conjunto_y_magnitud(eps, estrato, rng, N_NULO)
        D_media = D.mean()
        E_conj_exacto = q2_conj_exacto.mean()
        DS_media = (D + S).mean()
        DS_rec_media = (D + S_rec).mean()
        E_conj, E_mag = nulos["conjunto"].mean(), nulos["magnitud"].mean()
        err_conj = abs(E_conj_exacto - E_conj) / abs(E_conj)
        err_mag = abs(DS_rec_media - E_mag) / abs(E_mag)
        se_conj = float(nulos["conjunto"].std(ddof=1) / np.sqrt(N_NULO))
        se_mag = float(nulos["magnitud"].std(ddof=1) / np.sqrt(N_NULO))

        # --- termino estacional que la mbar global regalaba al acoplo ----
        est = termino_estacional(eps, mbar_h_t, mbar_global).mean()

        # --- razones R con MEDIA del nulo (no mediana) --------------------
        Q2_media = dsk["Q2"].mean()
        R_conj_media = Q2_media / E_conj_exacto
        RM_sobre_RJ_media = DS_rec_media / E_conj_exacto
        R_conj_mediana = Q2_media / np.median(nulos["conjunto"])
        RM_sobre_RJ_mediana = DS_rec_media / np.median(nulos["conjunto"])

        # --- bootstrap pareado ---------------------------------------------
        boot = boot_pareado(D, S_rec, K_rec, rng)
        boot8 = boot["by_block_weeks"]["8"]

        cuota_KSK = K_rec.mean() / (S_rec.mean() + K_rec.mean())
        medias_estrato = pd.Series(eps).groupby(estrato).mean().to_numpy(float)
        resultados.append({
            "activo": a, "flow_mode": modo,
            "D": float(D.mean()), "S": float(S_rec.mean()),
            "K": float(K_rec.mean()), "Q2": float(Q2_media),
            "S_sin_recentrado": float(S.mean()),
            "K_sin_recentrado": float(K.mean()),
            "C_rec": float(C_rec.mean()),
            "C_conjunto": float(C_conj.mean()),
            "max_abs_media_estrato": float(np.max(np.abs(medias_estrato))),
            "cuota_K_sobre_SK": float(cuota_KSK),
            "boot": boot,
            "verificacion": {
                "D_medido": float(D_media),
                "E_nulo_conjunto_analitico": float(E_conj_exacto),
                "E_nulo_conjunto_mc": float(E_conj),
                "error_relativo_D": float(err_conj),
                "DS_pre_recentrado": float(DS_media),
                "DS_recentrado_analitico": float(DS_rec_media),
                "E_nulo_magnitud": float(E_mag),
                "error_relativo_DS": float(err_mag),
                "se_mc_conjunto": se_conj, "se_mc_magnitud": se_mag,
                "z_mc_conjunto": float((E_conj - E_conj_exacto) / se_conj),
                "z_mc_magnitud": float((E_mag - DS_rec_media) / se_mag)},
            "termino_estacional_mbar_global": float(est),
            "termino_estacional_sobre_K": float(est / K.mean()) if K.mean() else None,
            "ratios_media": {
                "observado_sobre_nulo_conjunto": float(R_conj_media),
                "nulo_magnitud_sobre_nulo_conjunto": float(RM_sobre_RJ_media),
            },
            "ratios_mediana_conjunta": {
                "observado_sobre_nulo_conjunto": float(R_conj_mediana),
                "nulo_magnitud_sobre_nulo_conjunto": float(RM_sobre_RJ_mediana),
            },
        })

        print(f"\n  {a} ({modo})")
        print(f"    D={D.mean():.4e}  S_rec={S_rec.mean():.4e}  K_rec={K_rec.mean():.4e}")
        print(f"    C_rec={C_rec.mean():+.4e}  ({100*C_rec.mean()/Q2_media:+.3f}% de Q2)")
        print(f"    C_conjunto={C_conj.mean():+.4e}; "
              f"max |media de estrato|={np.max(np.abs(medias_estrato)):.4e}")
        print(f"    VERIFICACION DEL NULO OPERACIONAL:")
        print(f"      E[nulo conjunto] analitico={E_conj_exacto:.4e}  vs  MC={E_conj:.4e}"
              f"   error={100*err_conj:.2f}%  z_MC={(E_conj-E_conj_exacto)/se_conj:+.2f}")
        print(f"      D+S_rec={DS_rec_media:.4e}  vs  E[nulo magnitud]={E_mag:.4e}"
              f"   error={100*err_mag:.2f}%  z_MC={(E_mag-DS_rec_media)/se_mag:+.2f}")
        print(f"    termino estacional (lo que mbar global asignaba a K0): "
              f"{est:.4e}  ({100*est/K.mean():.1f}% de K0)")
        print(f"    cuota K_rec/(S_rec+K_rec) = {cuota_KSK:.3f}   "
              f"IC95 pareado (bloques 8 semanas) = "
              f"[{boot8['K_sobre_SK']['ic95'][0]:.3f}, "
              f"{boot8['K_sobre_SK']['ic95'][1]:.3f}]")
        print(f"    observado/nulo conjunto (media)={R_conj_media:.3f}  "
              f"nulo magnitud/nulo conjunto={RM_sobre_RJ_media:.3f}")
        print(f"    observado/nulo conjunto (mediana)={R_conj_mediana:.3f}  "
              f"nulo magnitud/nulo conjunto={RM_sobre_RJ_mediana:.3f}")

    print("\n\n  " + "=" * 84)
    print("  RESUMEN")
    print("  " + "-" * 84)
    print(f"  {'activo':<14}{'D%':>7}{'S%':>7}{'K%':>7}{'K/(S+K)':>10}"
          f"{'IC95':>20}{'err D':>8}{'err D+S':>9}")
    for r in resultados:
        q2 = r["Q2"]
        ic = r["boot"]["by_block_weeks"]["8"]["K_sobre_SK"]["ic95"]
        etiqueta = r["activo"] + "|" + r["flow_mode"]
        print(f"  {etiqueta:<14}{100*r['D']/q2:>7.1f}{100*r['S']/q2:>7.1f}"
              f"{100*r['K']/q2:>7.1f}{r['cuota_K_sobre_SK']:>10.3f}"
              f"  [{ic[0]:.3f},{ic[1]:.3f}]"
              f"{100*r['verificacion']['error_relativo_D']:>7.2f}%"
              f"{100*r['verificacion']['error_relativo_DS']:>8.2f}%")

    max_err = max(max(r["verificacion"]["error_relativo_D"],
                      r["verificacion"]["error_relativo_DS"]) for r in resultados)
    print(f"\n  Maximo error relativo entre expectativas analiticas y nulos: "
          f"{100*max_err:.2f}%")
    print("  (el nulo tiene ruido Monte Carlo de %d permutaciones; un error"
          % N_NULO)
    print("  de ese orden confirma la igualdad, no la refuta)")

    out = RAIZ / "output" / "exp24_null_recentring.json"
    out.write_text(json.dumps({"resultados": resultados}, indent=2,
        default=lambda o: float(o) if isinstance(o, np.floating)
        else int(o) if isinstance(o, np.integer) else str(o)), encoding="utf-8")
    print(f"\n  log: {out}")


if __name__ == "__main__":
    main()
