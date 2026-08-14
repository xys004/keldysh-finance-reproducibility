# -*- coding: utf-8 -*-
"""Genera las figuras 1-9 del paper desde los logs depositados.

REGLA DEL REPO: ninguna cifra sin log. Cada figura declara su fuente. Todo sale
de `output/exp0{6,7,8,9}*.json` (keldysh-finance) y de los fingerprints C1/C2 de
QTEOM (`examples/market_quench_output/*.json`). Si una figura contradice un log,
manda el log.

    py experiments/figuras_paper.py [ruta_qteom]

Salida: output/figuras/figNN_*.png. Cada figura se genera dentro de un try/except
propio: un fallo aislado no tira las demás.

Mapa (esqueleto_paper.md):
    Fig 1: C_eps bicomponente, R(tau), T_eff(tau)      [exp09]
    Fig 2: s(Q) por T (simetria GC)                      [exp06]
    Fig 3: A(T) medida vs A_gauss=2<Q>/Var               [exp06]
    Fig 4: Fano(T)                                        [exp06]
    Fig 5: m(t_w) Omori + tau_c(t_w) obs vs nulo          [exp07]
    Fig 6: perfiles de fase Floquet (sesgo/A1/R1)         [exp08]
    Fig 7: I(t) del quench del dispositivo                [QTEOM C1]
    Fig 8: Var(Q_T) dispositivo vs pendiente de mercado   [QTEOM C2]
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
FIGS = os.path.join(OUT, "figuras")
os.makedirs(FIGS, exist_ok=True)

# Ruta al repo de QTEOM (fingerprints de las vias C1/C2). Configurable por
# argumento; si no existe, las figs 7-8 se saltan con aviso.
QTEOM_DEFAULT = os.path.normpath(os.path.join(
    REPO, "..", "..", "quantum", "QuantumTransportEOM",
    "examples", "market_quench_output"))
QTEOM = sys.argv[1] if len(sys.argv) > 1 else QTEOM_DEFAULT

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
COL = {"BTC": "#e8833a", "ETH": "#3a6ee8", "BNB": "#d9b310", "SOL": "#12a594"}


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def load_qteom(name):
    with open(os.path.join(QTEOM, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    p = os.path.join(FIGS, stem + ".png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(p, REPO)}")


def series_1h(items):
    """Filtra los items cuya clave termina en |1h, en orden BTC,ETH,BNB,SOL."""
    orden = {"BTC": 0, "ETH": 1, "BNB": 2, "SOL": 3}
    got = [it for it in items if it["clave"].endswith("|1h")]
    return sorted(got, key=lambda it: orden.get(it["clave"].split("|")[0], 9))


# ---------------------------------------------------------------- Fig 1
def fig1():
    """C_eps bicomponente, R(tau), T_eff(tau) -- exp09."""
    d = load("exp09_msrjd_orden2.json")
    items = series_1h(d["forma_teff"])
    dr = d["dos_regimenes"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    # (a) R(tau) ~ plana
    for it in items:
        a = it["clave"].split("|")[0]
        ax[0].plot(it["R_curva"]["tau"], it["R_curva"]["R"], color=COL[a], lw=1.4, label=a)
    ax[0].axhline(0, color="0.6", lw=0.8, ls=":")
    ax[0].set_xlabel(r"$\tau$ (velas)"); ax[0].set_ylabel(r"$R(\tau)$ impacto")
    ax[0].set_title("(a) respuesta plana"); ax[0].legend(fontsize=8, ncol=2)

    # (b) T_eff(tau)
    for it in items:
        a = it["clave"].split("|")[0]
        ax[1].plot(it["teff_curva"]["tau"], it["teff_curva"]["teff"], color=COL[a], lw=1.4)
    ax[1].set_xlabel(r"$\tau$ (velas)"); ax[1].set_ylabel(r"$T_{\rm eff}(\tau)=-C'_\varepsilon/R$")
    ax[1].set_title("(b) temperatura efectiva")

    # (c) pendientes bicomponentes de C_eps: rapida [2,10] vs lenta [10,50]
    #     crudo vs desestacionalizado (el drive no las mueve).
    assets = ["BTC", "ETH", "BNB", "SOL"]
    x = np.arange(len(assets))
    for j, (win, mk, lab) in enumerate([("[2,10]", "o", r"$\tau\in[2,10]$ (rapida)"),
                                        ("[10,50]", "s", r"$\tau\in[10,50]$ (lenta)")]):
        yc = [dr[a]["crudo"][win]["pendiente"] for a in assets]
        ec = [dr[a]["crudo"][win]["se"] for a in assets]
        yd = [dr[a]["desestacionalizado"][win]["pendiente"] for a in assets]
        ax[2].errorbar(x - 0.09, yc, yerr=ec, fmt=mk, color="C3" if j else "C0",
                       capsize=3, label=lab)
        ax[2].errorbar(x + 0.09, yd, yerr=ec, fmt=mk, mfc="white",
                       color="C3" if j else "C0", capsize=3)
    ax[2].axhline(-0.9, color="C0", ls=":", lw=0.8)
    ax[2].axhline(-0.2, color="C3", ls=":", lw=0.8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(assets)
    ax[2].set_ylabel(r"pendiente local de $C_\varepsilon$")
    ax[2].set_title("(c) $C_\\varepsilon$ bicomponente\n(relleno=crudo, hueco=desestacionalizado)")
    ax[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Fig. 1 — Respuesta, temperatura efectiva y correlacion bicomponente [exp09]", fontsize=11)
    save(fig, "fig1_ceps_R_Teff")


# ---------------------------------------------------------------- Fig 2
def fig2():
    """s(Q)=ln[P(Q)/P(-Q)] por T -- simetria GC lineal (exp06)."""
    d = load("exp06_conteo_flujo.json")
    res = {r["clave"]: r for r in d["resultados"]}
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for k, clave in enumerate(["BTC|1h", "ETH|1h"]):
        r = res[clave]; a = clave.split("|")[0]
        sim = r["afinidad"]["simetria"]
        for T, mk in zip(sorted(sim, key=int), ["o", "s", "^"]):
            q = np.array(sim[T]["q"]); s = np.array(sim[T]["s"])
            ax[k].plot(q, s, mk, ms=5, label=f"T={T}")
            # recta A*Q anclada por minimos cuadrados sin ordenada
            A = np.sum(q * s) / np.sum(q * q)
            qq = np.linspace(0, q.max(), 20)
            ax[k].plot(qq, A * qq, "-", lw=1, color="0.5", alpha=0.7)
        ax[k].axhline(0, color="0.7", lw=0.8, ls=":")
        ax[k].set_xlabel(r"$Q$ (flujo firmado acumulado, norm.)")
        ax[k].set_ylabel(r"$s(Q)=\ln[P(Q)/P(-Q)]$")
        ax[k].set_title(f"({'ab'[k]}) {clave}"); ax[k].legend(fontsize=8)
    fig.suptitle("Fig. 2 — Simetria de fluctuacion del conteo [exp06]", fontsize=11)
    save(fig, "fig2_sQ_por_T")


# ---------------------------------------------------------------- Fig 3
def fig3():
    """A(T) medida vs A_gauss=2<Q>/Var(Q) -- el conteo es gaussiano 2o orden (exp06)."""
    d = load("exp06_conteo_flujo.json")
    res = {r["clave"]: r for r in d["resultados"]}
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for k, clave in enumerate(["BTC|1h", "ETH|1h"]):
        r = res[clave]; a = clave.split("|")[0]
        pt = r["afinidad"]["por_T"]
        Ts = sorted(pt, key=int)
        T = np.array([int(t) for t in Ts])
        A = np.array([pt[t]["A"] for t in Ts])
        seA = np.array([pt[t]["se_A"] for t in Ts])
        Ag = np.array([pt[t]["A_gauss"] for t in Ts])
        ax[k].errorbar(T, A, yerr=seA, fmt="o", color=COL[a], capsize=3, label=r"$A$ medida")
        ax[k].plot(T, Ag, "s--", color="0.35", mfc="white", label=r"$A_{\rm gauss}=2\langle Q\rangle/{\rm Var}$")
        ax[k].set_xscale("log", base=2)
        ax[k].set_xlabel("T (velas)"); ax[k].set_title(f"({'ab'[k]}) {clave}")
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel("afinidad A")
    fig.suptitle("Fig. 3 — La afinidad medida coincide con la gaussiana de 2o orden [exp06]", fontsize=11)
    save(fig, "fig3_A_vs_Agauss")


# ---------------------------------------------------------------- Fig 4
def fig4():
    """Fano(T) -- bunching fuera del alcance fermionico libre (exp06)."""
    d = load("exp06_conteo_flujo.json")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for it in series_1h(d["resultados"]):
        a = it["clave"].split("|")[0]
        tab = it["fano"]["tabla"]
        T = np.array([row["T"] for row in tab])
        F = np.array([row["fano"] for row in tab])
        pend = it["fano"]["pendiente_log"]
        ax.plot(T, F, "o-", color=COL[a], lw=1.3, label=f"{a}  (pend log {pend:.2f})")
    ax.axhline(1, color="0.5", ls="--", lw=1, label="Fano=1 (Poisson)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("T (velas)"); ax.set_ylabel("Fano  Var(Q)/|<Q>|")
    ax.set_title("Fig. 4 — Fano ~ $10^3$ creciendo con T (memoria/splitting) [exp06]")
    ax.legend(fontsize=8)
    save(fig, "fig4_fano")


# ---------------------------------------------------------------- Fig 5
def fig5():
    """m(t_w) Omori y tau_c(t_w) obs vs nulo -- la media envejece, la correlacion no (exp07)."""
    d = load("exp07_reloj_quench.json")
    r = next(it for it in d["resultados"] if it["clave"] == "POOL|1h")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))

    # (a) media m(t_w): relajacion Omori
    m = np.array(r["m"], dtype=float)
    tw = np.arange(1, len(m) + 1)
    ax[0].plot(tw, m, ".", ms=3, color="#3a6ee8", alpha=0.6)
    p = r["omori"]["p"]; minf = r["omori"]["m_inf"]
    ax[0].axhline(minf, color="0.5", ls="--", lw=1, label=fr"$m_\infty$; Omori $p={p:.2f}$")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$t_w$ = velas desde el shock"); ax[0].set_ylabel(r"$m(t_w)$ (media post-shock)")
    ax[0].set_title("(a) la MEDIA envejece"); ax[0].legend(fontsize=8)

    # (b) tau_c(t_w) obs + IC, y el nulo (Spearman) que lo desmonta
    tau = np.array(r["tau_c"], dtype=float)
    ic = np.array(r["tau_c_ic"], dtype=float)  # [ [lo,hi], ... ]
    bordes = r["bordes_tw"]
    xmid = [0.5 * (bordes[i] + bordes[i + 1]) for i in range(len(tau))]
    yerr = np.abs(np.vstack([tau - ic[:, 0], ic[:, 1] - tau]))
    ax[1].errorbar(xmid, tau, yerr=yerr, fmt="o-", color="#12a594", capsize=3,
                   label=r"$\tau_c(t_w)$ obs (Spearman $+1.0$)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$t_w$ (bin)"); ax[1].set_ylabel(r"$\tau_c(t_w)$")
    ax[1].set_title("(b) la CORRELACION no: el nulo da lo mismo")
    # inset: distribucion nula de Spearman (shocks al azar) -> mediana ~ +1
    nul = np.array(r["nulo_spearman"], dtype=float)
    iax = ax[1].inset_axes([0.55, 0.12, 0.42, 0.42])
    iax.hist(nul, bins=12, color="0.6", edgecolor="white")
    iax.axvline(np.median(nul), color="C3", lw=1.5)
    iax.set_title(f"nulo Spearman\n(mediana {np.median(nul):.2f})", fontsize=7)
    iax.tick_params(labelsize=6); iax.grid(False)
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 5 — Reloj del quench: envejece la media, no la correlacion [exp07]", fontsize=11)
    save(fig, "fig5_reloj_quench")


# ---------------------------------------------------------------- Fig 6
def fig6():
    """Perfiles de fase Floquet: sesgo, afinidad A1, impacto R1 (exp08)."""
    d = load("exp08_floquet.json")
    r = next(it for it in d["resultados"] if it["clave"] == "BTC|1h")
    fases = r["fases"]
    horas = [f["horas"] for f in fases]
    x = np.arange(len(fases))
    sesgo = np.array([f["sesgo"] for f in fases]); se_s = np.array([f["se_sesgo"] for f in fases])
    A1 = np.array([f["A1"] for f in fases]); se_A = np.array([f["se_A1"] for f in fases])
    R1 = np.array([f["R1"] for f in fases])
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax[0].errorbar(x, sesgo, yerr=se_s, fmt="o-", color="#e8833a", capsize=3)
    ax[0].axhline(0, color="0.7", lw=0.8, ls=":")
    ax[0].set_ylabel("sesgo (flujo neto)"); ax[0].set_title("(a) desbalance vendedor por fase")
    ax[1].errorbar(x, A1, yerr=se_A, fmt="s-", color="#3a6ee8", capsize=3)
    ax[1].set_ylabel(r"afinidad $A_1$"); ax[1].set_title("(b) afinidad por fase")
    ax[2].plot(x, R1, "^-", color="#12a594")
    ax[2].set_ylabel(r"impacto $R_1$"); ax[2].set_title("(c) impacto por fase")
    ax[2].set_xticks(x); ax[2].set_xticklabels(horas, rotation=45, ha="right")
    ax[2].set_xlabel("fase diaria (UTC)")
    fig.suptitle("Fig. 6 — El conductor forzado: transporte modulado por el ciclo diario [exp08, BTC|1h]", fontsize=11)
    save(fig, "fig6_floquet_perfiles")


# ---------------------------------------------------------------- Fig 7
def fig7():
    """I(t) del quench del dispositivo NEGF -- relajacion en dos regimenes (QTEOM C1)."""
    d = load_qteom("c1_quench_fingerprint.json")
    g = d["grid"]; dt = g["dt"]; t_eq = g["t_eq"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) corriente completa: eje de tiempo reconstruido, quench en t=0
    for name, col, lab in [("two_band", "#3a6ee8", "banda bicomponente"),
                           ("wide", "#e8833a", "solo Lorentziana ancha")]:
        sc = d["scenarios"].get(name)
        if not sc:
            continue
        cur = np.array(sc["current"], dtype=float)
        t = np.arange(len(cur)) * dt - t_eq
        ax[0].plot(t, cur, lw=1.2, color=col, label=lab)
    tb = d["scenarios"]["two_band"]["relaxation"]
    ax[0].axhline(tb["i_inf"], color="0.5", ls="--", lw=1, label=r"$I_\infty$")
    ax[0].axvline(0, color="0.6", ls=":", lw=1)
    ax[0].set_xlabel("t (quench en 0)"); ax[0].set_ylabel("I(t) corriente")
    ax[0].set_title("(a) corriente del quench"); ax[0].legend(fontsize=8)

    # (b) relajacion |I-I_inf| loglog: los dos regimenes (rapido/lento)
    t = np.array(tb["t"], dtype=float); delta = np.abs(np.array(tb["delta"], dtype=float))
    ax[1].loglog(t, delta, ".", ms=3, color="#3a6ee8", alpha=0.5)
    for key, col, lab in [("loglog_slope_fast", "C0", "rapido"),
                          ("loglog_slope_slow", "C3", "lento")]:
        sl = tb[key]; s = sl["value"]; w = sl["window"]
        mask = (t >= w[0]) & (t <= w[1])
        if mask.any():
            t0 = t[mask][len(t[mask]) // 2]; d0 = delta[mask][len(t[mask]) // 2]
            tt = np.array(w)
            ax[1].loglog(tt, d0 * (tt / t0) ** s, "-", color=col, lw=2,
                         label=fr"{lab}: pend {s:.2f}$\pm${sl['se']:.2f}")
    ax[1].set_xlabel("t post-quench"); ax[1].set_ylabel(r"$|I(t)-I_\infty|$")
    ax[1].set_title(f"(b) relajacion en dos regimenes (Omori {tb['loglog_slope_omori']['value']:.2f})")
    ax[1].legend(fontsize=8)
    fig.suptitle("Fig. 7 — Quench de sesgo del dispositivo minimo: la banda lenta parte la relajacion [QTEOM C1]", fontsize=11)
    save(fig, "fig7_quench_current")


# ---------------------------------------------------------------- Fig 8
def fig8():
    """Var(Q_T) del dispositivo vs pendiente de mercado [1.3,1.43] (QTEOM C2)."""
    d = load_qteom("c2_counting_fingerprint.json")
    mref = d["market_reference"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sc = d["scenarios"]["two_band"]["windowed"]
    T = np.array(sc["T_obs"], dtype=float)
    v = np.array(sc["var_q"], dtype=float)
    ax.loglog(T, v, "o-", color="#3a6ee8", lw=1.3, label="Var$(Q_T)$ dispositivo (two_band)")
    # banda de pendiente de mercado k2: Var ~ T^s, s in [1.3,1.43]
    s_lo, s_hi = mref["k2_slope_range"]
    T0, v0 = T[len(T) // 3], v[len(T) // 3]
    for s, ls in [(s_lo, ":"), (s_hi, "--")]:
        ax.loglog(T, v0 * (T / T0) ** s, ls, color="0.45", lw=1,
                  label=fr"mercado $T^{{{s}}}$")
    ax.set_xlabel(r"$T_{\rm obs}$"); ax.set_ylabel(r"Var$(Q_T)$")
    ax.set_title("Fig. 8 — Ruido de conteo: dispositivo vs mercado\n"
                 f"(Fano mercado T=1 en {mref['fano_range_T1']}) [QTEOM C2]")
    ax.legend(fontsize=8)
    save(fig, "fig8_varQ_dispositivo_vs_mercado")


# ---------------------------------------------------------------------------- Fig 9
def fig9():
    """Crossover del exponente de Var(Q_T): la prediccion de la derivacion.

    Para C_eps ~ tau^-a se tiene Var(Q_T) ~ T^(2-a), asi que un kernel
    bicomponente obliga a que la pendiente LOCAL derive de 2-a_f a 2-a_s.
    Los exponentes a_f, a_s se miden en el sector de correlacion (exp09) y la
    pendiente local en el de conteo (exp06): el test cruza familias.
    """
    d6 = load("exp06_conteo_flujo.json")
    d9 = load("exp09_msrjd_orden2.json")
    dr = d9["dos_regimenes"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) pendiente local de log Var vs log T, por serie 1h
    for it in series_1h(d6["resultados"]):
        a = it["clave"].split("|")[0]
        tab = {row["T"]: row["k2"] for row in it["pendiente_k2_full"]["tabla"]}
        Ts = sorted(tab)
        xm, sl = [], []
        for t1, t2 in zip(Ts[:-1], Ts[1:]):
            sl.append((np.log(tab[t2]) - np.log(tab[t1])) / (np.log(t2) - np.log(t1)))
            xm.append(np.sqrt(t1 * t2))
        ax[0].semilogx(xm, sl, "o-", color=COL[a], lw=1.3, label=a)
    # bandas predichas 2-a, con a de la correlacion (BTC como referencia)
    af = -dr["BTC"]["crudo"]["[2,10]"]["pendiente"]
    as_ = -dr["BTC"]["crudo"]["[10,50]"]["pendiente"]
    ax[0].axhline(2 - af, color="C0", ls="--", lw=1.2, label=fr"$2-a_f={2-af:.2f}$ (rapido)")
    ax[0].axhline(2 - as_, color="C3", ls="--", lw=1.2, label=fr"$2-a_s={2-as_:.2f}$ (lento)")
    ax[0].axhline(1.0, color="0.7", ls=":", lw=1)
    ax[0].set_xlabel("T (velas)"); ax[0].set_ylabel(r"pendiente local de $\log\,$Var$(Q_T)$")
    ax[0].set_title("(a) el exponente DERIVA: no es potencia pura")
    ax[0].legend(fontsize=7, ncol=2)

    # (b) test exacto de la dilucion de la afinidad: A16/A1 = 16*V1/V16
    claves, pred, obs = [], [], []
    for it in d6["resultados"]:
        tab = {row["T"]: row["k2"] for row in it["pendiente_k2_full"]["tabla"]}
        pt = it["afinidad"]["por_T"]
        if 1 not in tab or 16 not in tab or "1" not in pt or "16" not in pt:
            continue
        claves.append(it["clave"])
        pred.append(16.0 * tab[1] / tab[16])
        obs.append(pt["16"]["A"] / pt["1"]["A"])
    estable = [c.split("|")[0] in ("BTC", "ETH") for c in claves]
    for i, (c, p, o) in enumerate(zip(claves, pred, obs)):
        m = "o" if estable[i] else "^"
        col = COL[c.split("|")[0]]
        ax[1].plot(p, o, m, ms=9, color=col, mfc=col if estable[i] else "white")
        ax[1].annotate(c, (p, o), fontsize=6, xytext=(4, 4), textcoords="offset points")
    lim = [0.3, 1.0]
    ax[1].plot(lim, lim, "--", color="0.5", lw=1, label="prediccion = observado")
    ax[1].set_xlim(*lim); ax[1].set_ylim(*lim)
    ax[1].set_xlabel(r"prediccion $16\,V_1/V_{16}$"); ax[1].set_ylabel(r"observado $A_{16}/A_1$")
    ax[1].set_title("(b) dilucion de la afinidad\n(relleno: afinidad estable; hueco: nivel-ruido)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Fig. 9 — El conteo hereda los dos exponentes de la correlacion [exp06+exp09]", fontsize=11)
    save(fig, "fig9_crossover_exponente")


def main():
    print(f"QTEOM: {QTEOM}  ({'OK' if os.path.isdir(QTEOM) else 'NO ENCONTRADO -> figs 7-8 se saltan'})")
    figuras = [("Fig 1", fig1), ("Fig 2", fig2), ("Fig 3", fig3), ("Fig 4", fig4),
               ("Fig 5", fig5), ("Fig 6", fig6), ("Fig 7", fig7), ("Fig 8", fig8),
               ("Fig 9", fig9)]
    ok = 0
    for nombre, fn in figuras:
        try:
            print(nombre)
            fn(); ok += 1
        except FileNotFoundError as e:
            print(f"  SALTADA ({e.__class__.__name__}: {e})")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e.__class__.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/9 figuras generadas en {os.path.relpath(FIGS, REPO)}")


if __name__ == "__main__":
    main()
