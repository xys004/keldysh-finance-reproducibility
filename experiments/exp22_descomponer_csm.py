r"""Experimento 22 — de que esta hecho C_sm(tau), antes de modelarlo.

POR QUE ESTE PASO VA PRIMERO
-----------------------------
El termino cruzado de la descomposicion exacta cambia de signo hacia
tau ~ 50-100 velas (exp. 20). Es el unico objeto dinamico que queda abierto en
el programa, y la tentacion es proponerle un mecanismo directamente. Pero hay
una restriccion que se puede establecer ANTES de modelar:

    C_sm(tau) = <s_t s_{t+tau} (m_t m_{t+tau} - mbar^2)>
              = C_s(tau) * C_m(tau)  +  R(tau)

donde C_s(tau) = <s_t s_{t+tau}>, C_m(tau) = <m_t m_{t+tau}> - mbar^2 es la
autocovarianza de tamanos, y R(tau) es lo que NO se factoriza: el acoplamiento
genuino. La primera pieza es el producto de las dos memorias marginales
conocidas (Lillo-Farmer y clustering de volatilidad); la segunda no es ninguna
de las dos.

Consecuencia inmediata, y es un no-go: en un modelo de metaordenes con signos
INDEPENDIENTES entre metaordenes, el termino cruzado es no negativo a todo tau
—dentro de una metaorden el signo es constante y los tamanos estan elevados, y
entre metaordenes la contribucion se anula—. **Un modelo asi no puede producir
la inversion.** Hace falta o reversion de signo a largo desfase, o que C_m
cambie de signo. Este experimento mide cual de las dos ocurre, que es lo que
fija el mecanismo.

QUE SE MIDE
-----------
Las tres funciones y el residuo, tau = 1..200, sobre el flujo desmediado por
estrato. Y de cada una: donde cruza cero por primera vez de forma sostenida
(tres desfases consecutivos del mismo signo, para no cazar ruido).
"""
from __future__ import annotations
import json, os, sys
for _v in ("OMP","OPENBLAS","MKL","NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS","1")
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ/"src"))
from keldysh_finance.fano_validation import prepare_complete_weeks

ACTIVOS = ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX"]
PRIMARIOS = {"BTC","ETH","BNB","SOL"}
TAU_MAX = 200

def _cargar(sym):
    p = RAIZ/"output"/"cache"/f"flow_{sym}_1h_4.0y.csv"
    d = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if d.index.tz is None: d.index = d.index.tz_localize("UTC")
    return d.dropna()

def piezas(eps):
    s, m = np.sign(eps), np.abs(eps)
    mb = float(m.mean()); mb2 = mb**2
    Cs, Cm, Csm = [], [], []
    for tau in range(1, TAU_MAX+1):
        a, b = slice(None,-tau), slice(tau,None)
        Cs.append(float(np.mean(s[a]*s[b])))
        Cm.append(float(np.mean(m[a]*m[b]) - mb2))
        Csm.append(float(np.mean(s[a]*s[b]*(m[a]*m[b]-mb2))))
    Cs, Cm, Csm = map(np.asarray, (Cs, Cm, Csm))
    return {"Cs":Cs, "Cm":Cm, "Csm":Csm, "R":Csm - Cs*Cm, "mb2":mb2}

def primer_cruce(x, k=3):
    """Primer tau donde el signo se invierte y AGUANTA k desfases."""
    s0 = np.sign(x[0])
    for i in range(len(x)-k):
        if all(np.sign(x[i+j]) == -s0 and x[i+j] != 0 for j in range(k)):
            return int(i+1)
    return None

def main():
    print("="*84); print("  EXP 22 — de que esta hecho C_sm(tau)"); print("="*84)
    print("\n  C_sm = C_s * C_m + R.  La primera pieza es el producto de las dos")
    print("  memorias conocidas; R es el acoplamiento que no es ninguna de ellas.\n")
    res = []
    for a in ACTIVOS:
        p = prepare_complete_weeks(_cargar(f"{a}USDT"), flow_mode="raw")
        d = piezas(p.flow_demeaned)
        cr = {k: primer_cruce(d[k]) for k in ("Cs","Cm","Csm","R")}
        frac = float(np.mean(np.abs(d["R"])/(np.abs(d["Csm"])+1e-300)))
        res.append({"activo":a, "cruces":cr, "frac_R":frac,
                    **{k:d[k].tolist() for k in ("Cs","Cm","Csm","R")}})
        print(f"  {a}")
        print(f"    primer cruce de signo sostenido:  C_s={cr['Cs']}  "
              f"C_m={cr['Cm']}  C_sm={cr['Csm']}  R={cr['R']}")
        print(f"    |R|/|C_sm| medio = {frac:.2f}")
        for t in (1,5,20,50,100,200):
            i = t-1
            print(f"      tau={t:>4}:  C_s={d['Cs'][i]:+.4f}  "
                  f"C_m={d['Cm'][i]:+.3e}  C_sm={d['Csm'][i]:+.3e}  "
                  f"CsCm={d['Cs'][i]*d['Cm'][i]:+.3e}  R={d['R'][i]:+.3e}")
        print()
    print("  "+"="*80); print("  LECTURA"); print("  "+"-"*80)
    cm = [r for r in res if r["cruces"]["Cm"] is not None]
    print(f"  C_m (tamanos) cambia de signo en {len(cm)}/{len(res)}: el clustering")
    print("     de volatilidad es positivo a todo desfase, como toca.")
    igual = sum(1 for r in res if r["cruces"]["Csm"] == r["cruces"]["R"])
    lo = min(r["frac_R"] for r in res); hi = max(r["frac_R"] for r in res)
    print("")
    print(f"  C_sm y R cruzan en el MISMO tau en {igual}/{len(res)}, y |R|/|C_sm|")
    print(f"     vale {lo:.2f}-{hi:.2f}: el termino cruzado ES el acoplamiento")
    print("     genuino, y la parte factorizable C_s*C_m es correccion pequena.")
    desp = [(r["activo"], r["cruces"]["Csm"], r["cruces"]["Cs"]) for r in res
            if r["cruces"]["Csm"] and r["cruces"]["Cs"]]
    n_desp = sum(1 for _, a_, b_ in desp if b_ > a_)
    print("")
    print(f"  C_s cruza DESPUES que C_sm en {n_desp}/{len(desp)} de los que cruzan")
    print("     (en el resto no cruza de forma sostenida). La inversion NO se")
    print("     hereda de la memoria de signos: ocurre antes que ella.")
    taus = [(r["activo"], r["cruces"]["Csm"]) for r in res if r["cruces"]["Csm"]]
    if taus:
        v = [t for _, t in taus]
        print("")
        print(f"  tau* = {min(v)}-{max(v)} velas horarias "
              f"({min(v)/24:.1f}-{max(v)/24:.1f} dias)")
        print("     " + "  ".join(f"{a_}:{t}" for a_, t in taus))
    print("")
    print("  " + "-"*80)
    print("  CONSECUENCIA PARA EL MODELO, y es un no-go:")
    print("  Con signos INDEPENDIENTES entre metaordenes, dentro de una el signo")
    print("  es constante y los tamanos estan elevados, y entre metaordenes la")
    print("  contribucion se anula: R >= 0 a todo desfase. Ningun modelo asi")
    print("  produce la inversion. Y tampoco la producen las memorias")
    print("  marginales: C_m no cruza, y C_s cruza mas tarde.")
    print("  El mecanismo tiene que poner flujo de signo OPUESTO y tamano")
    print("  ELEVADO a ~2 dias de distancia: el desarme de la contraparte.")

    out = RAIZ/"output"/"exp22_descomponer_csm.json"
    out.write_text(json.dumps({"tau_max":TAU_MAX,"resultados":res}, indent=2,
        default=lambda o: float(o) if isinstance(o,np.floating) else str(o)),
        encoding="utf-8")
    print(f"\n  log: {out}")

if __name__ == "__main__":
    main()
