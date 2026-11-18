"""Modeli i bilancit të ujit për Liqenin e Përlepnicës dhe rregullat e reduktimit (hedging)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config as C

DATA = Path(__file__).parent / "data" / "climate_monthly.csv"


# ----------------------------------------------------------------- gjeometria
def vol_from_h(h):
    return C.V_MAX * (np.clip(h, 0, C.H_MAX) / C.H_MAX) ** C.M_SHAPE


def h_from_vol(v):
    return C.H_MAX * (np.clip(v, 0, C.V_MAX) / C.V_MAX) ** (1 / C.M_SHAPE)


def area_from_vol(v):
    """Sipërfaqja = dV/dh (në përputhje me kurbën V–h)."""
    return C.A_LAKE_FULL * (h_from_vol(v) / C.H_MAX) ** (C.M_SHAPE - 1)


V_MIN = vol_from_h(C.H_MIN_ALLOWED)
V_CRISIS = vol_from_h(C.H_CRISIS)
V_SLUDGE = vol_from_h(C.H_SLUDGE)


# ----------------------------------------------------------------- klima, prurjet
def load_climate():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    return df.loc[C.SIM_START:C.SIM_END]


def abcd_runoff(clim, a, b, c, d):
    """Rrjedhja mujore (mm) me modelin abcd + borë e thjeshtë."""
    S, G, snow = 100.0, 20.0, 0.0
    q = []
    for P, PET, T in zip(clim.P, clim.ET0, clim["T"]):
        frac_snow = 1.0 if T < -1 else 0.0 if T > 3 else (3 - T) / 4
        snow += P * frac_snow
        melt = snow * (0 if T < 0 else min(1.0, T / 4))
        snow -= melt
        W = P * (1 - frac_snow) + melt + S
        k = (W + b) / (2 * a)
        Y = k - np.sqrt(max(k * k - W * b / a, 0))
        S = Y * np.exp(-PET / b)
        avail = W - Y
        G = (G + c * avail) / (1 + d)
        q.append((1 - c) * avail + d * G)
    return pd.Series(q, index=clim.index)


def demand_series(index, factor=1.0):
    s = np.array(C.SEASONAL) / np.mean(C.SEASONAL)
    days = index.days_in_month.values
    return pd.Series(C.DEMAND_ANNUAL * s[index.month - 1] * days / 365.25 * factor, index=index)


# ----------------------------------------------------------------- rezervuari
def simulate(inflow, et0, demand, policy=None, v0=C.V_MAX, floor=None):
    """policy(t, V, month) -> faktori i furnizimit (0..1). floor = vëllimi fizik minimal."""
    floor = V_SLUDGE if floor is None else floor
    V = v0
    rows = []
    for t, (date, I) in enumerate(inflow.items()):
        f = 1.0 if policy is None else policy(t, V, date)
        D = demand.iloc[t]
        E = K_E * et0.iloc[t] / 1000 * area_from_vol(V)
        eco = C.ECO_FLOW * 86400 * date.days_in_month
        avail = V + I - E - eco - floor
        R = min(D * f, max(avail, 0))
        V_new = V + I - E - eco - R
        spill = max(V_new - C.V_MAX, 0)
        V_new = min(V_new, C.V_MAX)
        rows.append(dict(date=date, V=V_new, H=h_from_vol(V_new), I=I, E=E, R=R, D=D,
                         spill=spill, factor=R / D))
        V = V_new
    return pd.DataFrame(rows).set_index("date")


K_E = C.K_EVAP


ABCD = dict(C.ABCD)          # përditësohet nga calibrate()
_CLIM = None


def build(area_km2, params=None):
    global _CLIM
    if _CLIM is None:
        _CLIM = load_climate()
    clim = _CLIM
    q_mm = abcd_runoff(clim, **(params or ABCD))
    inflow = q_mm / 1000 * area_km2 * 1e6
    return clim, inflow


KRU_FILE = Path(__file__).parent / "data" / "kru_drought_triggers_2013.csv"
KRU_2025_FILE = Path(__file__).parent / "data" / "kru_2025.csv"


def load_kru():
    """Të dhënat zyrtare të KRU Hidromorava (Drought Plan 2013/14), vlerat në Mm³."""
    return pd.read_csv(KRU_FILE, parse_dates=["data"]).set_index("data")


def load_kru_2025():
    """Të dhënat zyrtare mujore të KRU Hidromorava për vitin 2025 (prodhimi m³ dhe niveli m)."""
    df = pd.read_csv(KRU_2025_FILE, parse_dates=["data"]).set_index("data")
    df["V_m3"] = vol_from_h(df["niveli_m"])
    df["marrja_ls"] = df["prodhimi_m3"] / (df.index.days_in_month * 86400) * 1000
    return df


def kru_curves():
    """Kurbat zyrtare të pragjeve sipas muajit kalendarik (m³): 75%, 50%, 25% e normales."""
    k = load_kru()
    k.index = k.index.month
    return pd.DataFrame({"C1_alarm": k.alert_75 * 1e6, "C2_paralajmerim": k.warning_50 * 1e6,
                         "C3_emergjence": k.emergency_25 * 1e6, "normal": k.normal * 1e6}).sort_index()


def fit_to_kru(A, dfac=1.0, params=None):
    """Simulon prill 2013 – mars 2014 duke nisur nga vëllimi i vëzhguar; kthen (sim, obs) në m³."""
    k = load_kru()
    clim, inflow = build(A, params)
    idx = k.index
    d = demand_series(idx, dfac)
    res = simulate(inflow.loc[idx].iloc[1:], clim.ET0.loc[idx].iloc[1:], d.iloc[1:],
                   v0=k.observed.iloc[0] * 1e6, floor=0)
    sim = np.r_[k.observed.iloc[0] * 1e6, res.V.values]
    return sim, k.observed.values * 1e6


CALIB = {}


def calibrate():
    """Sipërfaqja efektive e pellgut që minimizon RMSE ndaj 12 vëllimeve të vëzhguara
    prill 2013 – mars 2014 (KRU Drought Plan), me kërkesën zyrtare 3.30 Mm³/vit."""
    best = None
    for A in np.arange(2.0, 40.01, 0.25):
        sim, obs = fit_to_kru(A)
        r = float(np.sqrt(np.mean((sim - obs) ** 2)))
        if best is None or r < best[0]:
            best = (r, A)
    sim, obs = fit_to_kru(best[1])
    CALIB.update(A=best[1], rmse=best[0],
                 nse=float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)))
    return best[1]


# ----------------------------------------------------------------- skenarët
def uniform_reduction_policy(start, end, r):
    def pol(t, V, date):
        return 1 - r if start <= date <= end else 1.0
    return pol


def min_reduction_for_lead(inflow, et0, demand, crisis_month, lead, recover_month):
    """Reduktimi uniform minimal që, i nisur `lead` muaj para krizës, mban V >= V_MIN."""
    start = crisis_month - pd.DateOffset(months=lead)
    lo, hi = 0.0, 0.95
    for _ in range(40):
        r = (lo + hi) / 2
        res = simulate(inflow, et0, demand, uniform_reduction_policy(start, recover_month, r))
        if res.loc[start:recover_month, "V"].min() < V_MIN:
            lo = r
        else:
            hi = r
    res = simulate(inflow, et0, demand, uniform_reduction_policy(start, recover_month, hi))
    return hi, res, start


def optimal_hedging(inflow, et0, demand, start, end):
    """Optimumi (me njohje të plotë të prurjeve): min Σ((D-R)/D)^2, me kusht V >= V_MIN."""
    idx = inflow.loc[start:end].index
    n = len(idx)
    base = simulate(inflow, et0, demand)
    v0 = base.loc[:start].iloc[-2]["V"] if base.index[0] < pd.Timestamp(start) else C.V_MAX

    def run(x):
        sub_in, sub_et, sub_d = inflow.loc[idx], et0.loc[idx], demand.loc[idx]
        fac = dict(zip(idx, x))
        return simulate(sub_in, sub_et, sub_d, lambda t, V, d: fac[d], v0=v0, floor=0)

    obj = lambda x: np.sum((1 - x) ** 2)
    cons = {"type": "ineq", "fun": lambda x: (run(x)["V"].values - V_MIN) / 1e5}
    sol = minimize(obj, np.full(n, 0.9), bounds=[(0.3, 1.0)] * n, constraints=[cons],
                   method="SLSQP", options=dict(maxiter=300, ftol=1e-8))
    fac = dict(zip(idx, sol.x))
    pol = lambda t, V, d: fac.get(d, 1.0)
    return sol.x, simulate(inflow, et0, demand, pol)


# ----------------------------------------------------------------- kurbat e operimit
def rule_curves(inflow, et0, demand, pct=10, horizon=12):
    """Kurba kritike sipas muajit: vëllimi që, me prurje të vitit të thatë (percentili `pct`)
    dhe kërkesë të plotë, mjafton për `horizon` muaj pa rënë nën V_MIN."""
    dry = inflow.groupby(inflow.index.month).quantile(pct / 100)
    evap = (K_E * et0.groupby(et0.index.month).mean() / 1000 * C.A_LAKE_FULL * 0.6)
    dem = demand.groupby(demand.index.month).mean()
    eco = C.ECO_FLOW * 86400 * 30.4
    net = (dem + evap + eco - dry)            # humbja neto mujore në vit të thatë
    crit = {}
    for m in range(1, 13):
        cum, need = 0.0, 0.0
        for j in range(horizon):
            cum += net[(m - 1 + j) % 12 + 1]
            need = max(need, cum)
        crit[m] = min(V_MIN + need, C.V_MAX)
    c1 = pd.Series(crit)
    c2 = V_MIN + 0.66 * (c1 - V_MIN)
    c3 = V_MIN + 0.33 * (c1 - V_MIN)
    return pd.DataFrame({"C1_paralajmerim": c1, "C2_reduktim": c2, "C3_emergjence": c3})


ZONE_FACTORS = (1.0, 0.90, 0.75, 0.60)   # normale, alarm, paralajmërim, emergjencë (KRU)
ZONE_NAMES = ("Normale", "Alarm", "Paralajmërim", "Emergjencë")


def zone_of(V, month, curves):
    c = curves.loc[month]
    if V >= c.iloc[0]:
        return 0
    if V >= c.iloc[1]:
        return 1
    if V >= c.iloc[2]:
        return 2
    return 3


def rule_policy(curves):
    return lambda t, V, date: ZONE_FACTORS[zone_of(V, date.month, curves)]


def performance(res):
    ratio = res.R / res.D
    return dict(
        besueshmeria_pct=round(100 * (ratio > 0.999).mean(), 1),
        muaj_nen_Vmin=int((res.V < V_MIN - 1).sum()),
        deficit_max_pct=round(100 * (1 - ratio.min()), 1),
        indeksi_demit=round(float(((1 - ratio) ** 2).sum()), 3),
        deficit_total_Mm3=round(float((res.D - res.R).sum() / 1e6), 3),
    )
