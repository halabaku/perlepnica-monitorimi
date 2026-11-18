"""Paneli i monitorimit dhe paralajmërimit – Liqeni i Përlepnicës.
Nisja:  uv run --frozen --offline streamlit run app.py
"""
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import config as C
import model as M
import forecast as F
import alerts as AL

OBS = Path(__file__).parent / "data" / "observations.csv"
ZONES = list(M.ZONE_NAMES)
COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
MCM = 1e6
LS_PER_M3YR = 31.5576e3   # m3/vit -> l/s
PROB_ALERT = 0.20         # paralajmërim nëse P(nën minimum) > 20%
BILLING_2025_M3 = 1_201_805.02

st.set_page_config("Përlepnica – Paralajmërimi", "💧", layout="wide")


@st.cache_data
def setup():
    A = M.calibrate()
    clim, inflow = M.build(A)
    D = M.demand_series(inflow.index)
    curves = M.kru_curves()
    k25 = M.load_kru_2025()
    k13 = M.load_kru()
    return A, clim, inflow, D, curves, k25, k13


def load_obs():
    if OBS.exists():
        return pd.read_csv(OBS, parse_dates=["data"])
    return pd.DataFrame(columns=["data", "niveli_m", "marrja_ls"])


def add_zone_bands(fig, x, curves):
    cv = curves.loc[x.month].set_index(x)
    lo = np.full(len(x), M.V_MIN)
    bands = [(np.zeros(len(x)), lo, "#999999", "Nën minimum"),
             (lo, cv.C3_emergjence, COLORS[3], "Emergjencë"),
             (cv.C3_emergjence, cv.C2_paralajmerim, COLORS[2], "Paralajmërim"),
             (cv.C2_paralajmerim, cv.C1_alarm, COLORS[1], "Alarm"),
             (cv.C1_alarm, np.full(len(x), C.V_MAX), COLORS[0], "Normale")]
    for a, b, c, n in bands:
        fig.add_trace(go.Scatter(x=x, y=np.asarray(a) / MCM, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=np.asarray(b) / MCM, fill="tonexty", fillcolor=c, opacity=.3,
                                 line=dict(width=0), name=n, hoverinfo="skip"))


A, clim, inflow, D, curves, k25, k13 = setup()
try:
    smtp = AL.smtp_config(st.secrets.get("smtp"))
except Exception:
    smtp = AL.smtp_config()

# ---------------------------------------------------------------- hyrja
st.sidebar.header("Matja e re")
date = st.sidebar.date_input("Data", dt.date.today())
level = st.sidebar.number_input("Niveli i ujit mbi fund (m)", 0.0, C.H_MAX, float(k25.niveli_m.iloc[-1]), 0.1)
take_default = round(float(k25.marrja_ls.mean()), 1)
take = st.sidebar.number_input("Marrja aktuale nga liqeni (l/s)", 0.0, 300.0, take_default, 1.0)
st.sidebar.caption(f"Mesatarja 2025: {take_default} l/s | Kapaciteti 2018: {C.DEMAND_ANNUAL/LS_PER_M3YR:.1f} l/s")

start = pd.Timestamp(date).to_period("M").to_timestamp()
clim_3m = F.recent_precip_climatology(clim.P, start)
st.sidebar.header("Parashikimi i reshjeve")
use_analog = st.sidebar.checkbox("Përdor reshjet e 3 muajve të fundit (vitet analoge)", True)
recent = st.sidebar.number_input(f"Reshjet e 3 muajve të fundit (mm) – mesatarja: {clim_3m:.0f}",
                                 0.0, 800.0, float(round(clim_3m)), 5.0, disabled=not use_analog)

# ---------------------------------------------------------------- gjendja + ansambli
V = float(M.vol_from_h(level))
z = M.zone_of(V, date.month, curves)
f0 = take / (C.DEMAND_ANNUAL / LS_PER_M3YR)
years = list(F.scenarios(inflow, clim.ET0, start).keys())
W = F.analog_weights(clim.P, years, start, recent if use_analog else None)
ens0 = F.run_ensemble(V, start, inflow, clim.ET0, f0, W)
ens1 = F.run_ensemble(V, start, inflow, clim.ET0, f0 * M.ZONE_FACTORS[z], W)
p_below = F.prob_below(ens0, W, M.V_MIN)
q0, q1 = F.weighted_quantiles(ens0, W), F.weighted_quantiles(ens1, W)
below = q0[0.1] < M.V_MIN
days = (q0[0.1][below].index[0] - start).days if below.any() else None
alert_level = max(z, 2 if p_below > PROB_ALERT and z < 2 else z)

st.title("💧 Liqeni i Përlepnicës – Gjendja dhe paralajmërimi")
tab1, tab2, tab3 = st.tabs(["💧 Gjendja & Paralajmërimi", "📊 Të dhënat zyrtare 2025 (KRU)", "📉 Kriza 2013/14 (KRU)"])

with tab1:
    st.markdown(f"<div style='padding:14px;border-radius:10px;background:{COLORS[z]};color:#111;"
                f"font-size:22px;font-weight:700'>Zona: {ZONES[z]} – furnizimi i rekomanduar "
                f"{int(M.ZONE_FACTORS[z]*100)}% ({take*M.ZONE_FACTORS[z]:.0f} l/s)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vëllimi", f"{V/MCM:.2f} Mm³", f"{100*V/C.V_MAX:.0f}% e kapacitetit")
    c2.metric("Rezerva e shfrytëzueshme", f"{max(V-M.V_MIN,0)/MCM:.2f} Mm³")
    c3.metric("Rreziku: nën minimum në 12 muaj", f"{100*p_below:.0f}%")
    c4.metric("Ditë deri te minimumi (viti i thatë, P10)", "> 365" if days is None else f"~{days}")
    if p_below > PROB_ALERT:
        st.error(f"⚠️ Probabiliteti që liqeni të bjerë nën {C.H_MIN_ALLOWED} m brenda 12 muajve është "
                 f"{100*p_below:.0f}% – fillo reduktimet tani.")

    x = pd.date_range(start - pd.DateOffset(months=12), periods=25, freq="MS")
    fig = go.Figure()
    add_zone_bands(fig, x, curves)
    for q, name, col in [(q0, "Pa ndryshim", "#1f2d3d"), (q1, "Me reduktimin e rekomanduar", "#1f6fd1")]:
        fig.add_trace(go.Scatter(x=q.index, y=q[0.9] / MCM, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.1] / MCM, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(31,45,61,.15)" if col == "#1f2d3d" else "rgba(31,111,209,.2)",
                                 name=f"{name}: P10–P90"))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.5] / MCM, line=dict(width=3, color=col), name=f"{name}: mediana"))
    k25_win = k25[(k25.index >= x[0]) & (k25.index <= x[-1])]
    if len(k25_win):
        fig.add_trace(go.Scatter(x=k25_win.index, y=k25_win.V_m3 / MCM, mode="markers+lines",
                                 name="KRU 2025", marker=dict(size=6, color="#0d9488"), line=dict(color="#0d9488", width=2)))
    obs = load_obs()
    if len(obs):
        fig.add_trace(go.Scatter(x=obs.data, y=M.vol_from_h(obs.niveli_m) / MCM, mode="markers+lines",
                                 name="Matjet", marker=dict(size=7, color="black")))
    fig.add_trace(go.Scatter(x=[start], y=[V / MCM], mode="markers",
                             marker=dict(size=14, symbol="star", color="navy"), name="Sot"))
    fig.update_layout(height=500, yaxis_title="Vëllimi (Mm³)", legend=dict(orientation="h"),
                      title=f"Parashikimi ansambël 12-mujor ({len(ens0.columns)} vite historike)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Sa duhet reduktuar tani?")
    rows = []
    for r in [0, .05, .10, .15, .20, .25, .30, .40]:
        e = F.run_ensemble(V, start, inflow, clim.ET0, f0 * (1 - r), W)
        pb = F.prob_below(e, W, M.V_MIN)
        rows.append({"Reduktimi": f"{int(r*100)}%", "Furnizimi (l/s)": round(take * (1 - r)),
                     "P(nën minimum)": f"{100*pb:.0f}%",
                     "Niveli min. – viti i thatë P10 (m)": round(float(M.h_from_vol(F.weighted_quantiles(e, W)[0.1].min())), 1),
                     "Vlerësimi": "✅" if pb <= 0.10 else "⚠️" if pb <= PROB_ALERT else "❌"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Vitet analoge dhe peshat"):
        st.dataframe(pd.DataFrame({"Viti": list(W), "Pesha (%)": [round(100 * w, 1) for w in W.values()]})
                     .sort_values("Pesha (%)", ascending=False), hide_index=True)

with tab2:
    prod_tot = float(k25.prodhimi_m3.sum())
    nrw_m3 = prod_tot - BILLING_2025_M3
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prodhimi total 2025", f"{prod_tot/MCM:.2f} Mm³", f"mesatarisht {k25.marrja_ls.mean():.1f} l/s")
    m2.metric("Uji i faturuar 2025", f"{BILLING_2025_M3/MCM:.2f} Mm³", f"{100*BILLING_2025_M3/prod_tot:.1f}% e prodhimit")
    m3.metric("Uji i pafaturuar (NRW)", f"{nrw_m3/MCM:.2f} Mm³", f"{100*nrw_m3/prod_tot:.1f}% humbje")
    m4.metric("Niveli min / max 2025", f"{k25.niveli_m.min():.2f} / {k25.niveli_m.max():.2f} m", "Nëntor / Dhjetor")

    fig25 = go.Figure()
    add_zone_bands(fig25, k25.index, curves)
    fig25.add_trace(go.Scatter(x=k25.index, y=k25.V_m3 / MCM, mode="markers+lines", name="Vëllimi 2025 (nga niveli)",
                               line=dict(width=3, color="#0f172a"), marker=dict(size=8)))
    fig25.update_layout(height=420, yaxis_title="Vëllimi (Mm³)", legend=dict(orientation="h"),
                        title="Vëllimi i vëzhguar/vlerësuar i liqenit në vitin 2025 kundrejt pragjeve të KRU-së")
    st.plotly_chart(fig25, width="stretch")

    fig_pm = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pm.add_trace(go.Bar(x=k25.muaji, y=k25.prodhimi_m3 / 1000, name="Prodhimi (mijë m³)", marker_color="#3b82c4"),
                     secondary_y=False)
    fig_pm.add_trace(go.Scatter(x=k25.muaji, y=k25.niveli_m, name="Niveli i ujit (m)", mode="lines+markers",
                                line=dict(color="#d97706", width=3)), secondary_y=True)
    fig_pm.update_yaxes(title_text="Prodhimi (mijë m³)", secondary_y=False)
    fig_pm.update_yaxes(title_text="Niveli (m)", secondary_y=True)
    fig_pm.update_layout(height=380, title="Prodhimi mujor dhe niveli i ujit gjatë vitit 2025",
                         legend=dict(orientation="h"))
    st.plotly_chart(fig_pm, width="stretch")

    t25 = k25.copy()
    t25["% e normales"] = [round(100 * v / curves.loc[d.month, "normal"]) for d, v in t25.V_m3.items()]
    t25["Zona"] = [ZONES[M.zone_of(v, d.month, curves)] for d, v in t25.V_m3.items()]
    t25["Vëllimi (Mm³)"] = (t25.V_m3 / MCM).round(2)
    t25["Marrja (l/s)"] = t25.marrja_ls.round(1)
    st.dataframe(t25[["muaji", "prodhimi_m3", "Marrja (l/s)", "niveli_m", "Vëllimi (Mm³)", "% e normales", "Zona"]]
                 .rename(columns={"muaji": "Muaji", "prodhimi_m3": "Prodhimi (m³)", "niveli_m": "Niveli (m)"}),
                 hide_index=True, width="stretch")

with tab3:
    fig13 = go.Figure()
    add_zone_bands(fig13, k13.index, curves)
    fig13.add_trace(go.Scatter(x=k13.index, y=k13.observed, mode="markers+lines", name="E vëzhguar 2013/14 (KRU)",
                               line=dict(width=3, color="#0033cc"), marker=dict(size=8)))
    fig13.update_layout(height=420, yaxis_title="Vëllimi (Mm³)", legend=dict(orientation="h"),
                        title="Kriza 2013/14 – vëllimi i vëzhguar kundrejt pragjeve zyrtare (KRU Drought Plan)")
    st.plotly_chart(fig13, width="stretch")
    st.dataframe(k13[["muaji", "normal", "alert_75", "warning_50", "emergency_25", "observed", "perqindja"]]
                 .rename(columns={"muaji": "Muaji", "normal": "Normale (Mm³)", "alert_75": "Alarm 75%",
                                  "warning_50": "Paralajmërim 50%", "emergency_25": "Emergjencë 25%",
                                  "observed": "E vëzhguar (Mm³)", "perqindja": "%"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- ruajtja + email
body = (f"Liqeni i Përlepnicës – {date:%d.%m.%Y}\n\nNiveli: {level:.1f} m | Vëllimi: {V/MCM:.2f} Mm³\n"
        f"Zona: {ZONES[z]}\nRreziku nën minimum (12 muaj): {100*p_below:.0f}%\n"
        f"Furnizimi i rekomanduar: {take*M.ZONE_FACTORS[z]:.0f} l/s (aktualisht {take:.0f} l/s)\n")
if st.sidebar.button("💾 Ruaj matjen"):
    o = pd.concat([load_obs(), pd.DataFrame([dict(data=pd.Timestamp(date), niveli_m=level, marrja_ls=take)])])
    OBS.parent.mkdir(exist_ok=True)
    o.drop_duplicates("data", keep="last").sort_values("data").to_csv(OBS, index=False)
    prev = AL.last_zone()
    AL.save_zone(alert_level)
    st.sidebar.success("U ruajt.")
    if alert_level > prev:
        if smtp:
            try:
                AL.send(smtp, f"[Përlepnica] Paralajmërim: {ZONES[alert_level]}", body)
                st.sidebar.warning(f"📧 Email u dërgua: {ZONES[prev]} → {ZONES[alert_level]}")
            except Exception as ex:
                st.sidebar.error(f"Email dështoi: {ex}")
        else:
            st.sidebar.warning(f"Zona u përkeqësua ({ZONES[prev]} → {ZONES[alert_level]}), "
                               "por email-i nuk është konfiguruar.")
st.sidebar.header("Njoftimet")
st.sidebar.caption("Email: " + (f"aktiv → {smtp['to']}" if smtp else "jo i konfiguruar (.streamlit/secrets.toml)"))
if st.sidebar.button("📧 Dërgo email testues", disabled=not smtp):
    try:
        AL.send(smtp, "[Përlepnica] Test", body)
        st.sidebar.success("Email-i testues u dërgua.")
    except Exception as ex:
        st.sidebar.error(f"Dështoi: {ex}")

with tab1:
    with st.expander("Matjet e ruajtura"):
        st.dataframe(obs, hide_index=True)
st.caption(f"Model: bilanc mujor, prurjet abcd/ERA5 {inflow.index[0]:%Y}–{inflow.index[-1]:%Y}, pellg efektiv "
           f"{A:.1f} km². Kurba H–V: Hmax={C.H_MAX} m (nga dhjetori 2025), m={C.M_SHAPE} [SUPOZIM].")
