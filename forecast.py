"""Parashikimi ansambël i prurjeve bazuar në reshjet historike (metoda ESP).

Ideja: për çdo vit historik (2005–2024) marrim sekuencën e prurjeve mujore duke filluar
nga muaji aktual dhe e përdorim si një "të ardhme të mundshme". Vitet me reshje të
3 muajve të fundit të ngjashme me gjendjen aktuale marrin peshë më të madhe (analogët).
"""
import numpy as np
import pandas as pd
import model as M


def scenarios(inflow, et0, start, months=12):
    """Kthen dict {viti: (prurjet, ET0)} me sekuencat historike që nisin në muajin e `start`."""
    out = {}
    m0 = start.month
    for y in sorted(set(inflow.index.year)):
        s = pd.Timestamp(year=y, month=m0, day=1)
        idx = pd.date_range(s, periods=months, freq="MS")
        if idx[-1] > inflow.index[-1]:
            continue
        fut = pd.date_range(start, periods=months, freq="MS")
        out[y] = (pd.Series(inflow.loc[idx].values, fut), pd.Series(et0.loc[idx].values, fut))
    return out


def analog_weights(precip, years, start, recent_mm=None, lookback=3, bandwidth=0.5):
    """Peshat e viteve sipas ngjashmërisë së reshjeve të `lookback` muajve para fillimit.
    recent_mm: reshjet e vërteta të 3 muajve të fundit (nëse dihen); përndryshe pesha uniforme."""
    if recent_mm is None:
        return {y: 1 / len(years) for y in years}
    vals = {}
    for y in years:
        e = pd.Timestamp(year=y, month=start.month, day=1) - pd.DateOffset(months=1)
        b = e - pd.DateOffset(months=lookback - 1)
        if b < precip.index[0]:
            continue
        vals[y] = precip.loc[b:e].sum()
    mean = np.mean(list(vals.values()))
    w = {y: np.exp(-0.5 * ((v - recent_mm) / (bandwidth * mean)) ** 2) for y, v in vals.items()}
    tot = sum(w.values())
    return {y: x / tot for y, x in w.items()}


def run_ensemble(v0, start, inflow, et0, factor, weights, months=12):
    sc = scenarios(inflow, et0, start, months)
    traj = {}
    for y, (q, e) in sc.items():
        if y not in weights:
            continue
        d = M.demand_series(q.index, factor)
        traj[y] = M.simulate(q, e, d, v0=v0)["V"]
    return pd.DataFrame(traj)


def weighted_quantiles(df, weights, qs=(0.1, 0.5, 0.9)):
    w = np.array([weights[c] for c in df.columns])
    out = {}
    for q in qs:
        row = []
        for _, r in df.iterrows():
            o = np.argsort(r.values)
            cw = np.cumsum(w[o]) / w.sum()
            row.append(r.values[o][np.searchsorted(cw, q)])
        out[q] = pd.Series(row, df.index)
    return pd.DataFrame(out)


def prob_below(df, weights, threshold):
    """Probabiliteti që liqeni të bjerë nën `threshold` brenda horizontit."""
    hit = (df.min() < threshold)
    return float(sum(weights[c] for c in df.columns if hit[c]))


def recent_precip_climatology(precip, start, lookback=3):
    """Reshjet mesatare historike për 3 muajt para muajit të fillimit (vlerë e paracaktuar)."""
    vals = []
    for y in sorted(set(precip.index.year)):
        e = pd.Timestamp(year=y, month=start.month, day=1) - pd.DateOffset(months=1)
        b = e - pd.DateOffset(months=lookback - 1)
        if b >= precip.index[0] and e <= precip.index[-1]:
            vals.append(precip.loc[b:e].sum())
    return float(np.mean(vals))
