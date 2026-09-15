import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# STREAMLIT PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="PRECEDENT [ThrowMaster] - OANDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Palette Invariants
COL_BULL = "#0ABAB5"
COL_BEAR = "#FF4D6D"
COL_TP   = "#3DDC97"
COL_WARN = "#F2B705"
COL_GOLD = "#D4AF37"
COL_SLATE= "#7A8B99"

# ==============================================================================
# 1. OANDA API DATA & INSTRUMENT FETCHING
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_oanda_instruments(account_type: str, api_key: str) -> List[Dict[str, str]]:
    """
    Fetches all tradeable instruments for the account dynamically from OANDA.
    Cached for 1 hour to prevent excessive API calls.
    """
    base_url = (
        "https://api-fxpractice.oanda.com" if account_type == "Practice"
        else "https://api-fxtrade.oanda.com"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Step 1: Get authorized account ID
        acc_res = requests.get(f"{base_url}/v3/accounts", headers=headers, timeout=10)
        if acc_res.status_code != 200:
            return []
        
        accounts = acc_res.json().get("accounts", [])
        if not accounts:
            return []

        account_id = accounts[0]["id"]

        # Step 2: Get tradeable instruments for the account
        inst_res = requests.get(f"{base_url}/v3/accounts/{account_id}/instruments", headers=headers, timeout=10)
        if inst_res.status_code != 200:
            return []

        raw_instruments = inst_res.json().get("instruments", [])
        
        instruments = [
            {
                "name": item["name"],
                "displayName": item.get("displayName", item["name"])
            }
            for item in raw_instruments
        ]
        # Sort alphabetically by display name
        instruments.sort(key=lambda x: x["displayName"])
        return instruments

    except Exception:
        return []


def fetch_oanda_candles(
    account_type: str,
    api_key: str,
    instrument: str,
    granularity: str,
    total_bars: int = 2000
) -> pd.DataFrame:
    """Fetches historical candles from OANDA v20 API with pagination."""
    base_url = (
        "https://api-fxpractice.oanda.com" if account_type == "Practice"
        else "https://api-fxtrade.oanda.com"
    )
    endpoint = f"{base_url}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}

    all_candles = []
    to_time = None
    remaining = total_bars

    while remaining > 0:
        count = min(remaining, 5000)
        params = {"granularity": granularity, "count": count, "price": "M"}
        if to_time:
            params["to"] = to_time

        res = requests.get(endpoint, headers=headers, params=params, timeout=15)
        if res.status_code != 200:
            st.error(f"OANDA Candle API Error [{res.status_code}]: {res.text}")
            return pd.DataFrame()

        data = res.json().get("candles", [])
        if not data:
            break

        all_candles = data + all_candles
        remaining -= len(data)
        to_time = data[0]["time"]

        if len(data) < count:
            break

    if not all_candles:
        return pd.DataFrame()

    records = []
    for c in all_candles:
        if not c["complete"]:
            continue
        mid = c["mid"]
        records.append({
            "time": pd.to_datetime(c["time"]),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c["volume"])
        })

    df = pd.DataFrame(records).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ==============================================================================
# 2. TECHNICAL INDICATORS & REGIME ENGINE
# ==============================================================================
def compute_indicators(df: pd.DataFrame, cal_win: int = 300) -> pd.DataFrame:
    df = df.copy()

    # True Range & ATR
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr20"] = df["tr"].rolling(20).mean()

    # Range & Body Dynamics
    df["rng"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["wick_up"] = df["high"] - np.maximum(df["open"], df["close"])
    df["wick_dn"] = np.minimum(df["open"], df["close"]) - df["low"]

    # Flow Proxy / Volume Flow
    vol_sma = df["volume"].rolling(50).mean()
    vol_avail = (vol_sma > 0).all()
    body_ratio = (df["close"] - df["open"]) / np.maximum(df["rng"], 1e-5)
    df["flow_unit"] = body_ratio * df["volume"] if vol_avail else body_ratio

    # Self-Calibrating Thresholds (Percentiles)
    df["wick_up_thr"] = df["wick_up"].rolling(100).quantile(0.75)
    df["wick_dn_thr"] = df["wick_dn"].rolling(100).quantile(0.75)
    df["rng_thr"] = df["rng"].rolling(200).quantile(0.90)
    df["disp_thr"] = df["body"].rolling(100).quantile(0.85)

    # Kaufman Efficiency Ratio (Regime Engine)
    er_change = (df["close"] - df["close"].shift(20)).abs()
    er_vol = (df["close"] - df["close"].shift(1)).abs().rolling(20).sum()
    er_raw = np.where(er_vol > 0, er_change / er_vol, 0.0)

    er_series = pd.Series(er_raw)
    er_rank = er_series.rolling(cal_win).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False
    )
    df["er_rank"] = er_rank
    df["regime"] = np.select([er_rank >= 70, er_rank <= 40], [2, 0], default=1)

    # Volatility Squeeze
    bb_basis = df["close"].rolling(20).mean()
    bb_dev = 2.0 * df["close"].rolling(20).std()
    kc_u = bb_basis + 1.5 * df["atr20"]
    kc_l = bb_basis - 1.5 * df["atr20"]
    df["in_sqz"] = ((bb_basis + bb_dev) < kc_u) & ((bb_basis - bb_dev) > kc_l)

    sqz_cnt = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if df["in_sqz"].iloc[i]:
            sqz_cnt[i] = sqz_cnt[i-1] + 1
    df["sqz_cnt"] = sqz_cnt
    df["sqz_release"] = (~df["in_sqz"]) & (df["sqz_cnt"].shift(1) >= 5)

    return df


# ==============================================================================
# 3. LEVEL MAP & PRECEDENT ENGINE
# ==============================================================================
class Level:
    def __init__(self, price: float, is_swing=False, is_eq=False, is_ob=False, is_htf=False, fvg_n=0, touches=0, birth=0):
        self.price = price
        self.is_swing = is_swing
        self.is_eq = is_eq
        self.is_ob = is_ob
        self.is_htf = is_htf
        self.fvg_n = fvg_n
        self.touches = touches
        self.birth = birth
        self.alive = True

    def weight(self, current_bar: int) -> int:
        w = 0
        w += 1 if self.is_swing else 0
        w += 1 if self.is_eq else 0
        w += 1 if self.is_ob else 0
        w += 2 if self.is_htf else 0
        w += min(self.fvg_n, 3)
        w += min(self.touches, 3)
        w += 1 if (current_bar - self.birth) > 200 else 0
        return w


def find_level(levels: List[Level], price: float, tol: float) -> int:
    idx, best = -1, tol
    for i, l in enumerate(levels):
        if l.alive:
            d = abs(l.price - price)
            if d <= best:
                best = d
                idx = i
    return idx


def add_factor(levels: List[Level], price: float, tol: float, kind: int, bar_idx: int, cap: int = 60):
    i = find_level(levels, price, tol)
    if i >= 0:
        l = levels[i]
        if kind == 1: l.is_swing = True
        elif kind == 2:
            l.is_eq = True
            l.touches += 1
            l.price = (l.price + price) / 2.0
        elif kind == 3: l.fvg_n += 1
        elif kind == 4: l.is_ob = True
        elif kind == 5: l.is_htf = True
    else:
        nl = Level(price, is_swing=(kind==1), is_eq=(kind==2), is_ob=(kind==4), is_htf=(kind==5), fvg_n=(1 if kind==3 else 0), touches=(1 if kind==2 else 0), birth=bar_idx)
        levels.append(nl)
        if len(levels) > cap:
            levels.pop(0)


def process_precedent_engine(df: pd.DataFrame, horizon: int = 24, n_min: int = 20, maj_weight: int = 4):
    levels: List[Level] = []
    store, pend, projs, events = [], [], [], []
    EV_SWEEP, EV_SQUEEZE, EV_CLIMAX = 1, 3, 4
    last_ev = {k: -99999 for k in range(1, 7)}

    for i in range(len(df)):
        if i < 50:
            continue

        c_bar = i
        row = df.iloc[i]
        close, high, low, open_p = row["close"], row["high"], row["low"], row["open"]
        atr14 = row["atr14"]
        lev_tol = 0.25 * atr14 if not np.isnan(atr14) else 0.001

        # FVGs
        if i >= 2:
            if low > df.iloc[i-2]["high"] and df.iloc[i-1]["close"] > df.iloc[i-1]["open"]:
                add_factor(levels, (low + df.iloc[i-2]["high"]) / 2.0, lev_tol, 3, c_bar)
            elif high < df.iloc[i-2]["low"] and df.iloc[i-1]["close"] < df.iloc[i-1]["open"]:
                add_factor(levels, (high + df.iloc[i-2]["low"]) / 2.0, lev_tol, 3, c_bar)

        loc_w = max([l.weight(c_bar) for l in levels if l.alive and abs(l.price - close) <= lev_tol] or [0])
        loc_cls = 2 if loc_w >= maj_weight else (1 if loc_w >= 1 else 0)

        # Event Detection
        ev_id, ev_dir = 0, 0
        if row["wick_up"] >= row["wick_up_thr"] and high > close + lev_tol:
            ev_id, ev_dir = EV_SWEEP, -1
        elif row["wick_dn"] >= row["wick_dn_thr"] and low < close - lev_tol:
            ev_id, ev_dir = EV_SWEEP, 1
        elif row["rng"] >= row["rng_thr"] and row["body"] >= row["disp_thr"]:
            ev_id, ev_dir = EV_CLIMAX, (1 if close > open_p else -1)
        elif row["sqz_release"]:
            ev_id, ev_dir = EV_SQUEEZE, (1 if close > open_p else -1)

        if ev_id > 0 and (c_bar - last_ev[ev_id]) >= horizon:
            last_ev[ev_id] = c_bar
            regime = int(row["regime"])
            tier_v = 1

            events.append({"bar": c_bar, "time": row["time"], "price": close, "ev_id": ev_id, "dir": ev_dir})
            pend.append({"sBar": c_bar, "anchor": close, "r": atr14, "dir": ev_dir, "ev": ev_id, "loc": loc_cls, "rg": regime, "tier": tier_v, "mfe": 0.0, "mae": 0.0, "e4": 0.0})

            # Signature Matching (L3 -> L2 -> L1 -> L0)
            i0 = [o for o in store if o["ev"] == ev_id and o["dir"] == ev_dir]
            i1 = [o for o in i0 if o["rg"] == regime]
            i2 = [o for o in i1 if o["loc"] == loc_cls]
            i3 = [o for o in i2 if o["tier"] == tier_v]

            sel, lvl_used = [], -1
            if len(i3) >= n_min: sel, lvl_used = i3, 3
            elif len(i2) >= n_min: sel, lvl_used = i2, 2
            elif len(i1) >= n_min: sel, lvl_used = i1, 1
            elif len(i0) >= n_min: sel, lvl_used = i0, 0

            if lvl_used >= 0:
                mfe_arr = [o["mfe"] for o in sel]
                e4_arr = [o["e4"] for o in sel]
                d_tgt = max(0.5, float(np.median(mfe_arr)))
                projs.append({
                    "sBar": c_bar, "time": row["time"], "anchor": close, "dir": ev_dir,
                    "tgt": close + ev_dir * d_tgt * atr14,
                    "hitPct": (np.array(mfe_arr) >= d_tgt).mean() * 100,
                    "n": len(sel), "lvl": lvl_used,
                    "q75": float(np.percentile(e4_arr, 75)), "q25": float(np.percentile(e4_arr, 25))
                })

        for p in list(pend):
            age = c_bar - p["sBar"]
            p["mfe"] = max(p["mfe"], (high - p["anchor"] if p["dir"] > 0 else p["anchor"] - low) / max(p["r"], 1e-5))
            p["mae"] = max(p["mae"], (p["anchor"] - low if p["dir"] > 0 else high - p["anchor"]) / max(p["r"], 1e-5))
            if age >= horizon:
                p["e4"] = (close - p["anchor"]) * p["dir"] / max(p["r"], 1e-5)
                store.append(p)
                pend.remove(p)

    return events, projs, levels


# ==============================================================================
# 4. STREAMLIT UI & MAIN APPLICATION
# ==============================================================================
def main():
    st.sidebar.title("PRECEDENT Engine")
    st.sidebar.caption("OANDA v20 Multi-Timeframe Instrument Analytics")

    # --- 1. SECRETS & API KEY MANAGEMENT ---
    st.sidebar.subheader("OANDA Authentication")
    account_type = st.sidebar.selectbox("Account Type", ["Practice", "Live"])

    # Read API Key directly from Streamlit secrets, with UI text_input fallback
    secret_key = st.secrets.get("OANDA_API_KEY") or st.secrets.get("OANDA_TOKEN")
    if secret_key:
        api_key = secret_key
        st.sidebar.success("Loaded API Key from Secrets")
    else:
        api_key = st.sidebar.text_input(
            "OANDA API Key",
            type="password",
            help="To skip manual entry, add OANDA_API_KEY to App Secrets in Streamlit Cloud."
        )

    if not api_key:
        st.info("Enter your OANDA API Key or set `OANDA_API_KEY` in Streamlit Community Cloud Secrets.")
        st.stop()

    # --- 2. DYNAMIC INSTRUMENT SELECTION ---
    st.sidebar.subheader("Market Selection")
    
    with st.spinner("Loading OANDA instruments..."):
        instruments = fetch_oanda_instruments(account_type, api_key)

    if instruments:
        inst_names = [i["name"] for i in instruments]
        inst_labels = {i["name"]: f"{i['displayName']} ({i['name']})" for i in instruments}
        default_idx = inst_names.index("EUR_USD") if "EUR_USD" in inst_names else 0

        instrument = st.sidebar.selectbox(
            "Instrument",
            options=inst_names,
            index=default_idx,
            format_func=lambda x: inst_labels.get(x, x)
        )
    else:
        st.sidebar.warning("Unable to fetch instruments dynamically. Using default list.")
        fallback_list = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "XAU_USD", "SPX500_USD", "BCO_USD"]
        instrument = st.sidebar.selectbox("Instrument", fallback_list, index=0)

    # --- 3. TIMEFRAME & PARAMETERS ---
    granularity = st.sidebar.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W"], index=2)
    total_bars = st.sidebar.slider("Lookback Bars", min_value=500, max_value=5000, value=2000, step=500)

    st.sidebar.subheader("Engine Config")
    horizon = st.sidebar.number_input("Horizon H (bars)", min_value=8, max_value=100, value=24)
    n_min = st.sidebar.number_input("Min Sample n", min_value=5, max_value=100, value=20)
    maj_weight = st.sidebar.slider("MAJOR Level Weight", min_value=2, max_value=8, value=4)

    # --- DATA FETCH & CALCULATIONS ---
    with st.spinner(f"Downloading {total_bars} bars of {instrument}..."):
        df = fetch_oanda_candles(account_type, api_key, instrument, granularity, total_bars)

    if df.empty:
        st.warning("No candle data returned. Verify your API credentials and instrument selection.")
        st.stop()

    df = compute_indicators(df)
    events, projs, levels = process_precedent_engine(df, horizon=horizon, n_min=n_min, maj_weight=maj_weight)

    # --- KPI DISPLAY ---
    latest = df.iloc[-1]
    reg_txt = "TREND" if latest["regime"] == 2 else ("RANGE" if latest["regime"] == 0 else "TRANS")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Symbol", instrument)
    col2.metric("Regime", reg_txt, delta=f"ER Rank: {latest['er_rank']:.1f}%")
    col3.metric("ATR (14)", f"{latest['atr14']:.5f}")
    col4.metric("Active Projections", len(projs))

    # --- PLOTLY CHART ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color=COL_BULL, decreasing_line_color=COL_BEAR
    ))

    for l in levels:
        if l.alive:
            fig.add_shape(
                type="line",
                x0=df["time"].iloc[0], x1=df["time"].iloc[-1],
                y0=l.price, y1=l.price,
                line=dict(color=COL_GOLD if l.weight(len(df)) >= maj_weight else COL_SLATE, width=1, dash="dash")
            )

    ev_names = {1: "SWP", 3: "SQZ", 4: "CLX"}
    for ev in events:
        fig.add_annotation(
            x=ev["time"], y=ev["price"],
            text=f"{ev_names.get(ev['ev_id'], 'EV')}",
            showarrow=True, arrowhead=2,
            arrowcolor=COL_BULL if ev["dir"] > 0 else COL_BEAR,
            ax=0, ay=-25 if ev["dir"] > 0 else 25,
            font=dict(color="#FFFFFF", size=9),
            bgcolor=COL_BULL if ev["dir"] > 0 else COL_BEAR
        )

    if projs:
        last_p = projs[-1]
        s_time = last_p["time"]
        s_idx = df[df["time"] == s_time].index[0]
        fut_idx = min(len(df) - 1, s_idx + horizon)
        e_time = df["time"].iloc[fut_idx]
        anc = last_p["anchor"]
        atr = df["atr14"].iloc[s_idx]

        fig.add_trace(go.Scatter(
            x=[s_time, e_time], y=[last_p["tgt"], last_p["tgt"]],
            mode="lines+text", name=f"Target (Hit: {last_p['hitPct']:.0f}%)",
            line=dict(color=COL_TP, width=2),
            text=["", f"TGT: {last_p['tgt']:.5f}"], textposition="middle right"
        ))

        q75_y = anc + last_p["dir"] * last_p["q75"] * atr
        q25_y = anc + last_p["dir"] * last_p["q25"] * atr
        fig.add_trace(go.Scatter(
            x=[s_time, e_time, e_time, s_time],
            y=[anc, q75_y, q25_y, anc],
            fill="toself",
            fillcolor="rgba(61, 220, 151, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Interquartile Fan"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Projection Log")
    if projs:
        st.dataframe(pd.DataFrame(projs)[["time", "anchor", "dir", "tgt", "hitPct", "n", "lvl"]], use_container_width=True)


if __name__ == "__main__":
    main()