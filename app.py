import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from typing import List, Dict

# ==============================================================================
# STREAMLIT PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="PRECEDENT [ThrowMaster] - OANDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

COL_BULL = "#0ABAB5"
COL_BEAR = "#FF4D6D"
COL_TP   = "#3DDC97"
COL_GOLD = "#D4AF37"
COL_SLATE= "#7A8B99"

# ==============================================================================
# 1. OANDA API DATA FETCHING
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_oanda_instruments(account_type: str, api_key: str) -> List[Dict[str, str]]:
    base_url = "https://api-fxpractice.oanda.com" if account_type == "Practice" else "https://api-fxtrade.oanda.com"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        acc_res = requests.get(f"{base_url}/v3/accounts", headers=headers, timeout=10)
        if acc_res.status_code != 200:
            return []
        accounts = acc_res.json().get("accounts", [])
        if not accounts:
            return []

        account_id = accounts[0]["id"]
        inst_res = requests.get(f"{base_url}/v3/accounts/{account_id}/instruments", headers=headers, timeout=10)
        if inst_res.status_code != 200:
            return []

        raw_instruments = inst_res.json().get("instruments", [])
        instruments = [{"name": item["name"], "displayName": item.get("displayName", item["name"])} for item in raw_instruments]
        instruments.sort(key=lambda x: x["displayName"])
        return instruments
    except Exception:
        return []


def fetch_oanda_candles(account_type: str, api_key: str, instrument: str, granularity: str, total_bars: int = 2000) -> pd.DataFrame:
    base_url = "https://api-fxpractice.oanda.com" if account_type == "Practice" else "https://api-fxtrade.oanda.com"
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

    return pd.DataFrame(records).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)


# ==============================================================================
# 2. INDICATORS & ENGINE
# ==============================================================================
def compute_indicators(df: pd.DataFrame, cal_win: int = 300) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr20"] = df["tr"].rolling(20).mean()

    df["rng"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["wick_up"] = df["high"] - np.maximum(df["open"], df["close"])
    df["wick_dn"] = np.minimum(df["open"], df["close"]) - df["low"]

    df["wick_up_thr"] = df["wick_up"].rolling(100).quantile(0.75)
    df["wick_dn_thr"] = df["wick_dn"].rolling(100).quantile(0.75)
    df["rng_thr"] = df["rng"].rolling(200).quantile(0.90)
    df["disp_thr"] = df["body"].rolling(100).quantile(0.85)

    er_change = (df["close"] - df["close"].shift(20)).abs()
    er_vol = (df["close"] - df["close"].shift(1)).abs().rolling(20).sum()
    er_raw = np.where(er_vol > 0, er_change / er_vol, 0.0)

    er_series = pd.Series(er_raw)
    er_rank = er_series.rolling(cal_win).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False
    )
    df["er_rank"] = er_rank
    df["regime"] = np.select([er_rank >= 70, er_rank <= 40], [2, 0], default=1)

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


def process_precedent_engine(df: pd.DataFrame, horizon: int = 24, n_min: int = 20, maj_weight: int = 4):
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

        loc_cls = 1

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

    return events, projs, store


# ==============================================================================
# 3. STREAMLIT UI & DIAGNOSTIC TABLE
# ==============================================================================
def main():
    st.sidebar.title("PRECEDENT Engine")
    account_type = st.sidebar.selectbox("Account Type", ["Practice", "Live"])

    secret_key = st.secrets.get("OANDA_API_KEY") or st.secrets.get("OANDA_TOKEN")
    api_key = secret_key if secret_key else st.sidebar.text_input("OANDA API Key", type="password")

    if not api_key:
        st.info("Enter your OANDA API Key or set `OANDA_API_KEY` in Streamlit Community Cloud Secrets.")
        st.stop()

    instruments = fetch_oanda_instruments(account_type, api_key)
    if instruments:
        inst_names = [i["name"] for i in instruments]
        instrument = st.sidebar.selectbox("Instrument", options=inst_names, index=inst_names.index("EUR_USD") if "EUR_USD" in inst_names else 0)
    else:
        instrument = st.sidebar.selectbox("Instrument", ["EUR_USD", "GBP_USD", "USD_JPY"])

    granularity = st.sidebar.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W"], index=2)
    total_bars = st.sidebar.slider("Lookback Bars", min_value=500, max_value=5000, value=2000, step=500)

    df = fetch_oanda_candles(account_type, api_key, instrument, granularity, total_bars)
    if df.empty:
        st.warning("No candle data returned.")
        st.stop()

    df = compute_indicators(df)
    events, projs, store = process_precedent_engine(df)

    latest = df.iloc[-1]
    reg_txt = "TREND" if latest["regime"] == 2 else ("RANGE" if latest["regime"] == 0 else "TRANS")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Instrument", instrument)
    col2.metric("Regime", reg_txt, delta=f"ER Rank: {latest['er_rank']:.1f}%")
    col3.metric("ATR (14)", f"{latest['atr14']:.5f}")
    col4.metric("Completed Outcomes", len(store))

    # Plotly Chart
    fig = go.Figure(data=[go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # PRECEDENT DIAGNOSTIC TABLE (AXIS HEALTH & OUTCOME LEDGER)
    st.subheader("PRECEDENT Diagnostic & Outcome Ledger")

    if store:
        df_store = pd.DataFrame(store)
        ev_map = {1: "Sweep", 3: "Squeeze", 4: "Climax"}
        rg_map = {0: "RANGE", 1: "TRANS", 2: "TREND"}

        summary = []
        for (ev_id, rg), group in df_store.groupby(["ev", "rg"]):
            n = len(group)
            med_mfe = group["mfe"].median()
            med_mae = group["mae"].median()
            exp_val = group["e4"].mean()
            win_rate = (group["e4"] > 0).mean() * 100

            summary.append({
                "Event Type": ev_map.get(ev_id, f"EV_{ev_id}"),
                "Regime": rg_map.get(rg, "N/A"),
                "Sample Count (n)": n,
                "Win Rate (%)": f"{win_rate:.1f}%",
                "Median MFE (ATR)": f"{med_mfe:.2f}R",
                "Median MAE (ATR)": f"{med_mae:.2f}R",
                "Expectancy E[R]": f"{exp_val:+.2f}R"
            })

        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    else:
        st.info("Accumulating outcome data... Check back after more historical bars complete.")


if __name__ == "__main__":
    main()