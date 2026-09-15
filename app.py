
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PRECEDENT [ThrowMaster] - Streamlit/OANDA conversion
# ============================================================
# The Pine version is a bar-close measurement instrument. This
# Python version keeps the same event classes, level map, score,
# signature backoff, pending outcomes, forecast ledger, sequence
# statistics, and dashboard concepts.
#
# OANDA data is fetched as completed mid-price candles. Streamlit
# reruns the script, so all state is rebuilt deterministically from
# the downloaded history rather than relying on Pine's var state.
# ============================================================

st.set_page_config(page_title="PRECEDENT [ThrowMaster]", layout="wide")

# ---------- OANDA ----------
LIVE_URL = "https://api-fxtrade.oanda.com"
PRACTICE_URL = "https://api-fxpractice.oanda.com"

EV_NAMES = {
    1: "Sweep",
    2: "Shift",
    3: "Squeeze",
    4: "Climax",
    5: "Reject",
    6: "Diverge",
}

ST_NAMES = {0: "OPEN", 1: "HIT", 2: "ADV", 3: "AMB", 4: "EXP"}


def get_secret(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def get_oanda_credentials():
    # Preferred Community Cloud secrets:
    # OANDA_API_KEY = "..."
    # OANDA_ENV = "practice"
    #
    # The account ID is discovered automatically from the token via
    # GET /v3/accounts, so it is NOT required in Streamlit Secrets.
    token = get_secret("OANDA_API_KEY")
    env = str(get_secret("OANDA_ENV", "practice")).lower()

    if not token:
        token = get_secret("oanda_api_key")

    return token, env


@st.cache_data(ttl=300, show_spinner=False)
def discover_oanda_account(token: str, environment: str):
    """Discover an authorized OANDA account using the API token."""
    base = PRACTICE_URL if environment == "practice" else LIVE_URL
    url = f"{base}/v3/accounts"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    payload = r.json()
    accounts = payload.get("accounts", [])
    if not accounts:
        raise RuntimeError("OANDA returned no accounts authorized for this API token.")

    # Use the first account authorized by the token. The API token can access
    # all sub-accounts belonging to the OANDA user.
    return accounts[0].get("id")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_oanda_instruments(token: str, account_id: str, environment: str):
    """Fetch the complete tradeable instrument list for the authorized OANDA account."""
    base = PRACTICE_URL if environment == "practice" else LIVE_URL
    url = f"{base}/v3/accounts/{account_id}/instruments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    payload = r.json()

    instruments = payload.get("instruments", [])
    if not instruments:
        raise RuntimeError("OANDA returned no tradeable instruments for this account.")

    # Keep both the OANDA instrument name and display name. The dropdown
    # uses the API's actual instrument name so it can be passed directly
    # to the v20 candles endpoint.
    rows = []
    for item in instruments:
        name = item.get("name")
        if not name:
            continue
        rows.append({
            "name": name,
            "displayName": item.get("displayName", name),
            "type": item.get("type", ""),
        })

    rows.sort(key=lambda x: (x["type"], x["displayName"], x["name"]))
    return rows


@st.cache_data(ttl=20, show_spinner=False)
def fetch_oanda_candles(
    instrument: str,
    granularity: str,
    count: int,
    token: str,
    account_id: str,
    environment: str,
):
    base = PRACTICE_URL if environment == "practice" else LIVE_URL
    url = f"{base}/v3/accounts/{account_id}/instruments/{instrument}/candles"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }
    params = {
        "granularity": granularity,
        "count": min(int(count), 5000),
        "price": "M",
        "smooth": "false",
    }

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for c in payload.get("candles", []):
        if not c.get("complete", False):
            continue
        m = c.get("mid", {})
        rows.append(
            {
                "time": pd.to_datetime(c["time"], utc=True),
                "open": float(m["o"]),
                "high": float(m["h"]),
                "low": float(m["l"]),
                "close": float(m["c"]),
                "volume": float(c.get("volume", 0)),
            }
        )

    if not rows:
        raise ValueError("OANDA returned no completed candles.")

    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    return df.reset_index(drop=True)


# ---------- data classes ----------

@dataclass
class Level:
    price: float
    is_swing: bool = False
    is_eq: bool = False
    is_ob: bool = False
    is_htf: bool = False
    fvg_n: int = 0
    touches: int = 0
    birth: int = 0
    alive: bool = True


@dataclass
class Outcome:
    ev: int
    loc: int
    rg: int
    direction: int
    tier: int
    score: float
    e1: float
    e2: float
    e3: float
    e4: float
    mfe: float
    mae: float
    fp1: int
    fp2: int
    fp3: int


@dataclass
class Pending:
    ev: int
    loc: int
    rg: int
    direction: int
    tier: int
    score: float
    sbar: int
    anchor: float
    r: float
    e1: Optional[float] = None
    e2: Optional[float] = None
    e3: Optional[float] = None
    e4: Optional[float] = None
    mfe: float = 0.0
    mae: float = 0.0
    fp1: int = 0
    fp2: int = 0
    fp3: int = 0


@dataclass
class Projection:
    sbar: int
    anchor: float
    r: float
    direction: int
    target: float
    adverse: float
    ring: int
    hit_pct: float
    eta: int
    n: int
    lvl: int
    snapped: bool
    ev: int
    state: int = 0


@dataclass
class RunRec:
    ev: int
    direction: int
    rg: int
    cnt: int
    ext: float


# ---------- helpers ----------

def q(a, p):
    if not a:
        return 0.0
    return float(np.percentile(np.asarray(a, dtype=float), p))


def med(a):
    if not a:
        return 0.0
    return float(np.median(np.asarray(a, dtype=float)))


def percentile_rank(series, window):
    x = pd.Series(series, dtype=float)
    out = np.full(len(x), np.nan)
    vals = x.to_numpy()
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        w = vals[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0 or not np.isfinite(vals[i]):
            continue
        # Pine ta.percentrank is the percentage of values <= current.
        out[i] = 100.0 * np.sum(w <= vals[i]) / len(w)
    return pd.Series(out, index=x.index)


def rolling_percentile_nearest(series, length, pct):
    x = pd.Series(series, dtype=float)
    out = np.full(len(x), np.nan)
    vals = x.to_numpy()
    for i in range(len(vals)):
        lo = max(0, i - length + 1)
        w = vals[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            continue
        # Nearest-rank approximation used by the Pine function.
        k = max(1, int(math.ceil(pct / 100.0 * len(w)))) - 1
        out[i] = float(np.sort(w)[min(k, len(w) - 1)])
    return pd.Series(out, index=x.index)


def atr(df, n):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Pine ta.atr uses Wilder/RMA.
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rma(s, n):
    return pd.Series(s).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def pivot_high(s, left, right):
    a = s.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    for i in range(left + right, len(a)):
        p = i - right
        if p - left < 0:
            continue
        v = a[p]
        if np.isfinite(v) and v == np.max(a[p - left:p + right + 1]):
            # Require the pivot to be strictly greater than the right/left
            # neighborhood where possible, matching the practical Pine use.
            if v > np.max(a[p-left:p]) if left else True:
                pass
            out[i] = v
    return pd.Series(out, index=s.index)


def pivot_low(s, left, right):
    a = s.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    for i in range(left + right, len(a)):
        p = i - right
        if p - left < 0:
            continue
        v = a[p]
        if np.isfinite(v) and v == np.min(a[p - left:p + right + 1]):
            out[i] = v
    return pd.Series(out, index=s.index)


def add_level(levels, px, tol, kind, bi, cap):
    if not np.isfinite(px):
        return
    idx = find_level(levels, px, tol)
    if idx >= 0:
        l = levels[idx]
        if kind == 1:
            l.is_swing = True
        elif kind == 2:
            l.is_eq = True
            l.touches += 1
            l.price = (l.price + px) / 2.0
        elif kind == 3:
            l.fvg_n += 1
        elif kind == 4:
            l.is_ob = True
        elif kind == 5:
            l.is_htf = True
    else:
        levels.append(
            Level(
                price=float(px),
                is_swing=kind == 1,
                is_eq=kind == 2,
                is_ob=kind == 4,
                is_htf=kind == 5,
                fvg_n=1 if kind == 3 else 0,
                touches=1 if kind == 2 else 0,
                birth=bi,
            )
        )
        if len(levels) > cap:
            levels.pop(0)


def find_level(levels, px, tol):
    best = tol
    idx = -1
    for i, l in enumerate(levels):
        if l.alive:
            d = abs(l.price - px)
            if d <= best:
                best = d
                idx = i
    return idx


def level_weight(l, bi):
    w = 0
    w += 1 if l.is_swing else 0
    w += 1 if l.is_eq else 0
    w += 1 if l.is_ob else 0
    w += 2 if l.is_htf else 0
    w += min(l.fvg_n, 3)
    w += min(l.touches, 3)
    w += 1 if (bi - l.birth) > 200 else 0
    return w


def location_weight(levels, px, tol, bi):
    best = 0
    for l in levels:
        if l.alive and abs(l.price - px) <= tol:
            best = max(best, level_weight(l, bi))
    return best


def nearest_above(levels, px, max_d):
    res = np.nan
    best = max_d
    for l in levels:
        d = l.price - px
        if l.alive and d > 0 and d <= best:
            best = d
            res = l.price
    return res


def nearest_below(levels, px, max_d):
    res = np.nan
    best = max_d
    for l in levels:
        d = px - l.price
        if l.alive and d > 0 and d <= best:
            best = d
            res = l.price
    return res


def snap_level(levels, px, tol):
    i = find_level(levels, px, tol)
    return levels[i].price if i >= 0 else np.nan


def auto_htf(granularity):
    ladder = {
        "S5": "M1", "S10": "M1", "S15": "M1", "S30": "M1",
        "M1": "M15", "M2": "M15", "M4": "H1", "M5": "H1",
        "M10": "H1", "M15": "H1", "M30": "H1",
        "H1": "H4", "H2": "H4", "H3": "H4", "H4": "D",
        "H6": "D", "H8": "D", "H12": "W", "D": "W",
        "W": "M", "M": "M",
    }
    return ladder.get(granularity, "H4")


# ---------- HTF context from OANDA ----------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_htf_context(instrument, htf, token, account_id, environment, count=600):
    return fetch_oanda_candles(
        instrument, htf, min(count, 5000), token, account_id, environment
    )


def build_htf_arrays(df, htf_df, htf):
    # For each chart candle, attach the LAST COMPLETED HTF candle.
    # This is the same no-lookahead intent as request.security(... [1],
    # lookahead_on) in the Pine source.
    chart = df.copy()
    h = htf_df.copy()

    h["ema50"] = h["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    h["prev_close"] = h["close"].shift(1)
    h["prev_ema"] = h["ema50"].shift(1)

    # Use completed HTF bar preceding the chart candle's containing HTF bar.
    # Merge-asof with a one-period shifted timestamp accomplishes this.
    h_lookup = h[["time", "close", "ema50"]].copy()
    h_lookup["htf_close_completed"] = h_lookup["close"]
    h_lookup["htf_ema_completed"] = h_lookup["ema50"]
    h_lookup = h_lookup[
        ["time", "htf_close_completed", "htf_ema_completed"]
    ].dropna()

    # The OANDA candle timestamps are period starts. Shift the lookup by one
    # HTF period using the next candle start, so a chart candle maps to the
    # previous completed HTF candle.
    if len(h_lookup) >= 2:
        shifted = h_lookup.copy()
        shifted["time"] = h_lookup["time"].shift(-1)
        shifted = shifted.dropna(subset=["time"])
        chart = pd.merge_asof(
            chart.sort_values("time"),
            shifted.sort_values("time"),
            on="time",
            direction="backward",
        )
    else:
        chart["htf_close_completed"] = np.nan
        chart["htf_ema_completed"] = np.nan

    # Daily/weekly previous high/low.
    daily = fetch_oanda_candles(
        instrument, "D", 500, token, account_id, environment
    )
    weekly = fetch_oanda_candles(
        instrument, "W", 300, token, account_id, environment
    )

    daily["pdH"] = daily["high"].shift(1)
    daily["pdL"] = daily["low"].shift(1)
    daily_lookup = daily[["time", "pdH", "pdL"]].dropna()

    weekly["pwH"] = weekly["high"].shift(1)
    weekly["pwL"] = weekly["low"].shift(1)
    weekly_lookup = weekly[["time", "pwH", "pwL"]].dropna()

    chart = pd.merge_asof(
        chart.sort_values("time"),
        daily_lookup.sort_values("time"),
        on="time",
        direction="backward",
    )
    chart = pd.merge_asof(
        chart.sort_values("time"),
        weekly_lookup.sort_values("time"),
        on="time",
        direction="backward",
    )

    return chart


# ---------- indicator calculations ----------

def prepare_features(df, horizon, piv_ext, piv_int):
    x = df.copy().reset_index(drop=True)

    x["atr14"] = atr(x, 14)
    x["atr20"] = atr(x, 20)
    x["rng"] = x["high"] - x["low"]
    x["body"] = (x["close"] - x["open"]).abs()
    x["wickUp"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["wickDn"] = x[["open", "close"]].min(axis=1) - x["low"]

    x["volSma"] = x["volume"].rolling(50, min_periods=50).mean()
    x["volAvail"] = x["volSma"].gt(0) & x["volSma"].notna()
    x["bodyRatio"] = (x["close"] - x["open"]) / x["rng"].replace(0, np.nan)
    x["flowUnit"] = np.where(
        x["volAvail"],
        x["bodyRatio"] * x["volume"],
        x["bodyRatio"],
    )
    x["cumFlow"] = x["flowUnit"].fillna(0).cumsum()
    x["flowRank"] = percentile_rank(x["flowUnit"].abs(), 200)

    x["wickUpThr"] = rolling_percentile_nearest(x["wickUp"], 100, 75)
    x["wickDnThr"] = rolling_percentile_nearest(x["wickDn"], 100, 75)
    x["volThr"] = rolling_percentile_nearest(x["volume"], 200, 95)
    x["rngThr"] = rolling_percentile_nearest(x["rng"], 200, 90)
    x["dispThr"] = rolling_percentile_nearest(x["body"], 100, 85)

    er_change = (x["close"] - x["close"].shift(20)).abs()
    er_vol = x["close"].diff().abs().rolling(20, min_periods=20).sum()
    x["erRaw"] = (er_change / er_vol.replace(0, np.nan)).fillna(0)
    x["erRank"] = percentile_rank(x["erRaw"], 300)
    x["regime"] = np.select(
        [x["erRank"] >= 70, x["erRank"] <= 40],
        [2, 0],
        default=1,
    )
    x["regTxt"] = x["regime"].map({0: "RANGE", 1: "TRANS", 2: "TREND"})

    # Bollinger inside Keltner.
    x["bbBasis"] = x["close"].rolling(20, min_periods=20).mean()
    x["bbDev"] = 2.0 * x["close"].rolling(20, min_periods=20).std(ddof=1)
    x["bbU"] = x["bbBasis"] + x["bbDev"]
    x["bbL"] = x["bbBasis"] - x["bbDev"]
    x["kcU"] = x["bbBasis"] + 1.5 * x["atr20"]
    x["kcL"] = x["bbBasis"] - 1.5 * x["atr20"]
    x["inSqz"] = (x["bbU"] < x["kcU"]) & (x["bbL"] > x["kcL"])
    sq = []
    n = 0
    for v in x["inSqz"].fillna(False):
        n = n + 1 if v else 0
        sq.append(n)
    x["sqzCnt"] = sq
    x["sqzRelease"] = (~x["inSqz"].fillna(False)) & (
        pd.Series(x["sqzCnt"]).shift(1).fillna(0) >= 5
    )

    # FVGs.
    x["fvgBull"] = (
        (x["low"] > x["high"].shift(2))
        & (x["close"].shift(1) > x["open"].shift(1))
    )
    x["fvgBear"] = (
        (x["high"] < x["low"].shift(2))
        & (x["close"].shift(1) < x["open"].shift(1))
    )
    x["fvgBullMid"] = (x["low"] + x["high"].shift(2)) / 2
    x["fvgBearMid"] = (x["high"] + x["low"].shift(2)) / 2

    x["bullEngulf"] = (
        (x["close"] > x["open"])
        & (x["close"].shift(1) < x["open"].shift(1))
        & (x["close"] >= x["open"].shift(1))
        & (x["open"] <= x["close"].shift(1))
    )
    x["bearEngulf"] = (
        (x["close"] < x["open"])
        & (x["close"].shift(1) > x["open"].shift(1))
        & (x["close"] <= x["open"].shift(1))
        & (x["open"] >= x["close"].shift(1))
    )
    x["bullPin"] = (
        (x["wickDn"] >= x["wickDnThr"])
        & (x["wickDn"] > x["body"] * 2)
        & (x["close"] > (x["high"] + x["low"]) / 2)
    )
    x["bearPin"] = (
        (x["wickUp"] >= x["wickUpThr"])
        & (x["wickUp"] > x["body"] * 2)
        & (x["close"] < (x["high"] + x["low"]) / 2)
    )

    x["phExt"] = pivot_high(x["high"], piv_ext, piv_ext)
    x["plExt"] = pivot_low(x["low"], piv_ext, piv_ext)
    x["phInt"] = pivot_high(x["high"], piv_int, piv_int)
    x["plInt"] = pivot_low(x["low"], piv_int, piv_int)

    x["flowAtPiv"] = x["cumFlow"].shift(piv_ext)
    x["barAtPiv"] = np.arange(len(x)) - piv_ext

    x["htfAvail"] = x["htf_close_completed"].notna() & x["htf_ema_completed"].notna()
    x["htfBias"] = np.select(
        [
            ~x["htfAvail"],
            x["htf_close_completed"] > x["htf_ema_completed"],
            x["htf_close_completed"] < x["htf_ema_completed"],
        ],
        [0, 1, -1],
        default=0,
    )

    return x


# ---------- main sequential engine ----------

def run_engine(
    x,
    horizon,
    n_min,
    piv_ext,
    maj_weight,
    lev_tol_atr,
    max_levels,
):
    levels = []
    store = []
    pending = []
    projections = []
    runs = []
    last_ev = [-99999] * 7
    seq_cnt = [0] * 14

    last_ph = prev_ph = np.nan
    last_pl = prev_pl = np.nan
    last_phf = prev_phf = np.nan
    last_plf = prev_plf = np.nan
    bos_hi = bos_lo = np.nan
    last_phi = last_pli = np.nan

    run_dir = 0
    run_start = 0
    run_st_px = np.nan
    run_st_rg = 1

    led_hit = led_adv = led_amb = led_exp = 0
    led_proj_sum = 0.0
    led_proj_n = 0

    last_fired_ev = 0
    last_fired_dir = 0

    # Current projection display fields.
    dsp = dict(
        n=0, lvl=-1, hit=0.0, eta=0, ring=1, tgt=np.nan,
        ev=0, direction=0, bar=-1, snap=False,
        q50=0.0, q75=0.0, q95=0.0, q25=0.0, q05=0.0,
        tier=0, score=0.0,
    )

    # Per-row diagnostic data.
    event_rows = []

    for i in range(len(x)):
        row = x.iloc[i]
        atr14 = row["atr14"]
        if not np.isfinite(atr14) or atr14 <= 0:
            continue

        lev_tol = lev_tol_atr * atr14
        ph = row["phExt"]
        pl = row["plExt"]

        # ----- level map -----
        if np.isfinite(ph):
            ex = find_level(levels, ph, lev_tol)
            add_level(levels, ph, lev_tol, 1, int(row["barAtPiv"]), max_levels)
            if ex >= 0:
                add_level(levels, ph, lev_tol, 2, int(row["barAtPiv"]), max_levels)
            prev_ph, last_ph = last_ph, ph
            prev_phf, last_phf = last_phf, row["flowAtPiv"]
            bos_hi = ph

        if np.isfinite(pl):
            ex = find_level(levels, pl, lev_tol)
            add_level(levels, pl, lev_tol, 1, int(row["barAtPiv"]), max_levels)
            if ex >= 0:
                add_level(levels, pl, lev_tol, 2, int(row["barAtPiv"]), max_levels)
            prev_pl, last_pl = last_pl, pl
            prev_plf, last_plf = last_plf, row["flowAtPiv"]
            bos_lo = pl

        if np.isfinite(row["phInt"]):
            last_phi = row["phInt"]
        if np.isfinite(row["plInt"]):
            last_pli = row["plInt"]

        if bool(row["fvgBull"]):
            add_level(levels, row["fvgBullMid"], lev_tol, 3, i, max_levels)
        if bool(row["fvgBear"]):
            add_level(levels, row["fvgBearMid"], lev_tol, 3, i, max_levels)

        prev_down = i > 0 and x.iloc[i - 1]["close"] < x.iloc[i - 1]["open"]
        prev_up = i > 0 and x.iloc[i - 1]["close"] > x.iloc[i - 1]["open"]
        ob_low = x.iloc[i - 1]["low"] if i > 0 else np.nan
        ob_high = x.iloc[i - 1]["high"] if i > 0 else np.nan

        disp_up = (
            row["close"] > row["open"]
            and row["body"] >= row["dispThr"]
            and prev_down
            and np.isfinite(last_phi)
            and row["close"] > last_phi
        )
        disp_dn = (
            row["close"] < row["open"]
            and row["body"] >= row["dispThr"]
            and prev_up
            and np.isfinite(last_pli)
            and row["close"] < last_pli
        )
        if disp_up:
            add_level(levels, ob_low, lev_tol, 4, i, max_levels)
        if disp_dn:
            add_level(levels, ob_high, lev_tol, 4, i, max_levels)

        # New day/week. Use UTC date boundaries for deterministic OANDA candles.
        new_day = i > 0 and row["time"].date() != x.iloc[i - 1]["time"].date()
        new_week = i > 0 and row["time"].isocalendar().week != x.iloc[i - 1]["time"].isocalendar().week
        if new_day and np.isfinite(row.get("pdH", np.nan)):
            add_level(levels, row["pdH"], lev_tol, 5, i, max_levels)
            add_level(levels, row["pdL"], lev_tol, 5, i, max_levels)
        if new_week and np.isfinite(row.get("pwH", np.nan)):
            add_level(levels, row["pwH"], lev_tol, 5, i, max_levels)
            add_level(levels, row["pwL"], lev_tol, 5, i, max_levels)

        # ----- events -----
        lvl_above = nearest_above(levels, row["close"], 3.0 * atr14)
        lvl_below = nearest_below(levels, row["close"], 3.0 * atr14)
        loc_w = location_weight(levels, row["close"], lev_tol, i)

        sw_dn = (
            np.isfinite(lvl_above)
            and row["high"] > lvl_above
            and row["close"] < lvl_above
            and row["wickUp"] >= row["wickUpThr"]
        )
        sw_up = (
            np.isfinite(lvl_below)
            and row["low"] < lvl_below
            and row["close"] > lvl_below
            and row["wickDn"] >= row["wickDnThr"]
        )
        sh_up = np.isfinite(bos_hi) and row["close"] > bos_hi
        sh_dn = np.isfinite(bos_lo) and row["close"] < bos_lo

        if bool(row["volAvail"]) and np.isfinite(row["volThr"]):
            clx_on = row["volume"] >= row["volThr"] and row["rng"] >= row["rngThr"]
        else:
            clx_on = row["rng"] >= row["rngThr"] and row["body"] >= row["dispThr"]

        rj_up = (bool(row["bullEngulf"]) or bool(row["bullPin"])) and loc_w >= 1
        rj_dn = (bool(row["bearEngulf"]) or bool(row["bearPin"])) and loc_w >= 1

        dv_up = (
            np.isfinite(pl) and np.isfinite(prev_pl) and np.isfinite(prev_plf)
            and np.isfinite(last_pl) and np.isfinite(last_plf)
            and last_pl < prev_pl and last_plf > prev_plf
        )
        dv_dn = (
            np.isfinite(ph) and np.isfinite(prev_ph) and np.isfinite(prev_phf)
            and np.isfinite(last_ph) and np.isfinite(last_phf)
            and last_ph > prev_ph and last_phf < prev_phf
        )

        ev_id = 0
        ev_dir = 0
        if sw_up:
            ev_id, ev_dir = 1, 1
        elif sw_dn:
            ev_id, ev_dir = 1, -1
        elif dv_up:
            ev_id, ev_dir = 6, 1
        elif dv_dn:
            ev_id, ev_dir = 6, -1
        elif sh_up:
            ev_id, ev_dir = 2, 1
        elif sh_dn:
            ev_id, ev_dir = 2, -1
        elif clx_on:
            ev_id, ev_dir = 4, (1 if row["close"] > row["open"] else -1)
        elif bool(row["sqzRelease"]):
            ev_id, ev_dir = 3, (1 if row["close"] > row["bbBasis"] else -1)
        elif rj_up:
            ev_id, ev_dir = 5, 1
        elif rj_dn:
            ev_id, ev_dir = 5, -1

        if sh_up:
            bos_hi = np.nan
        if sh_dn:
            bos_lo = np.nan

        cal_ready = (
            i > 300
            and np.isfinite(row["erRank"])
            and np.isfinite(row["wickUpThr"])
        )
        sep_ok = ev_id > 0 and (i - last_ev[ev_id]) >= horizon
        ev_fired = ev_id > 0 and sep_ok and cal_ready

        # ----- score -----
        s_up_cnt = (
            (1 if np.isfinite(last_ph) and np.isfinite(prev_ph) and last_ph > prev_ph else 0)
            + (1 if np.isfinite(last_pl) and np.isfinite(prev_pl) and last_pl > prev_pl else 0)
        )
        s_dn_cnt = (
            (1 if np.isfinite(last_ph) and np.isfinite(prev_ph) and last_ph < prev_ph else 0)
            + (1 if np.isfinite(last_pl) and np.isfinite(prev_pl) and last_pl < prev_pl else 0)
        )
        s_score = s_up_cnt * 50.0 if ev_dir > 0 else s_dn_cnt * 50.0 if ev_dir < 0 else 0.0

        f_agree = (ev_dir > 0 and row["flowUnit"] > 0) or (
            ev_dir < 0 and row["flowUnit"] < 0
        )
        f_score = (
            row["flowRank"] if f_agree else row["flowRank"] * 0.2
        )
        h_score = (
            50.0 if row["htfBias"] == 0
            else 100.0 if row["htfBias"] == ev_dir
            else 0.0
        )

        f_half = 0.5 if ev_id in (4, 6) else 1.0
        w_fe = 30.0 * f_half
        spare = 30.0 - w_fe
        w_se = 35.0 + spare / 2
        w_he = 35.0 + spare / 2 if bool(row["htfAvail"]) else 0.0

        d_s = 0.70 if row["regime"] == 0 else 1.0
        d_f = 0.85 if row["regime"] == 2 else 1.0
        d_h = 0.80 if row["regime"] == 1 else 1.0

        den = w_se * d_s + w_fe * d_f + w_he * d_h
        score = (
            (w_se * d_s * s_score + w_fe * d_f * f_score + w_he * d_h * h_score) / den
            if den > 0 else 0.0
        )

        loc_cls = 2 if loc_w >= maj_weight else 1 if loc_w >= 1 else 0

        cp1 = max(1, round(horizon / 4))
        cp2 = max(2, round(horizon / 2))
        cp3 = max(3, round(horizon * 3 / 4))
        cp4 = horizon

        # ----- run tracker -----
        cur_seq = 0
        if ev_fired and ev_id == 2 and ev_dir != run_dir:
            if run_dir != 0:
                ext_v = (
                    0.0
                    if not np.isfinite(run_st_px)
                    else abs(row["close"] - run_st_px) / atr14
                )
                for c in range(1, 7):
                    for d in range(2):
                        cnt = seq_cnt[c * 2 + d]
                        if cnt > 0:
                            runs.append(
                                RunRec(c, 1 if d == 1 else -1, run_st_rg, cnt, ext_v)
                            )
                if len(runs) > 400:
                    runs.pop(0)

            seq_cnt = [0] * 14
            run_dir = ev_dir
            run_start = i
            run_st_px = row["close"]
            run_st_rg = int(row["regime"])

        if ev_fired:
            d_ix = 1 if ev_dir > 0 else 0
            s_ix = ev_id * 2 + d_ix
            if 0 <= s_ix < len(seq_cnt):
                seq_cnt[s_ix] += 1
                cur_seq = seq_cnt[s_ix]
            last_ev[ev_id] = i

        run_ext = (
            0.0
            if not np.isfinite(run_st_px)
            else abs(row["close"] - run_st_px) / atr14
        )

        # ----- issue projection -----
        if ev_fired:
            sc_arr = [o.score for o in store if o.ev == ev_id]
            tier_v = 0
            if len(sc_arr) >= 10:
                tier_v = 1 if score >= q(sc_arr, 60) else 0

            i0 = []
            i1 = []
            i2 = []
            i3 = []
            for idx, o in enumerate(store):
                if o.ev == ev_id and o.direction == ev_dir:
                    i0.append(idx)
                    if o.rg == int(row["regime"]):
                        i1.append(idx)
                        if o.loc == loc_cls:
                            i2.append(idx)
                            if o.tier == tier_v:
                                i3.append(idx)

            sel = []
            lvl_used = -1
            if len(i3) >= n_min:
                sel, lvl_used = i3, 3
            elif len(i2) >= n_min:
                sel, lvl_used = i2, 2
            elif len(i1) >= n_min:
                sel, lvl_used = i1, 1
            elif len(i0) >= n_min:
                sel, lvl_used = i0, 0

            pending.append(
                Pending(
                    ev_id, loc_cls, int(row["regime"]), ev_dir, tier_v,
                    float(score), i, float(row["close"]), float(atr14)
                )
            )

            if lvl_used >= 0:
                selected = [store[j] for j in sel]
                a1 = [o.e1 for o in selected]
                a2 = [o.e2 for o in selected]
                a3 = [o.e3 for o in selected]
                a4 = [o.e4 for o in selected]
                a_m = [o.mfe for o in selected]

                d_tgt = max(0.5, med(a_m))
                ring = max(1, min(3, int(round(d_tgt))))

                fp_vals = []
                for o in selected:
                    fpv = o.fp1 if ring == 1 else o.fp2 if ring == 2 else o.fp3
                    if fpv > 0:
                        fp_vals.append(fpv)

                hit_pct = 100.0 * len(fp_vals) / len(selected) if selected else 0.0
                eta_v = int(round(med(fp_vals))) if fp_vals else 0

                tgt_raw = row["close"] + ev_dir * d_tgt * atr14
                snap_px = snap_level(levels, tgt_raw, 0.5 * atr14)
                snapped = np.isfinite(snap_px)
                tgt_px = snap_px if snapped else tgt_raw
                adv_px = row["close"] - ev_dir * atr14

                projections.append(
                    Projection(
                        i, float(row["close"]), float(atr14), ev_dir,
                        float(tgt_px), float(adv_px), ring, hit_pct,
                        eta_v, len(selected), lvl_used, snapped, ev_id
                    )
                )
                if len(projections) > 40:
                    projections.pop(0)

                led_proj_sum += hit_pct
                led_proj_n += 1

                dsp.update(
                    n=len(selected), lvl=lvl_used, hit=hit_pct, eta=eta_v,
                    ring=ring, tgt=float(tgt_px), ev=ev_id, direction=ev_dir,
                    bar=i, snap=snapped, q50=q(a4, 50), q75=q(a4, 75),
                    q95=q(a4, 95), q25=q(a4, 25), q05=q(a4, 5),
                    tier=tier_v, score=float(score)
                )

            last_fired_ev = ev_id
            last_fired_dir = ev_dir

        # ----- update pending outcomes -----
        for p in pending[:]:
            age = i - p.sbar
            if p.r > 0:
                fav = (
                    (row["high"] - p.anchor) if p.direction > 0
                    else (p.anchor - row["low"])
                ) / p.r
                adv = (
                    (p.anchor - row["low"]) if p.direction > 0
                    else (row["high"] - p.anchor)
                ) / p.r

                p.mfe = max(p.mfe, fav)
                p.mae = max(p.mae, adv)

                if p.fp1 == 0 and p.mfe >= 1:
                    p.fp1 = age
                if p.fp2 == 0 and p.mfe >= 2:
                    p.fp2 = age
                if p.fp3 == 0 and p.mfe >= 3:
                    p.fp3 = age

                exc = (row["close"] - p.anchor) * p.direction / p.r
                if age == cp1:
                    p.e1 = exc
                if age == cp2:
                    p.e2 = exc
                if age == cp3:
                    p.e3 = exc
                if age >= cp4:
                    p.e4 = exc
                    store.append(
                        Outcome(
                            p.ev, p.loc, p.rg, p.direction, p.tier, p.score,
                            float(p.e1 or 0), float(p.e2 or 0),
                            float(p.e3 or 0), float(p.e4 or 0),
                            p.mfe, p.mae, p.fp1, p.fp2, p.fp3
                        )
                    )
                    if len(store) > 1500:
                        store.pop(0)
                    pending.remove(p)

        # ----- projection ledger -----
        for pr in projections:
            if pr.state == 0 and i > pr.sbar:
                hit_t = row["high"] >= pr.target if pr.direction > 0 else row["low"] <= pr.target
                hit_a = row["low"] <= pr.adverse if pr.direction > 0 else row["high"] >= pr.adverse

                if hit_t and hit_a:
                    pr.state = 3
                elif hit_t:
                    pr.state = 1
                elif hit_a:
                    pr.state = 2
                elif i - pr.sbar >= cp4:
                    pr.state = 4

                if pr.state != 0:
                    if pr.state == 1:
                        led_hit += 1
                    elif pr.state == 2:
                        led_adv += 1
                    elif pr.state == 3:
                        led_amb += 1
                    elif pr.state == 4:
                        led_exp += 1

        event_rows.append(
            {
                "time": row["time"],
                "event": EV_NAMES.get(ev_id, ""),
                "direction": "UP" if ev_dir > 0 else "DOWN" if ev_dir < 0 else "",
                "regime": row["regTxt"],
                "score": float(score),
                "location": "MAJOR" if loc_cls == 2 else "MINOR" if loc_cls == 1 else "NONE",
                "loc_w": loc_w,
                "sequence": cur_seq,
            }
        )

    # ----- last-bar stats -----
    tier_a = [o.e4 for o in store if o.tier == 1]
    tier_b = [o.e4 for o in store if o.tier != 1]
    loc_maj = [o.e4 for o in store if o.loc == 2]
    loc_min = [o.e4 for o in store if o.loc == 1]
    loc_none = [o.e4 for o in store if o.loc == 0]

    seq_vals = [
        float(r.cnt)
        for r in runs
        if r.ev == last_fired_ev and r.direction == last_fired_dir
    ]
    seq_b = [0, 0, 0, 0]
    if seq_vals:
        for v in seq_vals:
            if v <= 1:
                seq_b[0] += 1
            elif v <= 2:
                seq_b[1] += 1
            elif v <= 3:
                seq_b[2] += 1
            else:
                seq_b[3] += 1
        seq_b = [100 * v / len(seq_vals) for v in seq_b]

    live_open = False
    live_eta = 0
    if projections:
        pr = projections[-1]
        if pr.state == 0:
            live_open = True
            live_eta = max(0, pr.eta - (len(x) - 1 - pr.sbar))

    last_row = x.iloc[-1]
    calibration_real = (
        100 * led_hit / (led_hit + led_adv + led_exp)
        if (led_hit + led_adv + led_exp) > 0 else 0.0
    )
    calibration_proj = led_proj_sum / led_proj_n if led_proj_n else 0.0

    return {
        "levels": levels,
        "store": store,
        "pending": pending,
        "projections": projections,
        "runs": runs,
        "dsp": dsp,
        "last_fired_ev": last_fired_ev,
        "last_fired_dir": last_fired_dir,
        "last_row": last_row,
        "loc_w": location_weight(levels, last_row["close"], lev_tol_atr * last_row["atr14"], len(x)-1)
                  if np.isfinite(last_row["atr14"]) else 0,
        "tier_a": med(tier_a), "tier_b": med(tier_b),
        "tier_na": len(tier_a), "tier_nb": len(tier_b),
        "loc_maj": med(loc_maj), "loc_min": med(loc_min), "loc_none": med(loc_none),
        "loc_nmaj": len(loc_maj), "loc_nmin": len(loc_min), "loc_nnone": len(loc_none),
        "seq_med": med(seq_vals), "seq_runs": len(seq_vals), "seq_b": seq_b,
        "run_ext": (
            0.0 if not np.isfinite(run_st_px) else abs(last_row["close"] - run_st_px) / last_row["atr14"]
        ),
        "led_hit": led_hit, "led_adv": led_adv, "led_amb": led_amb, "led_exp": led_exp,
        "calibration_real": calibration_real, "calibration_proj": calibration_proj,
        "events": pd.DataFrame(event_rows),
    }


# ---------- chart ----------

def make_chart(df, result, show_levels=True, show_events=True, show_target=True):
    d = df.tail(500).copy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=d["time"],
                open=d["open"], high=d["high"],
                low=d["low"], close=d["close"],
                name="OANDA",
            )
        ]
    )

    levels = result["levels"]
    close = float(df.iloc[-1]["close"])
    atr14 = float(df.iloc[-1]["atr14"]) if np.isfinite(df.iloc[-1]["atr14"]) else 0

    if show_levels and atr14 > 0:
        drawn = 0
        for l in reversed(levels):
            if drawn >= 14:
                break
            if l.alive and abs(l.price - close) <= 6 * atr14:
                w = level_weight(l, len(df)-1)
                major = w >= MAJ_WEIGHT
                fig.add_hline(
                    y=l.price,
                    line_width=2 if major else 1,
                    line_dash="solid" if major else "dot",
                    opacity=0.65 if major else 0.35,
                    annotation_text=f"L {l.price:.5f} w{w}" if major else None,
                )
                drawn += 1

    if show_events and not result["events"].empty:
        ev = result["events"]
        ev = ev[ev["event"] != ""].tail(100)
        if not ev.empty:
            up = ev[ev["direction"] == "UP"]
            dn = ev[ev["direction"] == "DOWN"]
            if not up.empty:
                yy = d.set_index("time").reindex(up["time"])["low"].to_numpy()
                fig.add_trace(
                    go.Scatter(
                        x=up["time"], y=yy,
                        mode="markers+text",
                        text=up["event"],
                        textposition="bottom center",
                        marker_symbol="triangle-up",
                        marker_size=9,
                        name="Events UP",
                    )
                )
            if not dn.empty:
                yy = d.set_index("time").reindex(dn["time"])["high"].to_numpy()
                fig.add_trace(
                    go.Scatter(
                        x=dn["time"], y=yy,
                        mode="markers+text",
                        text=dn["event"],
                        textposition="top center",
                        marker_symbol="triangle-down",
                        marker_size=9,
                        name="Events DOWN",
                    )
                )

    if show_target and np.isfinite(result["dsp"]["tgt"]):
        fig.add_hline(
            y=result["dsp"]["tgt"],
            line_width=2,
            opacity=0.9,
            annotation_text=(
                f"TARGET {result['dsp']['tgt']:.5f} | "
                f"hit {result['dsp']['hit']:.0f}% | n={result['dsp']['n']}"
            ),
        )

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


# ---------- UI ----------

st.title("PRECEDENT [ThrowMaster]")
st.caption(
    "Measurement instrument — OANDA completed candles, empirical precedent, "
    "no forward-looking HTF values."
)

token, environment = get_oanda_credentials()

# Secrets can also be nested under [oanda].
try:
    if "oanda" in st.secrets:
        token = st.secrets["oanda"].get("api_key", token)
        environment = st.secrets["oanda"].get("environment", environment)
except Exception:
    pass

if not token:
    st.error("Missing OANDA_API_KEY in Streamlit Community Cloud Secrets.")
    st.code(
        '[oanda]\n'
        'api_key = "YOUR_OANDA_TOKEN"\n'
        'environment = "live"\n'
    )
    st.stop()

try:
    with st.spinner("Authenticating with OANDA..."):
        account_id = discover_oanda_account(token, environment)
except Exception as exc:
    st.error(f"Could not authenticate with OANDA: {exc}")
    st.stop()

try:
    with st.spinner("Loading OANDA instruments..."):
        oanda_instruments = fetch_oanda_instruments(token, account_id, environment)
except Exception as exc:
    st.error(f"Could not load OANDA instruments: {exc}")
    st.stop()

instrument_names = [x["name"] for x in oanda_instruments]
instrument_labels = {
    x["name"]: f"{x['displayName']} ({x['name']})"
    for x in oanda_instruments
}

with st.sidebar:
    st.header("Instrument")
    instrument = st.selectbox(
        "OANDA instrument",
        instrument_names,
        index=instrument_names.index("EUR_USD") if "EUR_USD" in instrument_names else 0,
        format_func=lambda x: instrument_labels.get(x, x),
    )
    granularity = st.selectbox(
        "Chart timeframe",
        ["M5", "M15", "M30", "H1", "H4", "D"],
        index=3,
    )

    st.header("Engine")
    horizon = st.slider("Horizon H (bars)", 8, 100, 24)
    n_min = st.slider("Minimum sample n", 5, 200, 20)
    piv_ext = st.slider("External pivot length", 3, 60, 21)
    piv_int = st.slider("Internal pivot length", 2, 30, 5)

    st.header("Level Map")
    MAJ_WEIGHT = st.slider("Major level weight threshold", 2, 12, 4)
    lev_tol_atr = st.slider("Level merge tolerance (ATR)", 0.05, 1.0, 0.25, 0.05)
    max_levels = st.slider("Maximum tracked levels", 20, 200, 60)

    st.header("HTF")
    htf_mode = st.radio("HTF selection", ["Auto", "Manual"], horizontal=True)
    htf_manual = st.selectbox(
        "Manual HTF",
        ["M15", "H1", "H4", "D", "W", "M"],
        index=2,
        disabled=htf_mode != "Manual",
    )

    st.header("Display")
    show_levels = st.checkbox("Show level map", True)
    show_events = st.checkbox("Show event markers", True)
    show_target = st.checkbox("Show target", True)
    compact = st.checkbox("Compact dashboard", False)
    history_count = st.slider(
        "OANDA candles to download", 1000, 5000, 5000, 500
    )

    refresh = st.button("Refresh OANDA data", use_container_width=True)

if refresh:
    st.cache_data.clear()

htf_used = auto_htf(granularity) if htf_mode == "Auto" else htf_manual

try:
    with st.spinner(f"Downloading {instrument} {granularity} from OANDA..."):
        df = fetch_oanda_candles(
            instrument, granularity, history_count,
            token, account_id, environment
        )

        htf_df = fetch_htf_context(
            instrument, htf_used, token, account_id, environment
        )
        df = build_htf_arrays(
            df, htf_df, htf_used
        )

    with st.spinner("Calculating precedent engine..."):
        df = prepare_features(df, horizon, piv_ext, piv_int)
        result = run_engine(
            df, horizon, n_min, piv_ext, MAJ_WEIGHT,
            lev_tol_atr, max_levels
        )

except requests.HTTPError as e:
    st.error(f"OANDA API error: {e}")
    try:
        st.json(e.response.json())
    except Exception:
        pass
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

last = result["last_row"]
dsp = result["dsp"]

# ---------- dashboard ----------
status_cal = len(result["store"]) < n_min or len(df) <= 300

cols = st.columns(5)
cols[0].metric("Instrument", instrument)
cols[1].metric("Timeframe", granularity)
cols[2].metric("HTF", f"{htf_used} ({htf_mode.lower()})")
cols[3].metric("Samples", f"{len(result['store'])} / {n_min}")
cols[4].metric("Regime", last["regTxt"])

event_name = EV_NAMES.get(result["last_fired_ev"], "none")
direction = "UP" if result["last_fired_dir"] > 0 else "DOWN" if result["last_fired_dir"] < 0 else ""
loc_w = result["loc_w"]
loc_txt = "MAJOR" if loc_w >= MAJ_WEIGHT else "MINOR" if loc_w >= 1 else "NONE"

st.divider()

dash_rows = [
    ("STATUS", "CALIBRATING" if status_cal else "ACTIVE"),
    ("EVENT", f"{event_name} {direction}"),
    ("LOCATION", f"{loc_txt}  w{loc_w}"),
    ("REGIME", f"{last['regTxt']}  score {dsp['score']:.0f}  {'A' if dsp['tier'] == 1 else 'B'}"),
    ("SIGNATURE", f"L{dsp['lvl']}  n={dsp['n']}" if dsp["lvl"] >= 0 else "-"),
    ("TARGET", f"{dsp['tgt']:.6f}  hit {dsp['hit']:.0f}%" if np.isfinite(dsp["tgt"]) else "-"),
    ("ETA", f"{max(0, dsp['eta'])} bars (+{dsp['ring']}R)" if dsp["n"] else "no open projection"),
    ("RUN EXT", f"{result['run_ext']:.1f} R"),
    ("FLOW SOURCE", "volume" if bool(last["volAvail"]) else "proxy"),
    ("LEDGER", f"hit {result['led_hit']}  adv {result['led_adv']}  amb {result['led_amb']}  exp {result['led_exp']}"),
    ("CALIBRATION", f"projected {result['calibration_proj']:.0f}%  realised {result['calibration_real']:.0f}%"),
]

if not compact:
    dash_rows.insert(5, ("WITH q50/q75/q95", f"{dsp['q50']:.1f} / {dsp['q75']:.1f} / {dsp['q95']:.1f} R"))
    dash_rows.insert(6, ("AGAINST q25/q05", f"{dsp['q25']:.1f} / {dsp['q05']:.1f} R"))
    if result["seq_runs"] >= 10:
        b = result["seq_b"]
        dash_rows.append(("SEQ 1/2/3/4+", f"{b[0]:.0f} / {b[1]:.0f} / {b[2]:.0f} / {b[3]:.0f}%"))
        dash_rows.append(("RUNS MEDIAN", f"{result['seq_med']:.1f}  n={result['seq_runs']}"))
    else:
        dash_rows.append(("SEQUENCE", f"calibrating {result['seq_runs']} / 10 runs"))

    ax_ok = (
        abs(result["tier_a"] - result["tier_b"]) >= 0.3
        and result["tier_na"] >= 15
        and result["tier_nb"] >= 15
    )
    dash_rows.append(
        ("TIER A / B",
         f"{result['tier_a']:.1f} R ({result['tier_na']}) / "
         f"{result['tier_b']:.1f} R ({result['tier_nb']})")
    )
    dash_rows.append(
        ("LOC MAJ/MIN/NONE",
         f"{result['loc_maj']:.1f} / {result['loc_min']:.1f} / {result['loc_none']:.1f} R")
    )

dash_df = pd.DataFrame(dash_rows, columns=["Metric", "Value"])
left, right = st.columns([1, 2])
with left:
    st.dataframe(dash_df, use_container_width=True, hide_index=True)

with right:
    st.plotly_chart(
        make_chart(df, result, show_levels, show_events, show_target),
        use_container_width=True,
        config={"displaylogo": False},
    )

# ---------- supporting panels ----------
st.subheader("Latest event history")
events = result["events"]
if events.empty:
    st.info("No events have fired yet.")
else:
    st.dataframe(
        events.tail(30).sort_values("time", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Resolved forecast ledger")
ledger_rows = []
for p in result["projections"][-40:]:
    ledger_rows.append(
        {
            "issued": df.iloc[p.sbar]["time"],
            "event": EV_NAMES.get(p.ev, str(p.ev)),
            "direction": "UP" if p.direction > 0 else "DOWN",
            "target": p.target,
            "adverse": p.adverse,
            "ring": p.ring,
            "hit_%": p.hit_pct,
            "n": p.n,
            "level": p.lvl,
            "state": ST_NAMES.get(p.state, "?"),
        }
    )
if ledger_rows:
    st.dataframe(pd.DataFrame(ledger_rows).sort_values("issued", ascending=False),
                 use_container_width=True, hide_index=True)
else:
    st.info("No projections have been issued yet.")

st.caption(
    f"Last completed OANDA candle: {last['time']} UTC | "
    f"close={last['close']:.6f} | downloaded={len(df)} candles | "
    f"data environment={environment}"
)
