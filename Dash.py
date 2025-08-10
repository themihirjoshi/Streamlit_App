# -*- coding: utf-8 -*-
import re, time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import cloudscraper
from bs4 import BeautifulSoup

# ---------------- Config ----------------
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_SLEEP = 2.0
YF_THREADS = True

INDEX_URLS = {
    "Nifty Auto":              "https://www.screener.in/company/CNXAUTO/#constituents",
    "Nifty Bank":              "https://www.screener.in/company/BANKNIFTY/#constituents",
    "Nifty Financial Services":"https://www.screener.in/company/CNXFINANCE/#constituents",
    "Nifty FMCG":              "https://www.screener.in/company/CNXFMCG/#constituents",
    "Nifty IT":                "https://www.screener.in/company/CNXIT/#constituents",
    "Nifty Media":             "https://www.screener.in/company/CNXMEDIA/#constituents",
    "Nifty Metal":             "https://www.screener.in/company/CNXMETAL/#constituents",
    "Nifty Pharma":            "https://www.screener.in/company/CNXPHARMA/#constituents",
    "Nifty Realty":            "https://www.screener.in/company/CNXREALTY/#constituents",
    "Nifty Commodities":       "https://www.screener.in/company/CNXCOMMOD/#constituents",
    "Nifty CPSE":              "https://www.screener.in/company/CNXCPSE/#constituents",
    "Nifty Energy":            "https://www.screener.in/company/CNXENERGY/#constituents",
    "Nifty PSE":               "https://www.screener.in/company/CNXPSE/#constituents",
    "Nifty Consumption":       "https://www.screener.in/company/CNXCONSUMPTION/#constituents",
    "Nifty Infrastructure":    "https://www.screener.in/company/CNXINFRA/#constituents",
    "Nifty MNC":               "https://www.screener.in/company/CNXMNC/#constituents",
    "Nifty Private Bank":      "https://www.screener.in/company/NIFPVTBANK/#constituents",
    "Nifty PSU Bank":          "https://www.screener.in/company/CNXPSUBANK/#constituents",
    "Nifty Services Sector":   "https://www.screener.in/company/NIFSERVICE/#constituents",
    "Nifty Alpha 50":          "https://www.screener.in/company/NIFTYALPHA50/#constituents",
}

# ---------------- HTTP ----------------
@st.cache_resource(show_spinner=False)
def _scraper():
    s = cloudscraper.create_scraper(browser={"browser":"chrome","platform":"windows","mobile":False})
    s.headers.update({"Referer":"https://www.screener.in/"})
    return s

# ---------------- Helpers ----------------
def _resolve_index(name_or_url: str) -> str:
    norm = re.sub(r"\s+", " ", name_or_url.strip()).casefold()
    for k, v in INDEX_URLS.items():
        if re.sub(r"\s+", " ", k.strip()).casefold() == norm:
            return v
    return name_or_url

def y_ticker(code: str) -> str: return f"{code}.NS"

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------- Screener constituents ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_constituents(name_or_url: str) -> pd.DataFrame:
    url = _resolve_index(name_or_url)
    if "#constituents" not in url:
        url = url.rstrip("/") + "/#constituents"
    html = _scraper().get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#constituents table") or soup.find("table")
    if table is None:
        return pd.DataFrame(columns=["Name","NSE Code","MktCapCr"])
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    name_col = next((h for h in headers if h.lower().startswith("name")), "Name")
    mcap_col = next((h for h in headers if "cap" in h.lower()), None)

    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds: continue
        row = dict(zip(headers, tds))
        name_text = row.get(name_col, "")
        if "Median" in name_text: continue
        a = tr.find("a", href=re.compile(r"^/company/"))
        code = None
        if a and a.has_attr("href"):
            m = re.search(r"/company/([^/]+)/", a["href"])
            if m: code = m.group(1).upper()
        mcap_val = None
        if mcap_col and mcap_col in row:
            s = (row[mcap_col].replace(",", "").replace("%", "")
                 .replace("₹", "").replace("Rs.", "")
                 .replace("Cr.", "").strip())
            try: mcap_val = float(s)
            except: mcap_val = None
        rows.append({"Name": name_text, "NSE Code": code, "MktCapCr": mcap_val})
    return pd.DataFrame(rows).dropna(subset=["NSE Code"]).reset_index(drop=True)

# ---------------- Prices (batched) ----------------
@st.cache_data(ttl=1800, show_spinner=False)
def robust_yf_download(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    for batch in chunked(tickers, BATCH_SIZE):
        tries = 0
        while True:
            try:
                df = yf.download(
                    tickers=batch, start=start.date(), end=end.date(),
                    auto_adjust=True, progress=False, group_by="ticker", threads=YF_THREADS,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    close = pd.concat({t: df[t]["Close"] for t in batch if t in df.columns.get_level_values(0)}, axis=1)
                else:
                    close = df[["Close"]].rename(columns={"Close": batch[0]})
                frames.append(close)
                break
            except Exception:
                tries += 1
                if tries >= MAX_RETRIES: break
                time.sleep(RETRY_SLEEP * tries)
        time.sleep(0.2)
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out.dropna(axis=1, how="all")

# ---------------- Indicators ----------------
def ema_df(df: pd.DataFrame, span: int = 200) -> pd.DataFrame:
    return df.ewm(span=span, adjust=False).mean()

def breadth200(close_wide: pd.DataFrame, span=200) -> pd.Series:
    valid = close_wide.dropna(how="all", axis=1)
    if valid.empty or len(valid) < span:
        return pd.Series(dtype=float, name="%Above200")
    ema_w = ema_df(valid, span=span)
    mask = valid.iloc[span-1:].to_numpy() > ema_w.iloc[span-1:].to_numpy()
    frac = mask.mean(axis=1) * 100.0
    return pd.Series(frac, index=valid.index[span-1:], name="%Above200")

def snapshot200(cons: pd.DataFrame, close_wide: pd.DataFrame, span=200) -> pd.DataFrame:
    tickers = [y_ticker(c) for c in cons["NSE Code"]]
    sub = close_wide.reindex(columns=tickers).dropna(axis=1, how="any")
    if sub.empty or len(sub) < span: return pd.DataFrame()
    ema_w = ema_df(sub, span=span)
    last_close = sub.iloc[-1]; last_ema = ema_w.iloc[-1]
    dist_pct = (last_close / last_ema - 1.0) * 100.0
    df = pd.DataFrame({"Yahoo": dist_pct.index, "Close": last_close.values,
                       "EMA200": last_ema.values, "Dist%": dist_pct.values})
    out = cons.assign(Yahoo=cons["NSE Code"].apply(y_ticker)).merge(df, on="Yahoo", how="inner")
    total = out["MktCapCr"].sum(skipna=True)
    out["Weight%"] = (out["MktCapCr"] / max(total, 1e-9) * 100.0).fillna(0)
    out["Status"] = np.where(out["Dist%"] >= 0, "Above", "Below")
    return out[["Name","NSE Code","Weight%","Status","Dist%","Close","EMA200","Yahoo"]]

# ---------------- Health summary + score ----------------
def index_health_summary(index_or_url: str, days=365, span=200):
    cons = fetch_constituents(index_or_url)
    if cons.empty: return {"error":"No constituents"}
    end = datetime.now(); start = end - timedelta(days=max(days, span) + 40)
    tickers = [y_ticker(c) for c in cons["NSE Code"]]
    close_wide = robust_yf_download(tickers, start, end)
    if close_wide.empty or len(close_wide) < span + 5: return {"error":"No price data"}
    close_wide = close_wide.dropna(axis=1, how="any")
    ema_w = ema_df(close_wide, span=span)
    last_close = close_wide.iloc[-1]; last_ema = ema_w.iloc[-1]
    dist_pct = (last_close / last_ema - 1.0) * 100.0
    above_now = (last_close > last_ema)

    cons2 = cons.assign(Yahoo=cons["NSE Code"].apply(y_ticker))
    cons2 = cons2[cons2["Yahoo"].isin(close_wide.columns)]
    total_mcap = cons2["MktCapCr"].sum(skipna=True)
    weights = (cons2.set_index("Yahoo")["MktCapCr"] / max(total_mcap, 1e-9)).reindex(close_wide.columns).fillna(0.0)

    # Breadth now
    eq_breadth_now = above_now.mean() * 100.0
    cap_breadth_now = (weights * above_now.astype(float)).sum() * 100.0

    # Avg distance vs 200EMA
    cap_avg_dist_now = (weights * dist_pct.reindex(weights.index)).sum()
    eq_avg_dist_now  = float(dist_pct.mean())

    # Returns
    look = 252 if len(close_wide) > 252 else max(20, len(close_wide)-1)
    base = close_wide.iloc[-look]; ret_1y = (last_close / base - 1.0)
    eq_ret_1y = ret_1y.mean() * 100.0
    cap_ret_1y = (weights * ret_1y).sum() * 100.0

    # 3m ago breadth
    look_3m = min(63, len(close_wide)-1)
    past_idx = -look_3m
    above_3m = (close_wide.iloc[past_idx] > ema_w.iloc[past_idx])
    eq_breadth_3m = above_3m.mean() * 100.0
    cap_breadth_3m = (weights * above_3m.astype(float)).sum() * 100.0

    return {
        "eq_breadth_now_%": float(eq_breadth_now),
        "cap_breadth_now_%": float(cap_breadth_now),
        "eq_avg_dist_now_%vsEMA": float(eq_avg_dist_now),
        "cap_avg_dist_now_%vsEMA": float(cap_avg_dist_now),
        "eq_ret_1y_%": float(eq_ret_1y),
        "cap_ret_1y_%": float(cap_ret_1y),
        "eq_breadth_3mago_%": float(eq_breadth_3m),
        "cap_breadth_3mago_%": float(cap_breadth_3m),
        "n_constituents_used": int((weights > 0).sum()),
        "span_days": int(span),
        "return_lookback_days": int(look),
        "close_wide": close_wide,   # pass through for plotting
        "ema_w": ema_w,
        "weights": weights,
        "cons": cons2
    }

def sustainability_score_v2(summary, mode: str = "hybrid"):
    """
    mode: 'hybrid' (60% cap + 40% eq), 'cap', or 'equal'
    """
    if not summary or "error" in summary:
        return {"score": float("nan"), "label": "N/A", "why": "no data"}

    if mode == "cap":
        breadth_now = summary["cap_breadth_now_%"]
        breadth_3m  = summary["cap_breadth_3mago_%"]
        dist        = summary["cap_avg_dist_now_%vsEMA"]
        ret1y       = summary["cap_ret_1y_%"]
        tag = "Cap-weight"
    elif mode == "equal":
        breadth_now = summary["eq_breadth_now_%"]
        breadth_3m  = summary["eq_breadth_3mago_%"]
        dist        = summary["eq_avg_dist_now_%vsEMA"]
        ret1y       = summary["eq_ret_1y_%"]
        tag = "Equal-weight"
    else:
        b_eq = summary["eq_breadth_now_%"]; b_cap = summary["cap_breadth_now_%"]
        b_eq_3m = summary["eq_breadth_3mago_%"]; b_cap_3m = summary["cap_breadth_3mago_%"]
        dist = 0.6*summary["cap_avg_dist_now_%vsEMA"] + 0.4*summary["eq_avg_dist_now_%vsEMA"]
        ret1y = 0.6*summary["cap_ret_1y_%"] + 0.4*summary["eq_ret_1y_%"]
        breadth_now = 0.6*b_cap + 0.4*b_eq
        breadth_3m  = 0.6*b_cap_3m + 0.4*b_eq_3m
        tag = "Hybrid 60/40"

    comp1 = breadth_now
    comp2 = float(np.clip((breadth_now - breadth_3m + 20)/40*100, 0, 100))
    comp3 = float((np.tanh(dist/5.0)+1)/2 * 100)
    comp4 = float(np.clip((ret1y + 20)/40*100, 0, 100))
    score = float(np.clip(0.50*comp1 + 0.20*comp2 + 0.15*comp3 + 0.15*comp4, 0, 100))
    label = "Broad & strong" if score>=75 else "Constructive" if score>=60 else "Mixed/fragile" if score>=40 else "Weak/narrow"
    why = f"{tag} — breadth {breadth_now:.0f}%, Δbreadth {breadth_now-breadth_3m:+.0f}pp, dist {dist:+.1f}%, 1y {ret1y:+.1f}%"
    return {"score": score, "label": label, "why": why}

# ---------------- Extra visuals ----------------
def plot_breadth_and_snapshot(index_name, br_series, df, days, eq_breadth, cap_breadth, cap_avg_dist):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[1.1, 1.4])
    fig.subplots_adjust(hspace=0.25)

    ax1.plot(br_series.index, br_series.values, linewidth=1.5)
    ax1.axhline(50, linestyle=":")
    ax1.set_ylim(0, 100); ax1.set_ylabel("% above 200-EMA")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title(f"{index_name}: Breadth (last {days} days)")
    ax1.text(br_series.index[-1], br_series.iloc[-1], f" {br_series.iloc[-1]:.1f}%", va="center")

    y = np.arange(len(df))
    for yi, x in zip(y, df["Dist%"].values):
        ax2.plot([0, x], [yi, yi], linewidth=1)
    sizes = ((df["Weight%"].fillna(0)+0.25)**1.15)*60
    is_above = df["Status"].eq("Above").to_numpy()
    ax2.scatter(df.loc[is_above,"Dist%"], y[is_above], s=sizes[is_above], marker="o", alpha=0.9, linewidths=0.6)
    ax2.scatter(df.loc[~is_above,"Dist%"], y[~is_above], s=sizes[~is_above], marker="X", alpha=0.9, linewidths=0.6)
    ax2.axvline(0, linestyle=":")
    lim = max(5, np.nanmax(np.abs(df["Dist%"])))*1.1
    ax2.set_xlim(-lim, lim)
    ax2.set_yticks(y); ax2.set_yticklabels(df["NSE Code"].values)
    ax2.set_xlabel("% vs 200-EMA")
    ax2.set_title(f"{index_name}: Constituent distance to 200-EMA (bubble = mcap weight)")
    ax2.grid(True, linestyle="--", alpha=0.35)

    metrics = (f"Equal-wt breadth: {eq_breadth:.1f}%\n"
               f"Cap-wt breadth:   {cap_breadth:.1f}%\n"
               f"Cap-wt avg dist:  {cap_avg_dist:.1f}%")
    fig.text(0.015, 0.02, metrics, fontsize=9, family="monospace")
    return fig

TRADING_LOOKBACKS = {"1y":252,"9m":189,"6m":126,"3m":63,"1m":21}

def compute_time_windows(summary, span=200):
    close_wide = summary["close_wide"]; ema_w = summary["ema_w"]; weights = summary["weights"]
    out = {}
    last_close = close_wide.iloc[-1]
    for label, look in TRADING_LOOKBACKS.items():
        if len(close_wide) <= look: continue
        base_close = close_wide.iloc[-look]
        ret = (last_close / base_close - 1.0)
        eq_ret = float(ret.mean()*100.0); cap_ret = float((weights*ret).sum()*100.0)
        past_idx = -look
        above_past = (close_wide.iloc[past_idx] > ema_w.iloc[past_idx]).astype(float)
        eq_b = float(above_past.mean()*100.0); cap_b = float((weights*above_past).sum()*100.0)
        out[label] = {"eq_ret%":eq_ret,"cap_ret%":cap_ret,"eq_breadth%":eq_b,"cap_breadth%":cap_b}
    return out

def fig_health_summary(summary, score_dict):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Index Health Summary", fontsize=16, fontweight="bold")

    axes[0,0].bar(["Eq Now","Eq 3m"], [summary["eq_breadth_now_%"], summary["eq_breadth_3mago_%"]], alpha=0.85, label="Equal-weight")
    axes[0,0].bar(["Cap Now","Cap 3m"], [summary["cap_breadth_now_%"], summary["cap_breadth_3mago_%"]], alpha=0.85, label="Cap-weight")
    axes[0,0].set_ylim(0,100); axes[0,0].set_ylabel("Breadth (%)"); axes[0,0].legend(); axes[0,0].set_title("Breadth Now vs 3m Ago")

    axes[0,1].barh(["Cap Avg Dist","Eq Avg Dist"], [summary["cap_avg_dist_now_%vsEMA"], summary["eq_avg_dist_now_%vsEMA"]], alpha=0.85)
    axes[0,1].axvline(0, linestyle="--")
    axes[0,1].set_xlabel("% vs 200-EMA"); axes[0,1].set_title("Distance from 200-EMA")

    axes[1,0].bar(["Equal-weight","Cap-weight"], [summary["eq_ret_1y_%"], summary["cap_ret_1y_%"]], alpha=0.85)
    axes[1,0].axhline(0, linestyle="--"); axes[1,0].set_ylabel("Return (%)"); axes[1,0].set_title("1-Year Returns")

    axes[1,1].axis("off")
    color = "#4caf50" if score_dict["score"]>=75 else "#ff9800" if score_dict["score"]>=60 else "#f44336" if score_dict["score"]>=40 else "#b71c1c"
    axes[1,1].text(0.5,0.6,f"{score_dict['score']:.1f}",fontsize=34,ha="center",va="center",fontweight="bold",
                   bbox=dict(facecolor=color, edgecolor="black", boxstyle="round,pad=0.6"))
    axes[1,1].text(0.5,0.28,score_dict["label"],ha="center",va="center",fontsize=12)
    return fig

def fig_quadrant_and_windows(summary, windows, title="Sector Health — Quadrant & Deterioration"):
    eq_now = summary["eq_breadth_now_%"]; cap_now = summary["cap_breadth_now_%"]
    labels = [w for w in ["1y","9m","6m","3m","1m"] if w in windows]
    eq_det = [eq_now - windows[w]["eq_breadth%"] for w in labels]
    cap_det = [cap_now - windows[w]["cap_breadth%"] for w in labels]
    eq_ret = [windows[w]["eq_ret%"] for w in labels]; cap_ret = [windows[w]["cap_ret%"] for w in labels]

    fig = plt.figure(figsize=(12, 6)); fig.suptitle(title, fontsize=14, fontweight="bold")

    axq = plt.subplot2grid((2,3),(0,0),rowspan=2)
    axq.scatter(cap_now, eq_now, s=180, alpha=0.9)
    axq.axvline(50, linestyle="--", alpha=0.6); axq.axhline(50, linestyle="--", alpha=0.6)
    axq.set_xlim(0,100); axq.set_ylim(0,100); axq.set_xlabel("Cap-weight breadth (%)"); axq.set_ylabel("Equal-weight breadth (%)")
    axq.set_title("Breadth Quadrant (Now)")
    axq.text(75,92,"Broad & strong",ha="center",fontsize=9); axq.text(25,92,"Broad / weak leaders",ha="center",fontsize=9)
    axq.text(75,8,"Narrow / strong leaders",ha="center",fontsize=9); axq.text(25,8,"Weak & narrow",ha="center",fontsize=9)
    axq.text(cap_now, eq_now, f"  ({cap_now:.0f}%, {eq_now:.0f}%)", va="center")

    axb = plt.subplot2grid((2,3),(0,1),colspan=2)
    x = np.arange(len(labels)); w = 0.4
    axb.bar(x-w/2, eq_det, w, label="Eq breadth Δ"); axb.bar(x+w/2, cap_det, w, label="Cap breadth Δ")
    axb.axhline(0, linestyle="--", alpha=0.6)
    axb.set_xticks(x, labels); axb.set_ylabel("Change (pp)"); axb.set_title("Breadth Deterioration / Improvement"); axb.legend()

    axr = plt.subplot2grid((2,3),(1,1),colspan=2)
    axr.bar(x-w/2, eq_ret, w, label="Equal-weight"); axr.bar(x+w/2, cap_ret, w, label="Cap-weight")
    axr.axhline(0, linestyle="--")
    axr.set_xticks(x, labels); axr.set_ylabel("Return (%)"); axr.set_title("Returns by Window"); axr.legend()
    return fig

# -------- All-sectors summaries & plots --------
@st.cache_data(ttl=1800, show_spinner=False)
def summarize_many(indices: dict, days: int, span: int):
    out = []
    for name in indices.keys():
        s = index_health_summary(name, days=days, span=span)
        if "error" in s:
            continue
        out.append({
            "Index": name,
            "cap_now": s["cap_breadth_now_%"],
            "eq_now":  s["eq_breadth_now_%"],
            "cap_3m":  s["cap_breadth_3mago_%"],
            "eq_3m":   s["eq_breadth_3mago_%"],
            "n":       s["n_constituents_used"]
        })
    return pd.DataFrame(out).sort_values("Index")

def fig_all_quadrant_cap_vs_eq(df: pd.DataFrame, title="All NSE sectors — Breadth Quadrant"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.axvline(50, linestyle="--", alpha=0.6); ax.axhline(50, linestyle="--", alpha=0.6)
    ax.set_xlabel("Cap-weight breadth (%)"); ax.set_ylabel("Equal-weight breadth (%)")
    sizes = (df["n"].clip(lower=8, upper=120) ** 1.05) * 2.7
    ax.scatter(df["cap_now"], df["eq_now"], s=sizes, alpha=0.85, linewidths=0.5)
    for _, r in df.iterrows():
        ax.text(r["cap_now"]+0.8, r["eq_now"]+0.8, r["Index"], fontsize=8)
    ax.text(75,92,"Broad & strong",ha="center",fontsize=9)
    ax.text(25,92,"Broad / weak leaders",ha="center",fontsize=9)
    ax.text(75,8,"Narrow / strong leaders",ha="center",fontsize=9)
    ax.text(25,8,"Weak & narrow",ha="center",fontsize=9)
    return fig

def fig_all_cap_only(df: pd.DataFrame, title="All sectors — Cap breadth (Now vs Δ3m)"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(0, linestyle="--", alpha=0.6)
    x, y = df["cap_now"], (df["cap_now"] - df["cap_3m"])
    sizes = (df["n"].clip(8,120)**1.05)*2.7
    ax.scatter(x, y, s=sizes, alpha=0.85, linewidths=0.5)
    ax.set_xlim(0,100); ax.set_xlabel("Cap breadth now (%)")
    ax.set_ylabel("Change vs 3m (pp)")
    for _, r in df.iterrows():
        ax.text(r["cap_now"]+0.8, (r["cap_now"]-r["cap_3m"])+0.8, r["Index"], fontsize=8)
    return fig

def fig_all_eq_only(df: pd.DataFrame, title="All sectors — Equal breadth (Now vs Δ3m)"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(0, linestyle="--", alpha=0.6)
    x, y = df["eq_now"], (df["eq_now"] - df["eq_3m"])
    sizes = (df["n"].clip(8,120)**1.05)*2.7
    ax.scatter(x, y, s=sizes, alpha=0.85, linewidths=0.5)
    ax.set_xlim(0,100); ax.set_xlabel("Equal breadth now (%)")
    ax.set_ylabel("Change vs 3m (pp)")
    for _, r in df.iterrows():
        ax.text(r["eq_now"]+0.8, (r["eq_now"]-r["eq_3m"])+0.8, r["Index"], fontsize=8)
    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="NSE Index Breadth & Health", layout="wide")
st.title("NSE Index Breadth & Health")
st.caption("Data: Screener constituents + Yahoo Finance via yfinance")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    idx = st.selectbox("Index", sorted(INDEX_URLS.keys()), index=sorted(INDEX_URLS.keys()).index("Nifty FMCG") if "Nifty FMCG" in INDEX_URLS else 0)
    custom = st.text_input("…or paste Screener #constituents URL (optional)", "")
    target = custom.strip() or idx
with col2:
    span = st.number_input("EMA span (days)", 50, 400, 200, 10)
    days = st.number_input("Breadth lookback (calendar days)", 120, 800, 365, 5)
with col3:
    topk = st.number_input("Bubble: topk each side", 5, 999, 12, 1)
    focus = st.selectbox("Bubble ranking", ["abs_dist","weight"])
    compare_all = st.checkbox("Compare all sectors", value=False)

# Score basis switch
basis = st.radio("Score basis", ["Hybrid 60/40", "Cap-weight only", "Equal-weight only"], horizontal=True)

go = st.button("Run analysis", type="primary")

if go:
    with st.spinner("Crunching…"):
        cons = fetch_constituents(target)
        if cons.empty:
            st.error("No constituents found."); st.stop()
        tickers = [y_ticker(c) for c in cons["NSE Code"]]
        end = datetime.now(); start = end - timedelta(days=days + span + 10)
        close_wide = robust_yf_download(tickers, start, end)
        if close_wide.empty: st.error("No price data."); st.stop()
        br = breadth200(close_wide, span=span)
        snap = snapshot200(cons, close_wide, span=span)
        if br.empty or snap.empty: st.error("Insufficient data."); st.stop()

        df = snap.copy()
        df["_sel"] = df["Weight%"] if focus=="weight" else df["Dist%"].abs()
        top = df[df["Dist%"]>=0].nlargest(int(topk), "_sel")
        bot = df[df["Dist%"]<0].nlargest(int(topk), "_sel")
        df_plot = pd.concat([bot, top]).sort_values("Dist%").reset_index(drop=True)

        eq_breadth = (snap["Status"]=="Above").mean()*100
        cap_breadth = (snap["Weight%"]/100 * (snap["Status"]=="Above")).sum()*100
        cap_avg_dist = (snap["Weight%"]*snap["Dist%"]).sum()/max(snap["Weight%"].sum(),1e-9)

        fig1 = plot_breadth_and_snapshot(target, br, df_plot, int(days), eq_breadth, cap_breadth, cap_avg_dist)
        st.pyplot(fig1, use_container_width=True)

        summary = index_health_summary(target, days=days, span=span)
        mode = "hybrid" if basis.startswith("Hybrid") else ("cap" if basis.startswith("Cap") else "equal")
        score = sustainability_score_v2(summary, mode=mode)

        st.subheader("Health Summary")
        st.write(f"**Score:** {score['score']:.1f} — {score['label']}  |  {score['why']}")
        st.pyplot(fig_health_summary(summary, score), use_container_width=True)

        windows = compute_time_windows(summary, span=span)
        st.pyplot(fig_quadrant_and_windows(summary, windows, title=f"{target} — Quadrant & Deterioration"),
                  use_container_width=True)

        if compare_all:
            st.subheader("All sectors — scatterviews")
            df_all = summarize_many(INDEX_URLS, days=int(days), span=int(span))
            tabs = st.tabs(["Cap vs Eq (quadrant)", "Cap-only (Now vs Δ3m)", "Equal-only (Now vs Δ3m)"])
            with tabs[0]:
                st.pyplot(fig_all_quadrant_cap_vs_eq(df_all), use_container_width=True)
            with tabs[1]:
                st.pyplot(fig_all_cap_only(df_all), use_container_width=True)
            with tabs[2]:
                st.pyplot(fig_all_eq_only(df_all), use_container_width=True)
            st.dataframe(
                df_all.rename(columns={"cap_now":"Cap now %","eq_now":"Eq now %","cap_3m":"Cap 3m %","eq_3m":"Eq 3m %","n":"# names"}),
                use_container_width=True
            )

        st.subheader("Constituent snapshot")
        st.dataframe(snap.sort_values("Weight%", ascending=False), use_container_width=True)
        csv = snap.to_csv(index=False).encode()
        st.download_button("Download snapshot CSV", csv, file_name="snapshot.csv", mime="text/csv")
