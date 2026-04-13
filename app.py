import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SIGNAL — Stock Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    background-color: #080810 !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.main { background-color: #080810 !important; }
.block-container { padding-top: 2rem !important; max-width: 1100px !important; }

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.12em !important; }

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.signal-bull { color: #00f5d4 !important; font-weight: 700; }
.signal-bear { color: #ff6b6b !important; font-weight: 700; }
.signal-neut { color: #f5a623 !important; font-weight: 700; }
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin: 2px;
}
.tag-bull { background: rgba(0,245,212,0.15); color: #00f5d4; border: 1px solid #00f5d430; }
.tag-bear { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid #ff6b6b30; }
.tag-neut { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid #f5a62330; }
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px;
    letter-spacing: 0.18em;
    color: #00f5d4;
    border-bottom: 1px solid #00f5d420;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
.verdict-box {
    background: rgba(255,214,10,0.06);
    border: 1px solid rgba(255,214,10,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 10px;
}
div[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 28px !important; }
div[data-testid="stMetricLabel"] { font-size: 11px !important; letter-spacing: 0.08em !important; color: #666 !important; }
.stButton>button {
    background: #00f5d4 !important;
    color: #080810 !important;
    border: none !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 16px !important;
    letter-spacing: 0.15em !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
}
.stTextInput>div>div>input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    letter-spacing: 0.1em !important;
}
.stSelectbox>div>div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #fff !important;
}
.stSpinner { color: #00f5d4 !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def signal_tag(label, direction):
    cls = {"bull": "tag-bull", "bear": "tag-bear", "neut": "tag-neut"}.get(direction, "tag-neut")
    return f'<span class="tag {cls}">{label}</span>'

def color_val(val, good_positive=True):
    if val is None: return "#888"
    if good_positive:
        return "#00f5d4" if val >= 0 else "#ff6b6b"
    else:
        return "#ff6b6b" if val >= 0 else "#00f5d4"

def fmt(val, decimals=2, prefix="", suffix=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"

def fmt_large(val):
    if val is None or pd.isna(val): return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval="1d")
    info = t.info
    return hist, info

@st.cache_data(ttl=600)
def fetch_intraday(ticker):
    t = yf.Ticker(ticker)
    hist_1d = t.history(period="5d", interval="15m")
    return hist_1d


# ─────────────────────────────────────────────
# TECHNICAL CALC
# ─────────────────────────────────────────────
def compute_indicators(df):
    df = df.copy()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    return df


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def build_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("", "VOLUME", "RSI")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00f5d4", decreasing_line_color="#ff6b6b",
        name="Price"
    ), row=1, col=1)

    # EMAs
    for col, color, name in [
        ("EMA_20", "#f5a623", "EMA 20"),
        ("EMA_50", "#c77dff", "EMA 50"),
        ("EMA_200", "#ff6b6b", "EMA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.2), opacity=0.85
            ), row=1, col=1)

    # BB bands
    if "BBU_20_2.0" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBU_20_2.0"], name="BB Upper",
            line=dict(color="#ffffff20", width=1, dash="dot"), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBL_20_2.0"], name="BB Lower",
            line=dict(color="#ffffff20", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,255,255,0.02)", showlegend=False
        ), row=1, col=1)

    # Volume
    colors = ["#00f5d4" if c >= o else "#ff6b6b"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7, showlegend=False
    ), row=2, col=1)

    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"], name="RSI",
            line=dict(color="#c77dff", width=1.5), showlegend=False
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff6b6b40", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00f5d440", row=3, col=1)

    fig.update_layout(
        plot_bgcolor="#0d0d1a",
        paper_bgcolor="#080810",
        font=dict(family="IBM Plex Mono", color="#888", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=560,
    )
    for i in range(1, 4):
        fig.update_xaxes(
            row=i, col=1,
            gridcolor="#111122", showgrid=True,
            zeroline=False, showspikes=True, spikecolor="#444"
        )
        fig.update_yaxes(
            row=i, col=1,
            gridcolor="#111122", showgrid=True,
            zeroline=False, side="right"
        )

    return fig


# ─────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────
def evaluate_signals(df, info):
    signals = []
    score = 0

    last = df.iloc[-1]
    price = last["Close"]

    # Trend via EMA
    if all(c in df.columns for c in ["EMA_20", "EMA_50", "EMA_200"]):
        e20, e50, e200 = last["EMA_20"], last["EMA_50"], last["EMA_200"]
        if price > e20 > e50 > e200:
            signals.append(("bull", "STRONG UPTREND (P>EMA20>50>200)"))
            score += 3
        elif price > e50 > e200:
            signals.append(("bull", "UPTREND (P>EMA50>200)"))
            score += 2
        elif price < e20 < e50 < e200:
            signals.append(("bear", "STRONG DOWNTREND (P<EMA20<50<200)"))
            score -= 3
        elif price < e50:
            signals.append(("bear", "BELOW KEY MAs"))
            score -= 1
        else:
            signals.append(("neut", "MIXED TREND"))

    # RSI
    if "RSI_14" in df.columns:
        rsi = last["RSI_14"]
        if rsi > 70:
            signals.append(("bear", f"RSI OVERBOUGHT ({rsi:.1f})"))
            score -= 1
        elif rsi < 30:
            signals.append(("bull", f"RSI OVERSOLD ({rsi:.1f})"))
            score += 1
        elif 45 <= rsi <= 60:
            signals.append(("bull", f"RSI HEALTHY ({rsi:.1f})"))
            score += 1
        else:
            signals.append(("neut", f"RSI NEUTRAL ({rsi:.1f})"))

    # MACD
    macd_col = [c for c in df.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
    sig_col  = [c for c in df.columns if "MACDs" in c]
    if macd_col and sig_col:
        macd_val = last[macd_col[0]]
        sig_val  = last[sig_col[0]]
        if macd_val > sig_val and macd_val > 0:
            signals.append(("bull", "MACD BULLISH ABOVE ZERO"))
            score += 2
        elif macd_val > sig_val:
            signals.append(("bull", "MACD BULLISH CROSSOVER"))
            score += 1
        elif macd_val < sig_val and macd_val < 0:
            signals.append(("bear", "MACD BEARISH BELOW ZERO"))
            score -= 2
        else:
            signals.append(("bear", "MACD BEARISH CROSSOVER"))
            score -= 1

    # Volume
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
    cur_vol = last["Volume"]
    if cur_vol > avg_vol * 1.5:
        signals.append(("bull" if last["Close"] >= last["Open"] else "bear",
                         f"HIGH VOLUME ({cur_vol/avg_vol:.1f}x AVG)"))

    # Bollinger
    if "BBU_20_2.0" in df.columns and "BBL_20_2.0" in df.columns:
        bbu = last["BBU_20_2.0"]
        bbl = last["BBL_20_2.0"]
        if price >= bbu:
            signals.append(("bear", "AT UPPER BB (OVERBOUGHT ZONE)"))
            score -= 1
        elif price <= bbl:
            signals.append(("bull", "AT LOWER BB (OVERSOLD ZONE)"))
            score += 1

    return signals, score


# ─────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────
def generate_verdict(score, info, df):
    price = df["Close"].iloc[-1]
    atr = df["ATRr_14"].iloc[-1] if "ATRr_14" in df.columns else df["Close"].pct_change().std() * price

    if score >= 4:
        action = "BUY / LONG"
        action_color = "#00f5d4"
        bias = "Bullish"
        stop = price - (atr * 2)
        target = price + (atr * 4)
    elif score <= -3:
        action = "AVOID / SHORT BIAS"
        action_color = "#ff6b6b"
        bias = "Bearish"
        stop = price + (atr * 2)
        target = price - (atr * 4)
    else:
        action = "WAIT / NEUTRAL"
        action_color = "#f5a623"
        bias = "Neutral"
        stop = price - (atr * 1.5)
        target = price + (atr * 3)

    return {
        "action": action,
        "color": action_color,
        "bias": bias,
        "score": score,
        "stop": stop,
        "target": target,
        "atr": atr,
        "rr": round(abs(target - price) / abs(price - stop), 2) if abs(price - stop) > 0 else 0,
    }


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
  <span style='font-family:Bebas Neue,sans-serif;font-size:38px;letter-spacing:0.15em;color:#fff'>SIGNAL</span>
  <span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#333;margin-left:12px;letter-spacing:0.2em'>STOCK INTELLIGENCE TERMINAL</span>
</div>
<div style='height:2px;background:linear-gradient(90deg,#00f5d4,transparent);width:200px;margin-bottom:28px'></div>
""", unsafe_allow_html=True)

col_in, col_period, col_btn = st.columns([3, 1.5, 1])
with col_in:
    ticker_input = st.text_input("", placeholder="TICKER — e.g. BBCA.JK, ^JKSE, AAPL, BTC-USD", label_visibility="collapsed")
with col_period:
    period = st.selectbox("", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, label_visibility="collapsed")
with col_btn:
    analyze_btn = st.button("ANALYZE")

# ── IDX QUICK SELECT ──
IDX_TICKERS = {
    "🇮🇩 INDICES": {
        "IHSG (^JKSE)": "^JKSE",
        "LQ45 (^JKLQ45)": "^JKLQ45",
    },
    "🏦 Banking": {
        "BBCA": "BBCA.JK", "BBRI": "BBRI.JK",
        "BMRI": "BMRI.JK", "BBNI": "BBNI.JK",
    },
    "⚡ Energy & Mining": {
        "ADRO": "ADRO.JK", "PTBA": "PTBA.JK",
        "INCO": "INCO.JK", "MEDC": "MEDC.JK",
    },
    "📱 Telco & Tech": {
        "TLKM": "TLKM.JK", "EXCL": "EXCL.JK",
        "GOTO": "GOTO.JK", "BUKA": "BUKA.JK",
    },
    "🏭 Consumer & Industrial": {
        "UNVR": "UNVR.JK", "ICBP": "ICBP.JK",
        "ASII": "ASII.JK", "KLBF": "KLBF.JK",
    },
}

with st.expander("🇮🇩  IDX QUICK SELECT — Indonesian Stocks & Indices"):
    st.markdown("""
    <div style='font-size:11px;color:#555;margin-bottom:12px;font-family:IBM Plex Mono,monospace'>
    Indonesian stocks use <b style='color:#f5a623'>.JK</b> suffix &nbsp;|&nbsp;
    IHSG index = <b style='color:#f5a623'>^JKSE</b> &nbsp;|&nbsp;
    Type directly in the box above or click a button below
    </div>
    """, unsafe_allow_html=True)

    quick_ticker = None
    for sector, tickers in IDX_TICKERS.items():
        st.markdown(f"<span style='font-size:11px;color:#444;letter-spacing:0.1em'>{sector}</span>", unsafe_allow_html=True)
        cols = st.columns(len(tickers))
        for col, (name, sym) in zip(cols, tickers.items()):
            if col.button(name, key=f"quick_{sym}"):
                quick_ticker = sym

    if quick_ticker:
        ticker_input = quick_ticker
        analyze_btn = True

    st.markdown("""
    <div style='font-size:10px;color:#333;margin-top:10px;font-family:IBM Plex Mono,monospace'>
    Other IDX tickers: just type the stock code + .JK (e.g. ANTM.JK, INDF.JK, PGAS.JK, SMGR.JK)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---", unsafe_allow_html=True)

if analyze_btn and ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching live data for {ticker}..."):
        try:
            hist, info = fetch_data(ticker, period)
            intraday = fetch_intraday(ticker)
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}. Check the ticker symbol.")
            st.stop()

    if hist.empty:
        st.error("No data returned. Check ticker symbol.")
        st.stop()

    hist = compute_indicators(hist)
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    price = last["Close"]
    change = price - prev["Close"]
    change_pct = (change / prev["Close"]) * 100

    # ── CURRENCY DETECTION ──
    currency = info.get("currency", "USD")
    is_idr = currency == "IDR" or ticker.endswith(".JK") or ticker in ["^JKSE","^JKLQ45"]
    ccy_symbol = "Rp " if is_idr else ("$" if currency in ["USD",""] else currency + " ")
    price_fmt = f"{price:,.0f}" if is_idr else f"{price:,.2f}"
    change_fmt = f"{abs(change):,.0f}" if is_idr else f"{abs(change):,.2f}"

    def fmt_ccy(val, decimals=None):
        if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
        d = 0 if is_idr else (decimals if decimals is not None else 2)
        return f"{ccy_symbol}{val:,.{d}f}"

    def fmt_large_ccy(val):
        if val is None or pd.isna(val): return "N/A"
        sym = ccy_symbol
        if is_idr:
            if val >= 1e12: return f"{sym}{val/1e12:.2f}T"
            if val >= 1e9:  return f"{sym}{val/1e9:.2f}M"
            if val >= 1e6:  return f"{sym}{val/1e6:.2f}Jt"
            return f"{sym}{val:,.0f}"
        else:
            if val >= 1e12: return f"{sym}{val/1e12:.2f}T"
            if val >= 1e9:  return f"{sym}{val/1e9:.2f}B"
            if val >= 1e6:  return f"{sym}{val/1e6:.2f}M"
            return f"{sym}{val:,.0f}"

    # ── HEADER ROW ──
    market_badge = '<span style="background:rgba(255,214,10,0.1);border:1px solid #ffd60a30;border-radius:4px;padding:2px 8px;font-size:10px;color:#ffd60a;letter-spacing:0.1em">IDX · INDONESIA</span>' if is_idr else ""
    st.markdown(f"""
    <div style='display:flex;align-items:baseline;gap:16px;margin-bottom:20px;flex-wrap:wrap'>
      <span style='font-family:Bebas Neue,sans-serif;font-size:32px;color:#00f5d4;letter-spacing:0.1em'>{ticker}</span>
      {market_badge}
      <span style='font-family:Bebas Neue,sans-serif;font-size:28px;color:#fff'>{ccy_symbol}{price_fmt}</span>
      <span style='font-family:IBM Plex Mono,monospace;font-size:14px;color:{"#00f5d4" if change>=0 else "#ff6b6b"}'>
        {"&#9650;" if change>=0 else "&#9660;"} {change_fmt} ({abs(change_pct):.2f}%)
      </span>
      <span style='font-size:11px;color:#333;margin-left:auto'>{info.get("longName","")}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── QUICK STATS ──
    m1, m2, m3, m4, m5 = st.columns(5)
    stats = [
        ("MKT CAP", fmt_large_ccy(info.get("marketCap"))),
        ("52W HIGH", fmt_ccy(info.get("fiftyTwoWeekHigh"))),
        ("52W LOW",  fmt_ccy(info.get("fiftyTwoWeekLow"))),
        ("AVG VOL",  fmt_large(info.get("averageVolume"))),
        ("BETA",     fmt(info.get("beta"), decimals=2)),
    ]
    for col, (label, val) in zip([m1,m2,m3,m4,m5], stats):
        col.metric(label, val)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHART ──
    st.markdown('<div class="section-header">📈 PRICE CHART + INDICATORS</div>', unsafe_allow_html=True)
    fig = build_chart(hist, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ── TECHNICAL + FUNDAMENTAL COLS ──
    col_tech, col_fund = st.columns([1, 1], gap="large")

    with col_tech:
        st.markdown('<div class="section-header">⚡ TECHNICAL SIGNALS</div>', unsafe_allow_html=True)
        signals, score = evaluate_signals(hist, info)
        tags_html = " ".join(signal_tag(s[1], s[0]) for s in signals)
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Key levels
        rsi_val = last.get("RSI_14", None)
        macd_cols = [c for c in hist.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
        macd_val = last[macd_cols[0]] if macd_cols else None

        rows = [
            ("PRICE",    f"${price:,.2f}",  "—"),
            ("EMA 20",   fmt(last.get("EMA_20"), prefix="$"),   "↑ Bullish" if price > last.get("EMA_20",0) else "↓ Bearish"),
            ("EMA 50",   fmt(last.get("EMA_50"), prefix="$"),   "↑ Bullish" if price > last.get("EMA_50",0) else "↓ Bearish"),
            ("EMA 200",  fmt(last.get("EMA_200"), prefix="$"),  "↑ Bullish" if price > last.get("EMA_200",0) else "↓ Bearish"),
            ("RSI 14",   fmt(rsi_val, decimals=1),              "Overbought" if rsi_val and rsi_val>70 else ("Oversold" if rsi_val and rsi_val<30 else "Neutral")),
            ("ATR 14",   fmt(last.get("ATRr_14"), prefix="$"),  "Volatility"),
        ]
        df_tech = pd.DataFrame(rows, columns=["Indicator", "Value", "Signal"])
        st.dataframe(df_tech, use_container_width=True, hide_index=True)

    with col_fund:
        st.markdown('<div class="section-header">🏦 FUNDAMENTALS</div>', unsafe_allow_html=True)
        fund_rows = [
            ("P/E RATIO",     fmt(info.get("trailingPE"), decimals=2)),
            ("FWD P/E",       fmt(info.get("forwardPE"), decimals=2)),
            ("PEG RATIO",     fmt(info.get("pegRatio"), decimals=2)),
            ("P/S RATIO",     fmt(info.get("priceToSalesTrailing12Months"), decimals=2)),
            ("P/B RATIO",     fmt(info.get("priceToBook"), decimals=2)),
            ("EPS (TTM)",     fmt_ccy(info.get("trailingEps"))),
            ("FWD EPS",       fmt_ccy(info.get("forwardEps"))),
            ("REV GROWTH",    fmt(info.get("revenueGrowth", 0)*100 if info.get("revenueGrowth") else None, suffix="%")),
            ("PROFIT MARGIN", fmt(info.get("profitMargins", 0)*100 if info.get("profitMargins") else None, suffix="%")),
            ("DEBT/EQUITY",   fmt(info.get("debtToEquity"), decimals=2)),
            ("DIVIDEND %",    fmt(info.get("dividendYield", 0)*100 if info.get("dividendYield") else None, suffix="%")),
            ("SHORT FLOAT",   fmt(info.get("shortPercentOfFloat", 0)*100 if info.get("shortPercentOfFloat") else None, suffix="%")),
        ]
        df_fund = pd.DataFrame(fund_rows, columns=["Metric", "Value"])
        st.dataframe(df_fund, use_container_width=True, hide_index=True)

    # ── VERDICT ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ TRADER\'S VERDICT</div>', unsafe_allow_html=True)
    v = generate_verdict(score, info, hist)

    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    vcol1.metric("ACTION", v["action"])
    vcol2.metric("SIGNAL SCORE", f"{v['score']:+d} / 10")
    vcol3.metric("STOP LOSS", fmt_ccy(v["stop"]))
    vcol4.metric("TARGET", fmt_ccy(v["target"]))

    atr_fmt = f"{v['atr']:,.0f}" if is_idr else f"{v['atr']:.2f}"
    entry_fmt = f"{ccy_symbol}{price_fmt}"
    stop_fmt  = f"{ccy_symbol}{v['stop']:,.0f}" if is_idr else f"{ccy_symbol}{v['stop']:,.2f}"
    tgt_fmt   = f"{ccy_symbol}{v['target']:,.0f}" if is_idr else f"{ccy_symbol}{v['target']:,.2f}"

    st.markdown(f"""
    <div class='verdict-box'>
      <span style='font-family:Bebas Neue,sans-serif;font-size:22px;color:{v["color"]};letter-spacing:0.1em'>{v["action"]}</span>
      <span style='font-size:12px;color:#666;margin-left:14px'>Signal Score: {v["score"]:+d} &nbsp;|&nbsp; R:R = 1:{v["rr"]} &nbsp;|&nbsp; ATR = {ccy_symbol}{atr_fmt}</span>
      <br><br>
      <span style='font-size:12px;color:#aaa;line-height:1.8'>
        &#128205; <b>Entry Zone:</b> {entry_fmt} (current) &nbsp;&nbsp;
        &#128721; <b>Stop Loss:</b> {stop_fmt} &nbsp;&nbsp;
        &#127919; <b>Target:</b> {tgt_fmt}
        <br>
        Bias is <b style='color:{v["color"]}'>{v["bias"]}</b> based on {len(signals)} technical signals scanned across EMA trend, RSI momentum, MACD crossover, volume, and Bollinger Bands.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── COMPANY INFO ──
    if info.get("longBusinessSummary"):
        with st.expander("🏢 COMPANY OVERVIEW"):
            st.markdown(f"""
            <div style='font-size:12px;color:#aaa;line-height:1.8;font-family:IBM Plex Mono,monospace'>
            {info.get("longBusinessSummary","")}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='font-size:10px;color:#222;text-align:center;letter-spacing:0.05em'>
    SIGNAL TERMINAL — DATA VIA YAHOO FINANCE — NOT FINANCIAL ADVICE — FOR EDUCATIONAL USE ONLY
    </div>
    """, unsafe_allow_html=True)

elif not ticker_input and not analyze_btn:
    st.markdown("""
    <div style='text-align:center;padding:60px 0;color:#1a1a2e'>
      <div style='font-size:52px;margin-bottom:16px'>📡</div>
      <div style='font-family:Bebas Neue,sans-serif;font-size:20px;letter-spacing:0.2em;color:#1e1e30'>AWAITING TICKER INPUT</div>
      <div style='font-size:11px;color:#151520;margin-top:10px;letter-spacing:0.08em'>
        Enter any stock ticker above — AAPL · TSLA · NVDA · BTC-USD · ^JKSE · BBCA.JK · TLKM.JK
      </div>
    </div>
    """, unsafe_allow_html=True)
