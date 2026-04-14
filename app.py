import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Rocky Signal — Stock Intelligence", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] { background-color: #080810 !important; color: #e0e0e0 !important; font-family: 'IBM Plex Mono', monospace !important; }
.main { background-color: #080810 !important; }
.block-container { padding-top: 0.5rem !important; max-width: 1100px !important; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.12em !important; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.1em; margin: 2px; }
.tag-bull { background: rgba(0,245,212,0.15); color: #00f5d4; border: 1px solid #00f5d430; }
.tag-bear { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid #ff6b6b30; }
.tag-neut { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid #f5a62330; }
.section-header { font-family: 'Bebas Neue', sans-serif; font-size: 18px; letter-spacing: 0.18em; color: #00f5d4; border-bottom: 1px solid #00f5d420; padding-bottom: 6px; margin-bottom: 14px; }
.verdict-box { background: rgba(255,214,10,0.06); border: 1px solid rgba(255,214,10,0.25); border-radius: 12px; padding: 20px 24px; margin-top: 10px; }
div[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; }
div[data-testid="stMetricLabel"] { font-size: 11px !important; letter-spacing: 0.08em !important; color: #666 !important; }
.stButton>button { background: #00f5d4 !important; color: #080810 !important; border: none !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 16px !important; letter-spacing: 0.15em !important; border-radius: 8px !important; padding: 10px 28px !important; }
.stTextInput>div>div>input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; color: #fff !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 15px !important; letter-spacing: 0.1em !important; }
footer { visibility: hidden; } #MainMenu { visibility: hidden; }
.metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0 16px; }
.metric-tile { text-align: center; padding: 8px 4px; }
.m-label { font-size: 11px; letter-spacing: 0.08em; color: #666; margin-bottom: 4px; }
.m-value { font-family: 'Bebas Neue', sans-serif !important; font-size: 26px; color: #e0e0e0; line-height: 1.1; }
/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {
    .block-container { max-width: 100% !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .rocky-hero { text-align: center !important; }
    .rocky-hero span { font-size: 44px !important; }
    .rocky-divider { width: 180px !important; margin: 0 auto 24px !important; }
    [data-testid="stMetric"] { text-align: center !important; display: flex !important; flex-direction: column !important; align-items: center !important; }
    [data-testid="stMetricValue"] { text-align: center !important; width: 100% !important; }
    [data-testid="stMetricLabel"] { text-align: center !important; width: 100% !important; }
    [data-testid="stMetricDelta"] { justify-content: center !important; }
    .rocky-hero .hero-subtitle { font-size: 11px !important; letter-spacing: 0.25em !important; }
    .section-header { text-align: center !important; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; justify-content: center !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { flex: 0 0 50% !important; max-width: 50% !important; min-width: 0 !important; box-sizing: border-box !important; text-align: center !important; }
    .verdict-box { padding: 14px 16px !important; }
    .js-plotly-plot, .plotly-graph-div { touch-action: pan-y !important; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
}
</style>
""", unsafe_allow_html=True)

# ── INDICATORS (pure numpy/pandas, no pandas-ta) ──
def calc_ema(s, span): return s.ewm(span=span, adjust=False).mean()
def calc_rsi(s, period=14):
    d = s.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    ag = gain.ewm(com=period-1, min_periods=period).mean()
    al = loss.ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + ag/al))
def calc_macd(s, fast=12, slow=26, sig=9):
    ml = calc_ema(s, fast) - calc_ema(s, slow); sl = calc_ema(ml, sig); return ml, sl, ml-sl
def calc_bb(s, period=20, std=2):
    m = s.rolling(period).mean(); sd = s.rolling(period).std(); return m+std*sd, m, m-std*sd
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

def compute_indicators(df):
    df = df.copy()
    df["EMA_20"] = calc_ema(df["Close"], 20)
    df["EMA_50"] = calc_ema(df["Close"], 50)
    df["EMA_200"] = calc_ema(df["Close"], 200)
    df["RSI"] = calc_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calc_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calc_bb(df["Close"])
    df["ATR"] = calc_atr(df["High"], df["Low"], df["Close"])
    return df

def fmt(val, decimals=2, prefix="", suffix=""):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"
def fmt_large(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9: return f"${val/1e9:.2f}B"
    if val >= 1e6: return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"
def signal_tag(label, direction):
    cls = {"bull":"tag-bull","bear":"tag-bear","neut":"tag-neut"}.get(direction,"tag-neut")
    return f'<span class="tag {cls}">{label}</span>'

@st.cache_data(ttl=300)
def fetch_data(ticker, period):
    t = yf.Ticker(ticker)
    hist = pd.DataFrame()
    try:
        hist = t.history(period=period, interval="1d")
        if not hist.empty and isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
    except Exception:
        pass
    if hist.empty:
        try:
            hist = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if not hist.empty and isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
        except Exception:
            pass
    try:
        info = t.info
        if not isinstance(info, dict) or len(info) <= 1:
            info = {}
    except Exception:
        info = {}
    return hist, info

def build_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55,0.25,0.20], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], increasing_line_color="#00f5d4", decreasing_line_color="#ff6b6b", name="Price"), row=1, col=1)
    for col, color, name in [("EMA_20","#f5a623","EMA 20"),("EMA_50","#c77dff","EMA 50"),("EMA_200","#ff6b6b","EMA 200")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line=dict(color=color, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="rgba(255,255,255,0.09)", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="rgba(255,255,255,0.09)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(255,255,255,0.02)", showlegend=False), row=1, col=1)
    colors = ["#00f5d4" if c >= o else "#ff6b6b" for c,o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.7, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#c77dff", width=1.5), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,107,107,0.31)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,245,212,0.31)", row=3, col=1)
    fig.update_layout(plot_bgcolor="#0d0d1a", paper_bgcolor="#080810", font=dict(family="IBM Plex Mono", color="#888", size=11), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)", font=dict(size=10)), margin=dict(l=0,r=0,t=30,b=0), height=560)
    for i in range(1,4):
        fig.update_xaxes(row=i, col=1, gridcolor="#111122", zeroline=False)
        fig.update_yaxes(row=i, col=1, gridcolor="#111122", zeroline=False, side="right")
    return fig

def evaluate_signals(df, info):
    signals, score = [], 0
    last = df.iloc[-1]; price = last["Close"]
    e20, e50, e200 = last["EMA_20"], last["EMA_50"], last["EMA_200"]
    if price > e20 > e50 > e200: signals.append(("bull","STRONG UPTREND (P>EMA20>50>200)")); score += 3
    elif price > e50 > e200: signals.append(("bull","UPTREND (P>EMA50>200)")); score += 2
    elif price < e20 < e50 < e200: signals.append(("bear","STRONG DOWNTREND")); score -= 3
    elif price < e50: signals.append(("bear","BELOW KEY MAs")); score -= 1
    else: signals.append(("neut","MIXED TREND"))
    rsi = last["RSI"]
    if not pd.isna(rsi):
        if rsi > 70: signals.append(("bear",f"RSI OVERBOUGHT ({rsi:.1f})")); score -= 1
        elif rsi < 30: signals.append(("bull",f"RSI OVERSOLD ({rsi:.1f})")); score += 1
        elif 45 <= rsi <= 60: signals.append(("bull",f"RSI HEALTHY ({rsi:.1f})")); score += 1
        else: signals.append(("neut",f"RSI NEUTRAL ({rsi:.1f})"))
    macd, sig = last["MACD"], last["MACD_Signal"]
    if not pd.isna(macd) and not pd.isna(sig):
        if macd > sig and macd > 0: signals.append(("bull","MACD BULLISH ABOVE ZERO")); score += 2
        elif macd > sig: signals.append(("bull","MACD BULLISH CROSSOVER")); score += 1
        elif macd < sig and macd < 0: signals.append(("bear","MACD BEARISH BELOW ZERO")); score -= 2
        else: signals.append(("bear","MACD BEARISH CROSSOVER")); score -= 1
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]; cur_vol = last["Volume"]
    if avg_vol > 0 and cur_vol > avg_vol * 1.5:
        signals.append(("bull" if last["Close"] >= last["Open"] else "bear", f"HIGH VOLUME ({cur_vol/avg_vol:.1f}x AVG)"))
    if not pd.isna(last["BB_Upper"]):
        if price >= last["BB_Upper"]: signals.append(("bear","AT UPPER BB")); score -= 1
        elif price <= last["BB_Lower"]: signals.append(("bull","AT LOWER BB")); score += 1
    return signals, score

def generate_verdict(score, df):
    price = df["Close"].iloc[-1]; atr = df["ATR"].iloc[-1]
    if pd.isna(atr) or atr == 0: atr = price * 0.02
    if score >= 4: action,color,bias,stop,target = "BUY / LONG","#00f5d4","Bullish",price-atr*2,price+atr*4
    elif score <= -3: action,color,bias,stop,target = "AVOID / SHORT BIAS","#ff6b6b","Bearish",price+atr*2,price-atr*4
    else: action,color,bias,stop,target = "WAIT / NEUTRAL","#f5a623","Neutral",price-atr*1.5,price+atr*3
    rr = round(abs(target-price)/abs(price-stop), 2) if abs(price-stop) > 0 else 0
    return {"action":action,"color":color,"bias":bias,"score":score,"stop":stop,"target":target,"atr":atr,"rr":rr}

# ── UI ──
st.markdown("<div class='rocky-hero' style='margin-bottom:12px'><span style='font-family:Bebas Neue,sans-serif;font-size:64px;letter-spacing:0.10em;color:#00f5d4;text-shadow:0 0 40px rgba(0,245,212,0.45)'>ROCKY</span><span style='font-family:Bebas Neue,sans-serif;font-size:64px;letter-spacing:0.10em;color:#fff;margin-left:14px'>SIGNAL</span><br><span class='hero-subtitle' style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#444;letter-spacing:0.25em'>STOCK INTELLIGENCE TERMINAL</span></div><div class='rocky-divider' style='height:2px;background:linear-gradient(90deg,#00f5d4,transparent);width:320px;margin-bottom:32px'></div>", unsafe_allow_html=True)

col_in, col_period, col_btn = st.columns([3, 1.5, 1])
with col_in: ticker_input = st.text_input("", placeholder="TICKER — e.g. BBCA.JK, ^JKSE, AAPL, BTC-USD", label_visibility="collapsed")
with col_period: period = st.selectbox("PERIOD", ["1mo","3mo","6mo","1y","2y"], index=2, label_visibility="collapsed", format_func=lambda x: f"PERIOD · {x}")
with col_btn: analyze_btn = st.button("ANALYZE")

IDX = {"🇮🇩 INDICES":{"IHSG":"^JKSE","LQ45":"^JKLQ45"},"🏦 Banking":{"BBCA":"BBCA.JK","BBRI":"BBRI.JK","BMRI":"BMRI.JK","BBNI":"BBNI.JK"},"⚡ Energy":{"ADRO":"ADRO.JK","PTBA":"PTBA.JK","INCO":"INCO.JK","MEDC":"MEDC.JK"},"📱 Telco":{"TLKM":"TLKM.JK","EXCL":"EXCL.JK","GOTO":"GOTO.JK","BUKA":"BUKA.JK"},"🏭 Consumer":{"UNVR":"UNVR.JK","ICBP":"ICBP.JK","ASII":"ASII.JK","KLBF":"KLBF.JK"}}
with st.expander("🇮🇩  IDX QUICK SELECT"):
    st.markdown('<div style="font-size:11px;color:#555;margin-bottom:10px">Indonesian stocks: <b style="color:#f5a623">CODE.JK</b> &nbsp;|&nbsp; IHSG index: <b style="color:#f5a623">^JKSE</b></div>', unsafe_allow_html=True)
    quick_ticker = None
    for sector, tickers in IDX.items():
        st.markdown(f"<span style='font-size:10px;color:#444'>{sector}</span>", unsafe_allow_html=True)
        cols = st.columns(len(tickers))
        for col, (name, sym) in zip(cols, tickers.items()):
            if col.button(name, key=f"q_{sym}"): quick_ticker = sym
    if quick_ticker: ticker_input = quick_ticker; analyze_btn = True

st.markdown("---")

if analyze_btn and ticker_input:
    ticker = ticker_input.strip().upper()
    with st.spinner(f"Fetching {ticker}..."):
        try: hist, info = fetch_data(ticker, period)
        except Exception as e: st.error(f"Could not fetch {ticker}. Check the ticker. ({e})"); st.stop()
    if hist.empty or len(hist) < 10: st.error("No data. Check ticker symbol."); st.stop()

    hist = compute_indicators(hist)
    last = hist.iloc[-1]; prev = hist.iloc[-2]
    price = last["Close"]; change = price - prev["Close"]; pct_chg = (change / prev["Close"]) * 100

    # Fall back to hist-derived values when info is sparse (common for IDX/indices)
    _52w_high = info.get("fiftyTwoWeekHigh") or hist["High"].max()
    _52w_low = info.get("fiftyTwoWeekLow") or hist["Low"].min()
    _avg_vol = info.get("averageVolume") or hist["Volume"].mean()

    currency = info.get("currency","USD")
    is_idr = currency=="IDR" or ticker.endswith(".JK") or ticker in ["^JKSE","^JKLQ45"]
    ccy = "Rp " if is_idr else ("$" if currency in ["USD",""] else currency+" ")

    def p(val):
        if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
        return f"{ccy}{val:,.0f}" if is_idr else f"{ccy}{val:,.2f}"
    def big(val):
        if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
        s = ccy
        if is_idr:
            if val>=1e12: return f"{s}{val/1e12:.2f}T"
            if val>=1e9: return f"{s}{val/1e9:.2f}M"
            return f"{s}{val:,.0f}"
        else:
            if val>=1e12: return f"{s}{val/1e12:.2f}T"
            if val>=1e9: return f"{s}{val/1e9:.2f}B"
            if val>=1e6: return f"{s}{val/1e6:.2f}M"
            return f"{s}{val:,.0f}"
    def pct(v): return fmt(v*100 if v else None, suffix="%") if v else "N/A"

    badge = '<span style="background:rgba(255,214,10,0.1);border:1px solid #ffd60a30;border-radius:4px;padding:2px 8px;font-size:10px;color:#ffd60a">IDX · INDONESIA</span>' if is_idr else ""
    st.markdown(f"<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:20px;flex-wrap:wrap'><span style='font-family:Bebas Neue,sans-serif;font-size:32px;color:#00f5d4'>{ticker}</span>{badge}<span style='font-family:Bebas Neue,sans-serif;font-size:28px;color:#fff'>{p(price)}</span><span style='font-size:14px;color:{'#00f5d4' if change>=0 else '#ff6b6b'}'>{'&#9650;' if change>=0 else '&#9660;'} {p(abs(change))} ({abs(pct_chg):.2f}%)</span><span style='font-size:11px;color:#333;margin-left:auto'>{info.get('longName','')}</span></div>", unsafe_allow_html=True)

    mvals = [("MKT CAP",big(info.get("marketCap"))),("52W HIGH",p(_52w_high)),("52W LOW",p(_52w_low)),("AVG VOL",fmt_large(_avg_vol)),("BETA",fmt(info.get("beta"),decimals=2))]
    st.markdown('<div class="metrics-grid">'+''.join(f'<div class="metric-tile"><div class="m-label">{lbl}</div><div class="m-value">{val}</div></div>' for lbl,val in mvals)+'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 PRICE CHART + INDICATORS</div>', unsafe_allow_html=True)
    st.plotly_chart(build_chart(hist), use_container_width=True, config={"scrollZoom": False, "staticPlot": True})

    ct, cf = st.columns(2, gap="large")
    with ct:
        st.markdown('<div class="section-header">⚡ TECHNICAL SIGNALS</div>', unsafe_allow_html=True)
        signals, score = evaluate_signals(hist, info)
        st.markdown(" ".join(signal_tag(s[1],s[0]) for s in signals), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        rsi_val = last["RSI"]
        rows = [("PRICE",p(price),"—"),("EMA 20",p(last["EMA_20"]),"↑ Bull" if price>last["EMA_20"] else "↓ Bear"),("EMA 50",p(last["EMA_50"]),"↑ Bull" if price>last["EMA_50"] else "↓ Bear"),("EMA 200",p(last["EMA_200"]),"↑ Bull" if price>last["EMA_200"] else "↓ Bear"),("RSI 14",fmt(rsi_val,1),"OB" if rsi_val>70 else ("OS" if rsi_val<30 else "Neutral")),("ATR 14",p(last["ATR"]),"Volatility"),("MACD",fmt(last["MACD"],2),"↑ Bull" if last["MACD"]>last["MACD_Signal"] else "↓ Bear")]
        st.dataframe(pd.DataFrame(rows, columns=["Indicator","Value","Signal"]), use_container_width=True, hide_index=True, height=285)
    with cf:
        st.markdown('<div class="section-header">🏦 FUNDAMENTALS</div>', unsafe_allow_html=True)
        fund_rows = [("P/E RATIO",fmt(info.get("trailingPE"),2)),("FWD P/E",fmt(info.get("forwardPE"),2)),("PEG RATIO",fmt(info.get("pegRatio"),2)),("P/S RATIO",fmt(info.get("priceToSalesTrailing12Months"),2)),("P/B RATIO",fmt(info.get("priceToBook"),2)),("EPS (TTM)",p(info.get("trailingEps"))),("FWD EPS",p(info.get("forwardEps"))),("REV GROWTH",pct(info.get("revenueGrowth"))),("PROFIT MARGIN",pct(info.get("profitMargins"))),("DEBT/EQUITY",fmt(info.get("debtToEquity"),2)),("DIVIDEND %",pct(info.get("dividendYield"))),("SHORT FLOAT",pct(info.get("shortPercentOfFloat")))]
        st.dataframe(pd.DataFrame(fund_rows, columns=["Metric","Value"]), use_container_width=True, hide_index=True, height=460)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ TRADER\'S VERDICT</div>', unsafe_allow_html=True)
    v = generate_verdict(score, hist)
    vc1,vc2,vc3,vc4 = st.columns(4)
    vc1.metric("ACTION",v["action"]); vc2.metric("SIGNAL SCORE",f"{v['score']:+d} / 10"); vc3.metric("STOP LOSS",p(v["stop"])); vc4.metric("TARGET",p(v["target"]))
    atr_str = f"{ccy}{v['atr']:,.0f}" if is_idr else f"{ccy}{v['atr']:.2f}"
    st.markdown(f"<div class='verdict-box'><span style='font-family:Bebas Neue,sans-serif;font-size:22px;color:{v['color']}'>{v['action']}</span><span style='font-size:12px;color:#666;margin-left:14px'>Score: {v['score']:+d} | R:R = 1:{v['rr']} | ATR = {atr_str}</span><br><br><span style='font-size:12px;color:#aaa;line-height:1.8'>&#128205; <b>Entry:</b> {p(price)} &nbsp; &#128721; <b>Stop:</b> {p(v['stop'])} &nbsp; &#127919; <b>Target:</b> {p(v['target'])}<br>Bias is <b style='color:{v['color']}'>{v['bias']}</b> from {len(signals)} signals: EMA · RSI · MACD · Volume · Bollinger Bands</span></div>", unsafe_allow_html=True)

    if info.get("longBusinessSummary"):
        with st.expander("🏢 COMPANY OVERVIEW"):
            st.markdown(f'<div style="font-size:12px;color:#aaa;line-height:1.8">{info["longBusinessSummary"]}</div>', unsafe_allow_html=True)

    st.markdown('<br><div style="font-size:10px;color:#222;text-align:center">ROCKY SIGNAL TERMINAL — DATA VIA YAHOO FINANCE — NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)

elif not ticker_input:
    st.markdown("<div style='text-align:center;padding:60px 0'><div style='font-size:52px;margin-bottom:16px'>📡</div><div style='font-family:Bebas Neue,sans-serif;font-size:20px;letter-spacing:0.2em;color:#1e1e30'>AWAITING TICKER INPUT</div><div style='font-size:11px;color:#151520;margin-top:10px'>AAPL · TSLA · NVDA · BTC-USD · ^JKSE · BBCA.JK · TLKM.JK</div></div>", unsafe_allow_html=True)
