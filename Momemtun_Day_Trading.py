import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pytz

# ========== CONFIGURATION ==========
MAX_WORKERS = 6
REQUEST_DELAY = (1, 3)
CACHE_TTL = 3600 * 6
TIMEZONE = 'Asia/Shanghai'

# ========== ENHANCED SYMBOL MAPPING ==========
def map_symbol(symbol, exchange):
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()

    if exchange in ["SHH", "SHA"]:
        if not symbol.endswith(".SS"):
            return f"{symbol}.SS"
    elif exchange in ["SHZ", "SHE"]:
        if not symbol.endswith(".SZ"):
            return f"{symbol}.SZ"
    elif exchange == "HKG":
        if not symbol.endswith(".HK"):
            if len(symbol) == 4 and symbol.isdigit():
                symbol = f"0{symbol}"
            return f"{symbol}.HK"
    return symbol

# ========== MOMENTUM CALCULATION ==========
def calculate_momentum(data):
    if data.empty or len(data) < 15:
        return None

    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema20_series = close.ewm(span=20).mean()
        ema20 = ema20_series.iloc[-1]

        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg_10 if vol_avg_10 != 0 else 1

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_status = "Bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "Bearish"

        price_gap_ema20 = ((close.iloc[-1] - ema20) / ema20) * 100

        above_ema20 = close.iloc[-1] > ema20
        broke_5d_high = close.iloc[-1] > high.rolling(5).max().iloc[-2]

        score = 0
        if ema5 > ema10 > ema20: score += 30
        if above_ema20: score += 20
        if 40 < rsi < 70: score += 20
        if vol_ratio > 1.5: score += 15
        if broke_5d_high: score += 10

        return {
            "EMA5": round(ema5, 2),
            "EMA10": round(ema10, 2),
            "EMA20": round(ema20, 2),
            "RSI": round(rsi, 1),
            "Volume_Ratio": round(vol_ratio, 2),
            "Above_EMA20": above_ema20,
            "5D_Breakout": broke_5d_high,
            "Momentum_Score": min(100, score),
            "Trend": "↑ Strong" if score >= 70 else "↑ Medium" if score >= 50 else "↗ Weak",
            "MACD_Signal": macd_status,
            "Price_Gap_EMA20_%": round(price_gap_ema20, 2)
        }

    except Exception as e:
        st.warning(f"Momentum calculation failed: {str(e)}")
        return None

# ========== DATA FETCHING ==========
@st.cache_data(ttl=CACHE_TTL)
def get_ticker_data(symbol, exchange):
    try:
        yf_symbol = map_symbol(symbol, exchange)
        ticker = yf.Ticker(yf_symbol)
        time.sleep(random.uniform(*REQUEST_DELAY))
        data = ticker.history(period="1mo")

        if data.empty:
            if "." in yf_symbol:
                alt_symbol = yf_symbol.split(".")[0]
                ticker = yf.Ticker(alt_symbol)
                data = ticker.history(period="1mo")

        if data.empty or len(data) < 15:
            return None

        momentum = calculate_momentum(data)
        if not momentum:
            return None

        return {
            "Symbol": symbol,
            "Exchange": exchange,
            "Price": round(data['Close'].iloc[-1], 2),
            "5D_Change": round((data['Close'].iloc[-1]/data['Close'].iloc[-5]-1)*100, 1) if len(data) >=5 else None,
            **momentum
        }

    except Exception as e:
        st.warning(f"Failed to process {symbol}: {str(e)}")
        return None

# ========== STREAMLIT APP ==========
def main():
    st.set_page_config(layout="wide")
    st.title("🌏 Global Momentum Scanner (China Supported)")

    uploaded_file = st.file_uploader("Upload Ticker List (Excel/CSV)", type=["xlsx", "csv"])
    if not uploaded_file:
        st.warning("Please upload a file with columns: Symbol, Exchange")
        return

    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    df = df.dropna(subset=["Symbol", "Exchange"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()

    st.subheader(f"Loaded {len(df)} Tickers")
    st.dataframe(df.head())

    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
    volume_filter = st.sidebar.checkbox("Volume Spike (>1.5x avg)", True)
    breakout_filter = st.sidebar.checkbox("Price Above EMA20", True)
    china_boost = st.sidebar.checkbox("Boost Chinese Stocks", True)

    if st.sidebar.button("🔁 Refresh Results"):
        st.session_state.apply_filters = True

    if "apply_filters" not in st.session_state:
        st.session_state.apply_filters = True

    if "results_df" not in st.session_state:
        st.subheader("Fetching Data...")
        progress = st.progress(0)
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_ticker_data, row["Symbol"], row["Exchange"]): row for _, row in df.iterrows()}
            for i, future in enumerate(as_completed(futures)):
                row = futures[future]
                data = future.result()
                if data:
                    if china_boost and row["Exchange"] in ["SHZ", "SHH", "SHE", "SHA"]:
                        data["Momentum_Score"] = min(100, data["Momentum_Score"] + 5)
                    results.append(data)
                progress.progress((i + 1) / len(futures))
        st.session_state.results_df = pd.DataFrame(results)
    else:
        st.success("✔️ Loaded cached results. Use the refresh button to reapply filters.")

    if st.session_state.apply_filters:
        results_df = st.session_state.results_df
        filtered = results_df[results_df["Momentum_Score"] >= min_score]
        if volume_filter:
            filtered = filtered[filtered["Volume_Ratio"] > 1.5]
        if breakout_filter:
            filtered = filtered[filtered["Above_EMA20"]]

        st.subheader(f"Results: {len(filtered)} Stocks")

        def color_positive(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'

        st.dataframe(
            filtered.sort_values("Momentum_Score", ascending=False).style.applymap(
                color_positive, subset=["5D_Change"]
            ),
            height=600,
            use_container_width=True
        )

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Results",
            csv,
            "momentum_results.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
