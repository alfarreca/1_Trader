import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import pytz
import numpy as np

# ========== CONFIGURATION ==========
MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 1.5)  # Faster delays
CACHE_TTL = 3600 * 4  # 4-hour cache (refresh daily)
TIMEZONE = 'America/New_York'
SCAN_PERIOD = "1mo"  # Shorter timeframe for swing trading

yf.set_tz_cache_location("cache")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_fetch(ticker, period=SCAN_PERIOD):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

def exchange_suffix(ex: str) -> str:
    suffix_map = {
        "SHZ": "SZ", "SHH": "SS", "KOE": "KS", "HKG": "HK", 
        "FRA": "F", "JPX": "T", "TOR": "TO", "ASX": "AX",
        "NMS": "", "NYQ": "", "PCX": "", "BTS": "", "ASE": "",
        "NGM": "", "NCM": "", "PAR": "PA", "LSE": "L"
    }
    return suffix_map.get(ex.upper(), "")

def map_to_yfinance_symbol(symbol: str, exchange: str) -> str:
    suffix = exchange_suffix(exchange)
    return f"{symbol}.{suffix}" if suffix else symbol

def calculate_momentum(hist):
    if hist.empty or len(hist) < 10:  # Reduced minimum bars
        return None
    
    close, high, low, volume = hist['Close'], hist['High'], hist['Low'], hist['Volume']
    
    # Short-term EMAs
    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema20 = close.ewm(span=20).mean().iloc[-1]
    
    # Volume analysis (10-day avg)
    vol_avg_10 = volume.rolling(10).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg_10 if vol_avg_10 != 0 else 1
    volume_spike = vol_ratio > 1.5  # Reduced from 2.0
    
    # Breakout conditions
    broke_5d_high = close.iloc[-1] > high.rolling(5).max().iloc[-2]  # Previous day's high
    above_ema20 = close.iloc[-1] > ema20  # Simpler breakout condition
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # Momentum Score (0-100)
    score = 0
    if ema5 > ema10 > ema20: score += 25  # Strong short-term trend
    if above_ema20: score += 20            # Price above EMA20
    if 40 < rsi < 70: score += 20          # Ideal RSI range
    if volume_spike: score += 15           # Volume confirms
    if broke_5d_high: score += 10          # Bonus for 5D breakout
    
    return {
        "EMA5": round(ema5, 2),
        "EMA10": round(ema10, 2),
        "EMA20": round(ema20, 2),
        "RSI": round(rsi, 1),
        "Volume_Ratio": round(vol_ratio, 2),
        "Volume_Spike": volume_spike,
        "Above_EMA20": above_ema20,
        "5D_Breakout": broke_5d_high,
        "Momentum_Score": min(100, score),
        "Trend": "↑ Strong" if score >= 70 else "↑ Medium" if score >= 50 else "↗ Weak",
    }

@st.cache_data(ttl=CACHE_TTL)
def get_ticker_data(_ticker, exchange, yf_symbol):
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = safe_yfinance_fetch(ticker_obj)
        if hist.empty:
            st.warning(f"No data for {_ticker} ({yf_symbol})")
            return None
        momentum = calculate_momentum(hist)
        if not momentum:
            return None
        return {
            "Symbol": _ticker,
            "Exchange": exchange,
            "Price": round(hist['Close'].iloc[-1], 2),
            "5D_Change": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100, 1) if len(hist) >= 5 else None,
            **momentum
        }
    except Exception as e:
        st.error(f"Error with {_ticker} ({yf_symbol}): {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Swing Momentum Scanner (2-5 Day Horizon)")
    
    # Upload and filter
    uploaded_file = st.file_uploader("Upload Ticker List (Excel/CSV)", type=["xlsx", "csv"])
    if not uploaded_file:
        st.warning("Upload a file with columns: Symbol, Exchange")
        return
    
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    df["YF_Symbol"] = df.apply(lambda r: map_to_yfinance_symbol(r["Symbol"], r["Exchange"]), axis=1)
    
    # Filters
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
    volume_filter = st.sidebar.checkbox("Volume Spike (>1.5x avg)", False)  # Reduced threshold
    breakout_filter = st.sidebar.checkbox("Price Above EMA20", True)  # More reliable than 5D breakout
    
    # Fetch data
    progress = st.progress(0)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"]) for _, row in df.iterrows()]
        for i, f in enumerate(as_completed(futures)):
            if data := f.result():
                results.append(data)
            progress.progress((i + 1) / len(futures))
    
    if not results:
        st.error("No data fetched. Check ticker symbols.")
        return
    
    # Apply filters
    results_df = pd.DataFrame(results)
    filtered = results_df[results_df["Momentum_Score"] >= min_score]
    if volume_filter:
        filtered = filtered[filtered["Volume_Spike"]]
    if breakout_filter:
        filtered = filtered[filtered["Above_EMA20"]]
    
    # Display
    st.metric("Stocks Found", len(filtered))
    st.dataframe(
        filtered.sort_values("Momentum_Score", ascending=False).reset_index(drop=True),
        column_config={
            "Volume_Spike": st.column_config.CheckboxColumn("Volume Spike?"),
            "Above_EMA20": st.column_config.CheckboxColumn("Above EMA20?")
        },
        use_container_width=True,
        height=600
    )
    
    # Export
    st.download_button(
        "💾 Download Results",
        filtered.to_csv(index=False),
        "momentum_scan.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
