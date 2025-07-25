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
REQUEST_DELAY = (1, 3)  # Longer delays for reliability
CACHE_TTL = 3600 * 6  # 6-hour cache
TIMEZONE = 'Asia/Shanghai'

# ========== ENHANCED SYMBOL MAPPING ==========
def map_symbol(symbol, exchange):
    """Enhanced symbol mapping with China-specific handling"""
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    
    # China A-shares
    if exchange in ["SHH", "SHA"]:  # Shanghai
        if not symbol.endswith(".SS"):
            return f"{symbol}.SS"
    elif exchange in ["SHZ", "SHE"]:  # Shenzhen
        if not symbol.endswith(".SZ"):
            return f"{symbol}.SZ"
    
    # Hong Kong
    elif exchange == "HKG":
        if not symbol.endswith(".HK"):
            # Pad with leading zero if 4 digits
            if len(symbol) == 4 and symbol.isdigit():
                symbol = f"0{symbol}"
            return f"{symbol}.HK"
    
    # US and others
    return symbol

# ========== MOMENTUM CALCULATION ==========
def calculate_momentum(data):
    """Complete momentum calculation with error handling"""
    if data.empty or len(data) < 15:
        return None
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # EMAs
        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        
        # Volume analysis
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg_10 if vol_avg_10 != 0 else 1
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Breakouts
        above_ema20 = close.iloc[-1] > ema20
        broke_5d_high = close.iloc[-1] > high.rolling(5).max().iloc[-2]
        
        # Score calculation
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
            "Trend": "↑ Strong" if score >= 70 else "↑ Medium" if score >= 50 else "↗ Weak"
        }
    except Exception as e:
        st.warning(f"Momentum calculation failed: {str(e)}")
        return None

# ========== DATA FETCHING ==========
@st.cache_data(ttl=CACHE_TTL)
def get_ticker_data(symbol, exchange):
    """Robust data fetcher with retry logic"""
    try:
        yf_symbol = map_symbol(symbol, exchange)
        ticker = yf.Ticker(yf_symbol)
        
        # Try with delay
        time.sleep(random.uniform(*REQUEST_DELAY))
        data = ticker.history(period="1mo")
        
        if data.empty:
            # Try alternative symbol format
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
    
    # Upload section
    uploaded_file = st.file_uploader("Upload Ticker List (Excel/CSV)", type=["xlsx", "csv"])
    if not uploaded_file:
        st.warning("Please upload a file with columns: Symbol, Exchange")
        return
    
    # Load and clean data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    df = df.dropna(subset=["Symbol", "Exchange"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
    
    # Show preview
    st.subheader(f"Loaded {len(df)} Tickers")
    st.dataframe(df.head())
    
    # Filters
    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
    volume_filter = st.sidebar.checkbox("Volume Spike (>1.5x avg)", True)
    breakout_filter = st.sidebar.checkbox("Price Above EMA20", True)
    china_boost = st.sidebar.checkbox("Boost Chinese Stocks", True)
    
    # Fetch data
    st.subheader("Fetching Data...")
    progress = st.progress(0)
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_ticker_data, row["Symbol"], row["Exchange"]): row for _, row in df.iterrows()}
        
        for i, future in enumerate(as_completed(futures)):
            row = futures[future]
            data = future.result()
            
            if data:
                # Apply China boost if enabled
                if china_boost and row["Exchange"] in ["SHZ", "SHH", "SHE", "SHA"]:
                    data["Momentum_Score"] = min(100, data["Momentum_Score"] + 5)
                results.append(data)
                
            progress.progress((i + 1) / len(futures))
    
    if not results:
        st.error("""
        No data fetched. Common solutions:
        1. Verify symbol formats (e.g., 600000.SS for Shanghai)
        2. Try fewer stocks at once
        3. Check your internet connection
        4. Some Chinese stocks may have data restrictions
        """)
        return
    
    # Process results
    results_df = pd.DataFrame(results)
    
    # Apply filters
    filtered = results_df[results_df["Momentum_Score"] >= min_score]
    if volume_filter:
        filtered = filtered[filtered["Volume_Ratio"] > 1.5]
    if breakout_filter:
        filtered = filtered[filtered["Above_EMA20"]]
    
    # Display results
    st.subheader(f"Results: {len(filtered)} Stocks")
    
    # Color formatting
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
    
    # Export
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download Results",
        csv,
        "momentum_results.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
