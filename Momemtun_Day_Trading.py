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

# ========== CHINESE TICKER FIXES ==========
def exchange_suffix(ex: str) -> str:
    """Enhanced exchange suffix mapping with focus on Chinese markets"""
    suffix_map = {
        # Chinese exchanges
        "SHZ": "SZ",  # Shenzhen
        "SHH": "SS",  # Shanghai
        "SHA": "SS",  # Alternative Shanghai
        "SHE": "SZ",  # Alternative Shenzhen
        
        # Other Asian
        "HKG": "HK",  # Hong Kong
        "KOE": "KS",  # Korea Exchange
        "TSE": "T",   # Tokyo
        "TW": "TW",   # Taiwan
        
        # European
        "FRA": "F",   # Frankfurt
        "EPA": "PA",  # Paris
        "LON": "L",   # London
        
        # North American
        "TOR": "TO",  # Toronto
        "TSX": "TO",
        "NMS": "",    # Nasdaq
        "NYQ": "",    # NYSE
        "NYSE": "",
        "NASDAQ": "",
        
        # Others
        "ASX": "AX",  # Australia
        "BTS": "", "PCX": "", "ASE": "", "NGM": "", "NCM": ""
    }
    return suffix_map.get(ex.upper(), "")

def map_to_yfinance_symbol(symbol: str, exchange: str) -> str:
    """Special handling for Chinese and other international tickers"""
    # Clean symbol
    symbol = str(symbol).strip().upper()
    
    # Handle Chinese A-shares
    if exchange.upper() in ["SHZ", "SHE"]:
        if not symbol.endswith(".SZ"):
            return f"{symbol}.SZ"
    elif exchange.upper() in ["SHH", "SHA"]:
        if not symbol.endswith(".SS"):
            return f"{symbol}.SS"
    
    # Handle Hong Kong stocks
    if exchange.upper() == "HKG":
        if not symbol.endswith(".HK"):
            # Handle 5-digit HK stocks (add leading zero)
            if len(symbol) == 4 and symbol.isdigit():
                symbol = f"0{symbol}"
            return f"{symbol}.HK"
    
    # Default mapping
    suffix = exchange_suffix(exchange)
    return f"{symbol}.{suffix}" if suffix else symbol

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_fetch(ticker, period=SCAN_PERIOD):
    """Fetch data with retries and special handling for Chinese stocks"""
    try:
        time.sleep(random.uniform(*REQUEST_DELAY))
        data = ticker.history(period=period)
        
        # Special handling for empty Chinese stock data
        if data.empty and hasattr(ticker, 'ticker'):
            symbol = ticker.ticker
            # Try alternative formats for Chinese stocks
            if ".SZ" in symbol:
                alt_symbol = symbol.replace(".SZ", ".SZ")
                if alt_symbol != symbol:
                    alt_ticker = yf.Ticker(alt_symbol)
                    data = alt_ticker.history(period=period)
            elif ".SS" in symbol:
                alt_symbol = symbol.replace(".SS", ".SS")
                if alt_symbol != symbol:
                    alt_ticker = yf.Ticker(alt_symbol)
                    data = alt_ticker.history(period=period)
                    
        return data
    except Exception as e:
        st.warning(f"Fetch failed for {ticker.ticker if hasattr(ticker, 'ticker') else ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_momentum(hist, exchange=None):
    """Enhanced momentum calculation with China-specific adjustments"""
    if hist.empty or len(hist) < 10:
        return None
    
    # Handle different column names from alternative sources
    close = hist.get('Close', hist.get('c', hist.get('close', None)))
    high = hist.get('High', hist.get('h', hist.get('high', None)))
    low = hist.get('Low', hist.get('l', hist.get('low', None)))
    volume = hist.get('Volume', hist.get('v', hist.get('vol', None)))
    
    if None in [close, high, low, volume]:
        return None
    
    # Convert to pandas Series if needed
    close = pd.Series(close) if not isinstance(close, pd.Series) else close
    volume = pd.Series(volume) if not isinstance(volume, pd.Series) else volume
    
    # Short-term EMAs
    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema20 = close.ewm(span=20).mean().iloc[-1]
    
    # Volume analysis (10-day avg)
    vol_avg_10 = volume.rolling(10).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg_10 if vol_avg_10 != 0 else 1
    volume_spike = vol_ratio > 1.5  # Reduced threshold for Chinese stocks
    
    # Breakout conditions
    broke_5d_high = close.iloc[-1] > high.rolling(5).max().iloc[-2]
    above_ema20 = close.iloc[-1] > ema20
    
    # RSI calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # Adjust scoring for Chinese stocks
    base_score = 0
    if ema5 > ema10 > ema20: base_score += 25
    if above_ema20: base_score += 20
    if 40 < rsi < 70: base_score += 20
    if volume_spike: base_score += 15
    if broke_5d_high: base_score += 10
    
    # Additional points for Chinese stocks showing strong momentum
    if exchange and exchange.upper() in ["SHZ", "SHH", "SHE", "SHA"]:
        if volume_spike and above_ema20:
            base_score += 10
    
    score = min(100, base_score)
    
    return {
        "EMA5": round(ema5, 2),
        "EMA10": round(ema10, 2),
        "EMA20": round(ema20, 2),
        "RSI": round(rsi, 1),
        "Volume_Ratio": round(vol_ratio, 2),
        "Volume_Spike": volume_spike,
        "Above_EMA20": above_ema20,
        "5D_Breakout": broke_5d_high,
        "Momentum_Score": score,
        "Trend": "↑ Strong" if score >= 70 else "↑ Medium" if score >= 50 else "↗ Weak",
    }

@st.cache_data(ttl=CACHE_TTL)
def get_ticker_data(_ticker, exchange, yf_symbol):
    """Enhanced ticker data fetcher with better error handling"""
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = safe_yfinance_fetch(ticker_obj)
        
        if hist.empty:
            st.warning(f"No data for {_ticker} ({yf_symbol})")
            return None
            
        momentum = calculate_momentum(hist, exchange)
        if not momentum:
            return None
            
        return {
            "Symbol": _ticker,
            "Exchange": exchange,
            "Price": round(hist['Close'].iloc[-1], 2),
            "5D_Change": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100, 1) 
                      if len(hist) >= 5 else None,
            **momentum
        }
    except Exception as e:
        st.error(f"Error processing {_ticker} ({yf_symbol}): {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Global Momentum Scanner (Including Chinese Stocks)")
    
    # Upload and filter
    uploaded_file = st.file_uploader("Upload Ticker List (Excel/CSV)", type=["xlsx", "csv"])
    if not uploaded_file:
        st.warning("Upload a file with columns: Symbol, Exchange")
        return
    
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    
    # Clean data
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
    
    # Map to Yahoo Finance symbols
    df["YF_Symbol"] = df.apply(lambda r: map_to_yfinance_symbol(r["Symbol"], r["Exchange"]), axis=1)
    
    # Show preview
    st.write(f"Loaded {len(df)} tickers. Sample:")
    st.dataframe(df.head())
    
    # Filters
    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
    volume_filter = st.sidebar.checkbox("Volume Spike (>1.5x avg)", False)
    breakout_filter = st.sidebar.checkbox("Price Above EMA20", True)
    china_boost = st.sidebar.checkbox("Boost Chinese Stocks", True)
    
    # Fetch data
    st.subheader("Fetching Data...")
    progress = st.progress(0)
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"]): row
            for _, row in df.iterrows()
        }
        
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
        st.error("No data fetched. Check ticker symbols and connections.")
        return
    
    # Apply filters
    results_df = pd.DataFrame(results)
    filtered = results_df[results_df["Momentum_Score"] >= min_score]
    
    if volume_filter:
        filtered = filtered[filtered["Volume_Spike"]]
    if breakout_filter:
        filtered = filtered[filtered["Above_EMA20"]]
    
    # Display results
    st.subheader(f"Results ({len(filtered)} stocks)")
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
