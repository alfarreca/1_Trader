import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ========== CONFIGURATION ==========
MAX_WORKERS = 4  # Conservative for stability
REQUEST_DELAY = (1, 3)  # Longer delays for Chinese data
CACHE_TTL = 3600 * 6  # 6-hour cache
TIMEZONE = 'Asia/Shanghai'

# ========== CHINA-SPECIFIC FUNCTIONS ==========
def map_china_symbol(symbol, exchange):
    """Proper symbol mapping for Chinese exchanges"""
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()
    
    # Handle Shanghai stocks
    if exchange in ["SHH", "SHA"]:
        if not symbol.endswith(".SS"):
            return f"{symbol}.SS"
    
    # Handle Shenzhen stocks
    elif exchange in ["SHZ", "SHE"]:
        if not symbol.endswith(".SZ"):
            return f"{symbol}.SZ"
    
    return symbol

def safe_fetch(symbol, exchange):
    """Robust fetcher with China-specific handling"""
    try:
        time.sleep(random.uniform(*REQUEST_DELAY))
        
        # Special handling for China
        if exchange in ["SHH", "SHA", "SHZ", "SHE"]:
            yf_symbol = map_china_symbol(symbol, exchange)
            ticker = yf.Ticker(yf_symbol)
            
            # Try with and without exchange suffix
            data = ticker.history(period="1mo")
            if data.empty:
                ticker = yf.Ticker(symbol)  # Try without suffix
                data = ticker.history(period="1mo")
                
            # Convert currency for China A-shares (RMB to USD)
            if not data.empty and 'Close' in data:
                data['Close'] = data['Close'] / 6.5  # Approximate conversion
                
            return data
            
        # For non-China stocks
        ticker = yf.Ticker(symbol)
        return ticker.history(period="1mo")
        
    except Exception as e:
        st.warning(f"Failed to fetch {symbol}: {str(e)}")
        return pd.DataFrame()

# ========== MAIN APP ==========
def main():
    st.set_page_config(layout="wide")
    st.title("📈 China-Compatible Momentum Scanner")
    
    # Upload section
    uploaded_file = st.file_uploader("Upload Ticker List", type=["xlsx", "csv"])
    if not uploaded_file:
        st.warning("Please upload a file with columns: Symbol, Exchange")
        return
    
    # Load data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    
    # Clean data
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
    
    # Show preview
    st.subheader(f"Loaded {len(df)} Tickers")
    st.dataframe(df.head())
    
    # Fetch data
    st.subheader("Fetching Data...")
    progress = st.progress(0)
    results = []
    
    def process_ticker(row):
        symbol, exchange = row["Symbol"], row["Exchange"]
        data = safe_fetch(symbol, exchange)
        
        if data.empty or len(data) < 10:
            return None
            
        # Calculate basic momentum
        close = data["Close"]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        price = close.iloc[-1]
        change_5d = (price/close.iloc[-5] - 1)*100 if len(close) >=5 else 0
        
        return {
            "Symbol": symbol,
            "Exchange": exchange,
            "Price": round(price, 2),
            "5D_Change": round(change_5d, 1),
            "EMA20": round(ema20, 2),
            "Above_EMA20": price > ema20,
            "Data_Points": len(data)
        }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, row) for _, row in df.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            progress.progress((i + 1) / len(futures))
    
    if not results:
        st.error("""
        No data fetched. Try these solutions:
        1. Verify symbols are correct (e.g., 600000.SS for Shanghai)
        2. Try fewer stocks at once
        3. Use VPN with Asian endpoint if outside China
        4. Some Chinese stocks may have data restrictions
        """)
        return
    
    # Display results
    results_df = pd.DataFrame(results)
    st.subheader(f"Found {len(results_df)} Stocks with Data")
    
    # Add color to 5D change
    def color_change(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    
    st.dataframe(
        results_df.style.applymap(color_change, subset=['5D_Change']),
        height=600,
        use_container_width=True
    )
    
    # Export
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Results",
        csv,
        "stock_results.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
