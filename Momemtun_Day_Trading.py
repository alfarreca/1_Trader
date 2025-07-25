import streamlit as st
import pandas as pd
import yfinance as yf
import akshare as ak  # Alternative data source for Chinese stocks
from datetime import datetime, timedelta
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ========== CONFIGURATION ==========
MAX_WORKERS = 6  # Reduced for stability with Chinese data
REQUEST_DELAY = (1, 3)  # Longer delays for Chinese data
CACHE_TTL = 3600 * 6  # 6-hour cache
TIMEZONE = 'Asia/Shanghai'  # Changed to China timezone

# ========== CHINA-SPECIFIC FUNCTIONS ==========
def get_china_stock_data(symbol, exchange):
    """Fetch data for Chinese stocks using AKShare"""
    try:
        # Convert symbol to AKShare format
        clean_symbol = symbol.replace(".SS", "").replace(".SZ", "")
        
        if exchange in ["SHZ", "SHE"]:
            market = "sz"
            ak_symbol = f"{clean_symbol}.{market}"
            df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="hfq")
        elif exchange in ["SHH", "SHA"]:
            market = "sh"
            ak_symbol = f"{clean_symbol}.{market}"
            df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="hfq")
        else:
            return pd.DataFrame()
            
        # Standardize column names
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df.set_index('Date', inplace=True)
        return df.last('1mo')  # Get last month of data
        
    except Exception as e:
        st.warning(f"AKShare failed for {symbol}: {str(e)}")
        return pd.DataFrame()

def safe_fetch(ticker, exchange, symbol):
    """Unified fetch function with fallback to AKShare"""
    try:
        # First try yfinance
        time.sleep(random.uniform(*REQUEST_DELAY))
        data = ticker.history(period="1mo")
        
        if data.empty and exchange in ["SHZ", "SHH", "SHE", "SHA"]:
            # Fallback to AKShare for Chinese stocks
            data = get_china_stock_data(symbol, exchange)
            
        return data
        
    except Exception as e:
        st.warning(f"Fetch failed for {symbol}: {str(e)}")
        return pd.DataFrame()

# ========== MAIN APP ==========
def main():
    st.set_page_config(layout="wide")
    st.title("📈 China A-Share Momentum Scanner")
    
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
        yf_symbol = f"{symbol}.{'SS' if exchange in ['SHH', 'SHA'] else 'SZ'}"
        ticker = yf.Ticker(yf_symbol)
        
        data = safe_fetch(ticker, exchange, symbol)
        if data.empty:
            return None
            
        # Calculate momentum (simplified)
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
            "Above_EMA20": price > ema20
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
        No data fetched. Common solutions:
        1. Try again later (temporary API issue)
        2. Verify symbols are active (some may be suspended)
        3. Install AKShare: `pip install akshare`
        4. Use VPN with China endpoint if outside China
        """)
        return
    
    # Display results
    results_df = pd.DataFrame(results)
    st.subheader(f"Found {len(results_df)} Stocks with Data")
    st.dataframe(
        results_df.sort_values("5D_Change", ascending=False),
        height=600,
        use_container_width=True
    )
    
    # Export
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Results",
        csv,
        "china_stocks.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
