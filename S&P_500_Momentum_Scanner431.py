import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import pytz
import numpy as np

# ========== CONFIGURATION ==========
MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 2.0)
CACHE_TTL = 3600 * 12  # 12 hours
MAX_RETRIES = 3
TIMEZONE = 'America/New_York'

yf.set_tz_cache_location("cache")

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_fetch(ticker, period="3mo"):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

def exchange_suffix(ex: str) -> str:
    suffix_map = {"ETR": "DE", "EPA": "PA", "LON": "L", "BIT": "MI", "STO": "ST", "SWX": "SW", "TSE": "TO", "ASX": "AX", "HKG": "HK"}
    return suffix_map.get(ex.upper(), "")

def map_to_yfinance_symbol(symbol: str, exchange: str) -> str:
    suffix = exchange_suffix(exchange)
    return f"{symbol}.{suffix}" if suffix else symbol

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(_ticker, exchange, yf_symbol):
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = safe_yfinance_fetch(ticker_obj)
        if hist.empty or len(hist) < 50:
            return None
        current_price = hist['Close'].iloc[-1]
        five_day_change = ((current_price/hist['Close'].iloc[-5]-1)*100) if len(hist) >= 5 else None
        twenty_day_change = ((current_price/hist['Close'].iloc[-20]-1)*100) if len(hist) >= 20 else None
        return {
            "Symbol": _ticker,
            "Exchange": exchange,
            "Price": round(current_price, 2),
            "5D_Change": round(five_day_change, 1) if five_day_change else None,
            "20D_Change": round(twenty_day_change, 1) if twenty_day_change else None,
        }
    except:
        return None

def main():
    st.set_page_config(page_title="S&P 500 Momentum Scanner", layout="wide")
    st.title("S&P 500 Momentum Scanner")

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Select sheet to use", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

        if "Symbol" not in df.columns or "Exchange" not in df.columns:
            st.error("The selected sheet must contain 'Symbol' and 'Exchange' columns.")
            return

        df.dropna(subset=["Symbol", "Exchange"], inplace=True)
        df.drop_duplicates(subset=["Symbol"], inplace=True)
        df["Exchange"] = df["Exchange"].str.strip().str.upper()

        df["YF_Symbol"] = df.apply(lambda row: map_to_yfinance_symbol(row["Symbol"], row["Exchange"]), axis=1)

        ticker_data = []
        progress = st.progress(0)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"]) for _, row in df.iterrows()]
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    ticker_data.append(result)
                progress.progress((i + 1) / len(df))

        progress.empty()
        results_df = pd.DataFrame(ticker_data)
        st.dataframe(results_df)

        st.download_button("Download CSV", results_df.to_csv(index=False), "results.csv")

if __name__ == "__main__":
    main()
