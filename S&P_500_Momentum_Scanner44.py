import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import pytz
import numpy as np
import plotly.express as px

# ========== CONFIGURATION ==========
MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 2.0)
CACHE_TTL = 3600 * 12
MAX_RETRIES = 3
TIMEZONE = 'America/New_York'

yf.set_tz_cache_location("cache")

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_fetch(ticker, period="3mo"):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

def exchange_suffix(ex: str) -> str:
    suffix_map = {
        "ETR": "DE", "EPA": "PA", "LON": "L", "BIT": "MI", "STO": "ST",
        "SWX": "SW", "TSE": "TO", "TSX": "TO", "TSXV": "V", "ASX": "AX",
        "HKG": "HK", "CNY": "SS", "TORONTO": "TO"
    }
    return suffix_map.get(ex.upper(), "")

def map_to_yfinance_symbol(symbol: str, exchange: str) -> str:
    if exchange.upper() in ["NYSE", "NASDAQ"]:
        return symbol
    suffix = exchange_suffix(exchange)
    return f"{symbol}.{suffix}" if suffix else symbol

def calculate_momentum(hist):
    if hist.empty or len(hist) < 50:
        return None
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']
    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
    macd_line_above_signal = macd.iloc[-1] > macd_signal.iloc[-1]
    vol_avg_20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_avg_20 if vol_avg_20 != 0 else 1
    return {
        "EMA20": round(ema20, 2), "EMA50": round(ema50, 2), "EMA200": round(ema200, 2),
        "RSI": round(rsi, 1), "MACD_Hist": round(macd_hist, 3),
        "Volume_Ratio": round(volume_ratio, 2)
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(_ticker, exchange, yf_symbol):
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = safe_yfinance_fetch(ticker_obj, period="6mo")
        if hist.empty or len(hist) < 50:
            return None
        momentum_data = calculate_momentum(hist)
        if not momentum_data:
            return None
        current_price = hist['Close'].iloc[-1]
        week_pct = (hist['Close'] / hist['Close'].iloc[0]) * 100
        return {
            "Symbol": _ticker, "Exchange": exchange, "Price": round(current_price, 2),
            "YF_Symbol": yf_symbol, "Hist": hist, "NormPerf": week_pct
        } | momentum_data
    except Exception as e:
        st.warning(f"Error processing {_ticker}: {str(e)}")
        return None

def plot_normalized_chart(results_df):
    norm_df = pd.DataFrame()
    for _, row in results_df.iterrows():
        perf = row["NormPerf"]
        if isinstance(perf, pd.Series):
            temp = perf.reset_index()
            temp.columns = ["Date", "Performance"]
            temp["Symbol"] = row["Symbol"]
            norm_df = pd.concat([norm_df, temp])
    if not norm_df.empty:
        fig = px.line(norm_df, x="Date", y="Performance", color="Symbol", title="Normalized Performance (Start = 100)")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="S&P 500 Momentum Scanner", layout="wide")
    st.title("S&P 500 Momentum Scanner")

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Select sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.write(f"Loaded rows from '{selected_sheet}':", len(df))

        if not {"Symbol", "Exchange"}.issubset(df.columns):
            st.error("Uploaded sheet must contain 'Symbol' and 'Exchange' columns")
            return

        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
        df.dropna(subset=["Symbol", "Exchange"], inplace=True)
        df.drop_duplicates("Symbol", inplace=True)
        df["YF_Symbol"] = df.apply(lambda row: map_to_yfinance_symbol(row["Symbol"], row["Exchange"]), axis=1)

        ticker_data = []
        progress = st.progress(0, text="Fetching ticker data...")
        total = len(df)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"])
                for _, row in df.iterrows()
            ]
            for i, f in enumerate(as_completed(futures)):
                data = f.result()
                if data:
                    ticker_data.append(data)
                progress.progress((i + 1) / total, text=f"Processed {i + 1}/{total} tickers")
        progress.empty()

        results_df = pd.DataFrame(ticker_data)
        if results_df.empty:
            st.warning("No valid data was fetched.")
            return

        st.dataframe(results_df[["Symbol", "Exchange", "Price", "EMA20", "EMA50", "EMA200", "RSI", "MACD_Hist", "Volume_Ratio"]], use_container_width=True)
        plot_normalized_chart(results_df)
    else:
        st.warning("Please upload a .xlsx file with your tickers.")

if __name__ == "__main__":
    main()
