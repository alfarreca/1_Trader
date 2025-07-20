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
    suffix_map = {
        "ETR": "DE", "EPA": "PA", "LON": "L", "BIT": "MI", "STO": "ST",
        "SWX": "SW", "TSE": "TO", "ASX": "AX", "HKG": "HK"
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

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = high.diff().where(lambda x: (x > 0) & (x > low.diff().abs()), 0)
    minus_dm = (-low.diff()).where(lambda x: (x > 0) & (x > high.diff().abs()), 0)
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean().iloc[-1] if not dx.isnull().all() else dx.mean()

    score = 0
    if close.iloc[-1] > ema20 > ema50 > ema200:
        score += 30
    elif close.iloc[-1] > ema50 > ema200:
        score += 20
    elif close.iloc[-1] > ema200:
        score += 10

    if 60 <= rsi < 80:
        score += 20
    elif 50 <= rsi < 60 or 80 <= rsi <= 90:
        score += 10

    if macd_hist > 0 and macd_line_above_signal:
        score += 15

    if volume_ratio > 1.5:
        score += 15
    elif volume_ratio > 1.2:
        score += 10

    if adx > 30:
        score += 20
    elif adx > 25:
        score += 15
    elif adx > 20:
        score += 10

    score = max(0, min(100, score))

    return {
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "EMA200": round(ema200, 2),
        "RSI": round(rsi, 1),
        "MACD_Hist": round(macd_hist, 3),
        "ADX": round(adx, 1) if not np.isnan(adx) else None,
        "Volume_Ratio": round(volume_ratio, 2),
        "Momentum_Score": score,
        "Trend": "↑ Strong" if score >= 80 else 
                 "↑ Medium" if score >= 60 else 
                 "↗ Weak" if score >= 40 else "→ Neutral"
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(_ticker, exchange, yf_symbol):
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        hist = safe_yfinance_fetch(ticker_obj)
        if hist.empty or len(hist) < 50:
            return None
        momentum_data = calculate_momentum(hist)
        if not momentum_data:
            return None
        current_price = hist['Close'].iloc[-1]
        five_day_change = ((current_price / hist['Close'].iloc[-5] - 1) * 100) if len(hist) >= 5 else None
        twenty_day_change = ((current_price / hist['Close'].iloc[-20] - 1) * 100) if len(hist) >= 20 else None
        return {
            "Symbol": _ticker,
            "Exchange": exchange,
            "Price": round(current_price, 2),
            "5D_Change": round(five_day_change, 1) if five_day_change else None,
            "20D_Change": round(twenty_day_change, 1) if twenty_day_change else None,
            **momentum_data,
            "Last_Updated": datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
            "YF_Symbol": yf_symbol
        }
    except Exception as e:
        st.warning(f"Error processing {_ticker}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="S&P 500 Momentum Scanner", layout="wide")
    st.title("S&P 500 Momentum Scanner")

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
    company_file = st.file_uploader("Upload Excel file with company names (optional)", type="xlsx")

    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Select sheet to use", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        if "Symbol" not in df.columns or "Exchange" not in df.columns:
            st.error("Uploaded sheet must contain 'Symbol' and 'Exchange' columns.")
            return
        df = df.dropna(subset=["Symbol", "Exchange"]).drop_duplicates("Symbol")
        df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
        df["YF_Symbol"] = df.apply(lambda row: map_to_yfinance_symbol(row["Symbol"], row["Exchange"]), axis=1)
    else:
        st.warning("Please upload a .xlsx file with your tickers.")
        return

    df_companies = pd.DataFrame()
    if company_file is not None:
        df_companies = pd.read_excel(company_file)
        name_col = next((col for col in df_companies.columns if "name" in col.lower()), None)
        if name_col:
            df_companies = df_companies.rename(columns={name_col: "Company Name"})
            df_companies = df_companies[["Symbol", "Company Name"]].drop_duplicates("Symbol")

    exchanges = sorted(df["Exchange"].unique().tolist())
    selected_exchange = st.sidebar.selectbox("Exchange", ["All"] + exchanges)
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)

    ticker_data = []
    progress = st.progress(0, text="Fetching ticker data...")
    total = len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_ticker_data, row["Symbol"], row["Exchange"], row["YF_Symbol"]) for _, row in df.iterrows()]
        for i, f in enumerate(as_completed(futures)):
            data = f.result()
            if data:
                ticker_data.append(data)
            progress.progress((i + 1) / total, text=f"Processed {i+1}/{total} tickers")
    progress.empty()

    results_df = pd.DataFrame(ticker_data)

    if not df_companies.empty:
        results_df = results_df.merge(df_companies, on="Symbol", how="left")

    if selected_exchange != "All":
        results_df = results_df[(results_df["Exchange"] == selected_exchange)]
    filtered_df = results_df[results_df["Momentum_Score"] >= min_score].copy()

    st.dataframe(filtered_df, use_container_width=True, height=600)

    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download Filtered Results as CSV", data=csv, file_name="momentum_scanner_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
