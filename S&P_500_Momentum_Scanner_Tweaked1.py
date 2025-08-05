from pathlib import Path

# Define the updated script content with market cap filter included
updated_script = """
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

# ========== CONFIGURATION ==========
MAX_WORKERS = 8
REQUEST_DELAY = (0.5, 2.0)
CACHE_TTL = 3600 * 12
MAX_RETRIES = 3
TIMEZONE = 'America/New_York'

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_fetch(ticker, period="3mo"):
    time.sleep(random.uniform(*REQUEST_DELAY))
    return ticker.history(period=period)

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
        "EMA20": round(ema20, 2), "EMA50": round(ema50, 2), "EMA200": round(ema200, 2),
        "RSI": round(rsi, 1), "MACD_Hist": round(macd_hist, 3),
        "ADX": round(adx, 1) if not np.isnan(adx) else None,
        "Volume_Ratio": round(volume_ratio, 2),
        "Momentum_Score": score,
        "Trend": "↑ Strong" if score >= 80 else "↑ Medium" if score >= 60 else "↗ Weak" if score >= 40 else "→ Neutral",
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ticker_data(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        market_cap = info.get("marketCap", None)
        hist = safe_yfinance_fetch(yf_ticker)
        if hist.empty or len(hist) < 50:
            return None
        momentum_data = calculate_momentum(hist)
        if not momentum_data:
            return None
        current_price = hist['Close'].iloc[-1]
        five_day_change = ((current_price / hist['Close'].iloc[-5] - 1) * 100) if len(hist) >= 5 else None
        twenty_day_change = ((current_price / hist['Close'].iloc[-20] - 1) * 100) if len(hist) >= 20 else None
        return {
            "Symbol": ticker,
            "Price": round(current_price, 2),
            "5D_Change": round(five_day_change, 1) if five_day_change else None,
            "20D_Change": round(twenty_day_change, 1) if twenty_day_change else None,
            "Market_Cap": market_cap,
            **momentum_data,
            "Last_Updated": datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.warning(f"Error processing {ticker}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="S&P 500 Momentum Scanner", layout="wide")
    st.title("S&P 500 Momentum Scanner")

    uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Select sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

        expected_cols = ["Symbol", "Name", "Sector", "Industry Group", "Industry", "Theme", "Country", "Asset_Type", "Notes"]
        if not set(expected_cols).issubset(df.columns):
            st.error("Uploaded sheet must contain all of the following columns: " + ", ".join(expected_cols))
            return

        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["Symbol"]).drop_duplicates("Symbol")
        df["YF_Symbol"] = df["Symbol"]

        st.write(f"Loaded rows from '{selected_sheet}':", len(df))
        st.dataframe(df.head())

        min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
        min_mcap = st.sidebar.number_input("Min Market Cap (in billions)", min_value=0.0, value=10.0, step=1.0)
        max_mcap = st.sidebar.number_input("Max Market Cap (in billions)", min_value=0.0, value=1000.0, step=10.0)

        ticker_data = []
        progress = st.progress(0, text="Fetching ticker data...")
        total = len(df)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(get_ticker_data, row["YF_Symbol"])
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
            st.warning("No valid results.")
            return

        filtered = results_df[
            (results_df["Momentum_Score"] >= min_score) &
            (results_df["Market_Cap"].notna()) &
            (results_df["Market_Cap"] >= min_mcap * 1e9) &
            (results_df["Market_Cap"] <= max_mcap * 1e9)
        ].copy()
        combined = df.set_index("Symbol").join(filtered.set_index("Symbol"), how="inner").reset_index()

        st.metric("Stocks Found", len(combined))
        st.metric("Avg Momentum Score", round(combined["Momentum_Score"].mean(), 1))
        st.dataframe(combined.sort_values("Momentum_Score", ascending=False), use_container_width=True)

        csv = combined.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv, file_name="momentum_results.csv", mime="text/csv")
    else:
        st.warning("Please upload a .xlsx file with your tickers.")

if __name__ == "__main__":
    main()
"""

# Save the updated script to a new file
output_path = Path("/mnt/data/S&P_500_Momentum_Scanner_With_MarketCap.py")
output_path.write_text(updated_script)
output_path.name
