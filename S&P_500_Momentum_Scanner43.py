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

def calculate_di_crossovers(hist, period=14):
    high = hist['High']
    low = hist['Low']
    close = hist['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - close.shift()),
        np.abs(low - close.shift())
    ])
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    bullish_crossover = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    bearish_crossover = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
    return plus_di, minus_di, bullish_crossover, bearish_crossover

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
    plus_di_c, minus_di_c, bullish_cross, bearish_cross = calculate_di_crossovers(hist)
    last_bullish = bool(bullish_cross.iloc[-1])
    last_bearish = bool(bearish_cross.iloc[-1])
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
    if last_bullish:
        score += 10
    if last_bearish:
        score -= 10
    score = max(0, min(100, score))
    return {
        "EMA20": round(ema20, 2), "EMA50": round(ema50, 2), "EMA200": round(ema200, 2),
        "RSI": round(rsi, 1), "MACD_Hist": round(macd_hist, 3),
        "ADX": round(adx, 1) if not np.isnan(adx) else None,
        "Volume_Ratio": round(volume_ratio, 2),
        "Momentum_Score": score,
        "Trend": "↑ Strong" if score >= 80 else "↑ Medium" if score >= 60 else "↗ Weak" if score >= 40 else "→ Neutral",
        "Bullish_Crossover": last_bullish,
        "Bearish_Crossover": last_bearish,
        "plus_di_last": round(plus_di_c.iloc[-1], 1) if not np.isnan(plus_di_c.iloc[-1]) else None,
        "minus_di_last": round(minus_di_c.iloc[-1], 1) if not np.isnan(minus_di_c.iloc[-1]) else None,
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
            "Symbol": _ticker, "Exchange": exchange, "Price": round(current_price, 2),
            "5D_Change": round(five_day_change, 1) if five_day_change else None,
            "20D_Change": round(twenty_day_change, 1) if twenty_day_change else None,
            **momentum_data,
            "Last_Updated": datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
            "YF_Symbol": yf_symbol
        }
    except Exception as e:
        st.warning(f"Error processing {_ticker}: {str(e)}")
        return None

def display_results(filtered_df):
    if filtered_df.empty:
        st.warning("No stocks match your current filters.")
        return
    st.metric("Stocks Found", len(filtered_df))
    st.metric("Avg Momentum Score", round(filtered_df["Momentum_Score"].mean(), 1))
    st.dataframe(filtered_df.sort_values("Momentum_Score", ascending=False), use_container_width=True, height=600)

def display_symbol_details(selected_symbol):
    if not selected_symbol:
        return
    try:
        symbol_data = st.session_state.filtered_results[
            st.session_state.filtered_results["Symbol"] == selected_symbol
        ].iloc[0]
        st.subheader(f"{selected_symbol} Detailed Analysis")
        st.json(symbol_data.to_dict())
    except Exception as e:
        st.error(f"Error loading {selected_symbol}: {str(e)}")

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
        st.dataframe(df.head())

        expected_cols = {"Symbol", "Exchange"}
        if not expected_cols.issubset(set(df.columns)):
            st.error(f"Uploaded sheet must contain these columns: {', '.join(expected_cols)}")
            return

        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Exchange"] = df["Exchange"].astype(str).str.strip().str.upper()
        before_drop = len(df)
        df = df.dropna(subset=["Symbol", "Exchange"]).drop_duplicates("Symbol")
        st.write(f"Dropped rows: {before_drop - len(df)} after cleaning.")
        df["YF_Symbol"] = df.apply(lambda row: map_to_yfinance_symbol(row["Symbol"], row["Exchange"]), axis=1)
    else:
        st.warning("Please upload a .xlsx file with your tickers.")
        return

    exchanges = sorted(df["Exchange"].unique().tolist())
    selected_exchange = st.sidebar.selectbox("Exchange", ["All"] + exchanges)
    min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)

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
    st.session_state["raw_results_df"] = results_df.copy()

    if not results_df.empty:
        if selected_exchange != "All":
            filtered = results_df[
                (results_df["Momentum_Score"] >= min_score) &
                (results_df["Exchange"] == selected_exchange)
            ].copy()
        else:
            filtered = results_df[results_df["Momentum_Score"] >= min_score].copy()
    else:
        filtered = pd.DataFrame()

    st.session_state.filtered_results = filtered
    display_results(filtered)

    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button("Download Filtered Results as CSV", data=csv, file_name="momentum_scanner_results.csv", mime="text/csv")

        symbol_options = ["— Select a symbol —"] + filtered["Symbol"].tolist()
        last_selected = st.session_state.get("symbol_select", symbol_options[0])
        if last_selected not in symbol_options:
            last_selected = symbol_options[0]

        selected = st.selectbox("Select a symbol for details", options=symbol_options, index=symbol_options.index(last_selected), key="symbol_select")
        if selected != symbol_options[0]:
            display_symbol_details(selected)

if __name__ == "__main__":
    main()
