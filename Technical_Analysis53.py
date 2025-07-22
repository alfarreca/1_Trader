import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ta
from io import BytesIO
import numpy as np

st.set_page_config(page_title="Stock Technical Analysis", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { max-width: 1200px; }
    .stSelectbox { margin-bottom: 20px; }
    .stFileUploader { margin-bottom: 20px; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
    .sheet-selector { margin-bottom: 15px; }
    .company-comparison { margin-top: 30px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Stock Technical Analysis Dashboard")

with st.sidebar:
    st.header("Upload Ticker List")
    uploaded_file = st.file_uploader("Choose an XLSX file with 'Symbol' and 'Exchange' columns", type=["xlsx"])
    st.header("OR")
    manual_ticker = st.text_input("Enter a single ticker (e.g. SPY, AAPL, 9618.HK)", help="For HKEX stocks use format XXXX.HK (e.g. 9618.HK)")
    st.header("Analysis Settings")
    analysis_type = st.radio("Analysis Type", ["Single Company", "Multi-Company Compare"])
    start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("today"))
    st.header("Technical Indicators")
    show_sma = st.checkbox("Show SMA (20, 50)", value=True)
    show_ema = st.checkbox("Show EMA (20)", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)

def fetch_live_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info['last_price']
    except:
        return None

@st.cache_data
def load_tickers_from_sheet(uploaded_file, selected_sheet):
    if uploaded_file and selected_sheet:
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        if 'Symbol' not in df.columns or 'Exchange' not in df.columns:
            st.error("Sheet must contain 'Symbol' and 'Exchange'.")
            return None
        df['YFinance_Symbol'] = df.apply(lambda row: f"{row['Symbol']}.HK" if row['Exchange'] == 'HKEX' else f"{row['Symbol']}", axis=1)
        df['Display_Name'] = df['YFinance_Symbol']
        return df
    return None

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        ticker = ticker.replace('.HK.HK', '.HK') if ticker.endswith('.HK.HK') else ticker
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        return data
    except:
        return None

def calculate_indicators(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    close_col = [c for c in df.columns if c.startswith('Close')][0]
    vol_col = [c for c in df.columns if c.startswith('Volume')][0]
    close_prices = df[close_col]
    if show_sma:
        df['SMA_20'] = ta.trend.sma_indicator(close_prices, 20)
        df['SMA_50'] = ta.trend.sma_indicator(close_prices, 50)
    if show_ema:
        df['EMA_20'] = ta.trend.ema_indicator(close_prices, 20)
    if show_rsi:
        df['RSI_14'] = ta.momentum.rsi(close_prices, 14)
    if show_macd:
        macd = ta.trend.MACD(close_prices)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
    if show_bollinger:
        bb = ta.volatility.BollingerBands(close_prices)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
    df['Volume_SMA_20'] = df[vol_col].rolling(window=20).mean()
    return df

def display_comparison_metrics(comparison_data, selected_companies):
    st.subheader("Comparison Metrics")
    metrics = []
    for company in selected_companies:
        if company in comparison_data:
            data = comparison_data[company]
            close_col = [c for c in data.columns if c.startswith('Close')][0]
            vol_col = [c for c in data.columns if c.startswith('Volume')][0]
            if not data.empty:
                try:
                    last_close = float(data[close_col].iloc[-1])
                    prev_close = float(data[close_col].iloc[-2]) if len(data) > 1 else last_close
                    change = last_close - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                    last_volume = int(data[vol_col].iloc[-1])

                    row = {
                        'Company': company,
                        'Price': f"${last_close:.2f}",
                        'Change': f"${change:.2f}",
                        'Pct Change': f"{pct_change:.2f}%",
                        'Volume': f"{last_volume:,}"
                    }

                    if show_sma and 'SMA_20' in data.columns:
                        row['SMA_20'] = f"{data['SMA_20'].iloc[-1]:.2f}"
                    if show_ema and 'EMA_20' in data.columns:
                        row['EMA_20'] = f"{data['EMA_20'].iloc[-1]:.2f}"
                    if show_rsi and 'RSI_14' in data.columns:
                        row['RSI_14'] = f"{data['RSI_14'].iloc[-1]:.2f}"
                    if show_macd and 'MACD' in data.columns:
                        row['MACD'] = f"{data['MACD'].iloc[-1]:.2f}"

                    metrics.append(row)

                except Exception as e:
                    st.error(f"Error processing metrics for {company}: {str(e)}")

    if metrics:
        st.dataframe(pd.DataFrame(metrics))
    else:
        st.warning("No metrics available for the selected companies")

# Main app continues as in your original version...
