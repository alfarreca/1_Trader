import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ta  # Technical analysis library
from io import BytesIO
import numpy as np

st.set_page_config(
    page_title="Stock Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
    <style>
    .main { max-width: 1200px; }
    .stSelectbox, .stFileUploader { margin-bottom: 20px; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
    .sheet-selector { margin-bottom: 15px; }
    .company-comparison { margin-top: 30px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Stock Technical Analysis Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Ticker List")
    uploaded_file = st.file_uploader("Choose an XLSX file with 'Symbol' and 'Exchange' columns", type=["xlsx"])
    st.header("OR")
    manual_ticker = st.text_input("Enter a single ticker (e.g. SPY, AAPL, 9618.HK)", help="For HKEX stocks use format XXXX.HK")
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

# --- Utility Functions ---
def fetch_live_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info['last_price']
    except Exception:
        return None

def get_sheet_names(uploaded_file):
    if uploaded_file:
        try:
            return pd.ExcelFile(uploaded_file).sheet_names
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return []

@st.cache_data
def load_tickers_from_sheet(uploaded_file, selected_sheet):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        if 'Symbol' not in df.columns or 'Exchange' not in df.columns:
            st.error("Sheet must contain 'Symbol' and 'Exchange' columns.")
            return None
        df['YFinance_Symbol'] = df.apply(lambda row: f"{row['Symbol']}.HK" if row['Exchange'] == 'HKEX' else row['Symbol'], axis=1)
        df['Display_Name'] = df['YFinance_Symbol']
        return df
    except Exception as e:
        st.error(f"Error reading sheet {selected_sheet}: {e}")
        return None

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        if ticker.endswith('.HK.HK'):
            ticker = ticker.replace('.HK.HK', '.HK')
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.warning(f"No data for {ticker}")
            return None
        return data.dropna()
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None

def calculate_indicators(df):
    if df is None or df.empty:
        return df

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    if show_sma:
        df['SMA_20'] = ta.trend.sma_indicator(close, 20)
        df['SMA_50'] = ta.trend.sma_indicator(close, 50)
    if show_ema:
        df['EMA_20'] = ta.trend.ema_indicator(close, 20)
    if show_rsi:
        df['RSI_14'] = ta.momentum.rsi(close, 14)
    if show_macd:
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
    if show_bollinger:
        bb = ta.volatility.BollingerBands(close)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()

    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    return df

def display_comparison_metrics(comparison_data, selected_companies):
    st.subheader("Price & Volume Metrics")
    rows = []
    for company in selected_companies:
        df = comparison_data.get(company)
        if df is not None and not df.empty:
            close = df['Close']
            vol = df['Volume']
            last = close.iloc[-1]
            prev = close.iloc[-2] if len(close) > 1 else last
            change = last - prev
            pct = (change / prev) * 100 if prev != 0 else 0
            rows.append({
                'Company': company,
                'Price': f"${last:.2f}",
                'Change': f"${change:.2f}",
                'Pct Change': f"{pct:.2f}%",
                'Volume': f"{vol.iloc[-1]:,.0f}"
            })
    if rows:
        st.table(pd.DataFrame(rows))

def display_comparison_indicators(comparison_data, selected_companies):
    st.subheader("Technical Indicators Comparison")
    rows = []
    for company in selected_companies:
        df = comparison_data.get(company)
        if df is not None and not df.empty:
            try:
                row = {
                    'Company': company,
                    'SMA_20': round(df['SMA_20'].iloc[-1], 2) if 'SMA_20' in df else None,
                    'SMA_50': round(df['SMA_50'].iloc[-1], 2) if 'SMA_50' in df else None,
                    'EMA_20': round(df['EMA_20'].iloc[-1], 2) if 'EMA_20' in df else None,
                    'RSI_14': round(df['RSI_14'].iloc[-1], 2) if 'RSI_14' in df else None,
                    'MACD': round(df['MACD'].iloc[-1], 2) if 'MACD' in df else None,
                    'MACD_Signal': round(df['MACD_Signal'].iloc[-1], 2) if 'MACD_Signal' in df else None,
                    'BB_Upper': round(df['BB_Upper'].iloc[-1], 2) if 'BB_Upper' in df else None,
                    'BB_Lower': round(df['BB_Lower'].iloc[-1], 2) if 'BB_Lower' in df else None,
                }
                rows.append(row)
            except Exception as e:
                st.warning(f"Error processing {company}: {e}")
    if rows:
        st.dataframe(pd.DataFrame(rows))

def plot_comparison_chart(comparison_data, selected_companies):
    fig = go.Figure()
    for company in selected_companies:
        df = comparison_data.get(company)
        if df is not None and not df.empty:
            norm_price = (df['Close'] / df['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=df.index, y=norm_price, mode='lines', name=company))
    fig.update_layout(title="Normalized Price Comparison (Base=100)",
                      xaxis_title="Date", yaxis_title="Normalized Price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- Main ---
def main():
    sheet_names = get_sheet_names(uploaded_file)
    selected_sheet = sheet_names[0] if len(sheet_names) == 1 else st.selectbox("Select Sheet", sheet_names) if sheet_names else None
    tickers_df = None
    if manual_ticker and not uploaded_file:
        tickers_df = pd.DataFrame({'Symbol': [manual_ticker], 'Exchange': ['MANUAL'], 'YFinance_Symbol': [manual_ticker], 'Display_Name': [manual_ticker]})
    elif uploaded_file:
        tickers_df = load_tickers_from_sheet(uploaded_file, selected_sheet)

    if tickers_df is not None:
        if analysis_type == "Single Company" or (manual_ticker and not uploaded_file):
            selected_display = tickers_df['Display_Name'].iloc[0] if len(tickers_df) == 1 else st.selectbox("Select Ticker", tickers_df['Display_Name'])
            ticker_symbol = tickers_df[tickers_df['Display_Name'] == selected_display]['YFinance_Symbol'].values[0]
            stock_data = load_stock_data(ticker_symbol, start_date, end_date)
            if stock_data is not None:
                stock_data = calculate_indicators(stock_data)
                st.dataframe(stock_data.tail())
        elif analysis_type == "Multi-Company Compare":
            selected_companies = st.multiselect("Select companies (2-5)", tickers_df['Display_Name'], default=tickers_df['Display_Name'].head(2).tolist())
            if 2 <= len(selected_companies) <= 5:
                comparison_data = {}
                for company in selected_companies:
                    ticker = tickers_df[tickers_df['Display_Name'] == company]['YFinance_Symbol'].values[0]
                    data = load_stock_data(ticker, start_date, end_date)
                    if data is not None:
                        comparison_data[company] = calculate_indicators(data)
                display_comparison_metrics(comparison_data, selected_companies)
                display_comparison_indicators(comparison_data, selected_companies)
                plot_comparison_chart(comparison_data, selected_companies)
            else:
                st.info("Please select 2-5 companies.")

if __name__ == '__main__':
    main()
