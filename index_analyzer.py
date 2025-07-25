import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Enable caching for data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_index_data(ticker, period='1y'):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {str(e)}")
        return None

# Optimized RSI calculation
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Main app function
def main():
    st.set_page_config(page_title="Index Analyzer", layout="wide")
    st.title("📈 Index Analysis Dashboard")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Ticker List", type=['xlsx', 'csv'])
        period = st.selectbox(
            "Time Period",
            ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            index=3
        )
        st.markdown("### Sample Tickers")
        st.code("^GSPC (S&P 500)\n^DJI (Dow Jones)\n^IXIC (NASDAQ)", language="text")

    # Sample data if no file uploaded
    if not uploaded_file:
        default_tickers = ['^GSPC', '^DJI', '^IXIC']
        st.info("ℹ️ Using sample data. Upload a file to analyze your own indices.")
    else:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            default_tickers = df.iloc[:, 0].tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            default_tickers = ['^GSPC', '^DJI', '^IXIC']

    # Analysis tabs
    tab1, tab2 = st.tabs(["Single Index", "Compare Indices"])

    with tab1:
        selected = st.selectbox("Select Index", default_tickers)
        if data := fetch_index_data(selected, period):
            data['RSI'] = calculate_rsi(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:,.2f}")
            with col2:
                rsi_value = data['RSI'].iloc[-1]
                status = "🟢 Oversold" if rsi_value < 30 else "🔴 Overbought" if rsi_value > 70 else "🟠 Neutral"
                st.metric("RSI (14)", f"{rsi_value:.1f}", status)

            # Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Price plot
            ax1.plot(data.index, data['Close'], color='royalblue')
            ax1.set_title(f"{selected} Price History")
            ax1.grid(True, alpha=0.3)
            
            # RSI plot
            ax2.plot(data.index, data['RSI'], color='purple')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_title("RSI (14-day)")
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Show raw data
            with st.expander("View Raw Data"):
                st.dataframe(data.sort_index(ascending=False))

    with tab2:
        st.subheader("Index Comparison")
        progress_bar = st.progress(0)
        all_data = {}
        
        for i, ticker in enumerate(default_tickers):
            if data := fetch_index_data(ticker, period):
                data['RSI'] = calculate_rsi(data)
                all_data[ticker] = data
            progress_bar.progress((i + 1) / len(default_tickers))
        
        if all_data:
            # Normalized price comparison
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            for ticker, data in all_data.items():
                norm_price = (data['Close'] / data['Close'].iloc[0]) * 100
                ax1.plot(data.index, norm_price, label=ticker)
            ax1.set_title("Normalized Price Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # RSI comparison
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            for ticker, data in all_data.items():
                ax2.plot(data.index, data['RSI'], label=ticker)
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_title("RSI Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            # Summary table
            summary = []
            for ticker, data in all_data.items():
                summary.append([
                    ticker,
                    f"${data['Close'].iloc[-1]:,.2f}",
                    f"{data['RSI'].iloc[-1]:.1f}",
                    "↑" if data['Close'].iloc[-1] > data['Close'].iloc[-2] else "↓"
                ])
            
            st.dataframe(
                pd.DataFrame(
                    summary,
                    columns=["Ticker", "Price", "RSI (14)", "Trend"]
                ).sort_values("RSI", ascending=False),
                hide_index=True
            )

if __name__ == "__main__":
    main()
