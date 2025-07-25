import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Enable caching for data fetching
@st.cache_data(ttl=3600)
def fetch_index_data(ticker, period='1y'):
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {str(e)}")
        return None

def calculate_rsi(data, window=14):
    try:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Add small value to avoid division by zero
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"RSI calculation failed: {e}")
        return None

def safe_get_price(data):
    try:
        if data is None or data.empty or 'Close' not in data:
            return "N/A"
        last_close = data['Close'].iloc[-1]
        return f"${last_close:,.2f}" if not pd.isna(last_close) else "N/A"
    except:
        return "N/A"

def safe_get_rsi(rsi_series):
    try:
        if rsi_series is None or rsi_series.empty:
            return "N/A"
        last_rsi = rsi_series.iloc[-1]
        return f"{last_rsi:.1f}" if not pd.isna(last_rsi) else "N/A"
    except:
        return "N/A"

def safe_get_trend(data):
    try:
        if data is None or data.empty or 'Close' not in data or len(data) < 2:
            return "N/A"
        current = data['Close'].iloc[-1]
        previous = data['Close'].iloc[-2]
        return "↑" if current > previous else "↓" if current < previous else "→"
    except:
        return "N/A"

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

    # Get tickers
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            tickers = df.iloc[:, 0].tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            tickers = ['^GSPC', '^DJI', '^IXIC']
    else:
        tickers = ['^GSPC', '^DJI', '^IXIC']
        st.info("ℹ️ Using sample data. Upload a file to analyze your own indices.")

    # Analysis tabs
    tab1, tab2 = st.tabs(["Single Index", "Compare Indices"])

    with tab1:
        selected = st.selectbox("Select Index", tickers)
        data = fetch_index_data(selected, period)
        
        if data is not None:
            try:
                data['RSI'] = calculate_rsi(data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Price", safe_get_price(data))
                with col2:
                    rsi_value = safe_get_rsi(data['RSI']) if 'RSI' in data else "N/A"
                    status = "🟢 Oversold" if rsi_value != "N/A" and float(rsi_value) < 30 else "🔴 Overbought" if rsi_value != "N/A" and float(rsi_value) > 70 else "🟠 Neutral"
                    st.metric("RSI (14)", rsi_value, status)

                # Plotting
                if not data.empty and 'Close' in data and 'RSI' in data:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    
                    ax1.plot(data.index, data['Close'], color='royalblue')
                    ax1.set_title(f"{selected} Price History")
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(data.index, data['RSI'], color='purple')
                    ax2.axhline(70, color='red', linestyle='--')
                    ax2.axhline(30, color='green', linestyle='--')
                    ax2.set_title("RSI (14-day)")
                    ax2.set_ylim(0, 100)
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                with st.expander("View Raw Data"):
                    st.dataframe(data.sort_index(ascending=False))
            except Exception as e:
                st.error(f"Error processing data: {e}")
        else:
            st.warning(f"No data available for {selected}")

    with tab2:
        st.subheader("Index Comparison")
        progress_bar = st.progress(0)
        all_data = {}
        
        for i, ticker in enumerate(tickers):
            data = fetch_index_data(ticker, period)
            if data is not None:
                try:
                    rsi = calculate_rsi(data)
                    if rsi is not None:
                        data['RSI'] = rsi
                        all_data[ticker] = data
                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")
            progress_bar.progress((i + 1) / len(tickers))
        
        if all_data:
            # Normalized price comparison
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            for ticker, data in all_data.items():
                if not data.empty and 'Close' in data and len(data) > 0:
                    norm_price = (data['Close'] / data['Close'].iloc[0]) * 100
                    ax1.plot(data.index, norm_price, label=ticker)
            ax1.set_title("Normalized Price Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # RSI comparison
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            for ticker, data in all_data.items():
                if not data.empty and 'RSI' in data and len(data) > 0:
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
                    safe_get_price(data),
                    safe_get_rsi(data['RSI']) if 'RSI' in data else "N/A",
                    safe_get_trend(data)
                ])
            
            st.dataframe(
                pd.DataFrame(
                    summary,
                    columns=["Ticker", "Price", "RSI (14)", "Trend"]
                ).sort_values("RSI", ascending=False, key=lambda x: pd.to_numeric(x.replace('N/A', '0'), errors='coerce')),
                hide_index=True
            )
        else:
            st.warning("No valid data available for comparison")

if __name__ == "__main__":
    main()
