import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure the app
st.set_page_config(layout="wide", page_title="Fibonacci Trader Pro")
st.title("📈 Fibonacci Trader Pro")
st.markdown("""
**Professional Fibonacci analysis with bulletproof swing detection**
""")

# ===== Enhanced Data Loading =====
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if data.empty:
                st.warning(f"No data returned for {ticker}. Trying alternative method...")
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    start=start_date, 
                    end=end_date + timedelta(days=1),
                    auto_adjust=True
                )
                if data.empty:
                    raise ValueError("Empty dataframe after alternative download")
            # Ensure we have numeric data
            if not pd.api.types.is_numeric_dtype(data['High']):
                data = data.apply(pd.to_numeric, errors='coerce')
            return data.dropna()
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt == max_retries - 1:
                st.error(f"Failed to download data for {ticker} after {max_retries} attempts")
                return None
            time.sleep(2)

# ===== Bulletproof Swing Detection =====
def detect_swings(data, lookback_period=20):
    try:
        if data is None or len(data) < 5:
            return None, None
        # Ensure we're working with the most recent data
        recent_data = data[-lookback_period:] if len(data) > lookback_period else data
        # Convert to numeric if needed
        highs = pd.to_numeric(recent_data['High'], errors='coerce').dropna()
        lows = pd.to_numeric(recent_data['Low'], errors='coerce').dropna()
        if len(highs) < 2 or len(lows) < 2:
            return None, None
        # Detect swing highs (peaks)
        swing_high_mask = (highs.shift(1) < highs) & (highs > highs.shift(-1))
        swing_highs = highs[swing_high_mask]
        # Detect swing lows (troughs)
        swing_low_mask = (lows.shift(1) > lows) & (lows < lows.shift(-1))
        swing_lows = lows[swing_low_mask]
        # Find the most recent valid swing pair
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            latest_high = swing_highs.iloc[-1]
            # Find the most recent low before the high
            prior_lows = swing_lows[swing_lows.index < swing_highs.index[-1]]
            latest_low = prior_lows.iloc[-1] if len(prior_lows) > 0 else swing_lows.iloc[-1]
            # Validate the swing pair
            if latest_high > latest_low:
                return float(latest_high), float(latest_low)
        # Fallback to recent extremes if no swings detected
        return float(highs.max()), float(lows.min())
    except Exception as e:
        st.error(f"Swing detection error: {str(e)}")
        return None, None

# ===== Fibonacci Calculation =====
def calculate_fib_levels(high, low):
    try:
        if high is None or low is None or high <= low:
            raise ValueError("Invalid swing points")
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100%': low,
            '161.8%': high - diff * 1.618
        }
    except Exception as e:
        st.error(f"Fibonacci calculation error: {str(e)}")
        return {}

# ===== Main Application =====
with st.sidebar:
    st.header("🔧 Settings")
    ticker = st.text_input("Stock/Crypto Ticker", "AAPL").strip().upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("End Date", datetime.today())
    analysis_mode = st.radio("Analysis Mode", ["Auto-Detect", "Manual"], index=0)
    if analysis_mode == "Manual":
        st.subheader("Manual Swing Points")
        manual_high = st.number_input("Swing High Price", value=0.0, step=0.01, min_value=0.0)
        manual_low = st.number_input("Swing Low Price", value=0.0, step=0.01, min_value=0.0)
    else:
        st.subheader("Auto-Detect Settings")
        sensitivity = st.slider("Lookback Period (days)", 5, 90, 20)

if st.button("Run Analysis", type="primary"):
    with st.spinner("Loading market data..."):
        data = load_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        # Get swing points
        if analysis_mode == "Auto-Detect":
            swing_high, swing_low = detect_swings(data, sensitivity)
            if swing_high is not None and swing_low is not None:
                st.success(f"📊 Auto-detected swings | High: {swing_high:.2f} | Low: {swing_low:.2f}")
            else:
                st.warning("⚠️ Could not detect valid swings - using recent extremes")
                swing_high, swing_low = float(data['High'].max()), float(data['Low'].min())
                st.info(f"Using recent extremes | High: {swing_high:.2f} | Low: {swing_low:.2f}")
        else:
            if manual_high > manual_low > 0:
                swing_high, swing_low = manual_high, manual_low
                st.success("✅ Using manual swing points")
            else:
                st.warning("⚠️ Invalid manual inputs - switching to auto-detection")
                swing_high, swing_low = detect_swings(data, sensitivity)
                if swing_high is None or swing_low is None:
                    swing_high, swing_low = float(data['High'].max()), float(data['Low'].min())

        # Calculate and display Fibonacci levels
        fib_levels = calculate_fib_levels(swing_high, swing_low)

        if fib_levels:
            with st.expander("📝 Fibonacci Levels", expanded=True):
                fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
                st.dataframe(
                    fib_df.style.format({'Price': '{:.2f}'}),
                    use_container_width=True
                )
                # Download button
                csv = fib_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Levels",
                    data=csv,
                    file_name=f"{ticker}_fib_levels.csv",
                    mime="text/csv"
                )

            # ====== CUSTOM COLORED FIBONACCI PLOTTING ======
            fib_colors = {
                '0%': 'purple',
                '23.6%': 'blue',
                '38.2%': 'teal',
                '50%': 'orange',
                '61.8%': 'gold',        # "Golden ratio" level
                '78.6%': 'magenta',
                '100%': 'red',
                '161.8%': 'green'
            }

            fig = go.Figure()

            # Candlestick trace
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))

            # Swing points
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[swing_high],
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-down'),
                name='Swing High'
            ))
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[swing_low],
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-up'),
                name='Swing Low'
            ))

            # Fibonacci levels with custom colors
            for level, price in fib_levels.items():
                color = fib_colors.get(level, 'purple')  # Fallback to purple if not mapped
                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=f"{level}",
                    annotation_position="right"
                )

            fig.update_layout(
                title=f"{ticker} Price with Fibonacci Levels",
                height=700,
                xaxis_rangeslider_visible=False,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("🔍 View Raw Data"):
            st.dataframe(data.sort_index(ascending=False))
    else:
        st.error("❌ Failed to load data. Please check:")
        st.markdown("""
        - Ticker symbol is correct
        - Date range is valid
        - Market was open during this period
        """)

st.markdown("---")
st.caption("ℹ️ For cryptocurrencies, use formats like 'BTC-USD' or 'ETH-USD'")
