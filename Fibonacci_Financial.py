import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(layout="wide", page_title="Fibonacci Trader Pro")
st.title("📈 Fibonacci Trader Pro")
st.markdown("**Professional Fibonacci analysis with bulletproof swing detection**")

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
            if not pd.api.types.is_numeric_dtype(data['High']):
                data = data.apply(pd.to_numeric, errors='coerce')
            return data.dropna()
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt == max_retries - 1:
                st.error(f"Failed to download data for {ticker} after {max_retries} attempts")
                return None
            time.sleep(2)

def detect_swings(data, lookback_period=20):
    try:
        if data is None or len(data) < 5:
            return None, None
        recent_data = data[-lookback_period:] if len(data) > lookback_period else data
        highs = pd.to_numeric(recent_data['High'], errors='coerce').dropna()
        lows = pd.to_numeric(recent_data['Low'], errors='coerce').dropna()
        if len(highs) < 2 or len(lows) < 2:
            return None, None
        swing_high_mask = (highs.shift(1) < highs) & (highs > highs.shift(-1))
        swing_highs = highs[swing_high_mask]
        swing_low_mask = (lows.shift(1) > lows) & (lows < lows.shift(-1))
        swing_lows = lows[swing_low_mask]
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            latest_high = swing_highs.iloc[-1]
            prior_lows = swing_lows[swing_lows.index < swing_highs.index[-1]]
            latest_low = prior_lows.iloc[-1] if len(prior_lows) > 0 else swing_lows.iloc[-1]
            if latest_high > latest_low:
                return float(latest_high), float(latest_low)
        return float(highs.max()), float(lows.min())
    except Exception as e:
        st.error(f"Swing detection error: {str(e)}")
        return None, None

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

with st.sidebar:
    st.header("🔧 Settings")
    ticker = st.text_input("Stock/Crypto Ticker", "AAPL").strip().upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("End Date", datetime.today())
    analysis_mode = st.radio("Analysis Mode", ["Auto-Detect", "Manual"], index=0)
    sensitivity = st.slider("Lookback Period (days)", 5, 90, 20) if analysis_mode == "Auto-Detect" else None

if st.button("Run Analysis", type="primary"):
    with st.spinner("Loading market data..."):
        data = load_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        # --- Step 1: Auto-detect swings and show initial chart ---
        swing_high, swing_low = detect_swings(data, sensitivity or 20)
        if swing_high is not None and swing_low is not None and swing_high > swing_low:
            fib_levels = calculate_fib_levels(swing_high, swing_low)
            st.info(f"Auto-detected swings | High: {swing_high:.2f} | Low: {swing_low:.2f}")

            # ----- Chart plotting -----
            fib_colors = {
                '0%': 'purple',
                '23.6%': 'blue',
                '38.2%': 'teal',
                '50%': 'orange',
                '61.8%': 'gold',
                '78.6%': 'magenta',
                '100%': 'red',
                '161.8%': 'green'
            }
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
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
            for level, price in fib_levels.items():
                color = fib_colors.get(level, 'purple')
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
            with st.expander("📝 Fibonacci Levels", expanded=True):
                fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
                st.dataframe(
                    fib_df.style.format({'Price': '{:.2f}'}),
                    use_container_width=True
                )
                csv = fib_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Levels",
                    data=csv,
                    file_name=f"{ticker}_fib_levels.csv",
                    mime="text/csv"
                )
            with st.expander("🔍 View Raw Data"):
                st.dataframe(data.sort_index(ascending=False))

            # --- Step 2: If Manual mode, prompt for swing points below the chart ---
            if analysis_mode == "Manual":
                st.subheader("Manual Swing Points")
                with st.form("manual_swings_form"):
                    manual_high = st.number_input("Swing High Price", value=float(swing_high), step=0.01, min_value=0.0)
                    manual_low = st.number_input("Swing Low Price", value=float(swing_low), step=0.01, min_value=0.0)
                    submitted = st.form_submit_button("Update with Manual Swings")
                if submitted:
                    if manual_high is not None and manual_low is not None and manual_high > manual_low > 0:
                        fib_levels_manual = calculate_fib_levels(manual_high, manual_low)
                        st.success(f"Manual swings | High: {manual_high:.2f} | Low: {manual_low:.2f}")
                        fig_manual = go.Figure()
                        fig_manual.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price'
                        ))
                        fig_manual.add_trace(go.Scatter(
                            x=[data.index[-1]],
                            y=[manual_high],
                            mode='markers',
                            marker=dict(color='green', size=12, symbol='triangle-down'),
                            name='Swing High'
                        ))
                        fig_manual.add_trace(go.Scatter(
                            x=[data.index[-1]],
                            y=[manual_low],
                            mode='markers',
                            marker=dict(color='red', size=12, symbol='triangle-up'),
                            name='Swing Low'
                        ))
                        for level, price in fib_levels_manual.items():
                            color = fib_colors.get(level, 'purple')
                            fig_manual.add_hline(
                                y=price,
                                line_dash="dot",
                                line_color=color,
                                annotation_text=f"{level}",
                                annotation_position="right"
                            )
                        fig_manual.update_layout(
                            title=f"{ticker} Price with Manual Fibonacci Levels",
                            height=700,
                            xaxis_rangeslider_visible=False,
                            showlegend=True
                        )
                        st.plotly_chart(fig_manual, use_container_width=True)
                        with st.expander("📝 Fibonacci Levels (Manual)", expanded=True):
                            fib_df_manual = pd.DataFrame.from_dict(fib_levels_manual, orient='index', columns=['Price'])
                            st.dataframe(
                                fib_df_manual.style.format({'Price': '{:.2f}'}),
                                use_container_width=True
                            )
                            csv_manual = fib_df_manual.reset_index().to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Levels (Manual)",
                                data=csv_manual,
                                file_name=f"{ticker}_fib_levels_manual.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning("Please enter valid manual swing points: high > low > 0.")
        else:
            st.error("Swing detection failed. Please adjust your lookback period or date range.")
    else:
        st.error("❌ Failed to load data. Please check:")
        st.markdown("""
        - Ticker symbol is correct
        - Date range is valid
        - Market was open during this period
        """)

st.markdown("---")
st.caption("ℹ️ For cryptocurrencies, use formats like 'BTC-USD' or 'ETH-USD'")
