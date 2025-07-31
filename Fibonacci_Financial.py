import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the app
st.set_page_config(layout="wide")
st.title("🎯 Fibonacci Retracement Tool")
st.markdown("""
**Manual Mode**: Set exact swing points  
**Auto Mode**: Let the app detect swings  
*Tip: Zoom in on the chart to fine-tune manual selections*
""")

# ===== Data Loading =====
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ===== Swing Point Detection =====
def detect_swings(data):
    highs = data['High']
    lows = data['Low']
    
    # Find swing highs (peak values)
    swing_highs = highs[(highs.shift(1) < highs) & (highs > highs.shift(-1))]
    # Find swing lows (trough values)
    swing_lows = lows[(lows.shift(1) > lows) & (lows < lows.shift(-1))]
    
    if len(swing_highs) > 0 and len(swing_lows) > 0:
        return swing_highs[-1], swing_lows[swing_lows.index < swing_highs.index[-1]][-1]
    return highs.max(), lows.min()

# ===== Fibonacci Calculation =====
def calculate_fib_levels(high, low):
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

# ===== User Interface =====
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=180))
    end_date = st.date_input("End Date", datetime.today())
    
    analysis_mode = st.radio("Analysis Mode", ["Manual", "Auto-Detect"], index=1)
    
    if analysis_mode == "Manual":
        st.subheader("Manual Swing Points")
        manual_high = st.number_input("Swing High Price", value=0.0, step=0.01)
        manual_low = st.number_input("Swing Low Price", value=0.0, step=0.01)
    else:
        st.subheader("Auto-Detect Settings")
        lookback = st.slider("Swing Sensitivity (days)", 5, 60, 20)

# ===== Main Execution =====
data = load_data(ticker, start_date, end_date)

if data is not None:
    # Get swing points based on selected mode
    if analysis_mode == "Auto-Detect":
        swing_high, swing_low = detect_swings(data[-lookback:])
        st.success(f"Auto-detected swings: High={swing_high:.2f}, Low={swing_low:.2f}")
    else:
        swing_high, swing_low = manual_high, manual_low
        if swing_high > 0 and swing_low > 0:
            st.success(f"Using manual swings: High={swing_high:.2f}, Low={swing_low:.2f}")
        else:
            st.warning("Enter both swing prices for manual mode")
            swing_high, swing_low = data['High'].max(), data['Low'].min()

    # Calculate Fibonacci levels
    fib_levels = calculate_fib_levels(swing_high, swing_low)
    
    # Display Fibonacci table
    with st.expander("Fibonacci Levels", expanded=True):
        fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
        st.dataframe(fib_df.style.format({'Price': '{:.2f}'}), use_container_width=True)

    # ===== Interactive Chart =====
    fig = go.Figure()
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Swing points markers
    fig.add_trace(go.Scatter(
        x=[data.index[-1]],
        y=[swing_high],
        mode='markers',
        marker=dict(color='green', size=12),
        name='Swing High'
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[-1]],
        y=[swing_low],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Swing Low'
    ))
    
    # Fibonacci levels
    for level, price in fib_levels.items():
        fig.add_hline(
            y=price,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"{level} ({price:.2f})",
            annotation_position="right"
        )
    
    fig.update_layout(
        title=f"{ticker} Fibonacci Retracements",
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ===== Data Export =====
    st.download_button(
        label="Download Fibonacci Levels",
        data=pd.DataFrame(fib_levels.items(), columns=['Level', 'Price']).to_csv().encode('utf-8'),
        file_name=f"{ticker}_fib_levels.csv",
        mime="text/csv"
    )
else:
    st.error("Couldn't load data. Check your inputs and try again.")

st.markdown("---")
st.caption("Tip: In manual mode, check the raw data below to find exact swing point values")

if data is not None and st.checkbox("Show Raw Data"):
    st.dataframe(data.sort_index(ascending=False))
