import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# App title and description
st.title('Fibonacci Financial Analyzer')
st.write("""
This app retrieves stock data, shows the price chart first, then calculates Fibonacci retracement levels based on user-defined swing points.
Now with **automated swing detection** suggestions!
""")

# Sidebar inputs
st.sidebar.header('User Input Parameters')

def get_user_input():
    ticker = st.sidebar.text_input('Stock Ticker (e.g., AAPL)', 'AAPL')
    end_date = st.sidebar.date_input('End Date', datetime.today())
    start_date = st.sidebar.date_input('Start Date', end_date - timedelta(days=365))
    show_volume = st.sidebar.checkbox('Show Volume', True)
    chart_style = st.sidebar.selectbox('Chart Style', ['Candlestick', 'Line'])
    return {
        'ticker': ticker.upper(),
        'start_date': start_date,
        'end_date': end_date,
        'show_volume': show_volume,
        'chart_style': chart_style
    }

user_input = get_user_input()

@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for this ticker and date range.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Swing detection
def detect_swings(data, window=10):
    if data is None or len(data) < window*2:
        return None, None
    highs = data['High'].rolling(window, center=True).max()
    lows = data['Low'].rolling(window, center=True).min()
    swing_high = data['High'][(data['High'] == highs) & (~highs.isna())].dropna()
    swing_low = data['Low'][(data['Low'] == lows) & (~lows.isna())].dropna()
    recent_high = swing_high[-1] if not swing_high.empty else float(data['High'].max())
    recent_low = swing_low[-1] if not swing_low.empty else float(data['Low'].min())
    return float(recent_high), float(recent_low)

# Load data
data = load_data(user_input['ticker'], user_input['start_date'], user_input['end_date'])

# Detect swings
suggested_high, suggested_low = detect_swings(data)

if data is not None:
    st.subheader(f"{user_input['ticker']} Price Chart")

    fig = go.Figure()

    if user_input['chart_style'] == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price'
        ))

    # --- Dynamic volume bar colors ---
    if user_input['show_volume']:
        vol_colors = ['#00B050' if close >= open_ else '#FF4B4B'
                      for close, open_ in zip(data['Close'], data['Open'])]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color=vol_colors,
            opacity=0.5
        ))

    fig.update_layout(
        title=f"{user_input['ticker']} Price from {user_input['start_date']} to {user_input['end_date']}",
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        height=600,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )

    if user_input['show_volume']:
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # --- Automated swing suggestion UI ---
    st.sidebar.subheader('Fibonacci Swing Points')

    if suggested_high and suggested_low:
        st.sidebar.markdown(
            f"**Suggested Swing High / Low:** {suggested_high:.2f} / {suggested_low:.2f}"
        )

    # Use session state for the autofill
    if 'swing_high' not in st.session_state:
        st.session_state.swing_high = float(suggested_high) if suggested_high else 0.0
    if 'swing_low' not in st.session_state:
        st.session_state.swing_low = float(suggested_low) if suggested_low else 0.0

    if st.sidebar.button("Use Suggested Swings"):
        st.session_state.swing_high = float(suggested_high) if suggested_high else 0.0
        st.session_state.swing_low = float(suggested_low) if suggested_low else 0.0

    swing_high = st.sidebar.number_input('Swing High Price', value=st.session_state.swing_high, key='high_in')
    swing_low = st.sidebar.number_input('Swing Low Price', value=st.session_state.swing_low, key='low_in')

    if swing_high > 0 and swing_low > 0:
        swing_diff = swing_high - swing_low
        fib_levels = {
            '0%': swing_high,
            '23.6%': swing_high - swing_diff * 0.236,
            '38.2%': swing_high - swing_diff * 0.382,
            '50%': swing_high - swing_diff * 0.5,
            '61.8%': swing_high - swing_diff * 0.618,
            '78.6%': swing_high - swing_diff * 0.786,
            '100%': swing_low
        }

        st.subheader('Fibonacci Retracement Levels')
        fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
        st.dataframe(fib_df.style.format({'Price': '{:.2f}'}))

        # Add Fibonacci lines to chart
        for level, price in fib_levels.items():
            fig.add_hline(y=price, line_dash="dot",
                          annotation_text=f"Fib {level} ({price:.2f})",
                          line_color="purple")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please enter Swing High and Swing Low prices to calculate Fibonacci retracements.")

    if st.checkbox('Show Raw Data'):
        st.subheader('Raw Data')
        st.write(data)
else:
    st.error("Failed to load data. Please check your inputs and try again.")

st.markdown("""
---
**Note:** This app is for educational purposes only. Financial data may be delayed.
""")
