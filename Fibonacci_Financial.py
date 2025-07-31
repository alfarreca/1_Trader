import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# App title and description
st.title('Fibonacci Financial Analyzer')
st.write("""
This app retrieves stock data, shows the price chart first, then calculates Fibonacci retracement levels based on user-defined swing points.
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
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load data
data = load_data(user_input['ticker'], user_input['start_date'], user_input['end_date'])

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

    if user_input['show_volume']:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(100, 100, 255, 0.3)'
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

    # User inputs for Fibonacci
    st.sidebar.subheader('Fibonacci Swing Points')
    swing_high = st.sidebar.number_input('Swing High Price', value=0.0)
    swing_low = st.sidebar.number_input('Swing Low Price', value=0.0)

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
