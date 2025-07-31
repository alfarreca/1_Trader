import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from requests.exceptions import RequestException

# Configure the app
st.set_page_config(layout="wide", page_title="Fibonacci Trader Pro")
st.title("📈 Fibonacci Trader Pro")
st.markdown("""
**Professional-grade Fibonacci analysis tool**  
*Now with robust error handling and retry logic*
""")

# ===== Improved Data Loading =====
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,  # Explicitly set to handle the warning
                progress=False
            )
            if data.empty:
                st.warning(f"No data returned for {ticker}. Trying alternative method...")
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start_date, end=end_date + timedelta(days=1))
                if data.empty:
                    raise ValueError("Empty dataframe after alternative download")
            
            return data
        
        except (RequestException, ValueError) as e:
            st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt == max_retries - 1:
                st.error(f"Failed to download data for {ticker} after {max_retries} attempts")
                return None
            time.sleep(2)  # Wait before retrying

# ===== Enhanced Swing Detection =====
def detect_swings(data, min_swing_period=5):
    try:
        if len(data) < min_swing_period:
            return data['High'].max(), data['Low'].min()
        
        highs = data['High']
        lows = data['Low']
        
        # More robust swing detection with minimum period between swings
        swing_highs = highs[
            (highs.shift(min_swing_period) < highs) & 
            (highs > highs.shift(-min_swing_period))
        ]
        
        swing_lows = lows[
            (lows.shift(min_swing_period) > lows) & 
            (lows < lows.shift(-min_swing_period))
        ]
        
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            # Find the most recent valid swing high/low pair
            latest_high = swing_highs[-1]
            valid_lows = swing_lows[swing_lows.index < swing_highs.index[-1]]
            
            if len(valid_lows) > 0:
                latest_low = valid_lows[-1]
                return latest_high, latest_low
        
        # Fallback to recent extremes if no swings detected
        return highs.max(), lows.min()
    
    except Exception as e:
        st.error(f"Swing detection error: {str(e)}")
        if len(data) > 0:
            return data['High'].max(), data['Low'].min()
        return 0, 0

# ===== Fibonacci Calculation =====
def calculate_fib_levels(high, low):
    try:
        if high <= low:
            raise ValueError("Swing high must be greater than swing low")
            
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100%': low,
            '161.8%': high - diff * 1.618,
            '261.8%': high - diff * 2.618
        }
    except Exception as e:
        st.error(f"Fibonacci calculation error: {str(e)}")
        return {}

# ===== User Interface =====
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
        default_high = st.number_input("Swing High Price", value=0.0, step=0.01)
        default_low = st.number_input("Swing Low Price", value=0.0, step=0.01)
    else:
        st.subheader("Auto-Detect Settings")
        sensitivity = st.slider("Swing Sensitivity", 1, 30, 10, 
                              help="Higher values detect larger swings")

# ===== Main Execution =====
if st.button("Run Analysis", type="primary"):
    with st.spinner("Fetching market data..."):
        data = load_data(ticker, start_date, end_date)
    
    if data is not None and not data.empty:
        # Get swing points
        if analysis_mode == "Auto-Detect":
            swing_high, swing_low = detect_swings(data[-sensitivity*3:])  # Lookback 3x sensitivity
            st.success(f"📊 Auto-detected swings | High: {swing_high:.2f} | Low: {swing_low:.2f}")
        else:
            if default_high > 0 and default_low > 0 and default_high > default_low:
                swing_high, swing_low = default_high, default_low
                st.success("✅ Using manual swing points")
            else:
                st.warning("⚠️ Invalid manual inputs - using auto-detection")
                swing_high, swing_low = detect_swings(data[-sensitivity*3:])
        
        # Calculate Fibonacci levels
        fib_levels = calculate_fib_levels(swing_high, swing_low)
        
        # Display Fibonacci table
        with st.expander("📝 Fibonacci Levels", expanded=True):
            if fib_levels:
                fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
                st.dataframe(
                    fib_df.style.format({'Price': '{:.2f}'}),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = fib_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Levels",
                    data=csv,
                    file_name=f"{ticker}_fib_levels.csv",
                    mime="text/csv"
                )
            else:
                st.error("Could not calculate Fibonacci levels")
        
        # ===== Interactive Chart =====
        st.subheader(f"📊 {ticker} Price Analysis")
        
        fig = go.Figure()
        
        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C'
        ))
        
        # Swing points markers if valid
        if swing_high > 0 and swing_low > 0 and swing_high > swing_low:
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[swing_high],
                mode='markers',
                marker=dict(color='#2ECC71', size=15, symbol='triangle-down'),
                name='Swing High'
            ))
            
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[swing_low],
                mode='markers',
                marker=dict(color='#E74C3C', size=15, symbol='triangle-up'),
                name='Swing Low'
            ))
            
            # Fibonacci levels
            for level, price in fib_levels.items():
                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color="#9B59B6",
                    annotation_text=f"{level} ({price:.2f})",
                    annotation_position="right",
                    annotation_font_size=10
                )
        
        fig.update_layout(
            title=f"{ticker} Price with Fibonacci Levels",
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data section
        with st.expander("🔍 View Raw Data"):
            st.dataframe(data.sort_index(ascending=False))
    
    else:
        st.error("❌ Failed to load data. Possible reasons:")
        st.markdown("""
        - Invalid ticker symbol
        - Market closed for selected date range
        - Temporary API issues
        - Delisted or non-existent symbol
        """)
        st.markdown("Try popular symbols like `AAPL`, `MSFT`, `BTC-USD`, or `ETH-USD`")

st.markdown("---")
st.caption("ℹ️ Tip: For cryptocurrencies, use formats like 'BTC-USD' or 'ETH-USD'")
