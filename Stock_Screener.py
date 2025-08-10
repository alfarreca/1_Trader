import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib  # For technical indicators

# Configure app
st.set_page_config(page_title="Advanced Stock Screener", layout="wide")
st.title("📈 Advanced Stock Screener with Technical Indicators")

# Cache data to improve performance
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

@st.cache_data
def fetch_yfinance_data(symbols):
    data = yf.download(tickers=list(symbols), period="1y", group_by='ticker')
    return data

# Calculate RSI
def calculate_rsi(data, window=14):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Main app
def main():
    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        # Filters
        st.sidebar.header("Filters")
        selected_sector = st.sidebar.multiselect("Sector", df['Sector'].unique())
        selected_industry = st.sidebar.multiselect("Industry", df['Industry'].unique())
        
        # Apply filters
        filtered_df = df.copy()
        if selected_sector:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sector)]
        if selected_industry:
            filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industry)]
            
        if st.button("Fetch Data"):
            with st.spinner("Downloading market data..."):
                symbols = filtered_df['Symbol'].tolist()
                market_data = fetch_yfinance_data(symbols)
                
                if not market_data.empty:
                    # Display results
                    tab1, tab2 = st.tabs(["📊 Summary", "📈 Technical Analysis"])
                    
                    with tab1:
                        st.dataframe(filtered_df)
                    
                    with tab2:
                        selected_stock = st.selectbox("Select Stock", symbols)
                        
                        if selected_stock:
                            stock_data = market_data[selected_stock]
                            
                            # Create subplots
                            fig = make_subplots(
                                rows=3, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=("Price with Moving Averages", "Volume", "RSI"),
                                row_heights=[0.6, 0.2, 0.2]
                            )
                            
                            # Price with Moving Averages
                            fig.add_trace(
                                go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['Close'],
                                    name='Close Price',
                                    line=dict(color='blue')
                                ),
                                row=1, col=1
                            )
                            
                            # Add moving averages
                            for window in [20, 50]:
                                ma = stock_data['Close'].rolling(window).mean()
                                fig.add_trace(
                                    go.Scatter(
                                        x=stock_data.index,
                                        y=ma,
                                        name=f'{window}-day MA',
                                        line=dict(width=1)
                                    ),
                                    row=1, col=1
                                )
                            
                            # Volume bars
                            fig.add_trace(
                                go.Bar(
                                    x=stock_data.index,
                                    y=stock_data['Volume'],
                                    name='Volume',
                                    marker_color='rgba(100, 100, 255, 0.4)'
                                ),
                                row=2, col=1
                            )
                            
                            # RSI
                            rsi = calculate_rsi(stock_data)
                            fig.add_trace(
                                go.Scatter(
                                    x=stock_data.index,
                                    y=rsi,
                                    name='RSI (14)',
                                    line=dict(color='purple')
                                ),
                                row=3, col=1
                            )
                            
                            # Add RSI reference lines
                            fig.add_hline(y=70, row=3, col=1, line_dash="dot", line_color="red")
                            fig.add_hline(y=30, row=3, col=1, line_dash="dot", line_color="green")
                            
                            # Update layout
                            fig.update_layout(
                                height=800,
                                showlegend=True,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
