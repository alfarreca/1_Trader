import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from curl_cffi.requests.exceptions import HTTPError
import os

# --- STATIC CONFIG ---
PROGRESS_FILE = "progress_results.csv"
EXTRA_METRICS = [
    'dividend_yield', 'eps', 'revenue', 'free_cash_flow', 'ebitda', 'gross_margin'
]
EXTRA_METRICS_LABELS = {
    'dividend_yield': 'Dividend Yield',
    'eps': 'EPS (TTM)',
    'revenue': 'Revenue',
    'free_cash_flow': 'Free Cash Flow',
    'ebitda': 'EBITDA',
    'gross_margin': 'Gross Margin',
}
SCORE_WEIGHTS = {
    'pe_ratio': 0.15,
    'peg_ratio': 0.15,
    'debt_to_equity': -0.1,
    'current_ratio': 0.05,
    'return_on_equity': 0.1,
    'profit_margin': 0.1,
    'dividend_yield': 0.05,
    'eps': 0.1,
    'revenue': 0.1,
    'free_cash_flow': 0.1,
    'ebitda': 0.05,
    'gross_margin': 0.05,
}

def clean_ticker(ticker):
    if not isinstance(ticker, str) or ticker.strip() == "":
        return ""
    ticker = ticker.replace('..', '.').strip()
    if ticker.endswith('.E'):
        ticker = ticker[:-2] + '.IS'
    return ticker

@st.cache_data(ttl=3600)
def get_tickers_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return tickers

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if not ticker or ticker.strip() == "":
        print(f"Skipping empty or invalid ticker: {ticker}")
        return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            'ticker': ticker,
            'pe_ratio': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'return_on_equity': info.get('returnOnEquity'),
            'profit_margin': info.get('profitMargins'),
            'company_name': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap'),
            'dividend_yield': info.get('dividendYield'),
            'eps': info.get('trailingEps'),
            'revenue': info.get('totalRevenue'),
            'free_cash_flow': info.get('freeCashflow'),
            'ebitda': info.get('ebitda'),
            'gross_margin': info.get('grossMargins'),
        }
        hist = stock.history(period="1d")
        fundamentals['price'] = hist['Close'].iloc[-1] if not hist.empty else None
        return fundamentals
    except HTTPError as e:
        st.warning(f"HTTP Error fetching info for {ticker}: {e}")
        return None
    except Exception as e:
        st.warning(f"Could not fetch data for {ticker}: {e}")
        return None

def normalize_value(value, min_val, max_val, reverse=False):
    if value is None or min_val is None or max_val is None:
        return 0
    if min_val == max_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return 1 - normalized if reverse else normalized

def calculate_scores(fundamentals_list):
    if not fundamentals_list:
        return []
    df = pd.DataFrame(fundamentals_list)
    for metric in SCORE_WEIGHTS:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    metrics = {metric: {
        'min': df[metric].min(),
        'max': df[metric].max(),
        'reverse': SCORE_WEIGHTS[metric] < 0
    } for metric in SCORE_WEIGHTS}
    for metric, params in metrics.items():
        df[f'{metric}_score'] = df[metric].apply(
            lambda x: normalize_value(x, params['min'], params['max'], params['reverse']) * abs(SCORE_WEIGHTS[metric])
        )
    df['total_score'] = df[[f'{m}_score' for m in SCORE_WEIGHTS]].sum(axis=1)
    min_score, max_score = df['total_score'].min(), df['total_score'].max()
    df['normalized_score'] = df['total_score'].apply(
        lambda x: ((x - min_score) / (max_score - min_score)) * 100 if max_score != min_score else 50
    )
    return df.to_dict('records')

def display_results(results):
    if not results:
        st.warning("No results to display")
        return
    df = pd.DataFrame([r for r in results if r is not None]).sort_values('normalized_score', ascending=False)
    if df.empty:
        st.warning("No valid data available for the selected tickers.")
        return
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else "N/A")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")

    st.columns(3)[0].metric("Total Stocks", len(df))
    st.columns(3)[1].metric("Highest Score", f"{df['normalized_score'].max():.1f}")
    st.columns(3)[2].metric("Average Score", f"{df['normalized_score'].mean():.1f}")

    min_score = st.slider("Minimum Score", 0, 100, 0)
    sectors = st.multiselect("Filter by Sector", df['sector'].unique(), df['sector'].unique())
    filtered_df = df[(df['normalized_score'] >= min_score) & (df['sector'].isin(sectors))]

    def add_icons(row):
        sector_mean = df[df['sector'] == row['sector']].mean(numeric_only=True)
        for m in EXTRA_METRICS:
            val, avg = row[m], sector_mean[m]
            if val is not None and avg is not None and pd.notnull(val) and pd.notnull(avg):
                if val > avg:
                    row[m] = f"🟢 {val:.3g}"
                elif val < avg:
                    row[m] = f"🔴 {val:.3g}"
                else:
                    row[m] = f"{val:.3g}"
            elif val is not None and pd.notnull(val):
                row[m] = f"{val:.3g}"
            else:
                row[m] = "N/A"
        return row

    table_df = filtered_df.copy().apply(add_icons, axis=1)

    st.dataframe(
        table_df[['ticker', 'company_name', 'sector', 'normalized_score', 'price', 'market_cap'] + EXTRA_METRICS]
            .rename(columns={
            'ticker': 'Ticker',
            'company_name': 'Company',
            'sector': 'Sector',
            'normalized_score': 'Score',
            'price': 'Price',
            'market_cap': 'Market Cap',
            **EXTRA_METRICS_LABELS
        }),
        use_container_width=True
    )

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"scores_{datetime.now().strftime('%Y%m%d')}.csv")
    if len(filtered_df) == 0:
        return

    tickers_list = filtered_df['ticker'].tolist()
    if ('selected_ticker' not in st.session_state) or (st.session_state.selected_ticker not in tickers_list):
        if tickers_list:
            st.session_state.selected_ticker = tickers_list[0]
        else:
            st.session_state.selected_ticker = ""

    selected_ticker = st.selectbox(
        "Select stock for details",
        tickers_list,
        index=tickers_list.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in tickers_list else 0,
        key="selected_ticker"
    )

    selected_stock = df[df['ticker'] == selected_ticker].iloc[0]

    st.markdown("### Company Details")
    col1, col2 = st.columns(2)
    col1.metric("Company", selected_stock['company_name'])
    col1.metric("Sector", selected_stock['sector'])
    col1.metric("Price", selected_stock['price'])
    col2.metric("Fundamental Score", f"{selected_stock['normalized_score']:.1f}")
    col2.metric("P/E Ratio", selected_stock['pe_ratio'])
    col2.metric("PEG Ratio", selected_stock['peg_ratio'])

    sector_df = df[df['sector'] == selected_stock['sector']]
    metric_vals = []
    sector_avgs = []
    metric_labels = []
    for metric in EXTRA_METRICS:
        val = selected_stock[metric]
        avg = sector_df[metric].mean() if not sector_df.empty else None
        if val is not None and avg is not None and pd.notnull(val) and pd.notnull(avg):
            metric_vals.append(val)
            sector_avgs.append(avg)
            metric_labels.append(EXTRA_METRICS_LABELS[metric])
    if metric_labels:
        fig, ax = plt.subplots(figsize=(7, 3))
        bar_width = 0.35
        x = range(len(metric_labels))
        ax.bar(x, metric_vals, width=bar_width, label="Selected Stock")
        ax.bar([i + bar_width for i in x], sector_avgs, width=bar_width, label="Sector Avg")
        ax.set_xticks([i + bar_width / 2 for i in x])
        ax.set_xticklabels(metric_labels, rotation=25, ha='right')
        ax.set_ylabel("Value")
        ax.set_title("Metrics: Stock vs Sector Avg")
        ax.legend()
        st.pyplot(fig)

    if not sector_df.empty:
        st.markdown("### Sector Averages (This Sector)")
        sector_means = sector_df[EXTRA_METRICS].mean()
        summary_df = pd.DataFrame({
            "Metric": [EXTRA_METRICS_LABELS[m] for m in EXTRA_METRICS],
            "Sector Average": [f"{sector_means[m]:.3g}" if pd.notnull(sector_means[m]) else "N/A" for m in EXTRA_METRICS]
        })
        st.dataframe(summary_df, hide_index=True)

    all_sector_avgs = df.groupby('sector')[EXTRA_METRICS].mean().reset_index()
    all_sector_avgs.columns = ['Sector'] + [EXTRA_METRICS_LABELS[m] for m in EXTRA_METRICS]
    with st.expander("See all sector averages (all S&P 500 sectors)"):
        st.dataframe(all_sector_avgs)

    sector_csv = all_sector_avgs.to_csv(index=False).encode('utf-8')
    st.download_button("Download All Sector Averages CSV", sector_csv, "sector_averages.csv")

    st.markdown("#### Top 5 Stocks per Metric")
    for m in EXTRA_METRICS:
        st.write(f"**{EXTRA_METRICS_LABELS[m]}:**")
        tops = df[['ticker', 'company_name', m]].dropna().sort_values(m, ascending=False).head(5)
        st.dataframe(tops.rename(columns={m: 'Value'}), hide_index=True)
    st.markdown("#### Bottom 5 Stocks per Metric")
    for m in EXTRA_METRICS:
        st.write(f"**{EXTRA_METRICS_LABELS[m]}:**")
        bottoms = df[['ticker', 'company_name', m]].dropna().sort_values(m, ascending=True).head(5)
        st.dataframe(bottoms.rename(columns={m: 'Value'}), hide_index=True)

    try:
        hist = yf.Ticker(selected_stock['ticker']).history(period="1y")
        if not hist.empty:
            st.markdown("### Price History (1Y)")
            st.line_chart(hist['Close'])
    except Exception:
        pass

    st.markdown("### Metric Comparison vs. Sector Average")
    for metric in EXTRA_METRICS:
        val = selected_stock[metric]
        avg = sector_df[metric].mean() if not sector_df.empty else None
        label = EXTRA_METRICS_LABELS[metric]
        if avg is not None and val is not None and pd.notnull(avg) and pd.notnull(val):
            diff = val - avg
            color = "🟢" if diff > 0 else "🔴"
            st.write(f"**{label}:** {val:.3g} | Sector Avg: {avg:.3g} {color} ({'above' if diff>0 else 'below'} sector avg)")
        else:
            st.write(f"**{label}:** {val if val is not None else 'N/A'} | Sector Avg: {'N/A'}")

def main():
    st.set_page_config(page_title="Stock Analyzer", layout="wide")
    st.title("📈 S&P 500 Fundamental Analysis (Pro Dashboard)")

    # --- UI: Sidebar controls for batch size and sleep interval ---
    with st.sidebar:
        st.markdown("### Batch Processing Settings")
        batch_size = st.number_input("Batch Size", min_value=10, max_value=500, value=100, step=10)
        sleep_between = st.slider("Sleep Between Requests (sec)", 0.5, 10.0, 1.5, step=0.1)

    # Load previously processed results if available
    if os.path.exists(PROGRESS_FILE):
        all_data = pd.read_csv(PROGRESS_FILE)
        processed_tickers = set(all_data['ticker'].astype(str).tolist())
    else:
        all_data = pd.DataFrame()
        processed_tickers = set()

    uploaded_file = st.file_uploader("Upload your S&P 500 Excel (.xlsx) file", type=["xlsx"])

    if uploaded_file is not None:
        tickers = get_tickers_from_excel(uploaded_file)
        st.success(f"Loaded {len(tickers)} tickers from file.")

        to_process = [t for t in tickers if clean_ticker(t) not in processed_tickers]
        st.info(f"{len(processed_tickers)} tickers already processed, {len(to_process)} remain in this run.")

        if st.button(f"Analyze Next Batch (Size: {batch_size})"):
            with st.spinner(f"Fetching data for next {batch_size} tickers..."):
                batch = to_process[:batch_size]
                data = []
                progress = st.progress(0)
                for i, t in enumerate(batch):
                    cleaned_ticker = clean_ticker(t)
                    if cleaned_ticker:
                        fundamentals = get_fundamentals(cleaned_ticker)
                    else:
                        fundamentals = None
                    if fundamentals is not None:
                        data.append(fundamentals)
                    else:
                        st.info(f"Skipping {t} due to data fetch issues.")
                    progress.progress((i + 1) / len(batch))
                    time.sleep(sleep_between)
                progress.empty()
                if data:
                    batch_df = pd.DataFrame(data)
                    # Merge to all_data and drop duplicates by 'ticker'
                    if not all_data.empty:
                        all_data = pd.concat([all_data, batch_df], ignore_index=True)
                        all_data = all_data.drop_duplicates(subset='ticker', keep='last')
                    else:
                        all_data = batch_df
                    # Save after every batch
                    all_data.to_csv(PROGRESS_FILE, index=False)
                    st.success(f"Batch processed and saved. {len(all_data)} total tickers now processed.")
                else:
                    st.error("No data was fetched in this batch.")

        # Show results if any available
        if not all_data.empty:
            results = calculate_scores(all_data.to_dict('records'))
            display_results(results)

    else:
        st.info("Please upload an Excel file with tickers in the first column.")

    # Show sector averages in sidebar if results available
    if not all_data.empty:
        df = pd.DataFrame(calculate_scores(all_data.to_dict('records')))
        with st.sidebar:
            st.markdown("### S&P 500 Sector Averages")
            for m in EXTRA_METRICS:
                overall_avg = df[m].mean()
                st.write(f"{EXTRA_METRICS_LABELS[m]}: {overall_avg:.3g}" if pd.notnull(overall_avg) else f"{EXTRA_METRICS_LABELS[m]}: N/A")

if __name__ == '__main__':
    main()
