import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_weeks(n):
    today = datetime.today()
    offset = (today.weekday() - 4) % 7  # 4 = Friday
    last_friday = today - timedelta(days=offset)
    weeks = [(last_friday - timedelta(weeks=i) - timedelta(days=4),
              last_friday - timedelta(weeks=i)) for i in reversed(range(n))]
    return weeks, last_friday

def fetch_friday_closes(symbol, weeks):
    start_date, end_date = weeks[0][0], weeks[-1][1]
    df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), interval="1d", progress=False)
    if df.empty or "Close" not in df.columns:
        return None
    closes = []
    for monday, friday in weeks:
        week_data = df[(df.index >= monday) & (df.index <= friday)]
        close = week_data[week_data.index.weekday == 4]["Close"].dropna()
        if not close.empty:
            closes.append(float(round(close.iloc[-1], 3)))
        else:
            fallback = week_data["Close"].dropna()
            closes.append(float(round(fallback.iloc[-1], 3)) if not fallback.empty else np.nan)
    return closes if sum(np.isnan(closes)) == 0 else None

def fetch_current_week_close(symbol, current_week_start):
    today = datetime.today()
    df = yf.download(symbol, start=current_week_start, end=today + timedelta(days=1), interval="1d", progress=False)
    if df.empty or "Close" not in df.columns:
        return np.nan
    return float(round(df["Close"].dropna().iloc[-1], 3)) if not df["Close"].dropna().empty else np.nan

def calculate_max_drawdown(prices):
    if len(prices) < 2: return 0.0
    arr = np.array(prices, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return drawdowns.min() * 100

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    sheet_choice = st.selectbox("Select sheet to analyze", [""] + sheet_names)

    if sheet_choice:
        df = pd.read_excel(xls, sheet_name=sheet_choice)
        if not all(col in df.columns for col in ["Symbol", "Exchange"]):
            st.error("Excel must contain 'Symbol' and 'Exchange'")
        else:
            symbols = df["Symbol"].dropna().unique().tolist()
            weeks, last_friday = get_last_n_weeks(6)
            current_week_start = last_friday

            all_data = {}
            for sym in symbols:
                closes = fetch_friday_closes(sym, weeks)
                current = fetch_current_week_close(sym, current_week_start)
                if closes is not None and len(closes) == 6:
                    all_data[sym] = closes + [current]
                else:
                    st.warning(f"⚠️ Ticker **{sym}**: insufficient data, skipped.")

            if all_data:
                labels = [f"{m.strftime('%b %d')}→{f.strftime('%b %d')}" for m, f in weeks]
                labels += [f"{current_week_start.strftime('%b %d')}→{datetime.today().strftime('%b %d')}"]
                price_df = pd.DataFrame(all_data).T
                price_df.columns = labels
                price_df.index.name = "Symbol"
                price_df = price_df.reset_index()

                for col in labels:
                    price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

                norm_df = price_df.set_index("Symbol")[labels]
                safe_norm = norm_df.copy()
                safe_norm = safe_norm.where(norm_df.iloc[:, 0] != 0)
                normed = safe_norm.div(norm_df.iloc[:, 0], axis=0)

                weekly_pct = norm_df.pct_change(axis=1) * 100

                # Clean-up: Drop last column if no change or same start/end
                if weekly_pct.columns[-1].split("→")[0] == weekly_pct.columns[-1].split("→")[1] or weekly_pct.iloc[:, -1].nunique() <= 1:
                    weekly_pct = weekly_pct.iloc[:, :-1]
                    norm_df = norm_df.iloc[:, :-1]
                    normed = normed.iloc[:, :-1]
                    labels = labels[:-1]

                tabs = st.tabs([
                    "📈 Price Trend",
                    "📊 Normalized Performance",
                    "📈 % Weekly Change",
                    "🎯 Ticker Scores",
                    "📉 Max Drawdown",
                    "📉 Volatility"
                ])

                with tabs[0]:
                    st.subheader("📈 Price Trend")
                    fig = go.Figure()
                    baseline = norm_df.iloc[:, 0]
                    pct_change_from_start = norm_df.subtract(baseline, axis=0).div(baseline, axis=0) * 100

                    for sym in norm_df.index:
                        fig.add_trace(go.Scatter(
                            x=labels,
                            y=norm_df.loc[sym],
                            customdata=pct_change_from_start.loc[sym].values.reshape(-1, 1),
                            mode='lines+markers',
                            name=sym,
                            hovertemplate=(
                                f"<b>{sym}</b><br>"
                                + "Price: %{y:.2f}<br>"
                                + "Change: %{customdata[0]:.2f}%"
                            )
                        ))
                    fig.update_layout(hovermode="x unified", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                with tabs[1]:
                    st.subheader("📊 Normalized Performance (Start = 100)")
                    norm_chart = go.Figure()
                    start_values = norm_df.iloc[:, 0]
                    last_values = norm_df.iloc[:, -1]

                    total_pct_change = ((last_values - start_values) / start_values) * 100
                    normed_pct_change = norm_df.divide(start_values, axis=0) * 100
                    pct_change_from_start = norm_df.subtract(start_values, axis=0).divide(start_values, axis=0) * 100

                    for sym in norm_df.index:
                        change = total_pct_change[sym]
                        label_name = f"{sym} ({change:+.2f}%)"
                        norm_chart.add_trace(go.Scatter(
                            x=labels,
                            y=normed_pct_change.loc[sym],
                            customdata=pct_change_from_start.loc[sym].values.reshape(-1, 1),
                            mode="lines",
                            name=label_name,
                            hovertemplate=(
                                f"<b>{sym}</b><br>"
                                + "Normalized: %{y:.2f}<br>"
                                + "Change: %{customdata[0]:.2f}%"
                            )
                        ))
                    norm_chart.update_layout(hovermode="closest", height=500)
                    st.plotly_chart(norm_chart, use_container_width=True)

                with tabs[2]:
                    st.subheader("📈 Weekly % Change")
                    st.dataframe(weekly_pct.round(2).reset_index(), use_container_width=True)

                with tabs[3]:
                    st.subheader("🎯 Ticker Scores")
                    scores = pd.DataFrame(index=norm_df.index)
                    scores["Momentum"] = (norm_df.iloc[:, -1] - norm_df.iloc[:, -2]).fillna(0)
                    scores["Volatility"] = norm_df.std(axis=1).fillna(0)
                    scores["Trend"] = norm_df.apply(lambda row: sum(row.diff().fillna(0) > 0), axis=1)
                    scores["Total Return"] = (norm_df.iloc[:, -1] - norm_df.iloc[:, 0]).fillna(0)
                    scores["All-Around"] = scores.sum(axis=1)
                    st.dataframe(scores.round(2).sort_values("All-Around", ascending=False), use_container_width=True)

                with tabs[4]:
                    st.subheader("📉 Max Drawdown")
                    drawdowns = norm_df.apply(lambda row: calculate_max_drawdown(row.dropna()), axis=1).dropna()
                    fig = go.Figure(go.Bar(x=drawdowns.index, y=drawdowns.values, marker_color="crimson",
                                           hovertemplate="%{x}<br>Drawdown: %{y:.2f}%"))
                    fig.update_layout(title="Drawdown (%)", yaxis_title="Drawdown", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(drawdowns.rename("Drawdown (%)").round(2).reset_index(), use_container_width=True)

                with tabs[5]:
                    st.subheader("📉 Volatility (Standard Deviation of Weekly % Change)")
                    volatility = weekly_pct.std(axis=1).fillna(0)
                    st.dataframe(volatility.rename("Volatility (%)").round(2).reset_index(), use_container_width=True)
