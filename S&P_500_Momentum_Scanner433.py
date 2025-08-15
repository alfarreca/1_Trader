# --- Inside main() replace upload/clean section ---
uploaded_file = st.file_uploader("Upload Excel file with tickers", type="xlsx")
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Select sheet to analyze", sheet_names)
    df = pd.read_excel(xls, sheet_name=selected_sheet)
    st.write(f"Loaded rows from '{selected_sheet}':", len(df))
    st.dataframe(df.head())

    # Expected columns without Exchange
    expected_cols = {"Symbol", "Name", "Sector", "Industry", "Theme", "Country", "Notes", "Asset_Type"}
    if not expected_cols.issubset(set(df.columns)):
        st.error(f"Uploaded sheet must contain these columns: {', '.join(expected_cols)}")
        return

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    before_drop = len(df)
    df = df.dropna(subset=["Symbol"]).drop_duplicates("Symbol")
    st.write(f"Dropped rows: {before_drop - len(df)} after cleaning.")

    # No Exchange column now — set YF_Symbol directly as Symbol (assuming US tickers)
    df["YF_Symbol"] = df["Symbol"]
else:
    st.warning("Please upload a .xlsx file with your tickers.")
    return

# Remove all Exchange filtering logic:
# Instead of building 'exchanges' list, remove sidebar Exchange filter
min_score = st.sidebar.slider("Min Momentum Score", 0, 100, 50)
