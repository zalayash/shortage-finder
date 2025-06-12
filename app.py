import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import requests

# -------- CONFIG -------- #
FOLDER_PATH = "data/"  # Folder with daily shortage files
MODEL_PATH = "models/shortage_predictor_model.joblib"
LEX_REF_FILE = "new lex code.csv"
MAX_FILES = 80

# -------- LOAD MODEL -------- #
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------- READ AND CLEAN DATA -------- #
@st.cache_data
def load_recent_data():
    files = sorted(glob.glob(os.path.join(FOLDER_PATH, "shortage_output_*.xlsx")), reverse=True)[:MAX_FILES]
    df_all = []
    for file in files:
        df = pd.read_excel(file)

        # Clean and deduplicate column names
        df.columns = pd.Series(df.columns).apply(lambda x: str(x).strip().lower().replace(" ", "_"))
        df = df.loc[:, ~df.columns.duplicated()]

        df['date'] = pd.to_datetime(Path(file).stem.split("_")[-1], format="%d%m%Y")
        df_all.append(df)
    return pd.concat(df_all, ignore_index=True)

data = load_recent_data()

# -------- LOAD LEX REFERENCE -------- #
@st.cache_data
def load_lex_reference():
    lex_ref = pd.read_csv(LEX_REF_FILE)
    lex_ref.columns = lex_ref.columns.str.strip().str.lower().str.replace(" ", "_")
    return lex_ref

lex_reference = load_lex_reference()
data = data.merge(lex_reference, on="lex_code", how="left")

# -------- SUPPLIER PRICE COLUMNS -------- #
supplier_cols = [col for col in data.columns if col.endswith('_price')]

# -------- DISPLAY LABEL -------- #
data['product_display'] = data['description_y'] if 'description_y' in data.columns else data['description']

# -------- LOCAL AI -------- #
def generate_local_ai_summary(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "No response from local AI.")
    except Exception as e:
        return f"⚠️ Local AI not available: {e}"

# -------- UI -------- #
st.title("Shortage Finder")
st.markdown("Search by any keyword: name, strength, form...")
query = st.text_input("🔍 Search Product", "")

if query:
    filtered = data[data['product_display'].str.lower().str.contains(query.lower())]
    latest = filtered.sort_values('date', ascending=False).drop_duplicates('lex_code')
    selected_label = st.selectbox("Select Product", latest['product_display'].values)
    selected_lex = latest[latest['product_display'] == selected_label]['lex_code'].values[0]
else:
    selected_lex = None

if selected_lex:
    product_data = data[data['lex_code'] == selected_lex].sort_values("date")
    product_data['avg_price'] = product_data[supplier_cols].replace(0, pd.NA).mean(axis=1, skipna=True)

    st.subheader(f"📦 {selected_label}")
    tab1, tab2, tab3 = st.tabs(["💰 Price", "🤖 AI Risk", "📊 Usage"])

    with tab1:
        st.markdown("### 💰 Average Price Trend")
        fig, ax = plt.subplots()
        ax.plot(product_data['date'], product_data['avg_price'], marker='o')
        ax.set_ylabel("Avg Price (GBP)")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        st.markdown("### 🤖 AI Shortage Insight")
        supplier_counts = product_data[supplier_cols].replace(0, pd.NA).notna().sum(axis=1)
        price_start = product_data['avg_price'].iloc[0]
        price_end = product_data['avg_price'].iloc[-1]
        concession_flag = product_data['concession'].max() if 'concession' in product_data.columns else 0
        shortage_flag = product_data['shortage'].max() if 'shortage' in product_data.columns else 0

        prompt = f"""
        You are an expert pharmacy data analyst.
        Give a short report based on the following:

        - Start price: £{price_start:.2f}
        - End price: £{price_end:.2f}
        - Supplier change: {int(supplier_counts.iloc[0])} to {int(supplier_counts.iloc[-1])}
        - Concession flag: {'Yes' if concession_flag else 'No'}
        - Shortage flag: {'Yes' if shortage_flag else 'No'}

        Provide a 2-line summary explaining if the product is at shortage risk and why.
        """

        ai_comment = generate_local_ai_summary(prompt)
        st.info(f"🧠 Local AI Insight: {ai_comment}")

    with tab3:
        st.markdown("### 📊 Usage (Reference Only)")
        if 'usage' in product_data.columns:
            fig2, ax2 = plt.subplots()
            ax2.plot(product_data['date'], product_data['usage'], color='green', marker='o')
            ax2.set_ylabel("Units")
            ax2.set_xlabel("Date")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.warning("No usage data available.")

    st.markdown("### 🏷️ Latest Supplier Prices")
    latest_day = product_data['date'].max()
    latest_row = product_data[product_data['date'] == latest_day]
    st.write(f"As of: {latest_day.date()}")
    display_columns = ['description'] + [col for col in ['strength', 'form', 'pack_size'] if col in latest_row.columns] + supplier_cols
    st.dataframe(latest_row[display_columns])

    with st.expander("📅 Full Daily History Table"):
        display_cols = ['date'] + [col for col in ['usage', 'concession', 'shortage', 'risk_label', 'avg_price'] if col in product_data.columns] + supplier_cols
        st.dataframe(product_data[display_cols].set_index('date'))

else:
    st.info("Search for a product above to view AI insights, pricing, and usage history.")
