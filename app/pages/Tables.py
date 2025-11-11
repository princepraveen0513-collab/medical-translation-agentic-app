import sqlite3
import pandas as pd
import streamlit as st
import os

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="📊 Database Tables",
    page_icon="🗃️",
    layout="wide"
)

DB_PATH = os.path.join("artifacts", "conversation_memory.db")

st.title("🗃️ Database Tables Explorer")
st.caption("View and inspect data stored in your app’s SQLite database.")

# ----------------------------------------
# Utility Functions
# ----------------------------------------

@st.cache_resource(show_spinner=False)
def get_connection():
    """Create a persistent SQLite connection (cached as resource)."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(show_spinner=False)
def list_tables():
    """Return all table names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cursor.fetchall()]
    return tables

@st.cache_data(show_spinner=False)
def fetch_table_data(table_name):
    """Fetch full table contents as DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df

@st.cache_data(show_spinner=False)
def fetch_table_schema(table_name):
    """Fetch schema info for a given table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    return schema

# ----------------------------------------
# Layout
# ----------------------------------------
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("📋 Tables")
    tables = list_tables()

    if not tables:
        st.warning("No tables found in the database.")
        st.stop()

    selected_table = st.selectbox("Select a table to view:", tables)

with col2:
    if selected_table:
        st.subheader(f"📄 Data: `{selected_table}`")
        df = fetch_table_data(selected_table)
        st.write(f"**Total Rows:** {len(df)}")

        # Display DataFrame
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # --- Schema Section ---
        with st.expander("🧱 View Table Schema"):
            schema = fetch_table_schema(selected_table)
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "default_value", "pk"])
            st.dataframe(schema_df, use_container_width=True)

# ----------------------------------------
# Footer
# ----------------------------------------
st.caption("🔍 Use the sidebar to navigate between pages. Data shown is read-only.")
