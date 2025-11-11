# streamlit_page_title: Conversation History

import os
import json
import sqlite3
import pandas as pd
import streamlit as st

# ----------------------------------------
# Configuration
# ----------------------------------------
DB_PATH = os.path.join("artifacts", "conversation_memory.db")

st.set_page_config(
    page_title="Conversation Database Explorer",
    page_icon="🗂️",
    layout="wide"
)

st.title("🗂️ Conversation Database Explorer")
st.caption("Browse, inspect, and audit saved conversations, translations, and RAG contexts.")

# ----------------------------------------
# Database connection helpers
# ----------------------------------------
def get_sessions():
    """Return distinct session IDs and timestamps."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT session_id,
                   MIN(timestamp) AS start_time,
                   MAX(timestamp) AS end_time,
                   COUNT(*) AS message_count
            FROM messages
            GROUP BY session_id
            ORDER BY end_time DESC
            """,
            conn,
        )
    return df


def get_messages(session_id: str):
    """Return all messages for a session ID."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            conn,
            params=(session_id,),
        )
    return df


# ----------------------------------------
# Sidebar: Session selection
# ----------------------------------------
with st.sidebar:
    st.header("📚 Session List")
    st.markdown("Select a conversation session to view full details.")
    try:
        sessions = get_sessions()
        if sessions.empty:
            st.warning("No sessions found yet. Start a conversation in the main app.")
            st.stop()
        session_id = st.selectbox(
            "Choose a Session ID:",
            sessions["session_id"],
            format_func=lambda x: f"{x[:8]}... ({int(sessions.loc[sessions.session_id == x, 'message_count'])} msgs)",
        )
    except Exception as e:
        st.error(f"Failed to read sessions: {e}")
        st.stop()

# ----------------------------------------
# Load messages for selected session
# ----------------------------------------
df_msgs = get_messages(session_id)

st.markdown(f"### 🧾 Session: `{session_id}`")
st.write(f"**Messages:** {len(df_msgs)} | **Time range:** {df_msgs['timestamp'].min()} → {df_msgs['timestamp'].max()}")

# ----------------------------------------
# Iterate through messages
# ----------------------------------------
for _, row in df_msgs.iterrows():
    role = "👨‍⚕️ Doctor" if row["speaker"] == "doctor" else "🧑‍🧍 Patient"

    with st.expander(f"{role}: {row['original'][:70]}{'...' if len(row['original']) > 70 else ''}", expanded=False):
        st.markdown(f"**Original:** {row['original']}")
        st.markdown(f"**De-identified:** {row.get('deidentified','')}")
        st.markdown(f"**Translation:** {row['translation']}")
        st.markdown(f"**Timestamp:** {row['timestamp']}")

        # Parse and pretty print context JSON
        ctx_text = row.get("context", "")
        try:
            ctx = json.loads(ctx_text) if ctx_text else {}
            med = ctx.get("medical", [])
            cult = ctx.get("cultural", [])
            if med:
                st.markdown("#### 🩺 Medical Context")
                for m in med:
                    st.markdown(f"- {m}")
            if cult:
                st.markdown("#### 🎭 Cultural Context")
                for c in cult:
                    st.markdown(f"- {c}")
        except Exception as e:
            st.write(f"(Invalid context JSON: {e})")

        # Optional prompt view
        if "prompt" in df_msgs.columns and row.get("prompt"):
            with st.expander("🧠 Show Prompt Sent to LLM"):
                st.code(row["prompt"], language="markdown")

st.markdown("---")
st.caption("💡 Tip: Expand each message to view detailed context and prompt text.")
