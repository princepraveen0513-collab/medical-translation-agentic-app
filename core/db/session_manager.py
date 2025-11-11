import sqlite3
import json
import datetime
from typing import Optional, Dict, Any, List


class SessionManager:
    """
    Handles persistent conversation memory.
    Uses SQLite — portable and deployable even on Streamlit Cloud.
    """

    def __init__(self, db_path: str = "artifacts/conversation_memory.db"):
        self.db_path = db_path
        self._init_db()

    # -------------------------------------------------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------------------------
    def _init_db(self):
        """Create tables if not exist."""
        with self._connect() as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                summary TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                speaker TEXT,
                original TEXT,
                deidentified TEXT,
                translation TEXT,
                context TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """)
            conn.commit()

    # -------------------------------------------------
    def create_session(self, session_id: str):
        now = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            conn.commit()

    # -------------------------------------------------
    def save_message(
        self,
        session_id: str,
        speaker: str,
        original: str,
        deidentified: str,
        translation: str,
        context: Dict[str, Any],
    ):
        """Save each message turn."""
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO messages (
                    session_id, timestamp, speaker, original, deidentified,
                    translation, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, ts, speaker, original, deidentified,
                translation, json.dumps(context, ensure_ascii=False)
            ))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (ts, session_id),
            )
            conn.commit()

    # -------------------------------------------------
    def get_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all messages for a given session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, speaker, original, translation FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
            return [
                {"timestamp": r[0], "speaker": r[1], "original": r[2], "translation": r[3]}
                for r in rows
            ]

    # -------------------------------------------------
    def count_messages(self, session_id: str) -> int:
        """Return number of messages in a session."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,))
            count = cur.fetchone()[0]
        return count

    # -------------------------------------------------
    def get_summary(self, session_id: str) -> Optional[str]:
        """Fetch stored summary for session."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT summary FROM sessions WHERE id=?", (session_id,))
            row = cur.fetchone()
        return row[0] if row and row[0] else None

    # -------------------------------------------------
    def save_summary(self, session_id: str, summary_text: str):
        """Save LLM-generated summary at end of conversation."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE id=?",
                (summary_text, datetime.datetime.utcnow().isoformat(), session_id),
            )
            conn.commit()

    # -------------------------------------------------
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with timestamps."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {"session_id": r[0], "created_at": r[1], "updated_at": r[2]}
                for r in rows
            ]


        # -------------------------------------------------
    def _init_db(self):
        """Create tables if not exist."""
        with self._connect() as conn:
            cur = conn.cursor()

            # existing tables
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                summary TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                speaker TEXT,
                original TEXT,
                deidentified TEXT,
                translation TEXT,
                context TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """)

            # 🆕 NEW: Reflexion Learner Table for Intent Classification
            cur.execute("""
            CREATE TABLE IF NOT EXISTS reflexion_medical_rag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message TEXT,
                label TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """)

            conn.commit()

    # -------------------------------------------------
    def save_medical_rag_reflexion(self, session_id: str, message: str, label: str, confidence: float):
        """Save the mini-LLM intent classification result for medical RAG requirement."""
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO reflexion_medical_rag (session_id, message, label, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, message, label, confidence, ts))
            conn.commit()
        print(f"🧩 Reflexion saved → session={session_id}, label={label}, conf={confidence}")
