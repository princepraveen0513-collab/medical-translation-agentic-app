import sqlite3
import bcrypt
import datetime
from typing import Optional


class AuthManager:
    """Simple user authentication system using SQLite + bcrypt."""

    def __init__(self, db_path: str = "artifacts/conversation_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create users table if not exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # -------------------------------------------------
    def register_user(self, username: str, password: str) -> bool:
        """Register new user with hashed password."""
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        created_at = datetime.datetime.utcnow().isoformat()

        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, hashed.decode("utf-8"), created_at),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # username already exists

    # -------------------------------------------------
    def verify_user(self, username: str, password: str) -> bool:
        """Verify login credentials."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,)
            ).fetchone()

        if not row:
            return False

        stored_hash = row[0].encode("utf-8")
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)

    # -------------------------------------------------
    def user_exists(self, username: str) -> bool:
        """Check if username already exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        return row is not None
