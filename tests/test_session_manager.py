import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from core.db.session_manager import SessionManager

if __name__ == "__main__":
    mgr = SessionManager()

    session_id = "test-session-001"
    mgr.create_session(session_id)

    mgr.save_message(
        session_id=session_id,
        speaker="patient",
        original="मुझे सिरदर्द है",
        deidentified="मुझे सिरदर्द है",
        translation="I have a headache",
        context={"medical": ["headache"], "cultural": []},
    )

    conv = mgr.get_conversation(session_id)
    print("🧾 Conversation:\n", conv)

    mgr.save_summary(session_id, "Patient reported headache, no red flags.")
    print("✅ Saved summary.")
