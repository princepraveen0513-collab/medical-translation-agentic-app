import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.agents.coordinator_agent import CoordinatorAgent

if __name__ == "__main__":
    agent = CoordinatorAgent()

    # 1. New message from patient (Hindi)
    res1 = agent.process_message(
        text="मेरा नाम प्रिंस प्रवीन है, मैं 26 साल का हूँ और मेरे सीने में जलन हो रही है।",
        speaker="patient",
    )
    print("\n🔹 Turn 1 Result:", {k: res1[k] for k in ["session_id", "deidentified_text", "translation"]})

    # 2. Doctor reply (English)
    res2 = agent.process_message(
        text="Please tell me if the pain increases when you walk or goes to your left arm or jaw.",
        speaker="doctor",
        session_id=res1["session_id"],
    )
    print("\n🔹 Turn 2 Result:", {k: res2[k] for k in ["session_id", "deidentified_text", "translation"]})

    # 3. Optional: summarize session
    summary = agent.summarize_session(res1["session_id"])
    print("\n📝 Session Summary:\n", summary)
