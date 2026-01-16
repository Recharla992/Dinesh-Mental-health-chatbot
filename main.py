# main.py
# Dinesh– AI Mental Health Assistant
# Developed by OpenAI ChatGPT
# =========================================================
# import necessary libraries
import streamlit as st
import pickle
import re
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from gtts import gTTS
import tempfile
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# =========================================================
# 📄 PDF SESSION REPORT GENERATOR
def generate_pdf_report(chat_history, mood_df, insight_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ---------------- Title ----------------
    story.append(Paragraph("<b>Aarya – Full Session Emotional Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # ---------------- Chat History ----------------
    story.append(Paragraph("<b>Chat History</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))

    for role, message in chat_history:
        story.append(Paragraph(f"<b>{role}:</b> {message}", styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))

    # ---------------- Mood Timeline ----------------
    if not mood_df.empty:
        story.append(Paragraph("<b>Daily Mood Timeline</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))

        table_data = [["Date", "Mood"]]
        for _, row in mood_df.iterrows():
            table_data.append([str(row["Date"]), row["Mood"]])

        table = Table(table_data, hAlign="LEFT")
        story.append(table)

    story.append(Spacer(1, 12))

    # ---------------- Emotional Insight ----------------
    story.append(Paragraph("<b>AI Emotional Insight</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(insight_text, styles["Normal"]))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
# =========================================================
# 1️⃣ PAGE CONFIG (MUST BE FIRST)
# =========================================================
st.set_page_config(
    page_title="Dinesh – AI Mental Health chatbot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2️⃣ GLOBAL CSS
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] { animation: none !important; }

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}

.chat-bubble {
    padding: 20px 25px;
    border-radius: 20px;
    margin: 15px 0;
    max-width: 80%;
    line-height: 1.6;
    animation: fadeIn 0.6s ease-in-out;
}

.user-bubble {
    background: linear-gradient(135deg, #1e40af, #1e3a8a);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 6px;
}

.assistant-bubble {
    background: linear-gradient(135deg, #064e3b, #022c22);
    color: #ecfdf5;
    margin-right: auto;
    border-bottom-left-radius: 6px;
}

.emergency {
    background: #7f1d1d;
    color: #fee2e2;
    padding: 18px;
    border-radius: 16px;
    animation: pulse 1.5s infinite;
}

.meta {
    font-size: 13px;
    opacity: 0.75;
    margin-bottom: 8px;
}

.footer {
    font-size: 12px;
    opacity: 0.6;
    text-align: center;
    margin-top: 40px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    70% { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3️⃣ SESSION STATE
# =========================================================
if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "emotional_state" not in st.session_state:
    st.session_state.emotional_state = []

if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0

if "daily_moods" not in st.session_state:
    st.session_state.daily_moods = {}

if "language" not in st.session_state:
    st.session_state.language = "English"

if "emotion_timeline" not in st.session_state:
    st.session_state.emotion_timeline = []


# =========================================================
# 🔓 CBT MODE SESSION STATE
# =========================================================
if "cbt_mode" not in st.session_state:
    st.session_state.cbt_mode = False

if "cbt_step" not in st.session_state:
    st.session_state.cbt_step = 0

if "cbt_data" not in st.session_state:
    st.session_state.cbt_data = {
        "thought": "",
        "emotion": "",
        "behavior": "",
        "reframe": ""
    }


# =========================================================
# 🔹 ADDITION STEP 1: FULL SESSION TRACKING (NEW)
# =========================================================
if "full_emotion_log" not in st.session_state:
    st.session_state.full_emotion_log = []

if "full_sentiment_log" not in st.session_state:
    st.session_state.full_sentiment_log = []

if "full_confidence_log" not in st.session_state:
    st.session_state.full_confidence_log = []

if "full_timestamp_log" not in st.session_state:
    st.session_state.full_timestamp_log = []

# =========================================================
# 4️⃣ INTRO SCREEN
# =========================================================
if not st.session_state.intro_seen:
    st.markdown("""
    <div style="text-align:center; padding:50px;">
        <h1>🩺 Hello, I’m Dinesh</h1>
        <p style="font-size:25px;">
        I’m here to listen — calmly, safely, and without judgment.
        </p>
        <hr style="opacity:0.3;">
        <p style="font-size:14px; opacity:0.7;">
        ⚠️ I’m not a medical professional.<br>
        If you are in immediate danger, please contact emergency services.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Begin Conversation 💬"):
        st.session_state.intro_seen = True
        st.rerun()

    st.stop()

# =========================================================
# 5️⃣ SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("🧠 Dinesh Control Panel")

st.session_state.language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", ]
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Session"):
    st.session_state.chat_history = []
    st.session_state.emotional_state = []
    st.session_state.daily_moods = {}
    st.session_state.negative_count = 0
    st.session_state.full_emotion_log = []
    st.session_state.full_sentiment_log = []
    st.session_state.full_confidence_log = []
    st.session_state.full_timestamp_log = []
    st.rerun()
    st.session_state.cbt_step = 0
    st.session_state.cbt_data = {
        "thought": "",
        "emotion": "",
        "behavior": "",
        "reframe": ""
    }
    st.session_state.cbt_mode = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Therapy Tools")

st.session_state.cbt_mode = st.sidebar.toggle(
    "🧠 Enable CBT Therapy Mode",
    value=st.session_state.cbt_mode
)


# 🔹 ADDITION STEP 5: FULL HISTORY DOWNLOAD
if st.session_state.full_sentiment_log:
    full_df = pd.DataFrame({
        "Timestamp": st.session_state.full_timestamp_log,
        "Sentiment": st.session_state.full_sentiment_log,
        "Confidence (%)": st.session_state.full_confidence_log,
        "Emotion": st.session_state.full_emotion_log
    })

    csv_full = full_df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        "⬇️ Download Full Session History",
        csv_full,
        "full_session_history.csv",
        "text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Future Upgrades")
st.sidebar.markdown("🎤 Voice Chat")
st.sidebar.markdown("🧩 CBT Therapy")
st.sidebar.markdown("👤 Secure Login")
st.sidebar.markdown("☁️ Cloud Sync")




# =========================================================
# 6️⃣ LOAD MODELS
# =========================================================
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

emotion_ai = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3
)

# =========================================================
# 7️⃣ NLP HELPERS
# =========================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = sentiment_model.predict(vec)[0]
    prob = sentiment_model.predict_proba(vec).max() * 100
    label = "Positive" if pred == 1 else "Negative"
    return ("Neutral", round(prob, 2)) if prob < 60 else (label, round(prob, 2))

EMERGENCY_WORDS = [
    "suicide", "kill myself", "end my life",
    "i want to die", "self harm"
]

def detect_emergency(text):
    return any(w in text.lower() for w in EMERGENCY_WORDS)

def nurse_reply(sentiment, negative_count):
    if sentiment == "Emergency":
        return (
            "🚨 I’m really concerned about your safety.\n\n"
            "📞 AASRA (India): 91-9820466726\n"
            "📞 Emergency: 112\n\n"
            "You are not alone."
        )
    if negative_count >= 3:
        return "💙 I’ve noticed this has been heavy for you. I’m here with you."
    if sentiment == "Negative":
        return "💭 That sounds really difficult. Want to share more?"
    if sentiment == "Positive":
        return "😊 I’m glad to hear that. What helped today?"
    return "🙂 I’m listening."



# =========================================================
# 🔊 NURSE VOICE (TEXT TO SPEECH)
# =========================================================
def speak_nurse_voice(text):
    try:
        tts = gTTS(text=text, lang="en", slow=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception:
        return None

# =========================================================
# 🧠 EMOTIONAL TREND ANALYZER
# =========================================================
def generate_emotional_insight(timeline):
    if len(timeline) < 3:
        return "🌱 Share a little more — I’ll understand your emotional pattern better."

    sentiments = [t["sentiment"] for t in timeline]

    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")
    neutral = sentiments.count("Neutral")

    if negative >= positive and negative >= neutral:
        return "💙 You’ve been feeling low frequently. Gentle care and rest may help."
    if positive > negative:
        return "🌸 Your emotional state shows improvement. Keep nurturing what helps you."
    if neutral >= positive and neutral >= negative:
        return "🙂 You seem emotionally steady. Consistency is a strength."

    return "🫂 I’m here with you, whatever you’re feeling."

# =========================================================
# 8️⃣ APP HEADER
# =========================================================
st.markdown("## 🧠 Dinesh – Your AI Mental Health Assistant")
st.markdown("*A calm, safe space to talk.*")

# =========================================================
# 9️⃣ CHAT FLOW
# =========================================================
user_input = st.text_input("How are you feeling today?")

if user_input:
    if detect_emergency(user_input):
        sentiment, confidence = "Emergency", 100
    else:
        sentiment, confidence = predict_sentiment(user_input)

    today = datetime.date.today().isoformat()
    st.session_state.daily_moods[today] = sentiment

    st.session_state.emotional_state.append(sentiment)
    st.session_state.emotion_timeline.append({
    "time": datetime.datetime.now().strftime("%H:%M:%S"),
    "sentiment": sentiment })
    st.session_state.negative_count = (
        st.session_state.negative_count + 1 if sentiment == "Negative" else 0
    )

    emotions = emotion_ai(user_input)[0]
    emotion_text = ", ".join(
        f"{e['label']} ({e['score']*100:.1f}%)" for e in emotions
    )

    # 🔹 ADDITION STEP 2: FULL HISTORY APPEND
    st.session_state.full_sentiment_log.append(sentiment)
    st.session_state.full_confidence_log.append(confidence)
    st.session_state.full_emotion_log.append(emotions[0]["label"])
    st.session_state.full_timestamp_log.append(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    reply = nurse_reply(sentiment, st.session_state.negative_count)
    reply = nurse_reply(sentiment, st.session_state.negative_count)
    voice_file = speak_nurse_voice(reply)
    # if voice_file:
    #     audio_bytes = open(voice_file, "rb").read()
    #     st.audio(audio_bytes, format="audio/mp3")
    #     os.remove(voice_file)


    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Aarya", reply))

# =========================================================
# 🔓 CBT THERAPY MODE
# =========================================================
if st.session_state.cbt_mode:
    st.markdown("## 🧩 CBT Therapy Session")
    st.markdown("*A guided, gentle reflection process*")

    # STEP 1 — THOUGHT
    if st.session_state.cbt_step == 0:
        thought = st.text_area(
            "💭 What troubling thought is bothering you right now?"
        )
        if st.button("Next ➡️"):
            st.session_state.cbt_data["thought"] = thought
            st.session_state.cbt_step = 1
            st.rerun()

    # STEP 2 — EMOTION
    elif st.session_state.cbt_step == 1:
        emotion = st.text_input(
            "💙 How does this thought make you feel?"
        )
        if st.button("Next ➡️"):
            st.session_state.cbt_data["emotion"] = emotion
            st.session_state.cbt_step = 2
            st.rerun()

    # STEP 3 — BEHAVIOR
    elif st.session_state.cbt_step == 2:
        behavior = st.text_input(
            "🔁 What do you usually do when you feel this way?"
        )
        if st.button("Next ➡️"):
            st.session_state.cbt_data["behavior"] = behavior
            st.session_state.cbt_step = 3
            st.rerun()

    # STEP 4 — REFRAME
    elif st.session_state.cbt_step == 3:
        reframe_text = (
            f"When you think **'{st.session_state.cbt_data['thought']}'**, "
            f"it makes you feel **{st.session_state.cbt_data['emotion']}**.\n\n"
            "Let’s gently reframe this thought:\n\n"
            "🌱 *Is there a kinder or more balanced way to see this situation?*"
        )

        st.info(reframe_text)

        reframe = st.text_area(
            "✨ Write a healthier alternative thought:"
        )

        if st.button("Finish CBT Session ✅"):
            st.session_state.cbt_data["reframe"] = reframe
            st.session_state.cbt_step = 4
            st.rerun()

    # SUMMARY
    elif st.session_state.cbt_step == 4:
        st.success("🩺 CBT Session Summary")

        st.markdown(f"""
        **💭 Thought:** {st.session_state.cbt_data['thought']}  
        **💙 Emotion:** {st.session_state.cbt_data['emotion']}  
        **🔁 Behavior:** {st.session_state.cbt_data['behavior']}  
        **✨ Reframed Thought:** {st.session_state.cbt_data['reframe']}
        """)

        if st.button("🔄 Start New CBT Session"):
            st.session_state.cbt_step = 0
            st.session_state.cbt_data = {
                "thought": "",
                "emotion": "",
                "behavior": "",
                "reframe": ""
            }
            st.rerun()


# =========================================================
# 🔟 CHAT HISTORY RENDER
# =========================================================
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(
            f"<div class='chat-bubble user-bubble'>👤 <b>You</b><br>{text}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
        f"<div class='chat-bubble assistant-bubble'>🩺 <b>Dinesh</b><br>{text}</div>",
        unsafe_allow_html=True
    )

    if "voice_file" in locals() and voice_file:
        audio_bytes = open(voice_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")


# 🔹 ADDITION STEP 3: EMOTION & CONFIDENCE UI
if user_input:
    st.markdown(
        f"<div class='meta'>🧠 {emotion_text} | 🔍 {sentiment} ({confidence}%)</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 1️⃣1️⃣ MOOD ANALYTICS
# =========================================================
if st.session_state.daily_moods:
    st.markdown("### 📊 Mood Distribution")

    mood_df = pd.DataFrame(
        st.session_state.daily_moods.items(),
        columns=["Date", "Mood"]
    )

    mood_counts = mood_df["Mood"].value_counts()

    fig, ax = plt.subplots()
    mood_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Mood")
    ax.set_ylabel("Count")
    ax.set_title("Mood Overview")

    st.pyplot(fig)

# 🔹 ADDITION STEP 4: FULL SESSION EMOTION CHART
if st.session_state.full_emotion_log:
    st.markdown("### 📊 Full Session Emotion Distribution")

    emotion_series = pd.Series(st.session_state.full_emotion_log)
    emotion_counts = emotion_series.value_counts()

    fig2, ax2 = plt.subplots()
    emotion_counts.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Emotion")
    ax2.set_ylabel("Count")
    ax2.set_title("Emotion Frequency (Entire Session)")

    st.pyplot(fig2)

# =========================================================
# 📥 DOWNLOAD SESSION REPORT
# =========================================================
if st.session_state.chat_history and st.session_state.daily_moods:
    mood_df = pd.DataFrame(
        st.session_state.daily_moods.items(),
        columns=["Date", "Mood"]
    )

    insight_text = generate_emotional_insight(
        st.session_state.emotion_timeline
    )

    pdf_file = generate_pdf_report(
        st.session_state.chat_history,
        mood_df,
        insight_text
    )

    st.download_button(
        "📄 Download Full Session Report (PDF)",
        pdf_file,
        file_name="aarya_session_report.pdf",
        mime="application/pdf"
    )


# =========================================================
# 🧠 AI EMOTIONAL INSIGHT
# =========================================================
if st.session_state.emotion_timeline:
    st.markdown("### 🧠 Emotional Insight")

    insight = generate_emotional_insight(
        st.session_state.emotion_timeline
    )

    st.info(insight)

# =========================================================
# 📄 PDF SESSION REPORT GENERATOR
# =========================================================
def generate_pdf_report(chat_history, mood_df, insight_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Aarya – Mental Health Session Report")

    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Date: {datetime.date.today().isoformat()}")

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Chat History")

    y -= 20
    c.setFont("Helvetica", 10)

    for role, text in chat_history:
        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"{role}: {text}")
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Mood Summary")

    y -= 20
    c.setFont("Helvetica", 10)

    for _, row in mood_df.iterrows():
        c.drawString(50, y, f"{row['Date']} → {row['Mood']}")
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "AI Emotional Insight")

    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, insight_text)

    c.save()
    buffer.seek(0)
    return buffer
if st.button("📄 Download Session Report (PDF)"):
    pdf_buffer = generate_pdf_report(
        st.session_state.chat_history,
        mood_df,
        insight
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="mental_health_session_report.pdf",
        mime="application/pdf"
    )

# =========================================================
# 🔹 FOOTER
# =========================================================
st.markdown("""
<div class='footer'>
⚠️ This AI assistant does not replace professional mental health care.
            </div>
""", unsafe_allow_html=True)
