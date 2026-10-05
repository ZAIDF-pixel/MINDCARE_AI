from pathlib import Path
from datetime import datetime
import logging
import os
import re
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "models" / "emotion_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
DB_PATH = Path(os.getenv("MINDCARE_DB_PATH", str(BASE_DIR / "mindcare.db"))).expanduser()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
logger = logging.getLogger(__name__)

app = FastAPI(title="MindCare AI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

SUPPORT_MESSAGES = {
    "positive": "It sounds like you are feeling positive. Keep doing activities that help you feel good and connected.",
    "neutral": "It sounds like things are fairly normal right now. A healthy routine and regular breaks can be helpful.",
    "negative": "It sounds like this experience has been difficult. Consider taking a moment for yourself and talking with someone you trust if you need support.",
    "stress": "You may be experiencing stress. Try breaking your work into smaller tasks and taking short breaks.",
    "sad": "It sounds like you may be having a difficult day. Consider talking to someone you trust and doing something gentle that helps you feel supported.",
    "anger": "It sounds like you are frustrated. Taking some time away from the situation and using slow breathing may help you respond calmly.",
    "anxiety": "You may be feeling worried or nervous. Focus on what you can control and consider talking to someone you trust."
}

SELF_CARE = {
    "positive": ["Keep a healthy routine", "Spend time with supportive people", "Continue activities you enjoy"],
    "neutral": ["Take regular breaks", "Maintain a regular sleep routine", "Do something enjoyable today"],
    "negative": ["Take a moment to care for yourself", "Talk with someone you trust", "Consider one small step that may improve how you feel"],
    "stress": ["Break large tasks into smaller steps", "Take short study breaks", "Try slow breathing for a few minutes"],
    "sad": ["Talk with someone you trust", "Do a gentle activity you enjoy", "Avoid isolating yourself"],
    "anger": ["Take some space from the situation", "Slow down before responding", "Try slow breathing"],
    "anxiety": ["Focus on what you can control", "Write down your worries", "Talk with someone you trust"]
}

SAFETY_TERMS = [
    "hurt myself", "harm myself", "kill myself",
    "end my life", "suicide", "suicidal"
]

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            message TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def safety_check(text):
    lowered = text.lower()
    return any(term in lowered for term in SAFETY_TERMS)

def predict_emotion(message):
    vector = tfidf.transform([message])
    emotion = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities) * 100)
    return emotion, confidence

def save_mood(message, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO mood_history(timestamp,message,emotion,confidence) VALUES (?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, emotion, confidence)
    )
    conn.commit()
    conn.close()

def get_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id,timestamp,message,emotion,confidence FROM mood_history ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df

def save_journal(title, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO journal(timestamp,title,content) VALUES (?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), title, content)
    )
    conn.commit()
    conn.close()


def clean_ai_summary(content):
    lines = []
    for line in content.splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"^\s*(?:[-*+•▪◦‣]|\d+[.)])\s+", "", line)
        if line.strip():
            lines.append(line.strip())

    plain_text = " ".join(lines)
    plain_text = re.sub(r"\*\*|__|[*_`#]", "", plain_text)
    plain_text = re.sub(r"\s+[-+]\s+", " ", plain_text)
    plain_text = re.sub(r"(?<=\s)\d+[.)]\s+", "", plain_text)
    plain_text = re.sub(r"[•▪◦‣]", "", plain_text)
    plain_text = re.sub(r"\s+", " ", plain_text).strip()
    sentence_endings = list(re.finditer(r"[.!?](?=\s|$)", plain_text))
    if not sentence_endings:
        return "The summary could not be formatted as complete sentences. Please try again."
    if not re.search(r"[.!?]\s*$", plain_text):
        plain_text = plain_text[:sentence_endings[-1].end()]
    return plain_text


def get_ai_summary(check_ins):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "AI summaries are disabled. Configure GROQ_API_KEY in your deployment environment to enable them."
    if Groq is None:
        logger.error("Groq summary requested, but the groq package is not installed.")
        return "AI summaries are temporarily unavailable because the Groq client is not installed."

    check_in_text = "\n".join(
        f"- {check_in['timestamp']}: {check_in['emotion']}"
        for check_in in check_ins
    )
    try:
        client = Groq(api_key=key, timeout=30.0, max_retries=1)
        prompt = f"""
You are a wellbeing-support assistant.
Write exactly two short, complete sentences in one plain-text paragraph about these recent broad, model-generated mood categories.
Describe an overall pattern without listing category counts.
Do not include headings, lists, bullets, asterisks, or decorative symbols.
Do not diagnose mental-health conditions.
Do not provide medical treatment.
Do not infer personal details or claim certainty from these categories.
Offer one gentle, general reflection and encourage trusted human support when appropriate.
End both sentences with normal punctuation.

Recent check-in timestamps and categories (no message text is included):
{check_in_text}
"""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350
        )
        content = response.choices[0].message.content
        if not content:
            logger.error("Groq returned an empty summary response.")
            return "The summary service returned an empty response. Please try again."
        return clean_ai_summary(content)
    except Exception:
        logger.exception("Groq summary request failed.")
        return "AI summary is temporarily unavailable. Please try again later."

init_db()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "page": "home"}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, message: str = Form(...)):
    message = message.strip()
    if not message:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "page": "home", "error": "Please enter how you are feeling."}
        )

    if safety_check(message):
        result = {
            "safety": True,
            "emotion": "Support Needed",
            "confidence": 0,
            "response": (
                "Please reach out to a trusted adult, family member, teacher, "
                "counselor, or another person who can stay with you and help. "
                "If you feel you may be in immediate danger, contact your local emergency service."
            ),
            "tips": []
        }
        save_mood(message, "Support Needed", 0)
    else:
        emotion, confidence = predict_emotion(message)
        result = {
            "safety": False,
            "emotion": emotion.title(),
            "confidence": round(confidence, 1),
            "response": SUPPORT_MESSAGES[emotion],
            "tips": SELF_CARE[emotion]
        }
        save_mood(message, emotion, confidence)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "page": "home", "result": result}
    )

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    df = get_history()
    records = df.to_dict("records")
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "page": "history", "records": records}
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    df = get_history()
    total = len(df)
    counts = df["emotion"].value_counts().to_dict() if not df.empty else {}
    avg_conf = round(float(df["confidence"].mean()), 1) if not df.empty else 0
    latest = df.iloc[0].to_dict() if not df.empty else None
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "page": "dashboard",
            "total": total,
            "avg_conf": avg_conf,
            "counts": counts,
            "latest": latest,
            "groq_enabled": bool(os.getenv("GROQ_API_KEY")) and Groq is not None,
        }
    )


@app.get("/summary", response_class=HTMLResponse)
async def summary_page(request: Request):
    return templates.TemplateResponse(
        "summary.html",
        {
            "request": request,
            "page": "summary",
            "summary": None,
            "groq_enabled": bool(os.getenv("GROQ_API_KEY")) and Groq is not None,
        }
    )


@app.get("/api/history")
async def history_api():
    df = get_history()
    if df.empty:
        return JSONResponse({"labels": [], "values": []})
    counts = df["emotion"].value_counts()
    return JSONResponse({
        "labels": counts.index.tolist(),
        "values": [int(v) for v in counts.values]
    })

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM journal ORDER BY id DESC LIMIT 30", conn
    )
    conn.close()
    return templates.TemplateResponse(
        "journal.html",
        {"request": request, "page": "journal", "entries": df.to_dict("records")}
    )

@app.post("/journal", response_class=HTMLResponse)
async def journal_save(
    request: Request,
    title: str = Form(""),
    content: str = Form(...)
):
    if content.strip():
        save_journal(title.strip(), content.strip())
    return templates.TemplateResponse(
        "journal.html",
        {"request": request, "page": "journal", "saved": True,
         "entries": []}
    )

@app.post("/summary", response_class=HTMLResponse)
@app.post("/ai-summary", response_class=HTMLResponse)
def ai_summary(request: Request):
    df = get_history(20)
    if df.empty:
        summary = "No mood check-ins are available yet."
    else:
        check_ins = df[["timestamp", "emotion"]].to_dict("records")
        summary = get_ai_summary(check_ins)
    return templates.TemplateResponse(
        "summary.html",
        {
            "request": request,
            "page": "summary",
            "summary": summary,
            "groq_enabled": bool(os.getenv("GROQ_API_KEY")) and Groq is not None,
        }
    )
