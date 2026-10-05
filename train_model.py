from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Customer_Sentiment_reference.csv"
MODEL_DIR = BASE_DIR / "models"
TEXT_COLUMN = "review_text"
LABEL_COLUMN = "sentiment"

df = pd.read_csv(DATA_PATH)
required_columns = {TEXT_COLUMN, LABEL_COLUMN}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}. "
        f"Available columns: {', '.join(df.columns)}"
    )

df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
df = df.loc[df[TEXT_COLUMN].ne("") & df[LABEL_COLUMN].ne("")]

if df.empty:
    raise ValueError("No usable review_text/sentiment rows were found in the dataset.")

class_counts = df[LABEL_COLUMN].value_counts()
if class_counts.min() < 2:
    raise ValueError(
        "Each sentiment class needs at least two records for a stratified train/test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df[LABEL_COLUMN],
    test_size=0.20,
    random_state=42,
    stratify=df[LABEL_COLUMN],
)

tfidf = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    ngram_range=(1, 2),
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("----- CUSTOMER SENTIMENT DATASET -----")
print(f"Usable reviews: {len(df)}")
print("Sentiment counts:")
print(class_counts.to_string())
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=model.classes_, zero_division=0))
print("Confusion matrix (class order: " + ", ".join(model.classes_) + "):")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_DIR / "emotion_model.pkl")
joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")

print(f"\nModel and vectorizer saved to: {MODEL_DIR}")
