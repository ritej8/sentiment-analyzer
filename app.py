import re
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "sentiment_model.pth"


# Text preprocessing (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Model architecture
class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

# Initialize model
input_dim = len(vectorizer.vocabulary_)
model = SentimentModel(input_dim)

# Load trained model
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():

    # Check JSON format
    if not request.is_json:
        return jsonify({
            "error": "Request must contain JSON data."
        }), 400

    data = request.get_json(silent=True)

    # Check JSON structure
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({
            "error": "Missing 'text' field."
        }), 400

    text = data["text"]

    # Validate input
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "'text' must be a non-empty string."
        }), 400

    # Preprocessing
    cleaned_text = clean_text(text)

    # Vectorization
    vectorized_text = vectorizer.transform([cleaned_text])

    input_tensor = torch.tensor(
        vectorized_text.toarray(),
        dtype=torch.float32
    )

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
