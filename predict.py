import re
from pathlib import Path

import joblib
import torch
import torch.nn as nn


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

# Initialize model with the actual vocabulary size
input_dim = len(vectorizer.vocabulary_)
model = SentimentModel(input_dim)

# Load trained weights
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()


def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    input_tensor = torch.tensor(
        vectorized_text.toarray(),
        dtype=torch.float32
    )

    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()

    return "Positive" if prediction == 1 else "Negative"


# Run interactive prediction only when executed directly
if __name__ == "__main__":
    review = input("Enter a movie review: ")
    result = predict_sentiment(review)
    print("Sentiment:", result)
