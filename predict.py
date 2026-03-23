import torch
import torch.nn as nn
import joblib

# ---- Model architecture (same as training) ----

class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super(SentimentModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---- Load vectorizer ----

vectorizer = joblib.load("vectorizer.pkl")


# ---- Load trained model ----

model = SentimentModel(input_dim=10000)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()


# ---- Prediction function ----

def predict_sentiment(text):

    # transform text → vector
    vec = vectorizer.transform([text])

    # convert to tensor
    X = torch.tensor(vec.toarray(), dtype=torch.float32)

    # model prediction
    with torch.no_grad():
        outputs = model(X)
        _, prediction = torch.max(outputs, 1)

    if prediction.item() == 1:
        return "Positive"
    else:
        return "Negative"


# ---- Test the model ----

review = input("Enter a movie review: ")

result = predict_sentiment(review)

print("Sentiment:", result)