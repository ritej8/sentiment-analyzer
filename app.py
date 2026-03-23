from flask import Flask, request, jsonify
import torch
import joblib
import torch.nn as nn

app = Flask(__name__)

# Model architecture
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


# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Load model
model = SentimentModel(input_dim=10000)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    text = data["text"]

    vec = vectorizer.transform([text])
    X = torch.tensor(vec.toarray(), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        _, prediction = torch.max(outputs, 1)

    sentiment = "Positive" if prediction.item() == 1 else "Negative"

    return jsonify({"sentiment": sentiment})


if __name__ == "__main__":
    app.run(debug=True)