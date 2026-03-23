import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
# Load the dataset
df = pd.read_csv("data/IMDB_Dataset.csv")
# Clean the reviews
df["clean_review"] = df["review"].apply(clean_text)

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})
# Split the dataset into training and testing sets
X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Vectorize the text data
vectorizer = CountVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
joblib.dump(vectorizer, "model/vectorizer.pkl")
X_train_array = X_train_vec.toarray()
X_test_array = X_test_vec.toarray()

X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
#dataset 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

input_dim = X_train_tensor.shape[1]

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
    
model = SentimentModel(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
#training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()        # reset gradients
        outputs = model(xb)          # forward pass
        loss = criterion(outputs, yb)

        loss.backward()              # compute gradients
        optimizer.step()             # update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in train_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

accuracy = correct / total
print(f"Train Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix
all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.numpy())
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

torch.save(model.state_dict(), "sentiment_model.pth")