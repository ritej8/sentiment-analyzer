# Sentiment Analyzer

**Sentiment analysis web application** using PyTorch and Flask API.  
Predicts whether a given text expresses **positive or negative sentiment**.



## Project Overview

This project demonstrates a full **ML workflow**:

1. Text preprocessing and vectorization  
2. Training a neural network on labeled sentiment data  
3. Building a Flask API for predictions  
4. End-to-end reproducibility



##  Features

- Preprocessing and vectorization of text input  
- Neural network classification (PyTorch)  
- Flask REST API to predict sentiment from user input  
- Clean, modular, and well-structured code  



##  Model Details

- **Input:** Vectorized text (Bag-of-Words)  
- **Hidden layer:** 128 neurons with ReLU activation  
- **Output:** 2 classes (Positive / Negative)  
- **Training:** 5 epochs, cross-entropy loss  
- **Performance:** ~99% train accuracy (for demonstration)



##  Project Structure

```text
sentiment-analyzer/
│
├── app.py            # Flask API
├── train.py          # Train the model
├── predict.py        # Prediction helper
├── requirements.txt  # Dependencies
├── README.md         # This file
└── .gitignore        # Ignored files (models, env)


##  How to run

1 Create a virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
2 Install dependencies
pip install -r requirements.txt
3 Train the model
python train.py

Generates the model file locally (sentiment_model.pth)

4 Run the Flask API
python app.py

The API will start at http://127.0.0.1:5000

5 Make predictions

Send a POST request to /predict with JSON:

{
  "text": "I love this product!"
}

Example response:

{
  "prediction": "Positive",
  "confidence": 0.93
}
 Tech Stack
Python – Core programming
PyTorch – Neural network modeling
Flask – API deployment
scikit-learn – Vectorization & preprocessing
pandas / NumPy – Data handling


