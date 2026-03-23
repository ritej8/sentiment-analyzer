# Sentiment Analyzer

**Sentiment analysis web application** using PyTorch and Flask API.  
Predicts whether a given text expresses **positive or negative sentiment**.

---

## Project Overview

This project demonstrates a full **ML workflow**:

1. Text preprocessing and vectorization  
2. Training a neural network on labeled sentiment data  
3. Building a Flask API for predictions  
4. End-to-end reproducibility

---

##  Features

- Preprocessing and vectorization of text input  
- Neural network classification (PyTorch)  
- Flask REST API to predict sentiment from user input  
- Clean, modular, and well-structured code  

---

## 🧠 Model Details

- **Input:** Vectorized text (Bag-of-Words)  
- **Hidden layer:** 128 neurons with ReLU activation  
- **Output:** 2 classes (Positive / Negative)  
- **Training:** 5 epochs, cross-entropy loss  
- **Performance:** ~99% train accuracy (for demonstration)

---

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
