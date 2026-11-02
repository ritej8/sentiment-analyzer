# Movie Review Sentiment Analysis

A binary sentiment classification project built with **PyTorch** and **Flask**. The model analyzes movie reviews and predicts whether their sentiment is positive or negative.

## Project Overview

This project implements a neural network-based sentiment classifier using a Bag-of-Words representation.

The pipeline includes:
- Text preprocessing and cleaning
- Feature extraction using `CountVectorizer`
- Binary sentiment classification with PyTorch
- Model evaluation using standard classification metrics
- A Flask REST API for sentiment prediction

## Technologies

- Python
- PyTorch
- Scikit-learn
- Pandas
- Flask
- Joblib

## Dataset

The project uses the IMDB Movie Reviews Dataset.

- **Total reviews:** 50,000
- **Positive reviews:** 25,000
- **Negative reviews:** 25,000
- **Training set:** 80% (40,000 reviews)
- **Test set:** 20% (10,000 reviews)

The dataset is split using `train_test_split` with `random_state=42` and stratification to preserve class proportions.

Expected dataset location:

`data/IMDB_Dataset.csv`

The CSV file must contain the columns `review` and `sentiment`.

## Model Architecture

The project uses a simple feedforward neural network.

**Text processing:**

1. Convert text to lowercase.
2. Remove HTML tags.
3. Remove non-alphabetic characters.
4. Normalize whitespace.
5. Transform text into numerical features using `CountVectorizer`.

**Vectorization parameters:**
- Maximum features: 10,000
- N-gram range: (1, 2)
- Stop words: English

**Neural network:**

| Layer | Configuration |
|---|---|
| Input | Vocabulary size |
| Fully Connected | 128 neurons |
| Activation | ReLU |
| Output | 2 classes |

**Training configuration:**
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 5
- Batch size: 32

## Project Structure

```text
sentiment-analyzer/
│
├── data/
│   └── IMDB_Dataset.csv
│
├── model/
│   ├── train.py
│   ├── vectorizer.pkl
│   └── sentiment_model.pth
│
├── app.py
├── predict.py
├── requirements.txt
└── README.md
```

The model artifacts (`vectorizer.pkl` and `sentiment_model.pth`) are generated after training.

## Installation

Clone the repository:

```bash
git clone https://github.com/ritej8/sentiment-analyzer.git
cd sentiment-analyzer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script from the project root:

```bash
python model/train.py
```

The script:
1. Loads and preprocesses the dataset.
2. Splits the data into training and test sets.
3. Fits the vectorizer on the training data.
4. Trains the neural network.
5. Evaluates the model on the test set.
6. Saves the vectorizer and trained model.

## Prediction

Run the interactive prediction script:

```bash
python predict.py
```

Enter a movie review when prompted.

Example:

```text
Enter a movie review: This movie was fantastic and very enjoyable!

Sentiment: Positive
```

## Flask REST API

Start the API:

```bash
python app.py
```

The API runs locally at:

`http://127.0.0.1:5000`

### Endpoint

`POST /predict`

**Request body:**

```json
{
  "text": "This movie was amazing!"
}
```

**Example response:**

```json
{
  "sentiment": "Positive"
}
```

The API returns HTTP 400 for invalid requests, such as missing or empty text.

## Model Evaluation

The following results were obtained on the held-out test set of 10,000 reviews.

| Metric | Score |
|---|---:|
| Training Accuracy | 99.84% |
| Test Accuracy | 87.88% |
| Negative Precision | 0.87 |
| Negative Recall | 0.88 |
| Negative F1-score | 0.88 |
| Positive Precision | 0.88 |
| Positive Recall | 0.87 |
| Positive F1-score | 0.88 |
| Macro F1-score | 0.88 |

### Confusion Matrix

```text
                 Predicted
                 Negative  Positive

Actual Negative    4420      580
Actual Positive     632     4368
```

The model achieves 87.88% accuracy on unseen test data.

The difference between training accuracy (99.84%) and test accuracy (87.88%) indicates overfitting. This suggests that the model learns the training examples very well but does not generalize equally well to unseen reviews.

## Limitations and Future Improvements

- Reduce overfitting through regularization and hyperparameter tuning.
- Compare Bag-of-Words with TF-IDF representations.
- Experiment with dropout and different hidden-layer sizes.
- Compare the neural network with traditional ML models such as Logistic Regression and SVM.
- Explore more advanced text representations, such as word embeddings and Transformer-based models.
- Improve API deployment and automated testing.

## Author

**Ritej Bouasker**

GitHub: [@ritej8](https://github.com/ritej8)
