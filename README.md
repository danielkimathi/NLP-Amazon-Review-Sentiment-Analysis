# Amazon-Sentiment-Analysis-NLP
**Amazon Product Review Sentiment Classification with Deep Learning**

## Project Overview
This project classifies Amazon product reviews into Positive, Neutral, or Negative sentiment using natural language processing (NLP) and machine learning models. It combines traditional classifiers (Logistic Regression, Naive Bayes) with advanced deep learning models (RNN, LSTM), while addressing common NLP challenges like class imbalance and preprocessing.

The notebook explores the full modeling pipeline, and includes an interactive script for making sentiment predictions from raw text inputs using a trained model.

### The notebook includes:
- **Exploratory Data Analysis (EDA)**: Score distributions, sentiment imbalance, and text length analysis.
- **Text preprocessing and vectorization** using spaCy, NLTK, and TF-IDF.
- **Class balancing with SMOTE** to address skewed sentiment labels.
- **Training and evaluation** of both baseline and deep learning models.
- **Visualizations**: Word clouds, bar plots, box plots, confusion matrices, and classification metrics.

---

## How to Use This Project
- **Outputs are fully included** in the Jupyter Notebook.
- This notebook is not intended to be re-run locally — no external data or setup is required.
- An interactive script (app.py) is available to run predictions locally using the trained LSTM model and custom review input.

---

## Files Included in This Repository
- Amazon-Sentiment-Analysis.ipynb – The full analysis notebook, with all outputs pre-rendered.
- app.py – script for local interactive sentiment prediction.
- sentiment_lstm_model.h5 – Trained LSTM model used in the interactive script.

---

## ⚠️ Known Limitation: Local App Resource Usage

The app.py script runs a deep learning LSTM model with 100D GloVe embeddings and a 20,000-word vocabulary. While the model performs well and delivers good predictions, repeated usage in limited environments (such as small VMs or shared servers) may cause occasional memory or performance issues.

If the app becomes unresponsive, restart the script or clear memory as needed.

---

## Project Highlights
- **Text Preprocessing**:
  - Tokenization, lemmatization, and stopword removal with spaCy and NLTK.
  
- **TF-IDF Vectorization + Hyperparameter Tuning**:
  - max_features and ngram_range explored for optimal performance.
  
- **Modeling**:
  - **Logistic Regression** and **Naive Bayes** for strong baselines.
  - **RNN and LSTM** models trained with embedded sequences using pre-trained GloVe vectors.
  
- **Evaluation**:
  - Classification reports, confusion matrices, word clouds, and accuracy/loss graphs across epochs.

---

## Results Summary
- **Logistic Regression**:
  - Outperformed Naive Bayes with a mean cross-validated F1-score of **0.846**.
  
- **LSTM Model**:
  - Achieved **85% test accuracy**, with strong precision and recall on the Positive class.
  
- **RNN Model**:
  - Demonstrated meaningful accuracy growth over training, but slightly less stable than LSTM.

---

## Final Notes
This project showcases a full sentiment classification pipeline using real-world product reviews. It includes both classical and deep learning approaches, and provides a local interactive interface for exploring model predictions without requiring additional data downloads or cloud deployment.

You can view the full analysis in the notebook and test the trained model locally using the provided script.


