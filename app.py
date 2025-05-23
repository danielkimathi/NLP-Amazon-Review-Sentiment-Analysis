import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load trained LSTM model
model = tf.keras.models.load_model("sentiment_lstm_model_2_12.keras", compile=False)

# Load tokenizer for preprocessing
with open("Data/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define sentiment labels
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to preprocess user input
def preprocess_text(text):
    max_length = 100  # Must match training sequence length
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding="post")
    return padded_sequence

# Streamlit App Layout
st.title("Amazon Review Sentiment Analysis")
st.subheader("Enter a review below to predict sentiment")

# User input
user_input = st.text_area("Enter a review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess input
        processed_input = preprocess_text(user_input)

        # Predict sentiment
        prediction = model.predict(processed_input)
        predicted_class = np.argmax(prediction)

        
        predicted_class = np.argmax(prediction)
        
        # Display results
        st.write(f"### Sentiment Prediction: **{sentiment_labels[predicted_class]}**")
        st.write(f"Confidence Scores: {prediction[0]}")
    else:
        st.warning("Please enter a review before analyzing.")
