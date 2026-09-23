"""Streamlit app: predict the sentiment of a movie review with the trained LSTM model."""
import pickle
import re

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200  # must match the padding length used in training


@st.cache_resource
def load_artifacts():
    model = load_model("imdb_lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def clean_text(text: str) -> str:
    """Apply the same cleaning used during training."""
    text = re.sub(r"<.*?>", "", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)     # keep letters only
    return text.lower().strip()


def predict_sentiment(text: str, model, tokenizer):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob_positive = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if prob_positive > 0.5 else "Negative"
    confidence = prob_positive if label == "Positive" else 1 - prob_positive
    return label, confidence


st.title("Movie Review Sentiment Analysis")
st.write("An LSTM neural network trained on 50,000 IMDB reviews (88% test accuracy).")

model, tokenizer = load_artifacts()
review = st.text_area("Enter a movie review")

if st.button("Predict"):
    if review.strip():
        label, confidence = predict_sentiment(review, model, tokenizer)
        st.success(f"Sentiment: {label}")
        st.write(f"Confidence: {confidence:.0%}")
    else:
        st.warning("Please enter a review first.")
