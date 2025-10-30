# app.py for sentiment prediction via Gradio

import gradio as gr
import torch
import joblib
from model import LSTMClassifier  # Put your LSTMClassifier definition in model.py or inside app.py
import numpy as np

# Load artifacts
vocab = joblib.load("vocab.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Model hyperparameters (match training)
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
NUM_CLASSES = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and load model weights
model = LSTMClassifier(len(vocab)+2, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS)
model.load_state_dict(torch.load("lstm_model.pt", map_location=device))
model.to(device)
model.eval()

# Copy preprocessing/tokenization from notebook
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_review(text, vocab):
    return [vocab.get(word, 1) for word in text.split()]

def predict_sentiment_gr(review):
    review_clean = clean_text(review)
    encoded = encode_review(review_clean, vocab)
    length = len(encoded)
    if length < MAX_LEN:
        encoded += [0] * (MAX_LEN - length)
    else:
        encoded = encoded[:MAX_LEN]
    tensor_review = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor_review)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
    sentiment = label_encoder.inverse_transform([pred_class])[0]
    return f"Sentiment: {sentiment} (confidence {confidence:.2f})"

# Gradio UI
iface = gr.Interface(
    fn=predict_sentiment_gr,
    inputs="text",
    outputs="text",
    title="IMDB Movie Sentiment Analysis",
    description="Enter a movie review to predict its sentiment (positive/negative) using an LSTM trained on IMDB dataset."
)

if __name__ == "__main__":
    iface.launch()
