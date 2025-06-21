import streamlit as st
import librosa
import os
import torch.nn as nn
import torch
import numpy as np
from testscript import extract_5channel_features  
NUM_CLASSES=8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmotionCNNBiLSTMWithAttention(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(input_size=32 * 32, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        x = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(x)
model = EmotionCNNBiLSTMWithAttention().to(device)
model.load_state_dict(torch.load(r"C:\Users\Raihan\OneDrive\Desktop\OPEN_PROJECT_AUDIO\scripts\cnn_bilstm_attention_model.pth", map_location=device))
model.eval()


label_map = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

# Streamlit UI
st.set_page_config(page_title="Emotion Predictor", page_icon="🎧")
st.title("🎧 Audio Emotion Recognition")
st.markdown("Upload a `.wav` file to detect emotion from voice!")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded audio temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    features = extract_5channel_features("temp.wav").to(device)
    features = (features - features.mean(dim=(2,3), keepdim=True)) / (features.std(dim=(2,3), keepdim=True) + 1e-6)
   
    # Predict
    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()
        emotion= label_map[pred]

    st.markdown(f"Predicted Emotion: **{emotion}**")
    st.audio("temp.wav", format="audio/wav")
    os.remove("temp.wav")