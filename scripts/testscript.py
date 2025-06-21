import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from moviepy import VideoFileClip
NUM_CLASSES=8
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

LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
DURATION = 4.0
FIXED_LEN = int(SAMPLE_RATE * DURATION)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_audio_from_video(video_path, output_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, fps=SAMPLE_RATE, nbytes=2, codec='pcm_s16le', logger=None)
    return output_path

def extract_5channel_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(data=y, size=FIXED_LEN)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel)
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

   
    T = mel_db.shape[1]
    T_TARGET = 200  

 
    pitches, _ = librosa.piptrack(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    pitch = np.max(pitches, axis=0, keepdims=True)  
    pitch = np.tile(pitch, (64, 1))  

    energy = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)  
    energy = np.tile(energy, (64, 1))  

    def fix_shape(x):
        if x.shape[1] < T_TARGET:
            return np.pad(x, ((0, 0), (0, T_TARGET - x.shape[1])), mode='constant')
        else:
            return x[:, :T_TARGET]

    mel_db = fix_shape(mel_db)
    delta = fix_shape(delta)
    delta2 = fix_shape(delta2)
    pitch = fix_shape(pitch)
    energy = fix_shape(energy)

    features = np.stack([mel_db, delta, delta2, pitch, energy], axis=0)  
    features = torch.tensor(features).unsqueeze(0).float() 
    return features


def predict_from_video(video_path: str, model_path: str = r"C:\Users\Raihan\OneDrive\Desktop\OPEN_PROJECT_AUDIO\scripts\cnn_bilstm_attention_model.pth") -> str:
    """Takes a .mp4 video path as input and returns predicted emotion"""
    assert os.path.exists(video_path), f"Video path does not exist: {video_path}"
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

    audio_path = extract_audio_from_video(video_path)
    features = extract_5channel_features(audio_path).to(device)

    model = EmotionCNNBiLSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(features)
        pred_idx = torch.argmax(output, dim=1).item()

    os.remove(audio_path)  
    return LABELS[pred_idx]

if __name__ == "__main__":
    test_video = r"C:\Users\Raihan\OneDrive\Desktop\01-02-03-01-02-01-20.mp4"
    print("🎯 Predicted Emotion:", predict_from_video(test_video))