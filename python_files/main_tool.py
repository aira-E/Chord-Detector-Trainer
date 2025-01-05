import os
import torch
import librosa
import numpy as np
import torch.nn as nn  # Import the torch.nn module
from sklearn.preprocessing import LabelEncoder

# Define the ChordDetector model class
class ChordDetector(nn.Module):
    def __init__(self, num_classes):
        super(ChordDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Dynamically calculate flattened size
        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Output logits for each chord class

    def _get_flattened_size(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 128, 100)
            x = self.pool(self.relu(self.conv1(sample_input)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            return x.view(-1).size(0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Logits
        return x


def load_model(filepath="trained_chord_detector.pth"):
    """
    Load the trained model and label encoder.
    """
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    encoder_classes = checkpoint['encoder_classes']
    num_classes = len(encoder_classes)

    # Initialize the model and load weights
    model = ChordDetector(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize the label encoder
    encoder = LabelEncoder()
    encoder.classes_ = encoder_classes

    print("Model and encoder loaded successfully.")
    return model, encoder


def preprocess_input(file_path, sr=16000, n_mels=128, max_len=100):
    """
    Preprocess the input .wav file into a Mel spectrogram.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio)  # Trim silence

    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Ensure consistent feature length
    if log_mel_spec.shape[1] < max_len:
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, max_len - log_mel_spec.shape[1])), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :max_len]

    # Add channel dimension
    feature = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return feature


def predict_chord(file_path, model, encoder):
    """
    Predict the chord being played in the input .wav file.
    """
    # Preprocess the input
    feature = preprocess_input(file_path)

    # Pass through the model
    with torch.no_grad():
        output = model(feature)
        predicted_class = torch.argmax(output, dim=1).item()

    # Decode the predicted class to the chord name
    chord = encoder.inverse_transform([predicted_class])[0]
    return chord


if __name__ == "__main__":
    # Path to the saved model
    model_filepath = "trained_chord_detector.pth"

    # Path to the input .wav file
    input_wav = r"D:\Aira-E\Chord-Detector-Trainer\Extracted_Chords\A_major\Major-A_Grezyl.wav"

    # Load the model and encoder
    model, encoder = load_model(model_filepath)

    # Predict the chord
    chord = predict_chord(input_wav, model, encoder)
    print(f"The predicted chord is: {chord}")
