import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import Counter


# Load and preprocess audio
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio)  # Trim silence
    return audio


def augment_audio(audio, sr):
    """
    Apply safe audio augmentations: time-stretching and volume adjustment.
    """
    if np.random.rand() < 0.5:
        audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))  # Time stretching
    if np.random.rand() < 0.5:
        audio = audio * np.random.uniform(0.8, 1.2)  # Volume adjustment
    return audio


def extract_features(audio, sr=16000, n_mels=128, max_len=100):
    """
    Extract log Mel spectrogram features and pad/truncate to a fixed size.
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Ensure consistent feature length
    if log_mel_spec.shape[1] < max_len:
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, max_len - log_mel_spec.shape[1])), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :max_len]

    return log_mel_spec


def encode_labels(labels):
    """
    Encode labels as integers.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder


def preprocess_dataset(dataset_path, sr=16000, n_mels=128, max_len=100):
    """
    Preprocess dataset: Load audio files, augment data, extract features, and encode labels.
    """
    features = []
    labels = []

    for chord_dir in os.listdir(dataset_path):
        chord_path = os.path.join(dataset_path, chord_dir)
        if os.path.isdir(chord_path):
            print(f"Processing chord class: {chord_dir}")
            for file in os.listdir(chord_path):
                file_path = os.path.join(chord_path, file)
                if file_path.endswith(".wav"):  # Only process .wav files
                    try:
                        # Load and preprocess audio
                        audio = load_audio(file_path, sr)
                        audio = augment_audio(audio, sr)  # Apply augmentation
                        feature = extract_features(audio, sr, n_mels, max_len)
                        features.append(feature)
                        labels.append(chord_dir)  # Use directory name as label
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Encode labels
    encoded_labels, encoder = encode_labels(labels)
    print(f"Class Distribution: {Counter(labels)}")  # Check class distribution

    return np.array(features), np.array(encoded_labels), encoder


class ChordDataset(Dataset):
    """
    PyTorch Dataset for Chord Detection.
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


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


def compute_classification_accuracy(model, val_loader, device):
    """
    Compute classification accuracy on the validation set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def train_model(train_dataset, val_dataset, num_classes, num_epochs=20, batch_size=32, lr=0.001):
    """
    Train the Chord Detector model using CrossEntropyLoss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ChordDetector(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        # Compute validation accuracy
        val_accuracy = compute_classification_accuracy(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training completed.")
    return model  # Return the trained model


def save_preprocessed_data(features, labels, encoder, output_path="preprocessed_data.pkl"):
    """
    Save preprocessed features, labels, and encoder to a file.
    """
    with open(output_path, "wb") as f:
        pickle.dump((features, labels, encoder), f)
    print(f"Preprocessed data saved to {output_path}")


def load_preprocessed_data(input_path="preprocessed_data.pkl"):
    """
    Load preprocessed features, labels, and encoder from a file.
    """
    with open(input_path, "rb") as f:
        features, labels, encoder = pickle.load(f)
    print(f"Preprocessed data loaded from {input_path}")
    return features, labels, encoder


def save_model(model, encoder, filepath="trained_chord_detector.pth"):
    """
    Save the trained model and label encoder.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_classes': encoder.classes_
    }, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    #Change the path depending on the user's path
    dataset_path = r"D:\Aira-E\Chord-Detector-Trainer\Extracted_Chords"

    features, labels, encoder = preprocess_dataset(dataset_path)
    save_preprocessed_data(features, labels, encoder, "preprocessed_data.pkl")

    # Load preprocessed data
    print("Loading preprocessed data...")
    features, labels, encoder = load_preprocessed_data("preprocessed_data.pkl")

    # Create PyTorch Dataset
    print("Creating PyTorch Dataset...")
    dataset = ChordDataset(features, labels)

    # Split dataset into training and validation sets
    print("Splitting dataset into training and validation sets...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Train the model
    print("Training the model...")
    model = train_model(
        train_dataset,
        val_dataset,
        num_classes=len(encoder.classes_),
        num_epochs=100,
        batch_size=32,
        lr=0.001
    )

    # Save the trained model
    save_model(model, encoder, filepath="trained_chord_detector.pth")
    print("Model training completed.")
