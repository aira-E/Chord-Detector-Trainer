# Chord-Detector-Trainer

This repository contains a project for training a chord detector using audio processing and machine learning. It leverages libraries such as Librosa, NumPy, PyTorch, and scikit-learn for audio preprocessing, feature extraction, and model training.

## Installation

Follow these steps to install the required dependencies for this project:

### 1. Install Librosa
Librosa is a library for audio and music analysis. Run the following command:
```bash
pip install librosa
```

### 2. Install NumPy
NumPy is required for numerical computations. Install it using:
```bash
pip install numpy
```

### 3. Install PyTorch
PyTorch is the machine learning framework used in this project. To install it, run:
```bash
pip install torch torchvision
```
> **Note:** Replace the above command with the appropriate one for your system and CUDA version. Check the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

### 4. Install scikit-learn
scikit-learn is used for preprocessing and model evaluation. Install it with:
```bash
pip install scikit-learn
```

### 5. Standard Python Libraries
`pickle` and `collections` are part of Python's standard library, so no additional installation is required.

## Usage

To use the chord detector trainer, ensure all dependencies are installed, and then execute the main Python file:
```bash
python main.py
```

## Contributing
If you'd like to contribute to this project, feel free to submit issues or pull requests. Ensure that your code follows the existing style and passes all tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
