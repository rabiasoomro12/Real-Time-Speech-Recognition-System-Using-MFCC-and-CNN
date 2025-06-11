Real-Time Speech Recognition System Using MFCC and CNN

This repository contains the implementation of a real-time speech recognition system designed to transcribe five voice commands ("cat," "no," "stop," "wow," "yes") for smart home applications. The project was developed as part of a Complex Engineering Problem (CEP) for the Bachelor of Engineering in Computer Systems Engineering, Spring 2025.
Project Overview
The goal of this project is to develop a real-time speech recognition system that accurately identifies five spoken commands in diverse acoustic environments, optimized for low latency and robustness to noise. The system uses Mel-Frequency Cepstral Coefficients (MFCCs) for feature extraction, a Convolutional Neural Network (CNN) for acoustic modeling, a bigram language model for sequence constraints, and TensorFlow Lite (TFLite) for real-time performance optimization.
Key Features

MFCC Feature Extraction: Extracts 13 or 20 MFCCs with a 16 kHz sampling rate, padded to 32 frames.
CNN Acoustic Model: Two Conv2D layers with 32 and 64 filters, followed by pooling, dropout, and dense layers.
Bigram Language Model: Constrains word sequences to improve transcription accuracy.
Real-Time Optimization: Achieves 0.30 ms inference latency using TFLite.
Evaluation Metrics: 68.25% test accuracy, Word Error Rate (WER) of 0.3125, and noise robustness analysis.

Methodology
The system pipeline consists of the following steps:

Dataset Preparation: Uses a balanced subset of 2000 samples (400 per command) from the Google Speech Commands Dataset, sampled at 16 kHz.
Feature Extraction: Computes MFCCs (n_mfcc=13 or 20) using Librosa with a 512-sample window, 256-sample hop length, and 32-frame padding.
Data Splitting: Splits data into 80% training (1600 samples) and 20% testing (400 samples) with stratified sampling.
CNN Model Design:
Two Conv2D layers (32 and 64 filters, 3x3 kernel, ReLU).
MaxPooling2D (2x2) after each Conv2D.
Flatten, Dense (128 units, ReLU), Dropout (0.5), and softmax output (5 classes).


Training: Trains for 20 epochs with Adam optimizer, batch size 32, and early stopping (patience=5).
Language Model: Implements a bigram model using NLTK, trained on simulated transcriptions.
Real-Time Optimization: Converts the Keras model to TFLite, reducing latency from 320.23 ms to 0.30 ms.
Noise Robustness: Tests accuracy at SNR levels (20, 10, 0, -10 dB) on 50 test samples.
Evaluation: Computes WER, precision, recall, F1-score, and visualizes results with confusion matrices and live prediction plots.

Core Implementation
The MFCC extraction and CNN inference are implemented in Python using Librosa and TensorFlow:
import librosa
import numpy as np
import tensorflow as tf

# MFCC Extraction
def extract_mfcc(audio_path, n_mfcc=13, n_fft=512, hop_length=256, num_frames=32):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] < num_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, num_frames - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :num_frames]
    return mfcc

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(13, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

Results
Quantitative Evaluation

Test Accuracy: 68.25% on 400 test samples.
Word Error Rate (WER): 0.3125.
Inference Latency: Reduced from 320.23 ms to 0.30 ms with TFLite.
Noise Robustness: Accuracy ranges from 0.1900 (10 dB) to 0.2075 (0 dB) under Gaussian noise.

Qualitative Evaluation

Classification Report:
"stop" and "yes" achieve high precision (0.91).
"cat" has high recall (0.89) but low precision (0.50).


Visualizations:
Confusion matrix shows misclassifications (e.g., "no" as "cat").
Live prediction plot displays confidence scores for real-time feedback.
Accuracy/loss plots confirm model convergence.



Visualizations

Label Distribution: Balanced dataset (400 samples per command).
MFCC Plot: Spectral characteristics of "cat" command.
Training Split: Pie chart of 80% training, 20% testing.
Accuracy/Loss: Training and validation curves.
Latency Comparison: Bar chart of pre- and post-TFLite optimization.
Noise Robustness: Accuracy vs. SNR plot.


Installation
To run the project, ensure Python 3.8+ is installed. Follow these steps:

Clone the repository:
git clone https://github.com/your-username/speech-recognition-mfcc-cnn.git
cd speech-recognition-mfcc-cnn


Install dependencies:
pip install -r requirements.txt


Requirements file (requirements.txt):
librosa==0.10.1
numpy==1.24.3
tensorflow==2.12.0
nltk==3.8.1
jiwer==2.5.1
matplotlib==3.7.1
seaborn==0.12.2



Usage

Download the Speech Commands Dataset and place it in the data/ folder.
Run the main script to train and evaluate the model:python main.py --data_path data/speech_commands --output_model model.h5


For real-time prediction:python predict.py --model model.tflite --audio sample.wav


The script will:
Extract MFCCs from audio.
Train or load the CNN model.
Generate visualizations (confusion matrix, accuracy plots, etc.).
Output predictions and confidence scores.



Example
import librosa
import tensorflow as tf

# Load audio and extract MFCC
audio_path = "sample.wav"
mfcc = extract_mfcc(audio_path, n_mfcc=13)
mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Shape: (1, 13, 32, 1)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Predict
interpreter.set_tensor(input_details[0]['index'], mfcc)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
print("Predicted command:", np.argmax(prediction))

Limitations

Noise Robustness: Low accuracy (0.19–0.21) under noise due to lack of noise-augmented training.
Language Model: Bigram model is simplistic, limiting sequence diversity.
Vocabulary Size: Restricted to five commands, limiting scalability.
Model Complexity: Basic CNN may underperform compared to deeper architectures.

Future Improvements

Noise Augmentation: Train with noisy data to improve robustness.
Advanced Language Model: Use LSTM or transformer-based models for better sequence prediction.
Feature Exploration: Test log-mel spectrograms or Wav2Vec features.
Scalability: Expand vocabulary with larger datasets like LibriSpeech.
Model Enhancement: Incorporate attention mechanisms or transfer learning.

References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
Rabiner, L., & Juang, B.-H. (1993). Fundamentals of Speech Recognition. Prentice Hall.
Speech Commands Dataset: https://www.tensorflow.org/datasets/catalog/speech_commands
TensorFlow: https://www.tensorflow.org
Librosa: https://librosa.org/doc/latest/index.html
NLTK: https://www.nltk.org
Jiwer: https://github.com/jitsi/jiwer

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Supervisor: Dr. Junaid Ahmed.
Libraries: Thanks to the developers of librosa, tensorflow, nltk, jiwer, matplotlib, and seaborn.


Explore the code, test it with your own audio, and contribute to enhancing the system! For questions, reach out via GitHub Issues.
