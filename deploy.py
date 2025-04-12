#!/usr/bin/env python

import os
import argparse
import importlib
import datetime
from loguru import logger

# looop the lines of ./ENV file and export all env variables
with open("env") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ[key] = value


import tensorflow as tf
import tensorflow_io as tfio

# List available models in the model directory
model_dir = "model"
available_models = [f for f in os.listdir(model_dir) if f.endswith(".keras")]

if not available_models:
    raise FileNotFoundError("No .keras models found in the model directory")

# Print available models
print("Available models:")
for i, model_name in enumerate(available_models):
    print(f"{i}: {model_name}")

# Get user choice
choice = int(input("Select model number: "))
if choice < 0 or choice >= len(available_models):
    raise ValueError("Invalid model selection")

MODEL_PATH = os.path.join(model_dir, available_models[choice])

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model input shape: {model.input_shape}")

# Check if the model is loaded correctly
if model is None:
    raise RuntimeError("Failed to load the model. Please check the model path.")


# Data loading and preprocessing_wav functions
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)  # Ensure sample_rate is int64
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess_wav(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


# Function to load and preprocess audio
def load_mp3_16k_mono(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    sample_rate = tf.cast(res.rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess_mp3(sample):
    sample_size = tf.shape(sample)[0]
    padding_size = tf.maximum(0, 48000 - sample_size)  # Ensure consistent input size
    zero_padding = tf.zeros(padding_size, dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    wav = wav[:48000]  # Truncate to 48000 samples if longer
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    spectrogram = tf.image.resize(
        spectrogram, [1491, 257]
    )  # Resize to match model input
    return spectrogram


# Function to make predictions
def predict_audio(file_path):
    # Identify if wav or mp3
    if file_path.endswith(".mp3"):
        wav = load_mp3_16k_mono(file_path)
    elif file_path.endswith(".wav"):
        wav = load_wav_16k_mono(file_path)
    else:
        # raise ValueError("Unsupported audio format. Please provide a .wav or .mp3 file.")
        print(
            "Unsupported audio format. Please provide a .wav or .mp3 file. Defaulting to mp3."
        )

    spectrogram = preprocess_mp3(wav)
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension
    prediction = model.predict(spectrogram)
    # print(f"Prediction shape: {prediction.shape}")
    return "Chainsaw" if prediction[0][0] > 0.99 else "Not Chainsaw"


# Create a class to handle audio predictions and file organization
class ChainSawClassifier:
    def __init__(self):
        self.chainsaw_files = []
        self.not_chainsaw_files = []

    def classify_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory)
                    prediction = predict_audio(file_path)
                    if prediction == "Chainsaw":
                        self.chainsaw_files.append(relative_path)
                    else:
                        self.not_chainsaw_files.append(relative_path)
                else:
                    print(f"Skipping unsupported file format: {filename}")

    def print_results(self):
        print("\nChainsaw Files:")
        for file in self.chainsaw_files:
            print(f"- {file}")

        print("\nNot Chainsaw Files:")
        for file in self.not_chainsaw_files:
            print(f"- {file}")


# Create instance and process files
classifier = ChainSawClassifier()
RECORDINGS_DIR = os.path.join("data", "Recordings")
classifier.classify_files(RECORDINGS_DIR)
classifier.print_results()
