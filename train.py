#!/usr/bin/env python

import os

# Ask user whether to use GPU or CPU
use_gpu = input("Do you want to use GPU for training? (yes/no): ").lower().strip()
if use_gpu in ['yes', 'y']:
    # Allow GPU usage (don't set CUDA_VISIBLE_DEVICES)
    print("Using GPU for training.")
else:
    # Disable GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU for training.")

# Authenticate with Kaggle using API credentials
kaggle_json_path = os.path.join(os.environ["HOME"], ".config", "kaggle", "kaggle.json")

# Get dirname of the kaggle.json file
kaggle_json_dir = os.path.dirname(kaggle_json_path)
# Check if the kaggle.json file exists
if not os.path.exists(kaggle_json_path):
    # If it doesn't exist, create the directory
    os.makedirs(kaggle_json_dir, exist_ok=True)
    # Create an empty kaggle.json file
    with open(kaggle_json_path, "w") as f:
        f.write("{}")
    print(
        f"kaggle.json file created at {kaggle_json_path}. Please add your Kaggle API credentials to this file."
    )
else:
    print(f"Kaggle API key found at {kaggle_json_path}")

# Set the KAGGLE_CONFIG_DIR environment variable to the directory containing kaggle.json
print(f"Setting KAGGLE_CONFIG_DIR to {kaggle_json_dir}")
os.environ["KAGGLE_CONFIG_DIR"] = kaggle_json_dir


import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU {gpu} is set to allow memory growth.")


import kagglehub
import kaggle

# # Import the Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime

api = KaggleApi()
# # Authenticate using Kaggle API
api.authenticate()

# # Download the private dataset from Kaggle
# # This dataset contains audio files for chainsaw and non-chainsaw sounds
# # Note: To use Kaggle API, you need to set up your Kaggle API credentials.
# #* 1. Go to your Kaggle account settings: https://www.kaggle.com/account
# #* 2. Scroll down to the "API" section and click on "Create New API Token".
# #* 3. This will download a file named `kaggle.json`.
# #* 4. Place this file in the `~/.kaggle/` directory (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).
# #* 5. Ensure the file has proper permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/Mac).
forest_watcher_db = kagglehub.dataset_download("almemara/forest-watcher")

# ? Sets the path to the dataset directory
PARSED_CHAINSAW_DIR = os.path.join(forest_watcher_db, "Kaggle", "POS")
NOT_PARSED_CHAINSAW_DIR = os.path.join( forest_watcher_db, "Kaggle", "NEG")

print(f"Path to dataset files {PARSED_CHAINSAW_DIR}/")
print(f"Path to dataset files {NOT_PARSED_CHAINSAW_DIR}/")

# Define paths
POS = os.path.join(PARSED_CHAINSAW_DIR)
NEG = os.path.join(NOT_PARSED_CHAINSAW_DIR)


# Data loading and preprocess_waving functions
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


# Create datasets
pos = tf.data.Dataset.list_files(os.path.join(POS, "*.wav"))
neg = tf.data.Dataset.list_files(os.path.join(NEG, "*.wav"))
positives = tf.data.Dataset.zip(
    (pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos))))
)
negatives = tf.data.Dataset.zip(
    (neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg))))
)
data = positives.concatenate(negatives)
data = data.map(preprocess_wav)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16).map(lambda x, y: (tf.ensure_shape(x, (None, 1491, 257, 1)), y))
data = data.prefetch(8)

# Split datasets
train = data.take(36)
test = data.skip(36).take(15)

# Build model
model = Sequential()
model.add(Input(shape=(1491, 257, 1)))
model.add(Conv2D(16, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    "Adam",
    loss="BinaryCrossentropy",
    metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
)

# Train model
model.fit(train, epochs=4, validation_data=test)

# Save model
MODEL_PATH = f"model/chainsaw_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))
model.save(MODEL_PATH)
