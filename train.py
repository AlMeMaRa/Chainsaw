#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Define paths
POS = os.path.join("data", "Parsed_Chainsaw_Clips")
NEG = os.path.join("data", "Parsed_Not_Chainsaw_Clips")


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
MODEL_PATH = "model/chainsaw_model.keras"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))
model.save(MODEL_PATH)
