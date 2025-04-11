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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

kagglehub = importlib.import_module("kagglehub")
tf = importlib.import_module("tensorflow")
tfio = importlib.import_module("tensorflow_io")


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


def main():
    parser = argparse.ArgumentParser(description="Train a chainsaw detection model.")
    parser.add_argument(
        "--database-path",
        type=str,
        help="Path to the database. If not provided, the dataset will be downloaded using kagglehub.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16).",
    )

    args = parser.parse_args()

    logger.info(f"Batch size set to: {args.batch_size}")

    # if KAGGLE_KEY is not set prompt forr the user to login
    if "KAGGLE_KEY" not in os.environ:
        logger.debug(
            "KAGGLE_KEY is not set. Please set it or provide the database path."
        )
        try:
            kagglehub.login()
            logger.debug("Successfully logged in to kagglehub.")
        except UnicodeEncodeError as e:
            logger.debug(
                "Error: Invalid characters in username or token. Please ensure they are valid."
            )
            raise e

    db_path = (
        args.database_path
        if args.database_path
        else kagglehub.dataset_download("almemara/forest-watcher")
    )

    PARSED_CHAINSAW_DIR = os.path.join(db_path, "POS")
    NOT_PARSED_CHAINSAW_DIR = os.path.join(db_path, "NEG")

    logger.debug(f"Path to dataset files {PARSED_CHAINSAW_DIR}/")
    logger.debug(f"Path to dataset files {NOT_PARSED_CHAINSAW_DIR}/")

    POS = os.path.join(PARSED_CHAINSAW_DIR)
    NEG = os.path.join(NOT_PARSED_CHAINSAW_DIR)

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
    data = data.batch(args.batch_size).map(
        lambda x, y: (tf.ensure_shape(x, (None, 1491, 257, 1)), y)
    )
    data = data.prefetch(8)

    train = data.take(36)
    test = data.skip(36).take(15)

    model = Sequential()

    model.add(Input(shape=(1491, 257, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        "Adam",
        loss="BinaryCrossentropy",
        metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )

    model.fit(train, epochs=4, validation_data=test)

    MODEL_PATH = (
        f"model/chainsaw_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
