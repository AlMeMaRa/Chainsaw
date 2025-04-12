#!/usr/bin/env python

import os
import argparse
import importlib
import datetime
import json
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
        "-d",
        type=str,
        help="Path to the database. If not provided, the dataset will be downloaded using kagglehub.",
    )
    parser.add_argument(
        "--database-version",
        "-dv",
        type=int,
        help="Version of the database to download. If not provided, the latest version will be used.",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=int,
        default=4,
        help="Number of epochs for training (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
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
    if args.database_path:
        db_path = args.database_path
    elif args.database_version is not None:
        db_path = kagglehub.dataset_download(f"almemara/forest-watcher/version/{args.database_version}")
    else:
        db_path = kagglehub.dataset_download("almemara/forest-watcher")

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

    # Adjust batch size dynamically based on dataset size
    dataset_size = len(list(data))
    batch_size = min(args.batch_size, dataset_size)

    data = data.batch(batch_size, drop_remainder=True).map(
        lambda x, y: (tf.ensure_shape(x, (None, 1491, 257, 1)), y)
    )
    data = data.prefetch(8)

    train = data.take(36)
    test = data.skip(36).take(15)

    # Load include array from include.json
    with open("include.json", "r") as include_file:
        include = json.load(include_file)

    # Load layer details from layers.json
    with open("layers.json", "r") as layer_file:
        layer_details = json.load(layer_file)

    # Filter layers based on the include array using the `name` key
    if include:
        layer_details = [
            layer for layer in layer_details if layer.get("name") in include
        ]

    # Log the layers being loaded
    logger.info(f"Layers loaded: {[layer.get('name') for layer in layer_details]}")

    # Dynamically add layers to the model
    model = Sequential()
    for layer in layer_details:
        layer_type = layer.pop("layer")  # Extract the layer type
        layer.pop("description", None)  # Remove the description field if it exists
        layer.pop("name", None)  # Remove the name field if it exists
        logger.info(f"Adding layer: {layer_type} with parameters: {layer}")
        if layer_type == "Input":
            model.add(Input(**layer))  # Dynamically pass arguments
        else:
            layer_class = getattr(
                tf.keras.layers, layer_type
            )  # Get the layer class dynamically
            model.add(layer_class(**layer))  # Dynamically pass arguments

    model.compile(
        "Adam",
        loss="BinaryCrossentropy",
        metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )

    # Train the model and capture the history
    history = model.fit(train, epochs=args.epoch, validation_data=test)

    # Evaluate the model on the test dataset
    evaluation = model.evaluate(test, return_dict=True)

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    MODEL_PATH = f"model/chainsaw_model_{date_time}.keras"
    MODEL_INFO_PATH = f"model/chainsaw_model_{date_time}.json"

    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    model.save(MODEL_PATH)

    # Save model details to a JSON file with training history and evaluation
    model_details = {
        "file_name": MODEL_PATH,
        "dataset_path": db_path,
        "dataset_name": os.path.basename(db_path) if db_path else "Default Dataset",
        "model_architecture": layer_details,
        "include": include,
        "training_history": history.history,
        "evaluation": evaluation,
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_details, f, indent=4)


if __name__ == "__main__":
    main()
