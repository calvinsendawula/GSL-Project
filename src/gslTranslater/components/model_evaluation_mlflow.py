import os
from dataclasses import dataclass
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import mlflow
import dagshub
import tensorflow as tf
from torch import nn
from sklearn.preprocessing import LabelEncoder
from gslTranslater.utils.common import read_yaml, create_directories, save_json
from gslTranslater.constants import *
from gslTranslater.entity.config_entity import EvaluationConfig
from urllib.parse import urlparse


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)

        # Filter for the selected unique words
        df = df[df['Gloss'].isin(df['Gloss'].unique()[:self.config.all_params['NUM_UNIQUE_WORDS']])]

        # Ensure we're selecting only one instance per word
        df = df.groupby('Gloss').head(1).reset_index(drop=True)

        data = []
        labels = []

        for _, row in df.iterrows():
            frames_path = os.path.join(self.config.root_dir, row['Path'])
            frames = sorted([os.path.join(frames_path, img) for img in os.listdir(frames_path) if img.endswith('.jpg')])

            sequence = []
            for frame in frames:
                image = tf.keras.preprocessing.image.load_img(frame, target_size=tuple(self.config.image_size[:-1]))
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = tf.keras.applications.resnet.preprocess_input(image)
                sequence.append(image)

            data.append(sequence)
            labels.append(row['Gloss'])

        # Pad sequences to ensure uniform shape
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=self.config.max_seq_length, padding='post', dtype='float32')
        labels = np.array(labels)

        return data, labels

    def encode_labels(self, labels):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = torch.tensor(labels)
        labels = nn.functional.one_hot(labels, num_classes=len(np.unique(labels))).float()
        return labels

    def load_model(self, path: Path):
        self.model = tf.keras.models.load_model(path)
        # No need to call self.model.eval() in TensorFlow/Keras

    def evaluate(self):
        # Load and preprocess the data using TensorFlow/Keras
        test_data, test_labels = self.load_data(self.config.test_csv)

        # Encode labels using TensorFlow/Keras utilities
        label_encoder = LabelEncoder()
        test_labels = label_encoder.fit_transform(test_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(np.unique(test_labels)))

        # Perform predictions using the Keras model
        predictions = self.model.predict(test_data)

        # Calculate loss and accuracy using TensorFlow/Keras
        loss = tf.keras.losses.CategoricalCrossentropy()(test_labels, predictions)
        accuracy = tf.keras.metrics.CategoricalAccuracy()(test_labels, predictions)

        avg_loss = loss.numpy()
        avg_accuracy = accuracy.numpy()

        print(f'Test Loss: {avg_loss}, Test Accuracy: {avg_accuracy * 100:.2f}%')

        return avg_loss, avg_accuracy


    def save_score(self, avg_loss, avg_accuracy):
        # Convert TensorFlow float32 to standard Python float
        scores = {
            "loss": float(avg_loss),  # Convert to float
            "accuracy": float(avg_accuracy)  # Convert to float
        }
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self, avg_loss, avg_accuracy):
        dagshub.init(repo_owner=self.config.dagshub_username, repo_name=self.config.dagshub_repo_name, mlflow=True)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"test_loss": avg_loss, "test_accuracy": avg_accuracy})

            # Register the model if not using a file store
            if tracking_url_type_store != "file":
                mlflow.tensorflow.log_model(self.model, "model", registered_model_name="CNN_LSTM_Model")
            else:
                mlflow.tensorflow.log_model(self.model, "model")
