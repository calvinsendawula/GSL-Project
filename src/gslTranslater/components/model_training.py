import os
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from gslTranslater.constants import *
from gslTranslater.utils.common import read_yaml, create_directories
from gslTranslater.entity.config_entity import (TrainingConfig)

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def load_or_process_dataset(self, csv_path, dataset_type, max_seq_length):
        # Define paths for saving the features and labels
        features_save_dir = os.path.join(self.config.root_dir, 'features')
        os.makedirs(features_save_dir, exist_ok=True)
        features_path = os.path.join(features_save_dir, f"{dataset_type}_features.npy")
        labels_path = os.path.join(features_save_dir, f"{dataset_type}_labels.npy")

        # Check if the features and labels already exist
        if os.path.exists(features_path) and os.path.exists(labels_path):
            print(f"Loading existing {dataset_type} features and labels...")
            data = np.load(features_path)
            labels = np.load(labels_path)
            return data, labels

        print(f"Processing {dataset_type} dataset...")

        df = pd.read_csv(csv_path)

        data = []
        labels = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_type}", unit="item", ncols=100):
            frames_path = os.path.join(self.config.data_dir, row['Path'].replace('/', '\\'))
            frames = sorted([os.path.join(frames_path, img) for img in os.listdir(frames_path) if img.endswith('.jpg')])

            sequence = []
            for frame in frames:
                image = tf.keras.preprocessing.image.load_img(frame, target_size=tuple(self.config.params_image_size[:-1]))
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = tf.keras.applications.resnet.preprocess_input(image)
                sequence.append(image)

            data.append(sequence)
            labels.append(row['Gloss'])

        # Pad sequences to ensure uniform shape
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=max_seq_length, padding='post', dtype='float32')
        labels = np.array(labels)

        # Save the features and labels
        np.save(features_path, data)
        np.save(labels_path, labels)

        return data, labels

    def encode_labels(self, labels):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.config.classes)
        return labels

    def load_model(self):
        # Load the CNN-LSTM model
        self.model = tf.keras.models.load_model(self.config.cnn_lstm_model_path)

    def train(self):
        # Load and process the datasets
        train_data, train_labels = self.load_or_process_dataset(self.config.train_csv, 'train', self.config.max_seq_length)
        validate_data, validate_labels = self.load_or_process_dataset(self.config.validate_csv, 'validate', self.config.max_seq_length)

        # Encode the labels to numeric values
        train_labels = self.encode_labels(train_labels)
        validate_labels = self.encode_labels(validate_labels)

        # Load the model
        self.load_model()

        # Compile the model with the desired optimizer and loss function
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        history = self.model.fit(
            train_data, train_labels,
            validation_data=(validate_data, validate_labels),
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size
        )

        # Save the trained model
        self.model.save(self.config.trained_model_path)
        print(f"Model saved successfully at {self.config.trained_model_path}")

        return history
