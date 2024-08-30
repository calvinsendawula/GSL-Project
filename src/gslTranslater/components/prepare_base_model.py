import os
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path
from gslTranslater.utils.common import read_yaml, create_directories
from gslTranslater.constants import *
from gslTranslater.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def prepare_model(self):
        # Load the pretrained ResNet50 model without the top layer
        cnn_base = tf.keras.applications.ResNet50(
            weights=self.config.params_weights,
            include_top=False,
            input_shape=tuple(self.config.params_image_size)
        )

        # Freeze the CNN layers
        for layer in cnn_base.layers:
            layer.trainable = False

        # Define the full model architecture
        model = tf.keras.Sequential()

        # TimeDistributed layer applies the CNN to each frame independently
        model.add(tf.keras.layers.TimeDistributed(cnn_base, input_shape=(None, *self.config.params_image_size)))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

        # Add Masking layer to handle padded frames
        model.add(tf.keras.layers.Masking(mask_value=0.0))

        # LSTM layer processes the sequence of frame features
        model.add(tf.keras.layers.LSTM(self.config.params_classes, return_sequences=False))

        # Save the model
        model.save(self.config.cnn_lstm_model_path)
        print(f"CNN-LSTM base model saved successfully at {self.config.cnn_lstm_model_path}")
