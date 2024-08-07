import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer, TFBertModel
import time
from gslTranslater.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_cnn_model(self):
        self.cnn_model = tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            pooling=self.config.params_pooling
        )
        self.save_model(path=self.config.cnn_model_path, model=self.cnn_model)

    def get_transformer_model(self, retries=5, delay=10):
        for attempt in range(retries):
            try:
                self.tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
                self.transformer_model = TFBertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
                self.transformer_model.save_pretrained(self.config.transformer_model_path)
                self.tokenizer.save_pretrained(self.config.tokenizer_path)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise e

    def _prepare_full_model(self, cnn_model, transformer_model, learning_rate, freeze_all, freeze_till):
        if freeze_all:
            for layer in cnn_model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in cnn_model.layers[:-freeze_till]:
                layer.trainable = False

        cnn_output = cnn_model.output
        flatten_cnn = tf.keras.layers.Flatten()(cnn_output)
        
        input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

        bert_output = transformer_model(input_ids, attention_mask=attention_mask)[0]
        flatten_bert = tf.keras.layers.Flatten()(bert_output)
        
        concat_output = tf.keras.layers.Concatenate()([flatten_cnn, flatten_bert])
        
        prediction = tf.keras.layers.Dense(
            units=cnn_model.output_shape[-1],
            activation="softmax"
        )(concat_output)

        full_model = tf.keras.models.Model(
            inputs=[cnn_model.input, input_ids, attention_mask],
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            cnn_model=self.cnn_model,
            transformer_model=self.transformer_model,
            learning_rate=self.config.params_learning_rate,
            freeze_all=True,
            freeze_till=None
        )
        self.save_model(path=self.config.updated_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)