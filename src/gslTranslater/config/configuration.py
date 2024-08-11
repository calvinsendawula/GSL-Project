from gslTranslater.constants import *
import os
from gslTranslater.utils.common import read_yaml, create_directories
from gslTranslater.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.tar_dir,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            cnn_model_path=Path(config.cnn_model_path),
            transformer_model_path=Path(config.transformer_model_path),
            tokenizer_path=Path(config.tokenizer_path),
            updated_model_path=Path(config.updated_model_path)
        )
        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        training_data = Path(training.training_data_dir)
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_image_size=self.params.IMAGE_SIZE
        )

        return training_config
    
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=self.config.training.trained_model_path,
            testing_data_dir=self.config.training.testing_data_dir,
            mlflow_uri=self.config.evaluation.mlflow_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
