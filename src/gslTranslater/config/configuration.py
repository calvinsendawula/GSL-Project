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
        csv_paths = config.csv_paths
        plot_paths = config.plot_paths
        
        # Create necessary directories
        create_directories([config.root_dir, csv_paths.analysis_dir, plot_paths.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.tar_dir,
            unzip_dir=config.unzip_dir,
            data_dir=Path(config.data_dir),
            analysis_dir=Path(csv_paths.analysis_dir),
            merged_csv=Path(csv_paths.merged_csv),
            confirmed_csv=Path(csv_paths.confirmed_csv),
            missing_csv=Path(csv_paths.missing_csv),
            gloss_distribution_csv=Path(csv_paths.gloss_distribution_csv),
            balanced_csv=Path(csv_paths.balanced_csv),
            summary_csv=Path(csv_paths.summary_csv),
            frame_count_csv=Path(csv_paths.frame_count_csv),
            analysis_txt=Path(csv_paths.analysis_txt),
            train_csv=Path(config.train_csv),
            test_csv=Path(config.test_csv),
            validate_csv=Path(config.validate_csv),
            plot_dir=Path(plot_paths.root_dir),
            gloss_distribution_plot=Path(plot_paths.gloss_distribution_plot),
            max_instances_per_class=self.params.MAX_INSTANCES_PER_CLASS,
            train_split=self.params.TRAIN_SPLIT,
            test_split=self.params.TEST_SPLIT,
            validate_split=self.params.VALIDATE_SPLIT
        )
        
        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(prepare_base_model.root_dir),
            cnn_lstm_model_path=Path(prepare_base_model.cnn_lstm_model_path),
            params_image_size=params.IMAGE_SIZE,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion
        params = self.params

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            cnn_lstm_model_path=Path(prepare_base_model.cnn_lstm_model_path),
            train_csv=Path(data_ingestion.train_csv),
            validate_csv=Path(data_ingestion.validate_csv),
            test_csv=Path(data_ingestion.test_csv),
            data_dir=Path(data_ingestion.data_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            max_seq_length=params.MAX_SEQ_LENGTH,
            learning_rate=params.LEARNING_RATE,
            classes=params.CLASSES
        )

        return training_config
    
    
    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        data_ingestion = self.config.data_ingestion
        return EvaluationConfig(
            root_dir=Path(data_ingestion.data_dir),
            path_of_model=Path(self.config.training.trained_model_path),
            test_csv=Path(data_ingestion.test_csv),
            mlflow_uri=evaluation.mlflow_uri,
            dagshub_username=evaluation.dagshub_username,
            dagshub_repo_name=evaluation.dagshub_repo_name,
            max_seq_length=self.params.MAX_SEQ_LENGTH,
            image_size=self.params.IMAGE_SIZE,
            batch_size=self.params.BATCH_SIZE,
            all_params=self.params
        )