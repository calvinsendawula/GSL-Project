from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    data_dir: Path
    analysis_dir: Path
    merged_csv: Path
    confirmed_csv: Path
    missing_csv: Path
    gloss_distribution_csv: Path
    balanced_csv: Path
    summary_csv: Path
    frame_count_csv: Path
    analysis_txt: Path
    train_csv: Path
    test_csv: Path
    validate_csv: Path
    plot_dir: Path
    gloss_distribution_plot: Path
    max_instances_per_class: int
    train_split: float
    test_split: float
    validate_split: float

    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    cnn_lstm_model_path: Path
    params_image_size: list
    params_weights: str
    params_classes: int
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    cnn_lstm_model_path: Path
    train_csv: Path
    validate_csv: Path
    test_csv: Path
    data_dir: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    max_seq_length: int
    learning_rate: float
    classes: int
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    test_csv: Path
    mlflow_uri: str
    dagshub_username: str
    dagshub_repo_name: str
    max_seq_length: int
    image_size: list
    batch_size: int
    all_params: dict
