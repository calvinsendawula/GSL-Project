from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    cnn_model_path: Path
    transformer_model_path: Path
    tokenizer_path: Path
    updated_model_path: Path