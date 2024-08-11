import os
import tarfile
import gdown
from gslTranslater import logger
from gslTranslater.utils.common import get_size
from gslTranslater.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_URL
            tar_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {tar_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, tar_download_dir, quiet=False)

            logger.info(f"Downloaded data from {dataset_url} into file {tar_download_dir}")

        except Exception as e:
            raise e

    def extract_tar_file(self):
        extract_path = self.config.unzip_dir
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(self.config.local_data_file, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)