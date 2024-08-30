from gslTranslater.config.configuration import ConfigurationManager
from gslTranslater.components.data_ingestion import DataIngestion
from gslTranslater import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.download_file()
        data_ingestion.extract_tar_file()
        merged_df = data_ingestion.merge_and_clean_csv_files()

        # Analyze raw gloss distribution
        data_ingestion.analyze_raw_gloss_distribution(merged_df)
        
        confirmed_df = data_ingestion.check_image_paths(merged_df)

        # Analyze and process frames
        data_ingestion.analyze_frames(confirmed_df)
        
        # Create the balanced dataset based on max_instances_per_class
        balanced_df = data_ingestion.create_balanced_dataset(confirmed_df)

        # Analyze gloss distribution after trimming
        data_ingestion.analyze_gloss_distribution(balanced_df)
        
        # Split the data into train, test, and validate
        data_ingestion.split_data(balanced_df)
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e