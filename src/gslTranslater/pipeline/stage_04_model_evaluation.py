from gslTranslater.config.configuration import ConfigurationManager
from gslTranslater.components.model_evaluation_mlflow import Evaluation
from gslTranslater import logger
import dagshub

dagshub.init(repo_owner='johndoe', repo_name='My-MLFlow-Repo-Name', mlflow=True)

STAGE_NAME = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e