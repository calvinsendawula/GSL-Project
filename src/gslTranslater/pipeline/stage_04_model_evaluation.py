from gslTranslater.config.configuration import ConfigurationManager
from gslTranslater.components.model_evaluation_mlflow import Evaluation
from gslTranslater import logger

STAGE_NAME = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        model_evaluation = Evaluation(config=evaluation_config)
        model_evaluation.load_model(evaluation_config.path_of_model)
        avg_loss, avg_accuracy = model_evaluation.evaluate()
        model_evaluation.save_score(avg_loss, avg_accuracy)
        model_evaluation.log_into_mlflow(avg_loss, avg_accuracy)


    
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