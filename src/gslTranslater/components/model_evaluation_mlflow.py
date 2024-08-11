from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from transformers import BertModel
from torchvision import datasets, transforms
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu
from gslTranslater.constants import *
from gslTranslater.entity.config_entity import EvaluationConfig
from gslTranslater.utils.common import save_json
from gslTranslater.components.sign_language_translator import SignLanguageTranslator
from urllib.parse import urlparse


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_model(self) -> torch.nn.Module:
        model = SignLanguageTranslator(
            cnn_model=models.resnet50(pretrained=False),
            transformer_model=BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1'),
            tokenizer_len=None
        )
        model.load_state_dict(torch.load(self.config.path_of_model))
        model.eval()
        return model

    def _test_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
        ])

        test_dataset = datasets.ImageFolder(root=self.config.testing_data_dir, transform=transform)
        return DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    def evaluation(self):
        self.model = self.load_model()
        test_loader = self._test_loader()
        all_preds = []
        all_labels = []
        all_bleu_scores = []

        for images, labels in test_loader:
            with torch.no_grad():
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

                # Calculate BLEU scores for the predictions
                for i, pred in enumerate(preds):
                    reference = [test_loader.dataset.classes[labels[i]]]
                    candidate = [test_loader.dataset.classes[pred]]
                    all_bleu_scores.append(sentence_bleu([reference], candidate))

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        bleu_score_avg = sum(all_bleu_scores) / len(all_bleu_scores)

        self.scores = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "bleu_score_avg": bleu_score_avg
        }
        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="SignLanguageTranslatorModel")
            else:
                mlflow.pytorch.log_model(self.model, "model")