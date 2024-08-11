import torch
from torch import nn
from torchvision import models
from transformers import BertTokenizer, BertModel
from gslTranslater.constants import *
from gslTranslater.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_cnn_model(self):
        self.cnn_model = models.resnet50(pretrained=True)
        self.cnn_model = nn.Sequential(*list(self.cnn_model.children())[:-1])
        torch.save(self.cnn_model.state_dict(), self.config.cnn_model_path)

    def get_transformer_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self.transformer_model = BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        torch.save(self.transformer_model.state_dict(), self.config.transformer_model_path)
        self.tokenizer.save_pretrained(self.config.tokenizer_path)

    def update_base_model(self):
        # Combine CNN and Transformer models
        self.cnn_model.load_state_dict(torch.load(self.config.cnn_model_path))
        self.transformer_model.load_state_dict(torch.load(self.config.transformer_model_path))

        # Define the full model combining both models
        class SignLanguageTranslator(nn.Module):
            def __init__(self, cnn_model, transformer_model):
                super(SignLanguageTranslator, self).__init__()
                self.cnn_model = cnn_model
                self.fc = nn.Linear(2048, 512)
                self.transformer_model = transformer_model
                self.classifier = nn.Linear(512 + transformer_model.config.hidden_size, len(self.tokenizer))

            def forward(self, features, input_ids, attention_mask):
                features = self.cnn_model(features)
                features = features.view(features.size(0), -1)
                features = torch.relu(self.fc(features))

                bert_outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                bert_cls = bert_outputs.last_hidden_state[:, 0, :]

                combined = torch.cat((features, bert_cls), dim=1)
                outputs = self.classifier(combined)
                return outputs

        self.full_model = SignLanguageTranslator(self.cnn_model, self.transformer_model)
        self.print_model_summary(self.full_model)
        self.save_model(self.full_model, self.config.updated_model_path)
    
    @staticmethod
    def save_model(model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)
        
    @staticmethod
    def print_model_summary(model: nn.Module):
        print(model)
