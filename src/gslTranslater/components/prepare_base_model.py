import torch
from torch import nn
from torchvision import models
from transformers import BertTokenizer, BertModel
from gslTranslater.constants import *
from gslTranslater.entity.config_entity import PrepareBaseModelConfig
from gslTranslater.components.sign_language_translator import SignLanguageTranslator

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

        self.full_model = SignLanguageTranslator(self.cnn_model, self.transformer_model, tokenizer_len=len(self.tokenizer))
        self.print_model_summary(self.full_model)
        self.save_model(self.full_model, self.config.updated_model_path)
    
    @staticmethod
    def save_model(model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)
        
    @staticmethod
    def print_model_summary(model: nn.Module):
        print(model)
