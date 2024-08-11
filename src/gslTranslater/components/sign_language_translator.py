import torch
from torch import nn
from torchvision import models
from transformers import BertModel

class SignLanguageTranslator(nn.Module):
    def __init__(self, cnn_model=None, transformer_model=None, tokenizer_len=None):
        super(SignLanguageTranslator, self).__init__()
        self.cnn_model = cnn_model if cnn_model else models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, 512)
        self.transformer_model = transformer_model if transformer_model else BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self.classifier = nn.Linear(512 + self.transformer_model.config.hidden_size, tokenizer_len)

    def forward(self, features, input_ids, attention_mask):
        features = self.cnn_model(features)
        features = features.view(features.size(0), -1)
        features = torch.relu(self.fc(features))

        bert_outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]

        combined = torch.cat((features, bert_cls), dim=1)
        outputs = self.classifier(combined)
        return outputs
