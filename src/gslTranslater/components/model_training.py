import os
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from transformers import BertModel
from gslTranslater.constants import *
from gslTranslater.utils.common import read_yaml, create_directories
from gslTranslater.components.sign_language_translator import SignLanguageTranslator
from gslTranslater.entity.config_entity import (TrainingConfig)

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = SignLanguageTranslator(
            cnn_model=models.resnet50(pretrained=False),
            transformer_model=BertModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1'),
            tokenizer_len=None
        )
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
        self.model.train()

    def train_valid_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder(root=self.config.training_data / "Train", transform=transform)
        valid_dataset = datasets.ImageFolder(root=self.config.training_data / "Test", transform=transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        best_val_loss = float('inf')

        for epoch in range(self.config.params_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.config.params_epochs}, Loss: {running_loss / len(self.train_loader)}")

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.valid_loader)
            print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

            # Checkpoint: Save the model if it has the best validation loss so far
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving checkpoint...")
                best_val_loss = val_loss
                self.save_model(path=self.config.trained_model_path, model=self.model)

        print("Training completed. Best validation loss was:", best_val_loss)