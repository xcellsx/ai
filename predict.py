import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class RobertaCNNClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_filters=128, kernel_size=3, dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size  # usually 768

        self.conv1d = nn.Conv1d(in_channels=self.hidden_size,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                padding=1)  # keep same length

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)  # output shape: (B, num_filters, 1)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters, 3)  # 3 class classification

    def forward(self, input_ids, attention_mask):
        # Encode with RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, T, H)

        x = last_hidden_state.permute(0, 2, 1)  # (B, H, T)
        x = self.conv1d(x)                      # (B, num_filters, T)
        x = self.relu(x)
        x = self.pool(x)                        # (B, num_filters, 1)
        x = x.squeeze(2)                        # (B, num_filters)

        x = self.dropout(x)
        logits = self.fc(x)                     # (B, 1)
        return logits

def predict_depression_multiclass(text, model, tokenizer, device='cpu', max_len=256):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)  # shape: (1, 3)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.squeeze().tolist()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    saved_model = RobertaCNNClassifier()
    saved_model.load_state_dict(torch.load('roberta_cnn_nf128_ks3_do40_lr1e-05_.pt'))
    saved_model.to(device=device)
    text = "I feel nothing about this life, there is nothing worth living for, everyday is just a bleak miserable existence"
    label, prob = predict_depression_multiclass(text, saved_model, roberta_tokenizer, device=device)

    print(f"Prediction: {label} (probability of depression = {prob})")