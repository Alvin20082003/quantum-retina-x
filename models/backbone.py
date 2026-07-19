import torch.nn as nn
from transformers import ViTModel

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, x, return_features=False):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(cls)
        if return_features:
            return logits, cls
        return logits
