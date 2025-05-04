# classifier.py
import torch.nn as nn

class Classifier(nn.Module):
    """A simple 2-class linear classifier head."""
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.fc(x)

