import torch
import torch.nn as nn
import torch.nn.functional as F


class DTCNet(nn.Module):
    """A simple DTC neural network for graph finger print classification"""

    def __init__(self, num_units1=1024, num_units2=512, num_units3=256, num_units4=128, num_classes=5, dropout_level=0.2):
        super().__init__()

        self.dense0 = nn.Linear(54, num_units1)
        self.dense1 = nn.Linear(num_units1, num_units2)
        self.dense2 = nn.Linear(num_units2, num_units3)
        self.dense3 = nn.Linear(num_units3, num_units4)
        self.output = nn.Linear(num_units4, num_classes)

        self.dropout = nn.Dropout(dropout_level)

    def forward(self, X):

        X = F.relu(self.dense0(X))
        X = self.dropout(X)

        X = F.relu(self.dense1(X))
        X = self.dropout(X)

        X = F.relu(self.dense2(X))
        X = self.dropout(X)

        X = F.relu(self.dense3(X))
        X = self.dropout(X)

        X = self.output(X)

        return X
