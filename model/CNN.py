import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) 
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x