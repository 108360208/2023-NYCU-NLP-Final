import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        # x = 1 + 9 * x  # 將值縮放到1到10之間
        return x
