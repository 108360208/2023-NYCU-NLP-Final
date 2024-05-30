import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)  # 添加 Dropout，丢弃概率为 0.5
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(2048, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)  # 在第一个全连接层后应用 Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)  # 在第二个全连接层后应用 Dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)  # 在第三个全连接层后应用 Dropout
        x = self.fc4(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))  # 添加一个隐藏层，并使用激活函数ReLU
        h = torch.relu(self.fc3(h))  # 添加一个隐藏层，并使用激活函数ReLU
        mu = self.fc_mu(h)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))  # 添加一个隐藏层，并使用激活函数ReLU
        h = torch.relu(self.fc3(h))  # 添加一个隐藏层，并使用激活函数ReLU
        x_recon = self.fc4(h)
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, logvar):
        mu = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, z
    
    
