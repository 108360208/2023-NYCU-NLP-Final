import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

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
class GaussianNoise(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.fc_mu1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu3 = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, x):
        h = torch.relu(self.fc_mu1(x))
        h = torch.relu(self.fc_mu2(h))
        mu = self.fc_mu3(h)
        return mu
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加一个隐藏层
        self.fc_mu = GaussianNoise(hidden_dim, latent_dim)
        self.fc_logvar = GaussianNoise(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))  # 添加一个隐藏层，并使用激活函数ReLU
        h = torch.relu(self.fc3(h))  # 添加一个隐藏层，并使用激活函数ReLU
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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

    def forward(self, x):
        mu , logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, z
    
    

class UnetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(UnetEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        return h1, h2, h3, mu, logvar

class UnetDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(UnetDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z, enc_h1, enc_h2, enc_h3):
        h1 = torch.relu(self.fc1(z))
        h1_concat = torch.cat([h1, enc_h3], dim=-1)
        h2 = torch.relu(self.fc2(h1_concat))
        h2_concat = torch.cat([h2, enc_h2], dim=-1)
        h3 = torch.relu(self.fc3(h2_concat))
        h3_concat = torch.cat([h3, enc_h1], dim=-1)
        x_recon = self.fc4(h3_concat)
        return x_recon

class UnetVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(UnetVAE, self).__init__()
        self.encoder = UnetEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = UnetDecoder(latent_dim, hidden_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_h1, enc_h2, enc_h3, mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, enc_h1, enc_h2, enc_h3)
        return x_recon, mu, logvar
    
    
class BERT_CNN(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits