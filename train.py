import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from utils.dataloader import CVATDataLoader 
from torch.utils.data import DataLoader
from model.CNN import CNN
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    valence_MAE = 0.0
    arousal_MAE = 0.0
    valence_pearson = 0.0
    arousal_pearson = 0.0
    for inputs, targets1, targets2 in train_loader:
        inputs, valence, arousal = inputs.to(device), targets1.to(device), targets2.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        pred_valence = outputs[:, 0]
        pred_arousal = outputs[:, 1]
        loss = criterion(outputs.float().squeeze(), torch.cat([valence, arousal], dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        valence_MAE += mean_absolute_error(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())
        arousal_MAE += mean_absolute_error(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())
        valence_pearson += pearsonr(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())[0]
        arousal_pearson += pearsonr(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())[0]
    return running_loss / len(train_loader), [valence_MAE/ len(train_loader), arousal_MAE/ len(train_loader)], [valence_pearson/ len(train_loader), arousal_pearson/ len(train_loader)]

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets1, targets2 in val_loader:
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.float().squeeze(), torch.cat([targets1, targets2], dim=1))
            running_loss += loss.item()
    return running_loss / len(val_loader)
folder_path = 'dataset'
dataset = CVATDataLoader(folder_path, tokenizer, embedding, "train")

model = CNN(input_dim=768)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 300

with tqdm(enumerate(kf.split(dataset)), total=k_folds, desc="Folds") as t:
    for fold, (train_index, val_index) in t:
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
        for epoch in range(num_epochs):
            train_loss, MAE, PCC = train(model, dataloader_train, criterion, optimizer, device)
        t.write(f"Fold [{fold+1}/{k_folds}], Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")
        val_loss = evaluate(model, dataloader_val, criterion)
        t.write(f"Fold [{fold+1}/{k_folds}], Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")


torch.save(model.state_dict(), "sentiment_analysis_model.pth")
