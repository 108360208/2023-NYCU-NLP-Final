import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel
from utils.dataloader import CVATDataLoader 
from torch.utils.data import DataLoader
from model.CNN import CNN
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter


writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')
# jina = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True) 
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=False)

def train(model, train_loader, criterion, optimizer, device, pbar):
    model.train()
    running_loss = 0.0

    for inputs, targets1, targets2 in train_loader:
        inputs, valence, arousal = inputs.to(device), targets1.to(device), targets2.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
  
        loss = criterion(outputs.float().squeeze(), torch.cat([valence, arousal], dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    valence_MAE = 0.0
    arousal_MAE = 0.0
    valence_pearson = 0.0
    arousal_pearson = 0.0
    with torch.no_grad():
        for inputs, targets1, targets2 in val_loader:
            inputs, valence, arousal = inputs.to(device), targets1.to(device), targets2.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.float().squeeze(), torch.cat([valence, arousal], dim=1))
            running_loss += loss.item()
            pred_valence = outputs[:, 0]
            pred_arousal = outputs[:, 1]
            valence_MAE += mean_absolute_error(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())
            arousal_MAE += mean_absolute_error(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())
            valence_pearson += pearsonr(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())[0]
            arousal_pearson += pearsonr(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())[0]
    return running_loss / len(val_loader),  [valence_MAE/ len(val_loader), arousal_MAE/ len(val_loader)], [valence_pearson/ len(val_loader), arousal_pearson/ len(val_loader)]

folder_path = 'dataset'
dataset = CVATDataLoader(folder_path, tokenizer, embedding, "train")

model = CNN(input_dim=768)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 300
valence_MAE = []
arousal_MAE = []
valence_pearson = []
arousal_pearson = []

indices = list(range(len(dataset)))

train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

with tqdm(total=(num_epochs), desc="Training progress") as pbar:
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, pbar)
        writer.add_scalar('Train Loss', train_loss, epoch)
        if((epoch % 20) == 0 or epoch == num_epochs - 1):
            val_loss , MAE, PCC= evaluate(model, val_loader, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Valence MAE', MAE[0], epoch)
            writer.add_scalar('Arousal MAE', MAE[1], epoch)
            writer.add_scalar('Valence PCC', PCC[0], epoch)
            writer.add_scalar('Arousal PCC', PCC[1], epoch)
        if(epoch == num_epochs - 1 ):
            valence_MAE.append(MAE[0])
            arousal_MAE.append(MAE[1])
            valence_pearson.append(PCC[0])
            arousal_pearson.append(PCC[1])
        pbar.set_postfix({'train_loss': train_loss})
        pbar.update(1)
# for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
#     train_dataset = torch.utils.data.Subset(dataset, train_index)
#     val_dataset = torch.utils.data.Subset(dataset, val_index)
#     dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     dataloader_val = DataLoader(val_dataset, batch_size=512, shuffle=False)
#     with tqdm(total=(num_epochs), desc="Training progress") as pbar:
#         for epoch in range(num_epochs):
#             train_loss = train(model, dataloader_train, criterion, optimizer, device, pbar)
       
#             if((epoch % 20) == 0 or epoch == num_epochs - 1):
#                 val_loss , MAE, PCC= evaluate(model, dataloader_val, criterion)
#                 print(f"Fold [{fold+1}/{k_folds}], Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")

#             if(epoch == num_epochs - 1 ):
#                 valence_MAE.append(MAE[0])
#                 arousal_MAE.append(MAE[1])
#                 valence_pearson.append(PCC[0])
#                 arousal_pearson.append(PCC[1])
#             pbar.set_postfix({'train_loss': train_loss})
#             pbar.update(1)
writer.close()
print("Valence MAE: ", np.mean(valence_MAE))
print("Arousal MAE: ", np.mean(arousal_MAE))
print("Valence Pearson: ", np.mean(valence_pearson))
print("Arousal Pearson: ", np.mean(arousal_pearson))
torch.save(model.state_dict(), "sentiment_analysis_model.pth")
