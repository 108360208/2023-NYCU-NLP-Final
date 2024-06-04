import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel
from utils.dataloader import CVATDataLoader 
from torch.utils.data import DataLoader
from model.CNN import CNN, VAE, UnetVAE
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.utils_loss import PearsonLoss, Reconstloss, KL_Loss


writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')
jina = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True) 
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    
def train(model, train_loader, scheduler, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    reconloss = Reconstloss()
    KLLoss = KL_Loss()
    for inputs, valence_mean, arousal_mean, valence_sd, arousal_ad in train_loader:
        inputs, valence, arousal = inputs.to(device), valence_mean.to(device), arousal_mean.to(device)
        optimizer.zero_grad()
        valence_sd = valence_sd.to(device)
        arousal_ad = arousal_ad.to(device)
        std = torch.cat([valence_sd, arousal_ad], dim=1)

        if (epoch <= -1):
            enc_h1, enc_h2, enc_h3, _, _ = model.encoder(inputs)
            latent_mean = torch.cat([valence, arousal], dim=1)
            logvar = torch.cat([valence_sd, arousal_ad], dim=1).unsqueeze(1)
            # _, logvar = model.encoder(inputs)
            latent_mean = latent_mean.unsqueeze(1)
        else:
            enc_h1, enc_h2, enc_h3, latent_mean, logvar = model.encoder(inputs)
        z = model.reparameterize(latent_mean, logvar)
        embedding_recon =  model.decoder(z, enc_h1, enc_h2, enc_h3)

        # embedding_recon, latent_mean, _ = model(inputs, std.unsqueeze(1))
        # loss = criterion(outputs.float().squeeze(), torch.cat([valence, arousal], dim=1))
        # print(embedding_recon.size(), inputs.size(), latent_mean.size(), valence.size(), arousal.size())
        recon_loss = reconloss(embedding_recon.float(), inputs, latent_mean, valence, arousal)
        
        kl_loss = KLLoss(latent_mean.squeeze(), logvar.squeeze(), torch.cat([valence, arousal], dim=1) , std, inputs.size(0))
        # print(recon_loss, kl_loss)  
        loss = recon_loss + 1.5 * kl_loss 
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if((epoch+1) % 5 == 0):
        scheduler.step()
    return running_loss / len(train_loader)

def evaluate(model, val_loader):
    model.eval()
    running_loss = 0.0
    valence_MAE = 0.0
    arousal_MAE = 0.0
    valence_pearson = 0.0
    arousal_pearson = 0.0
    with torch.no_grad():
        for inputs, valence_mean, arousal_mean, _ ,_ in val_loader:
            inputs, valence, arousal = inputs.to(device), valence_mean.to(device), arousal_mean.to(device)
            optimizer.zero_grad()
     
            std = torch.cat([valence, arousal], dim=1)
            embedding_recon, latent_mean, _ = model(inputs)
            # loss = criterion(outputs.float().squeeze(), torch.cat([valence, arousal], dim=1))
            running_loss += 0
            latent_mean = latent_mean.squeeze()
            pred_valence = latent_mean[:, 0]
            pred_arousal = latent_mean[:, 1]
            valence_MAE += mean_absolute_error(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())
            arousal_MAE += mean_absolute_error(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())
            valence_pearson += pearsonr(valence.float().squeeze().cpu().detach().numpy(), pred_valence.cpu().detach().numpy())[0]
            arousal_pearson += pearsonr(arousal.float().squeeze().cpu().detach().numpy(), pred_arousal.cpu().detach().numpy())[0]
    return running_loss / len(val_loader),  [valence_MAE/ len(val_loader), arousal_MAE/ len(val_loader)], [valence_pearson/ len(val_loader), arousal_pearson/ len(val_loader)]

folder_path = 'dataset'
dataset = CVATDataLoader(folder_path, tokenizer, embedding, "train")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 300
valence_MAE = []
arousal_MAE = []
valence_pearson = []
arousal_pearson = []

indices = list(range(len(dataset)))

# train_indices, test_indices = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)
# original_train_dataset = torch.utils.data.Subset(dataset, train_indices)
# test_dataset = torch.utils.data.Subset(dataset, test_indices)   
# test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=512, shuffle=False)
    model = UnetVAE(input_dim=768, hidden_dim=512, latent_dim = 2, output_dim = 768)
    model.apply(reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    model.to(device)
    with tqdm(total=(num_epochs), desc="Training progress") as pbar:
        for epoch in range(num_epochs):
            train_loss = train(model, dataloader_train, scheduler, optimizer, device, epoch)
            writer.add_scalar('Train Loss', train_loss, epoch)
            if(((epoch + 1) % 20) == 0 or epoch == (num_epochs - 1)):
                val_loss , MAE, PCC= evaluate(model, dataloader_val)
                print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")
                writer.add_scalar('Validation Loss', val_loss, epoch)
                writer.add_scalar('Valence MAE', MAE[0], epoch)
                writer.add_scalar('Arousal MAE', MAE[1], epoch)
                writer.add_scalar('Valence PCC', PCC[0], epoch)
                writer.add_scalar('Arousal PCC', PCC[1], epoch)
            if(epoch == (num_epochs - 1)):
                valence_MAE.append(MAE[0])
                arousal_MAE.append(MAE[1])
                valence_pearson.append(PCC[0])
                arousal_pearson.append(PCC[1])
                
            if epoch % 40 == 0:
                torch.save(model.state_dict(), f"sentiment_{fold+1}_fold_{epoch}.pth")
            pbar.set_postfix({'train_loss': train_loss})
            pbar.update(1)
            
        # val_loss , MAE, PCC= evaluate(model, test_loader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {val_loss:.4f}, Valence MAE: {MAE[0]:.4f}, Arousal MAE: {MAE[1]:.4f}, Valence PCC: {PCC[0]:.4f}, Arousal PCC: {PCC[1]:.4f}")
        # writer.add_scalar('Validation Loss', val_loss, epoch)
        # writer.add_scalar('Valence MAE', MAE[0], epoch)
        # writer.add_scalar('Arousal MAE', MAE[1], epoch)
        # writer.add_scalar('Valence PCC', PCC[0], epoch)
        # writer.add_scalar('Arousal PCC', PCC[1], epoch)


writer.close()
print("Valence MAE: ", np.mean(valence_MAE))
print("Arousal MAE: ", np.mean(arousal_MAE))
print("Valence Pearson: ", np.mean(valence_pearson))
print("Arousal Pearson: ", np.mean(arousal_pearson))
torch.save(model.state_dict(), "sentiment_analysis_model.pth")
