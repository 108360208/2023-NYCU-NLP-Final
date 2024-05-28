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

output_csv_path = "submission.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')

import csv

def test(model, val_loader, device):
    model.eval()
    csv_result = {}

    with torch.no_grad():
        for id, inputs in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            outputs_np = outputs.cpu().numpy()
            # print(outputs_np)
            for i in range(len(id)):
                csv_result[id[i]] = {'Valence': outputs_np[i, 0], 'Arousal': outputs_np[i, 1]}

    output_csv_path = "output.csv"
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Valence', 'Arousal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for id, values in csv_result.items():
            writer.writerow({'ID': id, 'Valence': f"{values['Valence']:.2f}", 'Arousal': f"{values['Arousal']:.2f}"})

            
folder_path = 'dataset'
dataset = CVATDataLoader(folder_path, tokenizer, embedding, "test")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
model = CNN(input_dim=768)
model.load_state_dict(torch.load('sentiment_analysis_model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test(model, dataloader, device)