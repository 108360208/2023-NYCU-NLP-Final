import os
import csv
import torch
import ast
from transformers import BertTokenizer, BertModel
import pandas as pd
class CVATDataLoader:
    def __init__(self, folder_path, tokenizer, embedding_model, mode):
        self.folder_path = folder_path
        self.mode = mode
        if(self.mode == "train"):
            self.data = self.load_train_data(tokenizer, embedding_model)
        elif self.mode == "test":
            self.data = self.load_test_data(tokenizer, embedding_model)
            
    def load_test_data(self, tokenizer, embedding_model):
        data = []
        all_files = os.listdir(self.folder_path)
        if("pre_test.csv" in all_files):
            file_name = "pre_test.csv"
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                print("Start processing file: ", file_name)
                
                for row in reader:
                    if 'Embedding' in row:
                        embedding = ast.literal_eval(row['Embedding'])
                        embedding = torch.tensor([embedding], dtype=torch.float32)
                        data.append((row['ID'],embedding))
                        continue
            return data
        else:
            file_name = "test.csv"

        file_path = os.path.join(self.folder_path, file_name)
        print("Start processing file: ", file_name)

        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
        if 'Embedding' not in df.columns:
            df['Embedding'] = None
        for index, row in df.iterrows():
            print(index, row['ID'])
            text = row['Text']
            # print(text)
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor([tokenized_text])
            
            with torch.no_grad():
                outputs = embedding_model(input_ids)
                embedding = outputs[0].mean(dim=1).squeeze()
            df.at[index, 'Embedding'] = embedding.numpy().tolist()
            data.append((row['ID'], embedding))

        df.to_csv("dataset/pre_test.csv", sep='\t', index=False, encoding='utf-8')

        return data
    
    def load_train_data(self, tokenizer, embedding_model):
        all_files = os.listdir(self.folder_path)
        if "train.csv" in all_files:    
            csv_files = ["train.csv"]
        else:
            csv_files = all_files
        print(all_files)
        data = []
        rows = []
        for file_name in csv_files:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                print("Start processing file: ", file_name)
                
                for row in reader:
                    if 'Embedding' in row:
                        embedding = ast.literal_eval(row['Embedding'])
                        embedding = torch.tensor([embedding], dtype=torch.float32)
                        valence = torch.tensor([float(row['Valence_Mean'])], dtype=torch.float32)
                        arousal = torch.tensor([float(row['Arousal_Mean'])], dtype=torch.float32)
                        data.append((embedding, valence, arousal))
                        continue

                    text = row['Text']
                    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
                    input_ids = torch.tensor([tokenized_text])
                    with torch.no_grad():
                        outputs = embedding_model(input_ids)
                        embedding = outputs[0].mean(dim=1).squeeze()
                    valence_mean = torch.tensor([float(row['Valence_Mean'])], dtype=torch.float32)
                    arousal_mean = torch.tensor([float(row['Arousal_Mean'])], dtype=torch.float32)
                    row['Embedding'] = embedding.numpy().tolist()
                    rows.append(row)
                    data.append((embedding, valence_mean, arousal_mean))
        if(len(rows) > 0):
            with open("dataset/train.csv", 'w', newline='', encoding='utf-8') as new_file:
                writer = csv.DictWriter(new_file, fieldnames=rows[0].keys(), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if(self.mode == "train"):
            embedding, valence, arousal = self.data[index]
            return embedding, valence, arousal
        elif self.mode == "test":
            id, embedding = self.data[index]
            return id, embedding
