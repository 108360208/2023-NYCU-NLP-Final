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
        if(self.mode == "train" or self.mode == "val"):
            self.data = self.load_train_data(tokenizer, embedding_model, mode)
        elif self.mode == "test":
            self.data = self.load_test_data(tokenizer, embedding_model)
            
    def load_test_data(self, tokenizer, embedding_model):
        data = []
        all_files = os.listdir(self.folder_path)
        
        file_name = "test.csv"

        file_path = os.path.join(self.folder_path, file_name)
        print("Start processing file: ", file_name)

        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
        if 'Embedding' not in df.columns:
            df['Embedding'] = None
        for index, row in df.iterrows():
            # print(index, row['ID'])
            text = row['Text']
            tokenized_text = tokenizer(text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
            data.append((row['ID'], tokenized_text))

        # df.to_csv("dataset/pre_test.csv", sep='\t', index=False, encoding='utf-8')
        return data
    
    def load_train_data(self, tokenizer, embedding_model,mode):
        all_files = os.listdir(self.folder_path)
        if "train_jina.csv" in all_files and mode == "train":    
            csv_files = ["train.csv"]
        elif "bert_val_train.csv" in all_files and mode == "val":
            csv_files = ["bert_val_train.csv"]
        else:
            csv_files = ['CVAT_3_SD.csv', 'CVAT_2_SD.csv', 'CVAT_1_SD.csv',  'CVAT_5_SD.csv', 'CVAT_4_SD.csv']
        print(all_files)
        data = []
        rows = []
        str_len = []
        
        for file_name in csv_files:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                print("Start processing file: ", file_name)
                
                for row in reader:
          
                    text = row['Text']
                    str_len.append(len(text))
                    tokenized_text = tokenizer(text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
    
                    valence_mean = torch.tensor([float(row['Valence_Mean'])], dtype=torch.float32)
                    arousal_mean = torch.tensor([float(row['Arousal_Mean'])], dtype=torch.float32)
                    valence_sd = torch.tensor([float(row['Valence_SD'])], dtype=torch.float32)
                    arousal_sd = torch.tensor([float(row['Arousal_SD'])], dtype=torch.float32)

                    data.append((tokenized_text, valence_mean, arousal_mean, valence_sd, arousal_sd))
        print("max len of text: ", max(str_len))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if(self.mode == "train" or self.mode == "val"):
            embedding, valence_mean, arousal_mean, valence_sd, arousal_sd = self.data[index]
            return embedding, valence_mean, arousal_mean, valence_sd, arousal_sd 
        elif self.mode == "test":
            id, embedding = self.data[index]
            return id, embedding
