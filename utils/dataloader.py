import os
import csv
import torch
import ast
from transformers import BertTokenizer, BertModel
import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

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
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-chinese', aug_p=0.2, top_k = 50, device='cuda:0')
            
        if "arg_trainff.csv" in all_files:    
            csv_files = ["train_bert.csv"]
        else:
            csv_files = ['CVAT_1_SD.csv','CVAT_2_SD.csv','CVAT_3_SD.csv','CVAT_4_SD.csv']
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
                        valence_mean = torch.tensor([float(row['Valence_Mean'])], dtype=torch.float32)
                        arousal_mean = torch.tensor([float(row['Arousal_Mean'])], dtype=torch.float32)
                        valence_sd = torch.tensor([float(row['Valence_SD'])], dtype=torch.float32)
                        arousal_sd = torch.tensor([float(row['Arousal_SD'])], dtype=torch.float32)
                        data.append((embedding, valence_mean, arousal_mean, valence_sd, arousal_sd))
                        augmented_text = aug.augment(row["Text"], n=2)
                        
                        for i in augmented_text:
                            tokenized_text = tokenizer.encode(i.replace(" ", ""), add_special_tokens=True)
                            input_ids = torch.tensor([tokenized_text])
                            with torch.no_grad():
                                outputs = embedding_model(input_ids)
                                embedding = outputs[0].mean(dim=1).squeeze()
              
                            row['Embedding'] = embedding.numpy().tolist()
                            rows.append(row)
                            data.append((embedding, valence_mean, arousal_mean, valence_sd, arousal_sd))
                        
                        continue

                    text = row['Text']
                    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
                    input_ids = torch.tensor([tokenized_text])
                    with torch.no_grad():
                        outputs = embedding_model(input_ids)
                        embedding = outputs[0].mean(dim=1).squeeze()
                    # embedding = tokenizer.encode(row["Text"])
                    # embedding = torch.tensor([embedding])
                    valence_mean = torch.tensor([float(row['Valence_Mean'])], dtype=torch.float32)
                    arousal_mean = torch.tensor([float(row['Arousal_Mean'])], dtype=torch.float32)
                    valence_sd = torch.tensor([float(row['Valence_SD'])], dtype=torch.float32)
                    arousal_sd = torch.tensor([float(row['Arousal_SD'])], dtype=torch.float32)
                    row['Embedding'] = embedding.numpy().tolist()
                    rows.append(row)
                    data.append((embedding, valence_mean, arousal_mean, valence_sd, arousal_sd))
                    
                    augmented_text = aug.augment(row["Text"], n=3)
                        
                    for i in augmented_text:
                        row = row.copy()
                        row['Text'] = i.replace(" ", "")
                        tokenized_text = tokenizer.encode(i.replace(" ", ""), add_special_tokens=True)
                        input_ids = torch.tensor([tokenized_text])
                        with torch.no_grad():
                            outputs = embedding_model(input_ids)
                            embedding = outputs[0].mean(dim=1).squeeze()
            
                        row['Embedding'] = embedding.numpy().tolist()
                        rows.append(row)
                        data.append((embedding, valence_mean, arousal_mean, valence_sd, arousal_sd))
        if(len(rows) > 0):
            with open("dataset/arg_train.csv", 'w', newline='', encoding='utf-8') as new_file:
                writer = csv.DictWriter(new_file, fieldnames=rows[0].keys(), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if(self.mode == "train"):
            embedding, valence_mean, arousal_mean, valence_sd, arousal_sd = self.data[index]
            return embedding, valence_mean, arousal_mean, valence_sd, arousal_sd 
        elif self.mode == "test":
            id, embedding = self.data[index]
            return id, embedding
