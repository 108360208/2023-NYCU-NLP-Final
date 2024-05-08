import os
import csv
import torch
import ast
from transformers import BertTokenizer, BertModel

class CVATDataLoader:
    def __init__(self, folder_path, tokenizer, embedding_model,train = True):
        self.folder_path = folder_path
        self.data = self.load_data(tokenizer, embedding_model,train)

    def load_data(self, tokenizer, embedding_model, train = True):
        all_files = os.listdir(self.folder_path)
        csv_files = ["CVAT_5_SD.csv"]
        if(train):
            csv_files = ["train.csv"]
        else:
            csv_files = ["test.csv"]
        data = []
        for file_name in csv_files:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                print("Start processing file: ", file_name)
                rows = []
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
        embedding, valence, arousal = self.data[index]
        return embedding, valence, arousal
