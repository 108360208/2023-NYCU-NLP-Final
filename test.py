import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from utils.dataloader import CVATDataLoader 
from torch.utils.data import DataLoader
from model.CNN import CNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')
text = "正面對撞結果現代車引擎蓋只凹一半，駕駛幾乎沒受傷vios好像連駕駛艙都完全潰縮駕駛就夾死在裡面你看看你看看假如低於標準的安全性，還可以販售有可能就是一個車廠是否尊重生命的表現"
tokenized_text = tokenizer.encode(text, add_special_tokens=True)
print(len(tokenized_text))
# print(tokenizer.decode(tokenized_text[0]))
out = embedding(torch.tensor([tokenized_text]))
print(out[0].shape)
print(out[0].mean(dim=1).squeeze().shape)
print(out)