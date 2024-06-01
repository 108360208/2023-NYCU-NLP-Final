
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import BertTokenizer, BertModel, AutoModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
embedding = BertModel.from_pretrained('bert-base-chinese')
model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-zh", # switch to en/zh for English or Chinese
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    '我很高興你來參加我的生日派對。',
    '你到底在搞什麼，什麼都做不好'
])
print(cos_sim(embeddings[0], embeddings[1]))
import torch
totokenized_text = tokenizer.encode('我很高興你來參加我的生日派對。', add_special_tokens=True)
totokenized_text2 = tokenizer.encode('你到底在搞什麼，什麼都做不好', add_special_tokens=True)
input_ids = torch.tensor([totokenized_text])
input_ids2 = torch.tensor([totokenized_text2])
with torch.no_grad():
    outputs = embedding(input_ids)
    outputs2 = embedding(input_ids2)
    embedding = outputs[0].mean(dim=1).squeeze()
    embedding2 = outputs2[0].mean(dim=1).squeeze()
print(cos_sim(embedding, embedding2))

