
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
from sentence_transformers import SentenceTransformer
text = '執政者無能，人民十分火大。'
# aug = naw.ContextualWordEmbsAug(model_path='bert-base-chinese', aug_p=0.2, top_k = 50)
# # text = 'Un rápido zorro marrón salta sobre el perro perezoso'
# augmented_text = aug.augment(text, n=1)
# print("Original:")
# print(text)
# print("Augmented Text:")
# #delete space between the words


# print((augmented_text[0].replace(" ", "")))

# Use BERT to get sentence embedding
model = SentenceTransformer('bert-base-chinese')
sentence_embedding = model.encode('執政者無能，人民十分火大。')
print("Sentence Embedding:")
print(sentence_embedding)
