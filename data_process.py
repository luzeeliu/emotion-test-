import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import re
from collections import Counter

path = os.path.join("data", "combined_shuffled.csv")
file = pd.read_csv(path)

hashmap = {
    "0":"sadness",
    "1":"joy",
    "2":"love",
    "3":"anger",
    "4":"fear",
    "5":"surprise"
}

#print(file.head())


###############BERT#################
#load the  protrain model 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TweetDataset_BERT(Dataset):
    #data process in BERT model
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#################LSTM#################
def tokenize(text):
    # Simple tokenizer: lowercase, remove non-letters, split by space
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

class LSTMDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=50):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = torch.tensor(encode(text, self.vocab, self.max_len), dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def processed_LSTM(df, max_len=50):
    vocab = build_vocab(df['text'])
    return LSTMDataset(df, vocab, max_len)

def processed_BERT(df):
    return TweetDataset_BERT(df, tokenizer)



