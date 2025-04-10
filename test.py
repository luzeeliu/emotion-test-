import os
import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from model_class import LSTM_model
from data_process import processed_LSTM
from evaluate import evaluate_model


device = torch.device("cuda")

# get test data from the datasetr
test_path = os.path.join("data", "combined_testdata.csv")
test_data = pd.read_csv(test_path)
trained_path = os.path.join("net", "lstm_model.pth")

# load the vocab
with open("net/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
# load data to tensor
test_datas, vocab = processed_LSTM(test_data, vocab=vocab, max_len= 50)
# when shuffle  = True in each epoch the order of your data will randomized
test_loader_LSTM = DataLoader(test_datas, batch_size= 32, shuffle=False, num_workers= 0, pin_memory=True)

# load LSTM model
model = LSTM_model(vocab_size= len(vocab), embed_dim= 100, num_hid= 128, num_out= 6)
# load trained model
model.load_state_dict(torch.load(trained_path))
model.to(device)
model.eval()

all_preds = []
all_label = []

with torch.no_grad():
    for batch in test_loader_LSTM:
        input_ids = batch['input_ids'].to(device)
        label = batch['labels'].to(device)
        
        # get output
        output = model(input_ids)
        preds = output.argmax(1)
        
        # save predict and label
        all_preds.extend(preds.cpu().numpy())
        all_label.extend(label.cpu().numpy())

evaluate_model(all_preds, all_label, "LSTM")

