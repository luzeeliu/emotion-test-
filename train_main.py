import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from torch.optim import Adam
from model_class import LSTM_model, BERT_model
from torchvision import datasets
from torch.utils.data import DataLoader
from data_process import processed_BERT, processed_LSTM


path = os.path.join("data", "combined_shuffled.csv")
file = pd.read_csv(path)

def train(model, device, train_loader, optimizer,citerion, epoch):
    #train model
    model.train()
    for i in range(epoch):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = citerion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {i+1}, Loss: {total_loss:.4f}")
    
def test():
    #test model
    return True

def main():
    #main train function first need load the load data and process it 
    #second need load train manipulate
    global file
    #dara process
    #bert
    file_bert = processed_BERT(file) 
    train_loader_BERT = DataLoader(file_bert, batch_size = 16, shuffle = True)
    #LSTM
    train_dataset_LSTM = processed_LSTM(file, max_lan = 50)
    train_loader_LSTM = DataLoader(train_dataset_LSTM, batch_size=16, shuffle=True)
    
    #device use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model select
    model1 = LSTM_model().to(device)
    model2 = BERT_model().to(device)
    
    #optimizer select
    optimizer = Adam(model2.parameters(), lr = 2e-5)
    
    #epoch set
    epoch = 10000
    
   
    #loss function
    citerion = nn.CrossEntropyLoss()
    
    #trainning
    train(model2, device, train_loader_BERT, optimizer,citerion, epoch)

if __name__ == '__main__':
    main()