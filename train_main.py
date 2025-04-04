import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from torch.optim import Adam
from model_class import LSTM_model
from torchvision import datasets
from torch.utils.data import DataLoader
from data_process import processed_LSTM


path = os.path.join("data", "combined_shuffled.csv")
path_save = os.path.join("net", "lstm_model.pth")
file = pd.read_csv(path)

def LSTM_train(model, device, train_loader, optimizer,citerion, epoch):
    #train model
    #let model training 
    model.train()
    
    #set the epoch
    for i in range(epoch):
        #recode the total loss
        total_loss = 0
        correct = 0
        total = 0
        #get data in loader like{input_ids,labels}
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids)
            loss = citerion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #evaluate
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"Epoch {i+1}, Loss: {total_loss:.4f}, average loss:{avg_loss:.4f}, accuracy:{acc:.2%}")
        
    torch.save(model.state_dict(), path_save)
    print("model is saved")
    
def test():
    #test model
    return True

def main():
    #main train function first need load the load data and process it 
    #second need load train manipulate
    #dara process
    #bert
    #file_bert = processed_BERT(file) 
    #train_loader_BERT = DataLoader(file_bert, batch_size = 16, shuffle = True)
    #LSTM
    train_dataset_LSTM, n = processed_LSTM(file, max_len = 50)
    train_loader_LSTM = DataLoader(train_dataset_LSTM, batch_size=16, shuffle=True)
    
    #device use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model select
    model1 = LSTM_model(vocab_size = n, embed_dim= 100, num_hid= 128, num_out= 6)
    """
    model1.load_state_dict(torch.load(path_save))
    model1.to(device)
    model1.eval()
    """
    #model2 = BERT_model().to(device)
    
    #optimizer select
    optimizer = Adam(model1.parameters(), lr = 2e-5)
    
    #epoch set
    epoch = 5
    
   
    #loss function
    citerion = nn.CrossEntropyLoss()
    
    #trainning
    LSTM_train(model1, device, train_loader_LSTM, optimizer,citerion, epoch)

if __name__ == '__main__':
    main()