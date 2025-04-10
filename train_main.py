import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import pickle
from torch.optim import Adam
from model_class import LSTM_model
from torchvision import datasets
from torch.utils.data import DataLoader
from data_process import processed_LSTM
from evaluate import evaluate_model


path_train = os.path.join("data", "combined_shuffled_8000.csv")
path_validation = os.path.join("data", "combined_shuffled_2000.csv")
path_save = os.path.join("net", "lstm_model.pth")
train_data = pd.read_csv(path_train)
v_data = pd.read_csv(path_validation)


def LSTM_train(model, device, train_loader, validation_loader_LSTM, optimizer,criterion, epoch):
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
            loss = criterion(outputs, labels)
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
        #test overfitting
        val_loss, val_acc,_,_ = validation(model, device, validation_loader_LSTM, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2%}")

    torch.save(model.state_dict(), path_save)
    print("model is saved")

#validation function to test overfitting
def validation(model, device, val_loader, criterion):
    # base attribute for accuracy and confusion matrix
    model.eval()  # Turn off dropout, etc.
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # set gradient not change
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            # in validation it dont need optimizer and backword
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

    

def main():
    #main train function first need load the load data and process it 
    #second need load train manipulate
    #dara process
    #bert
    #file_bert = processed_BERT(file) 
    #train_loader_BERT = DataLoader(file_bert, batch_size = 16, shuffle = True)
    #LSTM
    train_dataset_LSTM, vocab = processed_LSTM(train_data, max_len = 50)
    train_loader_LSTM = DataLoader(train_dataset_LSTM, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    n = len(vocab)
    # save vocab to pickel use it in test time
    with open("net/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    #load validation data
    validation_dataset_LSTM, _ =processed_LSTM(v_data,vocab=vocab, max_len= 50)
    validation_loader_LSTM = DataLoader(validation_dataset_LSTM, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    
    #device use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model select
    model1 = LSTM_model(vocab_size = n, embed_dim= 100, num_hid= 128, num_out= 6)
    #loade pretrain model

    model1.load_state_dict(torch.load(path_save, weights_only= True))
    model1.to(device)
    model1.eval()

    #model2 = BERT_model().to(device)
    
    #optimizer select
    optimizer = Adam(model1.parameters(), lr = 1e-3)
    
    #epoch set
    epoch = 5
    
   
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #trainning
    LSTM_train(model1, device, train_loader_LSTM,validation_loader_LSTM, optimizer,criterion, epoch)
    _,_, y_pred, y_true = validation(model1, device, validation_loader_LSTM, criterion)
    evaluate_model(y_pred, y_true, "LSTM")

if __name__ == '__main__':
    main()