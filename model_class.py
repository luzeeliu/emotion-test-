import torch
import torch.nn as nn
import math

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hid, num_out, batch_size=1, num_layers=1):
        #vocab_size is the total number of words in vocabulery
        #embed_dim is the size of each word vector
        #num_hid is the hidden layer size
        #the number of emotion classes
        #batch_size and num_layer will include for future scalability
        # inherit the model init from nn.module
        super().__init__()
        #initial the value in init
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.num_layer = num_layers
        
        #embedding the layer, convert a sequence of word induces into vectors
        #Input shape: (batch_size, seq_len) → Output: (batch_size, seq_len, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        #changer the parameter to configous LSTM
        #W is the input -to-hidden weight
        #U is the weight between hidden layer
        #hid_bias is bias for 4 LSTM gates
        self.W = nn.Parameter(torch.Tensor(embed_dim, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        
        #set output layer
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        
        self.init_weight()
        
        #after set the parameter of each layer, initial the weight
    def init_weight(self):
        stdv = 1.0/math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
    
    #define the forward function to puch forward
    def forward(self, x, init_state = None):
        #(bartch, seq_len, embed_dim)
        x = self.embedding(x)
        batch_size, seq_size, _= x.size()
        hidden_seq = []
        if init_state is None:
            #h_t is hidden state, adn c_ t is cell state
            
            h_t = torch.zeros(batch_size, self.num_hid).to(x.device)
            c_t = torch.zeros(batch_size, self.num_hid).to(x.device)
        else:
            h_t, c_t = init_state
        
        NH = self.num_hid
        #go over each word in the sequence
        for t in range(seq_size):
            #input vector at time step t :measns all bath, time step t and all feature in the embedding vector
            x_t = x[:, t, :]
            #calculate all 4 gates at once
            #LSTM have 4 gate use the weight between each layer and bias to calculate the gates
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            #input gate, forget gate, cell candidate and output gate
            i_t, f_t, g_t, o_t = (
                #use sigmoid to make partition keep (0,1) and use tanh to keep (-1,1)
                #c_t = f_t * c_t + i_t * g_t  # updated cell state
                #h_t = o_t * torch.tanh(c_t)  # updated hidden state (output)
                torch.sigmoid(gates[:,:NH]), #input gate
                torch.sigmoid(gates[:, NH:NH * 2]), #forget gate
                torch.tanh(gates[:, NH *2:NH *3]),  #cell candidate
                torch.sigmoid(gates[:, NH*3:])  #output gate
            )
            #standard LSTM equations
            #c_t combines past memory and new input
            #h_t filtered memory passed to the next step
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            #add a dimension in front of tensor
            #this step will give a room to save the length of the sequence
            hidden_seq.append(h_t.unsqueeze(0))
        #stack and reorder the hidden states to make them batch-first. cat is to replace the unsequeeze room in dim = 0
        hidden_seq = torch.cat(hidden_seq, dim= 0)
        #change the place of seq_len and batch size
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        #in last timestep and all batch and all feature
        final_hidden = hidden_seq[:, -1, :] 
        output = final_hidden @ self.V + self.out_bias
        
        return output            
        