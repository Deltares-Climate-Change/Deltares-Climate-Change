import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import *
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

#Importing the Data
data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)

models = np.array(X.coords['model'])
stations = np.array(X.coords['station'])
variables = np.array(X.coords['var'])


#class LSTMTranslator(nn.Module):
#    def __init__(self, vars_dim, hidden_dim):
#        super(LSTMTranslator, self).__init__()
#        self.hidden_dim = hidden_dim
#        self.vars_dim = vars_dim
#
#        self.encoder = nn.LSTM(vars_dim,hidden_dim)
#        self.decoder = nn.LSTM(vars_dim,hidden_dim)
#
#
#    def forward(self, x):
#        encoded, _ = self.encoder(x)
#


#Preprocessing
model_input = models[0]
model_output = models[1]


def normalize_data(X):
    X_prime =  X.sel(station = stations[0], model = model_input, exp = 'rcp45')
    Y_prime = X.sel(station = stations[0], model = model_output, exp = 'rcp45')

    xmeans = X_prime.mean(dim = 'time')
    ymeans = Y_prime.mean(dim = 'time')
    xstds = X_prime.std(dim = 'time')
    ystds = Y_prime.std(dim = 'time')

    for i in range(len(variables)):
        X_prime[dict(var = i)] = (X_prime[dict(var = i)] - xmeans[i])/xstds[i]
        Y_prime[dict(var = i)] = (Y_prime[dict(var = i)] -ymeans[i])/ystds[i]

    print(X_prime.mean(dim= 'time'))
    print(Y_prime.mean(dim = 'time'))

    return X_prime, Y_prime, xstds, ystds, xmeans, ymeans


X_prime, Y_prime, xstds, ystds, xmeans, ymeans = normalize_data(X)




X_prime = torch.from_numpy(np.array(X_prime, dtype = 'float32')).cuda()
Y_prime = torch.from_numpy(np.array(Y_prime, dtype = 'float32')).cuda()




def seq_fetch(data, labels, treshold, seq_len):
    start_idx = np.random.randint(low = 0, high = treshold-seq_len)
    in_seq = data[start_idx:start_idx+seq_len, :].cuda()
    out_seq = labels[start_idx:start_idx+seq_len, :].cuda()
    return in_seq.unsqueeze(1), out_seq.unsqueeze(1)



class LSTMAutoencoder(nn.Module):
    def __init__(self, vars_dim, hidden_dim, input_length):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vars_dim = vars_dim
        self.input_length = input_length

        self.encoder = nn.LSTM(vars_dim,hidden_dim)
        self.decoder = nn.LSTM(hidden_dim,hidden_dim)
        self.dense = nn.Linear(hidden_dim*input_length, input_length*vars_dim)


    def forward(self, x):
        encoded, _ = self.encoder(x)
        hidden_rep = encoded[-1,:,:].repeat(x.shape[0],1,1)
        decoded, _ = self.decoder(hidden_rep)
        out = self.dense(torch.flatten(decoded))
        return out



#Training the LSTM
treshold = 1000
seq_len = 30

#lstm = nn.LSTM(len(variables), len(variables),num_layers = 2).cuda()
lstm = LSTMAutoencoder(len(variables), 10, seq_len).cuda()
print("Number of parameters " + str(sum(p.numel() for p in lstm.parameters())))
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)


#in_seq, out_seq = seq_fetch(X_prime,Y_prime, treshold, seq_len)
#encoded = lstm.encoder(in_seq)[0]
#hidden_rep = encoded[-1,:,:].repeat(target.shape[0],
#decoded, _ = lstma.decoder(hidden_rep)
#lstma.dense(torch.flatten(decoded))

writer = SummaryWriter()

for epoch in range(100000):
    #Reset Gradients
    lstm.zero_grad()

    #Fetch input-output sequence pair of desired length
    in_seq, out_seq = seq_fetch(X_prime,Y_prime, treshold, seq_len)

    #Predict and compute loss
    predictions = lstm(in_seq)
    loss = F.mse_loss(predictions.view(seq_len,1,7), out_seq)
    if(epoch % 500 == 0):
        print("LOSS " + str(loss.item()))
    if(epoch% 10 == 0):
        writer.add_scalar('Train/loss', loss.item(), epoch)
    loss.backward()
    optimizer.step()



test_in, test_out = seq_fetch(X_prime,Y_prime, treshold, seq_len)




plt.plot(np.arange(seq_len),lstm(test_in).view(seq_len,1,7)[:,0,1].cpu().detach().numpy())
plt.plot(np.arange(seq_len),test_out[:,0,0].cpu())


lstm(test_in).view(seq_len,1,7)[:,0,:].shape

preds = np.empty((seq_len,len(variables)))

for i in range(treshold,X_prime.shape[0]-seq_len,seq_len):
    to_predict = torch.tensor(X_prime[i:i+30,:]).unsqueeze(1)
    preds = np.vstack((preds,lstm(to_predict).view(seq_len,1,len(variables))[:,0,:].cpu().detach().numpy()))



sel_var = 1

preds_scaled = (preds[:,sel_var] * ystds[sel_var].values)+ ymeans[sel_var].values
pd.DataFrame({'y' : preds_scaled}).ewm(alpha = 0.001).mean().plot()





preds.shape
Y_prime[treshold:,:].shape


Y_prime.shape
X_prime.shape

"""
SIMULATION STUDY TO CHECK THE MODELS CAPACITY

#autoenc = nn.LSTM(7,100)
#target = torch.randint(0,10,(10,1), dtype = torch.float32, requires_grad = True).unsqueeze(1) #10 'words' with representational size 1 i.e. one number one word
lstma = LSTMAutoencoder(7,100, seq_len)
optimizer = torch.optim.SGD(lstma.parameters(), lr = 0.1)
target = torch.randn(seq_len,1,7)

encoded = lstma.encoder(target)[0]
hidden_rep = encoded[-1,:,:].repeat(target.shape[0],1,1)
decoded, _ = lstma.decoder(hidden_rep)
lstma.dense(torch.flatten(decoded))


for epoch in range(50):
    #Reset Gradients
    optimizer.zero_grad()

    predictions = lstma.forward(target)
    loss = F.mse_loss(predictions.view(seq_len,1,7), target)
    print("LOSS " + str(loss.item()))
    loss.backward()
    optimizer.step()
"""
