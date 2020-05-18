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


#Preprocessing
model_input = models[1]
model_output = models[2]


def normalize_data(X):
    X_prime =  X.sel(station = stations[0], model = model_input, exp = 'rcp85')
    Y_prime = X.sel(station = stations[0], model = model_output, exp = 'rcp85')

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




def seq_fetch(data, labels, treshold, seq_len, var = -1):
    start_idx = np.random.randint(low = 0, high = treshold-seq_len)
    if (var == -1):
        in_seq = data[start_idx:start_idx+seq_len, :].cuda()
        out_seq = labels[start_idx:start_idx+seq_len, :].cuda()
    else:
        in_seq = data[start_idx:start_idx+seq_len, var].cuda()
        out_seq = labels[start_idx:start_idx+seq_len, var].cuda()
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
        decoded, _ = self.decoder(encoded)
        out = self.dense(torch.flatten(decoded))
        return out



#Training the LSTM
treshold = 10000
seq_len = 30

#lstm = nn.LSTM(len(variables), len(variables),num_layers = 2).cuda()
lstm = LSTMAutoencoder(len(variables), 100, seq_len).cuda()
print("Number of parameters " + str(sum(p.numel() for p in lstm.parameters())))
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)


#in_seq, out_seq = seq_fetch(X_prime,Y_prime, treshold, seq_len)
#encoded = lstm.encoder(in_seq)[0]
#hidden_rep = encoded[-1,:,:].repeat(target.shape[0],
#decoded, _ = lstma.decoder(hidden_rep)
#lstma.dense(torch.flatten(decoded))

writer = SummaryWriter()

for epoch in range(50000):
    #Reset Gradients
    lstm.zero_grad()

    #Fetch input-output sequence pair of desired length
    in_seq, out_seq = seq_fetch(X_prime,Y_prime, treshold, seq_len)

    #Predict and compute loss
    predictions = lstm(in_seq)
    loss = F.mse_loss(predictions.view(seq_len,1,7), out_seq)
    if(epoch % 2000 == 0):
        print("LOSS " + str(loss.item()))
    if(epoch% 10 == 0):
       writer.add_scalar('Train/loss', loss.item(), epoch)
    loss.backward()
    optimizer.step()




#test_in, test_out = seq_fetch(X_prime,Y_prime, treshold, seq_len,1)
#plt.plot(np.arange(seq_len),lstm(test_in).view(seq_len,1,7)[:,0,1].cpu().detach().numpy())
#plt.plot(np.arange(seq_len),test_out[:,0,1].cpu().numpy())
#preds = np.empty((seq_len,len(variables)))

for i in range(treshold,X_prime.shape[0]-seq_len,seq_len):
    if (i == treshold):
        to_predict = torch.tensor(X_prime[i:i+seq_len,:]).unsqueeze(1)
        preds = lstm(to_predict).view(seq_len,1,len(variables))[:,0,:].cpu().detach().numpy()
    else:
        to_predict = torch.tensor(X_prime[i:i+seq_len,:]).unsqueeze(1)
        preds = np.vstack((preds,lstm(to_predict).view(seq_len,1,len(variables))[:,0,:].cpu().detach().numpy()))



#Plotting

fig, axs = plt.subplots(len(variables),1, figsize = (30,50))
for sel_var in range(len(variables)):

    alpha = 0.05
    min_periods = 30

    preds_scaled = (preds[:,sel_var] * ystds[sel_var].values)+ ymeans[sel_var].values
    preds_scaled = np.array(pd.DataFrame({'y' : preds_scaled}).ewm(alpha = alpha, min_periods = min_periods).mean())

    ytrue = ((Y_prime[:,sel_var].cpu().numpy()) * np.array(ystds[sel_var]) + np.array(ymeans[sel_var]))
    ytrue = np.array(pd.DataFrame({'y' : ytrue}).ewm(alpha = alpha, min_periods = min_periods).mean())

    plt.sca(axs[sel_var])

    plt.plot(preds_scaled, label = 'Network Predictions')
    plt.plot(ytrue[treshold:], label = 'Ground Truth')
    plt.title(str(variables[sel_var]))
    plt.legend()





#Saving predictions for visualizing later
for i in range(7):
    if (i == 0):
        lstm_preds = (preds[:,i] * ystds[i].values)+ ymeans[i].values
        ytrue = ((Y_prime[:,i].cpu().numpy()) * np.array(ystds[i]) + np.array(ymeans[i]))
    else:
        preds_scaled = (preds[:,i] * ystds[i].values)+ ymeans[i].values
        lstm_preds = np.vstack((lstm_preds, preds_scaled))
        ytrue = np.vstack((ytrue, ((Y_prime[:,i].cpu().numpy()) * np.array(ystds[i]) + np.array(ymeans[i]))))


#np.save('lstm_preds.npy', lstm_preds)
np.save('ytrue10k', ytrue[:,treshold:])
