import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import *
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


#Importing the Data
data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)

models = np.array(X.coords['model'])
stations = np.array(X.coords['station'])
variables = np.array(X.coords['var'])


#Simple dense network class
class Densenet(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(Densenet, self).__init__()
        self.dense = nn.Sequential(
                            nn.Linear(inputsize, 128),
                            nn.ReLU(),
                            nn.Linear(128, 256),
                            nn.ReLU(),
                            nn.Linear(256, 256),
                            nn.ReLU(),
                            nn.Linear(256,outputsize))

    def forward(self,x):
        x = self.dense(x)
        return x


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

#Splitting into Train and Test data
X_train = torch.tensor(np.array(X_prime[:1000,:]), dtype = torch.float32)
Y_train = torch.tensor(np.array(X_prime[:1000,:]), dtype = torch.float32)

X_test = torch.tensor(np.array(X_prime[1000:,:]), dtype = torch.float32)
Y_test = torch.tensor(np.array(X_prime[1000:,:]), dtype = torch.float32)


#Training Loop
net = Densenet(len(variables), len(variables))
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)


for epoch in range(1000):
    optimizer.zero_grad()
    batch_idx = np.random.randint(low = 0,high = 1000, size = 50)
    minibatch = X_train[batch_idx,:]
    output = net.forward(minibatch)
    loss = F.mse_loss(output, Y_train[batch_idx])
    if (epoch % 200 == 0):
        print("Epoch: " + str(epoch))
        print("Loss: " + str(loss))

    loss.backward()
    optimizer.step()


train_error = F.mse_loss(net.forward(X_train), Y_train)
test_error = F.mse_loss(net.forward(X_test), Y_test)



#Plotting
fig, axs = plt.subplots(len(variables),1, figsize = (30,50))
for sel_var in range(len(variables)):

    alpha = 0.01
    min_periods = 30

    predictions = ((net.forward(X_test)[:,sel_var].detach().numpy()) * ystds[sel_var].values)+ ymeans[sel_var].values

    ypred = pd.DataFrame(predictions).ewm(alpha = alpha, min_periods = min_periods).mean()
    ypred = np.array(ypred)[:,0]

    ytrue = pd.DataFrame(Y_test[:,sel_var] * ystds[sel_var].values + ymeans[sel_var].values).ewm(alpha = alpha, adjust = True, min_periods = min_periods).mean()
    ytrue = np.array(ytrue)[:,0]


    plt.sca(axs[sel_var])
    plt.plot(ypred, label = 'Netword Predictions')
    plt.plot(ytrue, label = 'Ground Truth')
    plt.title(str(variables[sel_var]))
    plt.legend()


fig.suptitle('Predicting ' + models[0] + 'from the first 1000 days of ' + models[1] + '. Exponential Weighted Average Plot with smoothing factor ' + str(alpha) + '.', fontsize = 20)
