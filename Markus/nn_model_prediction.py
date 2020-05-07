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


#Simple dense network class
class Densenet(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(Densenet, self).__init__()
        self.dense = nn.Sequential(
                            nn.Linear(inputsize, 256),
                            nn.ReLU(),
                            nn.Linear(256, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512,256),
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
assert(torch.cuda.is_available()), 'GPU not available, aborting...'

t_size = 1000 #Size of training set

X_train = torch.tensor(np.array(X_prime[:t_size,:]), dtype = torch.float32).cuda()
Y_train = torch.tensor(np.array(Y_prime[:t_size,:]), dtype = torch.float32).cuda()

X_test = torch.tensor(np.array(X_prime[t_size:,:]), dtype = torch.float32).cuda()
Y_test = torch.tensor(np.array(Y_prime[t_size:,:]), dtype = torch.float32).cuda()


#Training Loop
net = Densenet(len(variables), len(variables)).cuda()
writer = SummaryWriter()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)


for epoch in range(2000):
    optimizer.zero_grad()
    batch_idx = np.random.randint(low = 0,high = t_size, size = 50)
    minibatch = X_train[batch_idx,:]
    output = net.forward(minibatch)
    loss = F.mse_loss(output, Y_train[batch_idx])
    if (epoch % 100 == 99):
        print("Epoch: " + str(epoch))
        print("Loss: " + str(loss))
        writer.add_scalar('Train/loss', loss.item(), epoch)

    loss.backward()
    optimizer.step()
writer.close()

train_error = F.mse_loss(net.forward(X_train), Y_train)
test_error = F.mse_loss(net.forward(X_test), Y_test)



#Plotting
fig, axs = plt.subplots(len(variables),1, figsize = (30,50))
for sel_var in range(len(variables)):

    alpha = 0.01
    min_periods = 30

    predictions = ((net.forward(X_test)[:,sel_var].cpu().detach().numpy()) * ystds[sel_var].values)+ ymeans[sel_var].values

    ypred = pd.DataFrame(predictions).ewm(alpha = alpha, min_periods = min_periods).mean()
    ypred = np.array(ypred)[:,0]

    ytrue = pd.DataFrame(Y_test[:,sel_var].cpu() * ystds[sel_var].values + ymeans[sel_var].values).ewm(alpha = alpha, adjust = True, min_periods = min_periods).mean()
    ytrue = np.array(ytrue)[:,0]


    plt.sca(axs[sel_var])
    plt.plot(ypred, label = 'Netword Predictions')
    plt.plot(ytrue, label = 'Ground Truth')
    plt.title(str(variables[sel_var]))
    plt.legend()


fig.suptitle('Predicting ' + model_output + ' from the first ' + str(t_size) +  ' days of ' + model_input + '. Exponential Weighted Average Plot with smoothing factor ' + str(alpha) + '.', fontsize = 20)


#Explicitly plotting averaged temperature
Y_pred = ((net.forward(X_test)[:,1].cpu().detach().numpy()) * ystds[1].values)+ ymeans[1].values
pd.DataFrame({'y' : Y_pred}).ewm(alpha = 0.0001, min_periods= 2000).mean().plot()
plt.title('Exponential weighted Average of predicted tas averaged over all time steps')
