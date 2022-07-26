#!/usr/bin/python3

import numpy as np
from numpy import vstack
import math
from pandas import read_csv
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
import torch
from torch.nn import Sigmoid,Softmax,ReLU,Linear,Tanh
from torch.nn import Module
from torch.optim import SGD,Adam
from torch.nn import BCELoss,NLLLoss,CrossEntropyLoss,MSELoss
from torch.nn.init import kaiming_uniform_,xavier_uniform_
from sklearn.model_selection import train_test_split
import time

fgen=open("general_input.in")
lines=fgen.readlines()
for i in range(len(lines)):
    lines[i]=lines[i].partition("#")[0]
    lines[i]=lines[i].rstrip()
n_inputs = int(lines[2])
H = int(lines[3])
lrate = float(lines[4])
epochs = int(lines[5])

fgen.close()

#n_inputs = 46         # number of spin orbitals 
#H = 40                # number of hidden nodes

# Columns of input csv file 
# 0 : line number; 1-23: alpha occupancies; 24-46: beta occupancies; 47 coe ci; 48: log(abs(ci)); 49 sign, 50-51 : configuration

# dataset loading
class CSVDataset(Dataset):
    def __init__(self, path):
        df  = read_csv(path,usecols=range(1,47), header=None)
        df_ci  = read_csv(path,usecols=[48], header=None)
        df_det = read_csv(path,usecols=[0,50,51], header=None)
        

        self.X = df.values[:, :]
        self.y = df_ci.values[:,:]
        self.det = df_det.values[:,:]


        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx],self.det[idx]]

    def get_splits(self, n_test=0.5):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])
############################################################################################################
# model definition
class Network(Module):
    def __init__(self,n_inputs):
        super(Network,self).__init__()

        #input descriptor
        self.hidden1 = Linear(n_inputs, H)                           # input to 1st hidden layer
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.relu = ReLU()
        self.output = Linear(H, 1)                                  # 2nd hidden layer to output
        xavier_uniform_(self.output.weight)
        self.relu = ReLU()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.relu(X)
        X = self.output(X)
        X = self.relu(X)


        return X

#########################################################################################


# prepare the dataset
def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=len(test), shuffle=True)
    return train_dl, test_dl
######################################### validation ############################
def validation(test_dl, model):                                          #send the test dataset through the network
    predictions, actuals = list(), list()
    for i, (inputs, targets,dets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = mean_squared_error(actuals, predictions)
    return acc                                                         

################################### TRAINING ########################################################
def train_model(train_dl, test_dl, model):
    criterion = MSELoss()                                               #loss function
    optimizer = Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999))
    min_valid_loss = np.inf
    arr = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    floss = open("Training_loss","w")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets,dets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
        epoch_loss = running_loss/len(train_dl)
        validation_error = validation(test_dl, model)

        if (epoch % 10==0):
            min_arr = min(arr)
            print(min_arr)
            if min_arr < min_valid_loss :
                min_valid_loss = min_arr
            else:
                torch.save(model.state_dict(),'saved_model.pth')
                floss.write(f'Validation Loss Decreased({min_valid_loss:.6f}--->{validation_error:.6f}) \t Saving The Model \n')
                break
        floss.write(f'Epoch {epoch}  epoch loss {epoch_loss}  val error{validation_error}\n')
        arr[epoch%10]=validation_error


    floss.close()
    print('Finished Training')

######################################### MODEL EVALUATION / Testing ##############################################################

def evaluate_model(test_dl, model):                                          #send the test dataset through the network
    f1 = open("validation.out","w")
    predictions, actuals = list(), list()
    for i, (inputs, targets,dets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
            f1.write(str(int(det[j][0]))+"     "+str(int(det[j][1]))+"     "+str(int(det[j][2]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(yhat[j][0]*-1))+"\n")
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = mean_squared_error(actuals, predictions)
    f1.close()
    return acc




def evaluate_trainmodel(train_dl, model):                                     #send the train dataset through the network
    f2 = open("train.out","w")
    predictions, actuals = list(), list()
    for i, (inputs, targets,dets) in enumerate(train_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
            f2.write(str(int(det[j][0]))+"     "+str(int(det[j][1]))+"     "+str(int(det[j][2]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(yhat[j][0]*-1))+"\n")
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = mean_squared_error(actuals, predictions)
    f2.close()
    return acc

###############################################################################################################
def train_ANN():
    
    train_dl, test_dl = prepare_data("csv_data/input.csv")

    # define the two networks
    model = Network(n_inputs)

    # train the models
    train_model(train_dl, test_dl, model)
    model.load_state_dict(torch.load("saved_model.pth"))
    acc = evaluate_model(test_dl, model)
    acc1 = evaluate_trainmodel(train_dl , model)
