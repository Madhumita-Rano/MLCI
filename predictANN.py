#!/usr/bin/python3
import numpy
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

fgen.close()

f = open("Pred.out","w")

# 0 : line number; 1-23: alpha occupancies; 24-46: beta occupancies; 47 fci; 48-49 : deci equivalent

# dataset loading
class CSVDataset(Dataset):
    def __init__(self, path):
        df  = read_csv(path,usecols=range(1,47), header=None)
        df_det = read_csv(path,usecols = [0,48,49], header = None)

        self.X = df.values
        self.det = df_det.values

        self.X = self.X.astype('float32')
        self.det = self.det.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.det[idx],self.X[idx]]

    def get_splits(self, n_test=1.0):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])
############################################################################################################


class Network(Module):
    def __init__(self,n_inputs):
        super(Network,self).__init__()

        #input descriptor
        self.hidden1 = Linear(n_inputs, H)                           # input to 1st hidden layer
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.relu = ReLU()
        # Output layer
        self.output = Linear(H, 1)                                  # 2nd hidden layer to output
        xavier_uniform_(self.output.weight)
        self.relu = ReLU()
    def forward(self, X):
        X = self.hidden1(X)
        X = self.relu(X)
        X = self.output(X)
        X = self.relu(X)


        return X

################################################################################################################

# prepare the dataset
def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    test_dl = DataLoader(test, batch_size=len(test), shuffle=False)
    return test_dl


################################### Prediction  ########################################################

def evaluate_model(test_dl, model):                                          #send the test dataset through the network
    for i, (dets,inputs) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        dets = dets.numpy()
        for j in range(len(yhat)):
            f.write(str(int(dets[j][0]))+"   "+str(int(dets[j][1]))+"   "+str(int(dets[j][2]))+"   "+str(10**(-1.0*yhat[j][0]))+"\n")
    return True                                                         




################################################### Sign prediction ###################################################
# prepare the data
test_dl = prepare_data("csv_data/whole_space.csv")

# load the two networks
model = Network(n_inputs)
model.load_state_dict(torch.load("saved_model.pth"))

# evaluate the model
evaluate_model(test_dl, model)

f.close()
