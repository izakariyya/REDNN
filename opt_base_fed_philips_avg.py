# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:02:22 2021

@author: 1804499
"""

import syft as sy
import numpy as np
import time
import memory_profiler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
hook = sy.TorchHook(torch)

#Create couple of workers

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id='alice')
secure_worker = sy.VirtualWorker(hook, id="secure_worker")


#Get data set

def data():
    benign = np.loadtxt("benign_traffic.csv", delimiter = ",")
    mirai = np.loadtxt("mirai_traffic.csv", delimiter = ",")
    gafgyt = np.loadtxt("gafgyt_traffic.csv", delimiter = ",")
    alldata = np.concatenate((benign, gafgyt, mirai))
    j = len(benign[0])
    data = alldata[:, 1:j] 
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2, random_state = 42)
    return bendata, benmir, benlabel, benslabel


traind, testd, trainlbl, testlbl =  data()

traind = torch.FloatTensor(traind)

testd = torch.FloatTensor(testd)

trainlbl = torch.FloatTensor(trainlbl)

#testlbl = torch.FloatTensor(testlbl)
torch.manual_seed(0)
# Define network dimensions
n_input_dim = traind.shape[1]
# Layer size
n_hidden1 = 83
n_hidden2 = 128 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier


#Build and initialize network (model)
model = nn.Sequential(
    nn.Linear(n_input_dim, n_hidden1),
    nn.ReLU(),
    nn.Linear(n_hidden1, n_hidden2),
    nn.Linear(n_hidden2, n_hidden1),
    nn.Linear(n_hidden1, n_output),
    nn.Sigmoid()) 

# Define the loss function
#loss_fn = torch.nn.BCELoss() 
learning_rate = 0.001
eps = 0.001
epochs = 4
worker_iter = 30
batch_size = 128
m_batch_size = 4

# Cross Entropy Cost Function

def cross_entropy(input, target, eps):
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - (target * torch.log(input + eps) + (1 - target + eps) * torch.log(1 - input))
    return torch.mean(bce)

# Regularized Cost

def cross_reg(input, target, eps, lambd):
    rloss = cross_entropy(input, target, eps)
    rloss = rloss * lambd
    return rloss
         
#Full Training

def train_base(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size):
    #Create data for Bob and Alice
    size = int(len(traind) / 2)

    bobs_data = traind[0:size]
    bobs_target = trainlbl[0:size]

    alices_data = traind[size:]
    alices_target = trainlbl[size:]
    
    #batch_number = bobs_data.size()[0] // batch_size
    
    for i in range(epochs):
        # X is a torch Variable
        #indices = epochs % batch_number
        permutation = torch.randperm(bobs_data.size()[0])
        indices = permutation[i:i+batch_size]
        bobs_data_batch, bobs_target_batch = bobs_data[indices].send(bob), bobs_target[indices].send(bob)
        alices_data_batch, alices_target_batch = alices_data[indices].send(alice), alices_target[indices].send(alice)
        #Send model to Alice and Bob
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)
        boptim = torch.optim.SGD(bobs_model.parameters(), lr=learning_rate)
        aoptim = torch.optim.SGD(alices_model.parameters(), lr=learning_rate)
        # Training virtual workers script
        for i in range(worker_iter):
            #Bobs Training
            boptim.zero_grad()
            b_yhat = bobs_model(bobs_data_batch) 
            bloss = cross_entropy(b_yhat.reshape(-1), bobs_target_batch, eps)
            bloss.backward()
        
            boptim.step()
            bloss = bloss.get().data
                
            #Alices Training
            aoptim.zero_grad()
            a_yhat = alices_model(alices_data_batch)
            aloss = cross_entropy(a_yhat.reshape(-1), alices_target_batch, eps)
            aloss.backward()
        
            aoptim.step()
            aloss = aloss.get().data
           
        #Send Both Updated Models to a Secure Worker
   
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)
    
        #obtaining model weights and averaging them
        paramb = []
        for param in bobs_model.parameters():
            paramb.append(param.view(-1))
        paramb = torch.cat(paramb)
        parama = []
        for param in alices_model.parameters():
            parama.append(param.view(-1))
        parama = torch.cat(parama)
        #Averaging model weights
        (parama + paramb) / 2
    return bloss, aloss, model

def train_efficient(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size, m_batch_size):
    #Create data for Bob and Alice
    size = int(len(traind) / 2)
    #W_c = 0.01
    #W_t = 0.01
    lambd = 0.01
    lambi = 0.01

    bobs_data = traind[0:size]
    bobs_target = trainlbl[0:size]

    alices_data = traind[size:]
    alices_target = trainlbl[size:]
    
   
    for i in range(epochs):
        
        permutation = torch.randperm(bobs_data.size()[0])
        indices = permutation[i:i+batch_size]
        # Mini-Batch
        bobs_data_batch, bobs_target_batch = bobs_data[indices].send(bob), bobs_target[indices]
        alices_data_batch, alices_target_batch = alices_data[indices].send(alice), alices_target[indices]
        
        # Micro-Batch
        mindices = indices / m_batch_size
        
        bobs_data_batch, bobs_target_batch = bobs_data[mindices].send(bob), bobs_target[mindices].send(bob)
        alices_data_batch, alices_target_batch = alices_data[mindices].send(alice), alices_target[mindices].send(alice)
        
        #Send model to Alice and Bob
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)
        boptim = torch.optim.SGD(bobs_model.parameters(), lr=learning_rate)
        aoptim = torch.optim.SGD(alices_model.parameters(), lr=learning_rate)
        # Training virtual workers script
        for i in range(worker_iter):
            #Bobs Training
            boptim.zero_grad()
            b_yhat = bobs_model(bobs_data_batch) 
            bloss = cross_reg(b_yhat.reshape(-1), bobs_target_batch, eps, lambd)
            bloss.backward()
        
            boptim.step()
            bloss = bloss.get().data
                
            #Alices Training
            aoptim.zero_grad()
            a_yhat = alices_model(alices_data_batch)
            aloss = cross_reg(a_yhat.reshape(-1), alices_target_batch, eps, lambd)
            aloss.backward()
        
            aoptim.step()
            aloss = aloss.get().data
            
            if bloss <= bl:
                lambd = lambd + lambi
            if aloss <= al:
                lambd = lambd + lambi
           
        #Send Both Updated Models to a Secure Worker
   
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)
    
        #obtaining model weights and averaging them
        paramob = []
        for param in bobs_model.parameters():
            paramob.append(param.view(-1))
        paramob = torch.cat(paramob)
        paramoa = []
        for param in alices_model.parameters():
            paramoa.append(param.view(-1))
        paramoa = torch.cat(paramoa)
        #Averaging model weights
        (paramoa + paramob) / 2
    return bloss, aloss, model

#Baseline Computational Resources
    
starttbase = time.time()
startmbase = memory_profiler.memory_usage()

bl, al, modelb = train_base(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size)

endtbase =time.time()
endmbase = memory_profiler.memory_usage()
traintime_base = endtbase - starttbase
train_memory_base = endmbase[0] - startmbase[0]

print("Training time base: {:2f} sec".format(traintime_base))
print("Training memory base: {:2f} mb".format(train_memory_base))


#Optimized Computational Resources

starttefi = time.time()
startmefi = memory_profiler.memory_usage()

obl, oal, modelo = train_efficient(traind, trainlbl, model, epochs, worker_iter, learning_rate, batch_size, m_batch_size)

endtefi = time.time()
endmefi = memory_profiler.memory_usage()
traintime_efi = endtefi - starttefi
train_memory_efi = endmefi[0] - startmefi[0]

print("Training time optimize: {:2f} sec".format(traintime_efi))
print("Training memory optimize: {:2f} mb".format(train_memory_efi))


def predict(model, X, Y):
    y_hat = model(X)
    y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
    accuracy = np.sum(Y.reshape(-1,1) ==y_hat_class) / len(Y)
    return accuracy
 
acc_b = predict(modelb, testd, testlbl)

acc_o = predict(modelo, testd, testlbl)

print("Test accuracy base: {:2f}".format(acc_b))
print("Test accuracy optimize: {:2f}".format(acc_o))