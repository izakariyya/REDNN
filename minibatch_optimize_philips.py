# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:13:21 2020

@author: 1804499
"""

import numpy as np
from sklearn.model_selection import train_test_split
import memory_profiler
import time


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


# implemented densely connected neural network
    
import fully_connected_nn

#neural network architecture
    
traind, testd, trainlbl, testlbl =  data()

#traind   = np.float32(traind)
#testd    = np.float32(testd)

#n = len(traind[0])

NN_ARCHITECTURE = [
    {"input_dim": 115, "output_dim": 83, "activation": "relu"},
    {"input_dim": 83, "output_dim": 83, "activation": "relu"},
    {"input_dim": 83, "output_dim":128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 83, "activation": "relu"},
    {"input_dim": 83, "output_dim": 1, "activation": "sigmoid"}
]

#dataset transformation

traind = np.transpose(traind)
trainlbl = np.transpose(trainlbl.reshape((trainlbl.shape[0], 1)))
testd = np.transpose(testd)
testlbl = np.transpose(testlbl.reshape((testlbl.shape[0], 1)))

N_EPOCHS = 100
LR = 0.001
eps= 0.001

#Baseline model (Basic Gradient)

starttb = time.time()
startmb = memory_profiler.memory_usage()

params_values_b, cost_history_b, accuracy_history_b, grad_a = fully_connected_nn.train(traind, trainlbl, NN_ARCHITECTURE, N_EPOCHS, LR,128)

endtb = time.time()
endmb = memory_profiler.memory_usage()
train_time_b = endtb - starttb
train_memory_b = endmb[0] - startmb[0]

print("Train set accuracy base: {:2f}".format(accuracy_history_b[-1]))
print("Training time base: {:2f} sec".format(train_time_b))
print("Training memory base: {:2f} mb".format(train_memory_b))

starttesttb = time.time()
starttestmb = memory_profiler.memory_usage()

Y_test_hat_b, _, _ =   fully_connected_nn.full_forward_propagation(testd, params_values_b, NN_ARCHITECTURE)
acc_test_b = fully_connected_nn.get_accuracy_value(Y_test_hat_b, testlbl)
pre_test_b, rec_test_b, F1_test_b = fully_connected_nn.get_performance_value(Y_test_hat_b, testlbl)

endttestb = time.time()
endtestmb = memory_profiler.memory_usage()
test_time_b = endttestb - starttesttb
test_memory_b = endtestmb[0] - starttestmb[0]

print("Test set accuracy base: {:2f}".format(acc_test_b))
print("Test set precision base: {:2f}".format(pre_test_b))
print("Test set recall base: {:2f}".format(rec_test_b))
print("Test set score base: {:2f}".format(F1_test_b))
print("Testing time base: {:2f} sec".format(test_time_b))
print("Testing memory base: {:2f} mb".format(test_memory_b))

def advsry(X, Y, epsi):
       Y_hat, cashe, _ = fully_connected_nn.full_forward_propagation(X, params_values_b, NN_ARCHITECTURE)
       grads_values, grad_w = fully_connected_nn.full_backward_propagation(Y_hat, Y, cashe, params_values_b, NN_ARCHITECTURE)
       pert = np.resize(grad_w, X.shape)   
       pertubated_data = X + np.sign(epsi * pert)
       pertubated_data = np.clip(pertubated_data, 0, 1)
       return pertubated_data 

X_attack = advsry(testd, testlbl, 1.0)
Y_test_hat_b_r, _, _  = fully_connected_nn.full_forward_propagation(X_attack, params_values_b, NN_ARCHITECTURE)
acc_test_b_r = fully_connected_nn.get_accuracy_value(Y_test_hat_b_r, testlbl)
pre_test_b_r, rec_test_b_r, F1_test_b_r = fully_connected_nn.get_performance_value(Y_test_hat_b_r, testlbl)

print("Adversarial set accuracy base: {:2f}".format(acc_test_b_r))
print("Adversarial set precision base: {:2f}".format(pre_test_b_r))
print("Adversarial set recall base: {:2f}".format(rec_test_b_r))
print("Adeverarial set  score base: {:2f}".format(F1_test_b_r))
#Y_test_hat_base, _, _  = fully_connected_nn.full_forward_propagation(adv_b, params_values_b, NN_ARCHITECTURE)
#acc_train_base = fully_connected_nn.get_accuracy_value(Y_test_hat_base, trainlbl)
#print("Adversarial train accuracy base: {:2f}".format(acc_train_base))

#epsis = [0, 0.01, 0.1, 0.15]
#acc_testbr = []

#for epsi in epsis:
#    X_attack_b = fully_connected_nn.advsry(testd,epsi, grad_a)
#    Y_test_hat_b_r, _, _  = fully_connected_nn.full_forward_propagation(X_attack_b, params_values_b, NN_ARCHITECTURE)
#    acc_test_b_r = fully_connected_nn.get_accuracy_value(Y_test_hat_b_r, testlbl)
#    acc_testbr.append(acc_test_b_r)
#    print(f"Epsilon: {epsi}, adversarial accuracy base:{acc_test_b_r:.3f}")



def train_batch(X, Y, nn_architecture, epochs, learning_rate, batch_size, m_batch_size, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = fully_connected_nn.init_layers(nn_architecture, 2)
    # initiation of regularization parameters
    W_tre = 0.01
    lambd = 0.01
    lambi = 0.01
   
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    examples_size = X.shape[1]
    batch_number = examples_size // batch_size
    m_batch_number = batch_number // m_batch_size
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        batch_idx = epochs % batch_number
        # Mini-Batch
        X_batch = X[:, batch_idx * batch_size : (batch_idx +1) * batch_size]
        Y_batch = Y[:, batch_idx * batch_size : (batch_idx +1) * batch_size]
        
        # Micro Batch
        
        batch_midx = epochs % m_batch_number
        X_batch = X[:, batch_midx * m_batch_size : (batch_midx +1) * m_batch_size]
        Y_batch = Y[:, batch_midx * m_batch_size : (batch_midx +1) * m_batch_size]
        # step forward
        Y_hat, cashe, W_curr = fully_connected_nn.full_forward_propagation(X_batch, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = fully_connected_nn.compute_cost_with_regularization(Y_hat, Y_batch, eps, W_curr, W_tre, lambd)
        cost_history.append(cost)
        accuracy = fully_connected_nn.get_accuracy_value(Y_hat, Y_batch)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values, grado = fully_connected_nn.full_backward_propagation(Y_hat, Y_batch, cashe, params_values, nn_architecture)
        # updating model state
        params_values = fully_connected_nn.update(params_values, grads_values, nn_architecture, learning_rate)
        
        if cost <= cost_history_b[-1]:
            lambd = lambd + lambi
            
        if(i % 50 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)
            
    return params_values, cost_history, accuracy_history, grado

starttbatch = time.time()
startmbatch = memory_profiler.memory_usage()

params_values_batch, cost_history_batch, accuracy_history_batch, grado = train_batch(traind, trainlbl, NN_ARCHITECTURE, N_EPOCHS, LR, 128, 4)

endtbatch = time.time()
endmbatch = memory_profiler.memory_usage()
train_time_batch = endtbatch - starttbatch
train_memory_batch = endmbatch[0] - startmbatch[0]

print("Train set accuracy optimize: {:2f}".format(accuracy_history_batch[-1]))
print("Training time optimize: {:2f} sec".format(train_time_batch))
print("Training memory optimize: {:2f} mb".format(train_memory_batch))

starttestbatch = time.time()
starttestmbatch = memory_profiler.memory_usage()

Y_test_hat_batch, _, _  = fully_connected_nn.full_forward_propagation(testd, params_values_batch, NN_ARCHITECTURE)
acc_test_batch = fully_connected_nn.get_accuracy_value(Y_test_hat_batch, testlbl)
pre_test_batch, rec_test_batch, F1_test_batch = fully_connected_nn.get_performance_value(Y_test_hat_batch, testlbl)

endttestbatch = time.time()
endtestmbatch = memory_profiler.memory_usage()
test_time_batch = endttestbatch - starttestbatch
test_memory_batch = endtestmbatch[0] - starttestmbatch[0]

print("Test set accuracy optimize: {:2f}".format(acc_test_batch))
print("Test set precision optimize: {:2f}".format(pre_test_batch))
print("Test set recall optimize: {:2f}".format(rec_test_batch))
print("Test set score optimize: {:2f}".format(F1_test_batch))
print("Testing time optimize: {:2f} sec".format(test_time_batch))
print("Testing memory optimize: {:2f} mb".format(test_memory_batch))


Y_test_hat_batch_r, _, _  = fully_connected_nn.full_forward_propagation(X_attack, params_values_batch, NN_ARCHITECTURE)
acc_test_batch_r = fully_connected_nn.get_accuracy_value(Y_test_hat_batch_r, testlbl)
pre_test_batch_r, rec_test_batch_r, F1_test_batch_r = fully_connected_nn.get_performance_value(Y_test_hat_batch_r, testlbl)
print("Adversarial set accuracy optimize: {:2f}".format(acc_test_batch_r))
print("Adversarial set  precision optimize: {:2f}".format(pre_test_batch_r))
print("Adeversarial set recall optimize: {:2f}".format(rec_test_batch_r))
print("Adversarial set score optimize: {:2f}".format(F1_test_batch_r))


#X_attack_o = fully_connected_nn.advsry(testd, 0.15)
#Y_test_hat_batch_r, _, _  = fully_connected_nn.full_forward_propagation(X_attack_o, params_values_batch, NN_ARCHITECTURE)
#acc_test_batch_r = fully_connected_nn.get_accuracy_value(Y_test_hat_batch_r, testlbl)
#print("Adversarial set accuracy optimize: {:2f}".format(acc_test_batch_r))

#acc_testbatchr = []
#for epsi in epsis:
#    X_attack_o = fully_connected_nn.advsry(testd, eps, grado)
#    Y_test_hat_batch_r, _, _  = fully_connected_nn.full_forward_propagation(X_attack_o, params_values_batch, NN_ARCHITECTURE)
#    acc_test_batch_r = fully_connected_nn.get_accuracy_value(Y_test_hat_batch_r, testlbl)
#    acc_testbatchr.append(acc_test_batch_r)
#    print(f"Epsilon: {epsi}, adversarial accuracy optimize:{acc_test_batch_r:.3f}")

#print("Test adverasarial accuracy optimize: {:2f}".format(acc_test_batch_r))

    




