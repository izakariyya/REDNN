# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:13:48 2022

@author: 1804499
"""

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf

import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
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

traind, testd, trainlbl, testlbl =  data()

# Building a model
model = Sequential()
model.add(Dense(128, input_dim=115,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

history = model.fit(traind, trainlbl, batch_size = 128, epochs=100, verbose=0)

starttc = time.time()
startmc = memory_profiler.memory_usage()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

endtc = time.time()
endmc = memory_profiler.memory_usage()
train_time_c = endtc - starttc
train_memory_c = endmc[0] - startmc[0]

print("Training time lite: {:2f} sec".format(train_time_c))
print("Training memory lite: {:2f} mb".format(train_memory_c))

tflite_models_dir = pathlib.Path("/tmp/pt737_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"pt737_model.tflite"
tflite_model_file.write_bytes(tflite_model)


startto = time.time()
startmo = memory_profiler.memory_usage()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

endto = time.time()
endmo = memory_profiler.memory_usage()
train_time_o = endto - startto
train_memory_o = endmo[0] - startmo[0]

print("Training time lite optimize: {:2f} sec".format(train_time_o))
print("Training memory lite optimize: {:2f} mb".format(train_memory_o))


interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  
  # Run predictions on every image in the "test" dataset.
  prediction_traffics = []   
  for test_i in testd:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_i = np.expand_dims(test_i, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_i)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    traf = np.argmax(output()[0])
    prediction_traffics.append(traf)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_traffics)):
    if prediction_traffics[index] == testlbl[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_traffics)

  return accuracy

starttesttb = time.time()
starttestmb = memory_profiler.memory_usage()

print(evaluate_model(interpreter))

endttestb = time.time()
endtestmb = memory_profiler.memory_usage()
test_time_b = endttestb - starttesttb
test_memory_b = endtestmb[0] - starttestmb[0]
print("Testing time base: {:2f} sec".format(test_time_b))
print("Testing memory base: {:2f} mb".format(test_memory_b))