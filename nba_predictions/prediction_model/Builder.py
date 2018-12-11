import numpy as np
import pandas as pd

from keras import layers
from keras import models
import matplotlib.pyplot as plt

#function takes in training data and output and returns a trained NN
def BuildNeuralNet(training_features,training_labels):
    model = models.Sequential()
    neural_nodes=[256,128,64,32,12,8]
    model.add(layers.Dense(128,activation='relu',input_shape=(18,)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(.25))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dropout(.25))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dropout(.25))
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(2,activation='softmax'))
    model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])

    model.fit(training_features,training_labels,validation_split=0.15, epochs=8,batch_size=100)
    
    return model
#functino to predict winning team. Output is 0 for first team and 1 for second team
def Predict(game_vector, model):
    game_vector=np.array(game_vector)
    game_vector=game_vector[np.newaxis]
    probabilities=model.predict(game_vector)
    winner= np.argmax(probabilities[0])
    
    print('Probability of first team: '+ str(probabilities[0][0]))
    print('Probability of second team: '+ str(probabilities[0][1]))
    print('favored winner: '+ str(winner+1))
    return winner
    
    