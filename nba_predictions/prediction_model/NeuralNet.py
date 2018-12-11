
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from keras import layers
from keras import models
from process_data import process_data
import matplotlib.pyplot as plt
#stored training/testing feature/label data. Created model fit to our data
ingames = pd.read_csv('/home/bigpandas/nba_ingame.csv')
lastmeeting = pd.read_csv('/home/bigpandas/nba_last_meeting.csv')
meta = pd.read_csv('/home/bigpandas/nba_meta.csv')
series = pd.read_csv('/home/bigpandas/nba_series.csv')

nba_df = ingames.merge(lastmeeting, on ='GAME_ID').merge(meta, on = 'GAME_ID').merge(series, on = 'GAME_ID')

features= pd.read_csv('/home/bigpandas/features2.csv')
labels= pd.read_csv('/home/bigpandas/labels.csv')
training= int(len(features)*0.8)
testing = int(len(features)*0.2)

training_features= features[:training]
training_labels = labels[:training]

test_features= features[training:]
test_labels= labels[training:]
training


# In[2]:


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

    model.fit(training_features,training_labels,validation_split=0.05, epochs=8,batch_size=100)
    
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
    
    


# In[3]:


test_list=[]
for i in range(0,200):
    test_list.append(test_features.iloc[i])
    
for i in range(0,200):
    Predict(test_list[i],module_test)


# In[5]:


from process_data import process_data
nbadict= process_data(nba_df)
nbadict.keys()


# In[6]:


model_training_input= np.array(nbadict['model_input'])[:training]
model_testing_input= np.array(nbadict['model_input'])[training:]

model_training_output = np.array(nbadict['labels'])[:training]
model_testing_output = np.array(nbadict['labels'])[training:]


# In[7]:


grant_model_test= BuildNeuralNet(model_training_input, model_training_output)


# In[8]:


Predict(model_testing_input[0], grant_model_test)


# In[ ]:


nbadict['complete info']

