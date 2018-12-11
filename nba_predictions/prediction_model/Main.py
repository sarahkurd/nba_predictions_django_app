
# coding: utf-8

# In[4]:


from process_data import *
from scraping import *
import Builder

import numpy as np
import pandas as pd

ingames = pd.read_csv('/home/bigpandas/nba_ingame.csv')
lastmeeting = pd.read_csv('/home/bigpandas/nba_last_meeting.csv')
meta = pd.read_csv('/home/bigpandas/nba_meta.csv')
series = pd.read_csv('/home/bigpandas/nba_series.csv')

nba_df = ingames.merge(lastmeeting, on ='GAME_ID').merge(meta, on = 'GAME_ID').merge(series, on = 'GAME_ID')

nbadict= process_data(nba_df)
model_training_input= np.array(nbadict['model_input'])[:30076]
model_testing_input= np.array(nbadict['model_input'])[30076:]
nbadict['model_input']


model_training_output = np.array(nbadict['labels'])[:30076]
model_testing_output = np.array(nbadict['labels'])[30076:]
grant_model_test= Builder.BuildNeuralNet(model_training_input, model_training_output)


# In[3]:


nba_scrape = NBAScraper
nbadict= process_data(nba_df)
model_training_input= np.array(nbadict['model_input'])[:30076]
model_testing_input= np.array(nbadict['model_input'])[30076:]

model_training_output = np.array(nbadict['labels'])[:30076]
model_testing_output = np.array(nbadict['labels'])[30076:]
grant_model_test.evaluate(model_testing_input,model_testing_output)


# In[4]:


Builder.Predict(model_testing_input[4],grant_model_test)

