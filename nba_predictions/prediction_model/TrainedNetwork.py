from nba_predictions.prediction_model import process_data
from nba_predictions.prediction_model import process_data
from nba_predictions.prediction_model import Builder
from keras.models import model_from_json

import numpy as np
import pandas as pd

class Network:
    
    def build_df():
        ingames = pd.read_csv('/Users/sarahkurdoghlian/Desktop/Data301/nba-django-app/mysite/nba_predictions/prediction_model/csvs/nba_ingame.csv')
        lastmeeting = pd.read_csv('/Users/sarahkurdoghlian/Desktop/Data301/nba-django-app/mysite/nba_predictions/prediction_model/csvs/nba_last_meeting.csv')
        meta = pd.read_csv('/Users/sarahkurdoghlian/Desktop/Data301/nba-django-app/mysite/nba_predictions/prediction_model/csvs/nba_meta.csv')
        series = pd.read_csv('/Users/sarahkurdoghlian/Desktop/Data301/nba-django-app/mysite/nba_predictions/prediction_model/csvs/nba_series.csv')
        nba_df = ingames.merge(lastmeeting, on ='GAME_ID').merge(meta, on ='GAME_ID').merge(series, on = 'GAME_ID')
        return nba_df
    
    def build_model(nba_df):
        nbadict= process_data.process_data(nba_df)
        model_training_input= np.array(nbadict['model_input'])[:30076]
        model_testing_input= np.array(nbadict['model_input'])[30076:]
        nbadict['model_input']

        model_training_output = np.array(nbadict['labels'])[:30076]
        model_testing_output = np.array(nbadict['labels'])[30076:]
        model= Builder.BuildNeuralNet(model_training_input, model_training_output)
        return model
    
    def save_model(model):
        # serialize model to json
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("model.h5")
