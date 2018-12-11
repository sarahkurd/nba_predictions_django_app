from django.shortcuts import render
import requests
from lxml import html
import datetime
import pandas as pd
import numpy as np
from nba_predictions.prediction_model import scraping
from nba_predictions.prediction_model import TrainedNetwork
from nba_predictions.prediction_model import process_data
global graph,model
import tensorflow as tf
from keras.models import model_from_json

myMap = {'Los Angeles': 'LAL', 'Atlanta': 'ATL', 'Boston': 'BOS', 'Charlotte': 'CHA', 'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN', 'Detroit': 'DET', 'LA': 'LAC', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND', 'Memphis': 'MEM', 'Miami': 'MIA', 'Milwaukee': 'MIL', 'Minnesota': 'MIN', 'New Orleans': 'NOP', 'New York': 'NYK', 'Oklahoma City': 'OKC', 'Orlando': 'ORL', 'Philadelphia': 'PHI', 'Phoenix': 'PHX', 'Portland': 'POR', 'Sacramento': 'SAC', 'San Antonio': 'SAS', 'Toronto': 'TOR', 'Utah': 'UTA', 'Washington': 'WAS', 'Brooklyn': 'BKN'}

# Create your views here.
def home(request):
    rows = scrapeDailyGames('2018', '12', '07')
    game_date = datetime.date(2018, 12, 7).strftime("%A, %B %d")
    return render(request, 'nba_predictions/home.html', {'rows': rows, 'date': game_date})

def changeDate(month, day, year):
    if(int(day) < 15):
        month = str(int(month) - 1)
        day = str(30 - (15-int(day)))
    else:
        day = int(day) - 15
    return month + '/' + str(day) + '/' + year

###### helper functions ######
def scrapeDailyGames(year, month, day):
    
#    pageContent=requests.get('http://www.espn.com/nba/schedule')
    date = year + month + day
    print(date)
    pageContent=requests.get('http://www.espn.com/nba/schedule/_/date/' + date)
    tree = html.fromstring(pageContent.content)
    
    rows_array = []
    # check if result column has been updated yet
    rows = tree.xpath('//div[contains(@class, "responsive-table-wrap")][1]/table/tbody/tr')
    for row in rows:
        away_pic = row.xpath('.//td[1]/div[contains(@class, "teams")]/a/img/@src')[0]
        away = row.xpath('.//td[1]/a/span/text()')[0]
        home_pic = row.xpath('.//td[2]/div[contains(@class, "home-wrapper")]/div[contains(@class, "teams")]/a/img/@src')[0]
        home = row.xpath('.//td[2]/div[contains(@class, "home-wrapper")]/a/span/text()')[0]
        
        start = changeDate(month, day, year)
        end = month + '/' + day + '/' + year
        print(start)
        print(end)
        # run prediction model on 2 team names
        our_winner = runModel(home, away, start, end)
        
        if(tree.xpath('//div[contains(@class, "responsive-table-wrap")][1]/table/thead/tr/th[3]/span/text()')[0] == 'result'):
            result = row.xpath('.//td[3]/a/text()')[0]
            rows_array.append({'awaypic': away_pic, 'away': away, 'homepic': home_pic, 'home': home, 'result': result, 'winner': our_winner})
        else:
#            time = row.xpath('.//td[3]/a/text()')
#            rows_array.append({'awaypic': away_pic, 'away': away, 'homepic': home_pic, 'home': home, 'time': time})
            rows_array.append({'awaypic': away_pic, 'away': away, 'homepic': home_pic, 'home': home, 'winner': our_winner})
          
    return rows_array

def runModel(home_team, away_team, start_date, end_date):
    # scrape last 15 days
    scraper = scraping.NBAScraper
    scraper.scrape(start_date, end_date)
    
    # process data
    df = pd.read_csv('/Users/sarahkurdoghlian/Desktop/Data301/nba-django-app/mysite/nba_df.csv')
    processed = process_data.process_data(df)
    
    processed['model_input'].reset_index(inplace=True)
    
    # home team    
    df1 = processed['complete info'][processed['complete info']['TEAM_ABBREVIATION_x'] == myMap[home_team]]
    df1.reset_index(inplace=True)
    df1 = df1[df1['GAME_ID'] == df1['GAME_ID'].max()]
    ID=df1['GAME_ID']
    ID=ID.iloc[0]
    team1=processed['model_input'][processed['model_input']['GAME_ID'] == ID]
    
    # away team
    df2 = processed['complete info'][processed['complete info']['TEAM_ABBREVIATION_x'] == myMap[away_team]]
    df2.reset_index(inplace=True)
    df2 = df2[df2['GAME_ID'] == df2['GAME_ID'].max()]
    ID=df2['GAME_ID']
    ID=ID.iloc[0]
    team2=processed['model_input'][processed['model_input']['GAME_ID'] == ID]
        
    team2=team2.drop(['GAME_ID','HOME_TEAM_LOSSES_x', 'PTS_x', 'FG_PCT_x', 'FT_PCT_x', 'FG3_PCT_x', 'AST_x', 'REB_x', 'TOV_x', 'HomeTeam_x'],axis=1)
    team1=team1.drop(['HOME_TEAM_WINS_y', 'PTS_y', 'FG_PCT_y', 'FT_PCT_y', 'FG3_PCT_y', 'AST_y', 'REB_y', 'TOV_y', 'HomeTeam_y'],axis=1)
    processed['model_input'].set_index(['GAME_ID'], inplace=True)
    
    # get our model and evaluate
    nba_df = TrainedNetwork.Network.build_df()
    model = TrainedNetwork.Network.build_model(nba_df)
    TrainedNetwork.Network.save_model(model)
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    # compile model
    loaded_model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
    
    # evaluate
    loaded_model.evaluate(processed['model_input'],processed['labels'])

    team1['merge']=1.121234
    team2['merge']=1.121234
    
    final=team1.merge(team2,on='merge')
    final=final.drop(['merge'],axis=1)
    final.drop(['GAME_ID'],axis=1, inplace=True)
    
    vector = loaded_model.predict(np.array(final))
        
    if(vector[0][0] > vector[0][1]):
        winning_team = home_team
    else:
        winning_team = away_team
    
    return winning_team








    