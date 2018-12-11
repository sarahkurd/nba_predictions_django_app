import requests
import numpy as np 
import pandas as pd 
import json
import datetime

# class NBAScraper defines scraping function
# to get various game data tables from nba stats website
# from 1989-1990 season to 2017-2018 season
# url for getting your user agent: https://stackoverflow.com/questions/46781563/how-to-obtain-a-json-response-from-the-stats-nba-com-api

class NBAScraper:

	# DataFrame column labels
	ingame_columns = 	['GAME_DATE_EST', 'GAME_SEQUENCE', 'GAME_ID', 'TEAM_ID','TEAM_ABBREVIATION', 'TEAM_CITY_NAME', 'TEAM_WINS_LOSSES', 'PTS_QTR1',
					'PTS_QTR2', 'PTS_QTR3','PTS_QTR4', 'PTS_OT1','PTS_OT2','PTS_OT3','PTS_OT4', 'PTS_OT5', 'PTS_OT6', 'PTS_OT7','PTS_OT8', 'PTS_OT9',
					'PTS_OT10', 'PTS','FG_PCT',	'FT_PCT','FG3_PCT',	'AST','REB','TOV']

	meta_columns = ['GAME_DATE_EST', 'GAME_SEQUENCE', 'GAME_ID', 'GAME_STATUS_ID',
				'GAME_STATUS_TEXT', 'GAMECODE', 'HOME_TEAM_ID','VISITOR_TEAM_ID', 'SEASON',
				'LIVE_PERIOD', 'LIVE_PC_TIME', 'NATL_TV_BROADCASTER_ABBREVIATION', 'LIVE_PERIOD_TIME_BCAST', 'WH_STATUS']

	series_columns = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST',
				 'HOME_TEAM_WINS', 'HOME_TEAM_LOSSES', 'SERIES_LEADER']

	last_meeting_columns = ['GAME_ID', 'LAST_GAME_ID', 'LAST_GAME_DATE_EST', 'LAST_GAME_HOME_TEAM_ID',
						'LAST_GAME_HOME_TEAM_CITY', 'LAST_GAME_HOME_TEAM_NAME', 'LAST_GAME_HOME_TEAM_ABBREVIATION',
						'LAST_GAME_HOME_TEAM_POINTS', 'LAST_GAME_VISITOR_TEAM_ID', 'LAST_GAME_VISITOR_TEAM_CITY',
						'LAST_GAME_VISITOR_TEAM_NAME', 'LAST_GAME_VISITOR_TEAM_CITY1', 'LAST_GAME_VISITOR_TEAM_POINTS']


	# input date range
	def scrape(start_date, end_date):
		# These empty dataframe are repeatedly concatenated in the for loop below with the information from each days games
		nba_ingame_stats = pd.DataFrame(columns = NBAScraper.ingame_columns)
		nba_meta_stats = pd.DataFrame(columns = NBAScraper.meta_columns)
		nba_series_stats = pd.DataFrame(columns  = NBAScraper.series_columns)
		nba_last_meeting_stats = pd.DataFrame(columns = NBAScraper.last_meeting_columns)

		# range of dates
		dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%m/%d/%y')
		for date in dates:
			try:
				#headers = {'User-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Mobile Safari/537.36'}
				headers = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}
				request = requests.get("https://stats.nba.com/stats/scoreboard/", 
							params={'GameDate':date, 'LeagueID':'00', 'DayOffset':'0'}, 
							headers=headers)

				# check status code for OK response
				print("Getting " + date)
				print(request.status_code)
				json = request.json()

				# Meta info about game (e.g. who is hometeam)
				todays_meta_data = json['resultSets'][0]['rowSet']
				todays_meta_data_df = pd.DataFrame(todays_meta_data, columns=NBAScraper.meta_columns)
				nba_meta_stats = pd.concat([nba_meta_stats,todays_meta_data_df],ignore_index=True)

				# Info about the seriese between two teams
				todays_series_info = json['resultSets'][2]['rowSet']
				todays_series_info_df = pd.DataFrame(todays_series_info,columns = NBAScraper.series_columns)
				nba_series_stats = pd.concat([nba_series_stats,todays_series_info_df],ignore_index=True)

				# Info about the ingame statistics
				todays_stats = json['resultSets'][1]['rowSet']
				todays_stats_df = pd.DataFrame(todays_stats,columns=NBAScraper.ingame_columns)
				nba_ingame_stats = pd.concat([nba_ingame_stats,todays_stats_df],ignore_index=True)

				# Info about the lastgame statistics
				todays_last_meeting_stats = json['resultSets'][3]['rowSet']
				todays_last_meeting_stats_df = pd.DataFrame(todays_last_meeting_stats,columns=NBAScraper.last_meeting_columns)
				nba_last_meeting_stats = pd.concat([nba_last_meeting_stats,todays_last_meeting_stats_df],ignore_index=True)

			except requests.exceptions.ReadTimeout:
				print("=(")

		# combine all tables on game id and write to csv
		nba_df = nba_ingame_stats.merge(nba_last_meeting_stats, on='GAME_ID').merge(nba_meta_stats, on='GAME_ID').merge(nba_series_stats, on='GAME_ID')	
		nba_df.to_csv('nba_df.csv')	


