import pandas as pd
import numpy as np 





# Read in data
#ingames = pd.read_csv('/home/bigpandas/nba_ingame.csv')
#lastmeeting = pd.read_csv('/home/bigpandas/nba_last_meeting.csv')
#meta = pd.read_csv('/home/bigpandas/nba_meta.csv')
#series = pd.read_csv('/home/bigpandas/nba_series.csv')
#
## Merge CSVs
#nba_df = ingames.merge(lastmeeting, on ='GAME_ID').merge(meta, on = 'GAME_ID').merge(series, on = 'GAME_ID')


def process_data(nba_df):

	# this dictionary should hold our data that can be fed into a model, our targets, and a dataframe with more complete information
	data_dict = {}

	# Subset columns
	nba_df = nba_df[['GAME_DATE_EST_x','GAME_ID','TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY_NAME','PTS_QTR1', 
	       'PTS_QTR2', 'PTS_QTR3', 'PTS_QTR4','PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST',
	       'REB', 'TOV', 'SEASON','HOME_TEAM_WINS', 'HOME_TEAM_LOSSES','HOME_TEAM_ID_y', 'VISITOR_TEAM_ID_y']]

	# Get the rows that have more than one na
	null_rows = nba_df[nba_df.isna().sum(axis=1)>0]
	# Get game ids of the rows with nulls in them
	null_GAME_IDS = null_rows.GAME_ID.unique()


	def testGameIds(id):
	    for gid in null_GAME_IDS:
	        if gid == id:
	            return False
	    return True


	# Create Boolean array where a value is False if there is a null in it
	bools = nba_df['GAME_ID'].aggregate(testGameIds)

	# Now there are no nan's in our df
	nba_df = nba_df[bools] 

	# Change game dates to panda data time stamps
	nba_df['GAME_DATE_EST_x'] = nba_df['GAME_DATE_EST_x'].aggregate(pd.to_datetime)
	nba_df = nba_df.set_index('GAME_DATE_EST_x')

	# games_back specifies how many days of games we want to include in our averages
	games_back = 5

	# Calculates average of ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST','REB', 'TOV'] over the last "games_back" games
	rolling_averages= nba_df.groupby(['TEAM_ID'])['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST','REB', 'TOV'].rolling(window=games_back).mean()
	rolling_averages = rolling_averages.dropna()

	# Drop columns ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST','REB', 'TOV'] so they can be replaced with their averages
	nba_df_w_o_stats = nba_df.drop(columns=['FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST','REB', 'TOV', 'PTS_QTR1', 'PTS_QTR2', 'PTS_QTR3', 'PTS_QTR4'])
	nba_df_w_o_stats = nba_df_w_o_stats.rename(columns={'PTS':'Game PTS'})

	rolling_averages.reset_index()
	# Fill in avg's of ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST','REB', 'TOV'] back into our dataframe
	nba_df_w_avg_stats = nba_df_w_o_stats.merge(rolling_averages, on=['TEAM_ID','GAME_DATE_EST_x'])

	# Create a boolean column that represents whether or not a team is the home team
	nba_df_w_avg_stats['HomeTeam'] = nba_df_w_avg_stats['HOME_TEAM_ID_y'] == nba_df_w_avg_stats['TEAM_ID']

	# Get average points from each game
	temp = nba_df_w_avg_stats.groupby('GAME_ID')['Game PTS'].mean().to_frame()
	temp = temp.reset_index()
	temp.columns=['GAME_ID', 'AVG PTS']

	# Merge so that average pts are now in each row
	nba_df_w_avg_stats = nba_df_w_avg_stats.merge(temp, on='GAME_ID')

	# Sort values by Game ID. Now, each team's information from each game are next to each other
	nba_df_w_avg_stats.sort_values('GAME_ID')

	dropped_rows = nba_df_w_avg_stats[nba_df_w_avg_stats['Game PTS'] == nba_df_w_avg_stats['AVG PTS']]['GAME_ID']
	nba_df_w_avg_stats = nba_df_w_avg_stats.set_index('GAME_ID')

	nba_df_w_avg_stats = nba_df_w_avg_stats.drop(np.array(dropped_rows))

	nba_df_w_avg_stats['Winner'] = nba_df_w_avg_stats['AVG PTS'] < nba_df_w_avg_stats['Game PTS']

	# Split data in alternating steps
	# This is ok bc the two vectors from each game are always right next to each other bc of our sort on GAME_ID
	firstHalf = nba_df_w_avg_stats.iloc[0::2]
	secondfirstHalf = nba_df_w_avg_stats.iloc[1::2]

	# Concatenates the two vectors from each game so that the y make one vector
	trainable_data = firstHalf.merge(secondfirstHalf, on='GAME_ID')

	data_dict['complete info'] = trainable_data

	# Drop superfluous information
	trainable_data = trainable_data.drop(columns=['TEAM_ABBREVIATION_x', 'TEAM_CITY_NAME_x', 'Game PTS_x',
       'SEASON_x','HOME_TEAM_ID_y_x', 'VISITOR_TEAM_ID_y_x','HOME_TEAM_ID_y_y','SEASON_y', 'TEAM_CITY_NAME_y'
		,'TEAM_ABBREVIATION_y', 'Game PTS_y', 'VISITOR_TEAM_ID_y_y'])



	labels = pd.concat([trainable_data['Winner_x'],trainable_data['Winner_y']], axis = 1)
	labels = np.array(labels.astype(int))



	trainable_data = trainable_data.drop(['TEAM_ID_x','TEAM_ID_y','Winner_x', 'Winner_y','AVG PTS_x','AVG PTS_y',
                                           'HOME_TEAM_LOSSES_y', 'HOME_TEAM_WINS_x'], axis = 1)
	data_dict['model_input'] = trainable_data
	data_dict['labels'] = labels

	return data_dict





