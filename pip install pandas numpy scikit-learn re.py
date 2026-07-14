import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Step 3: Fetch and preprocess data
nba_teams = teams.get_teams()
team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}
all_games = pd.DataFrame()

for team in nba_teams:
    team_id = team['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    all_games = pd.concat([all_games, games], ignore_index=True)

# Display the columns to ensure we have the correct column names
print(all_games.columns)

# Step 4: Preprocess data
all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
all_games['PTS'] = all_games['PTS'].astype(float)
all_games['Points_Per_Game'] = all_games.groupby('TEAM_ID')['PTS'].transform('mean')

     
Index(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'],
      dtype='object')

# Derive OPPONENT_TEAM_ID from MATCHUP
def get_opponent_team_id(matchup, team_abbr_to_id, team_id):
    if '@' in matchup:
        opponent_abbr = matchup.split(' @ ')[-1]
    else:
        opponent_abbr = matchup.split(' vs. ')[-1]
    return team_abbr_to_id.get(opponent_abbr, team_id)

all_games['OPPONENT_TEAM_ID'] = all_games.apply(
    lambda row: get_opponent_team_id(row['MATCHUP'], team_abbr_to_id, row['TEAM_ID']), axis=1
)

# Add HOME_GAME feature
all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

# Add LAST_GAME_RESULT feature
all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)


print(all_games.columns)

le = LabelEncoder()
all_games['TEAM_ID'] = le.fit_transform(all_games['TEAM_ID'])
all_games['OPPONENT_TEAM_ID'] = le.fit_transform(all_games['OPPONENT_TEAM_ID'])


     
Index(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'WIN',
       'Points_Per_Game', 'OPPONENT_TEAM_ID', 'HOME_GAME', 'LAST_GAME_RESULT'],
      dtype='object')

# Step 6: Split data
X = all_games[['TEAM_ID', 'OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME', 'LAST_GAME_RESULT']]
y = all_games['WIN']  # Target variable (win/loss)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
     
RandomForestClassifier(random_state=42)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
     
Accuracy: 0.5959557781347444
precision    recall  f1-score   support

           0       0.59      0.61      0.60     10525
           1       0.60      0.58      0.59     10641

    accuracy                           0.60     21166
   macro avg       0.60      0.60      0.60     21166
weighted avg       0.60      0.60      0.60     21166


from nba_api.stats.static import teams

# Fetch all NBA teams
nba_teams = teams.get_teams()

# Extract team abbreviations
team_abbreviations = [team['abbreviation'] for team in nba_teams]

# Print all team abbreviations
print("Team Abbreviations:", team_abbreviations)
     
Team Abbreviations: ['ATL', 'BOS', 'CLE', 'NOP', 'CHI', 'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MIA', 'MIL', 'MIN', 'BKN', 'NYK', 'ORL', 'IND', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'OKC', 'TOR', 'UTA', 'MEM', 'WAS', 'DET', 'CHA']

feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)
     
Feature Importances:
                   importance
OPPONENT_TEAM_ID    0.492184
HOME_GAME           0.292690
Points_Per_Game     0.092389
TEAM_ID             0.079427
LAST_GAME_RESULT    0.043310

team_abbr = 'LAL'
opponent_abbr = 'BOS'
average_points_per_game = 110.5  # Replace with the actual average points per game

new_data = pd.DataFrame({
    'TEAM_ID': [le.transform([team_abbr_to_id[team_abbr]])[0]],
    'OPPONENT_TEAM_ID': [le.transform([team_abbr_to_id[opponent_abbr]])[0]],
    'Points_Per_Game': [average_points_per_game],
    'HOME_GAME': [1],  # Replace with 1 if home game, else 0
    'LAST_GAME_RESULT': [1]  # Replace with last game result (1 if win, 0 if loss)
})

predictions = model.predict(new_data)
prediction_probabilities = model.predict_proba(new_data)

print("Predictions: ", predictions)
print("Prediction Probabilities: ", prediction_probabilities)
     
Predictions:  [0]
Prediction Probabilities:  [[0.50777377 0.49222623]]
