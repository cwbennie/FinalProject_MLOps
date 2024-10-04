from typing import List
import os
from fastapi import FastAPI
import uvicorn
import mlflow
import pickle
import scripts.model_inference as model_inf
import pandas as pd
import numpy as np
from pydantic import BaseModel


app = FastAPI(
    title="EPL Match Outcome Predictions",
    description="Create a prediction for which team will win a given match.",
    version="0.1",
)


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for predicting outcomes of EPL matches'}


class request_body(BaseModel):
    teams: List[str]


@app.on_event('startup')
# want to load the model here
def load_artifacts():
    global epl_model, team_history, encoder, valid_teams
    RUN_ID = os.environ['MLFLOW_RUNID']
    EXP_NAME = os.environ['MLFLOW_EXPNAME']
    mlflow.set_tracking_uri('http://192.168.5.6:8080')
    mlflow.set_experiment(EXP_NAME)

    model_uri = f'runs:/{RUN_ID}/better_models'

    epl_model = mlflow.sklearn.load_model(model_uri)
    mlflow.artifacts.download_artifacts(run_id=RUN_ID,
                                        artifact_path='x_train.pkl',
                                        dst_path='mlflow_data')

    with open('mlflow_data/x_train.pkl', 'rb') as f:
        team_history = pickle.load(f)

    with open('mlflow_data/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    valid_teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Chelsea',
                   'Crystal Palace', 'Everton', 'Leicester', 'Liverpool',
                   'Man City', 'Man United', 'Newcastle', 'Norwich',
                   'Southampton', 'Stoke', 'Sunderland', 'Swansea',
                   'Tottenham', 'Watford', 'West Brom', 'West Ham']


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data: request_body):
    # function to predict the outcome of a match
    # want to take two teams in, get their most recent stats and history
    assert len(data.teams) == 2
    home_team, away_team = data.teams

    assert home_team in valid_teams and away_team in valid_teams

    # function to lookup most recent stats for teams
    test_data = model_inf.create_prediction_row(home_team=home_team,
                                                away_team=away_team,
                                                data=team_history)

    model_out = epl_model.predict_proba(test_data)

    chk = encoder.inverse_transform([np.argmax(model_out)])

    if chk[0] == 'H':
        pred = 'Home'
        win_team = home_team
    elif chk[0] == 'A':
        pred = 'Away'
        win_team = away_team
    else:
        pred = 'Draw'
        win_team = 'Draw'

    model_pred = pd.DataFrame([{'Result': pred,
                                'Winning Team': win_team,
                                'Win Probability': max(model_out[0].tolist())}])
    model_pred.to_csv('predictions/new_prediction.csv')

    # generate prediction
    return {'Predictions': pred,
            'Probabilities': model_out[0].tolist()}
