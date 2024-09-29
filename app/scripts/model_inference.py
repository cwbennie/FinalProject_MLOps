from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def predict(model, test_data: pd.DataFrame, test_y: pd.DataFrame,
            partial: bool = True, test_size: float = 0.2) \
                -> Tuple[pd.DataFrame, float, float]:
    if partial:
        _, test_set, _, test_y = train_test_split(test_data, test_y,
                                                  test_size=test_size)
    else:
        test_set = test_data

    predictions = model.predict(test_set)
    accuracy = accuracy_score(test_y, predictions)
    mod_f1 = f1_score(test_y, predictions, average='weighted')

    return predictions, accuracy, mod_f1


def get_recent_results(team_name: str, data: pd.DataFrame):
    try:
        away_res = data.loc[data[f'AwayTeam_{team_name}'] == 1].tail(1)
    except KeyError:
        away_res = None
    try:
        home_res = data.loc[data[f'HomeTeam_{team_name}'] == 1].tail(1)
    except KeyError:
        home_res = None

    if away_res is None or home_res is None:
        return away_res or home_res

    if away_res.index[0] > home_res.index[0]:
        return away_res
    else:
        return home_res


def create_team_cols(column_list) -> Tuple[List, List, List]:

    home_cols = list()

    for col in column_list:
        if col.startswith('HT') or col.startswith('HM'):
            home_cols.append(col)
        elif col in ['DiffPts', 'DiffFormPts', 'HomeWins',
                     'HomeStanding', 'MW']:
            home_cols.append(col)

    away_cols = list()
    for col in column_list:
        if col.startswith('AT') or col.startswith('AM'):
            away_cols.append(col)
        elif col in ['AwayWins', 'AwayStanding']:
            away_cols.append(col)

    all_cols = home_cols + away_cols
    return home_cols, away_cols, all_cols


def create_prediction_row(home_team: str, away_team: str,
                          data: pd.DataFrame) -> pd.DataFrame:

    home_res = get_recent_results(home_team, data)
    away_res = get_recent_results(away_team, data)

    home_col = f'HomeTeam_{home_team}'
    away_col = f'AwayTeam_{away_team}'

    home_cols, away_cols, all_cols = create_team_cols(list(data.columns))
    all_cols.extend([home_col, away_col])

    new_row = {}
    for col in home_cols:
        try:
            new_row[col] = home_res[col].values[0]
        except KeyError:
            continue
    for col in away_cols:
        try:
            new_row[col] = away_res[col].values[0]
        except KeyError:
            continue

    new_row[home_col] = 1
    new_row[away_col] = 1

    for col in set(data.columns).difference(set(all_cols)):
        new_row[col] = 0

    pred_row = pd.DataFrame([new_row])
    pred_row = pred_row[data.columns]
    return pred_row
