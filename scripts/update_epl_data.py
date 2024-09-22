import pickle
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split


def get_result(row: pd.DataFrame) -> str:
    """Function to return the result of the match"""
    if row['FTHG'] > row['FTAG']:
        return 'H'
    elif row['FTHG'] < row['FTAG']:
        return 'A'
    else:
        return 'D'


def get_home_wins(row: pd.DataFrame) -> int:
    cols = ['HM1', 'HM2', 'HM3', 'HM4', 'HM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def get_away_wins(row: pd.DataFrame) -> int:
    cols = ['AM1', 'AM2', 'AM3', 'AM4', 'AM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def update_cols(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # label encoding for categorical features
    cols = [col for col in data.columns if data[col].dtype == 'object']
    le_dict = dict()

    for col in cols:
        label_enc = LabelEncoder()
        data.loc[:, col] = label_enc.fit_transform(data[col])
        le_dict[col] = label_enc

    return data, le_dict


def update_standings(standings: pd.DataFrame) -> pd.DataFrame:
    addl_standings = pd.DataFrame(
        {'Team': ['Man United', 'Arsenal', 'Leeds',
                  'Liverpool', 'Chelsea',
                  'Aston Villa', 'Sunderland', 'Leicester',
                  'West Ham', 'Tottenham',
                  'Newcastle', 'Middlesbrough', 'Everton',
                  'Coventry', 'Southampton',
                  'Derby', 'Bradford', 'Wimbledon', 
                  'Sheffield Wednesday', 'Watford'],
         '1999': np.arange(1.0, 21.0, 1.0)}
         )
    standings = addl_standings.merge(right=standings, how='right',
                                     left_on='Team', right_on='Team')
    standings = standings.fillna(21)
    return standings


def get_away_hist(row: pd.Series, standings: pd.DataFrame) -> float:
    try:
        yr = str(int(row['Year']) - 1)
        team = row['AwayTeam']
        standing = standings.loc[standings['Team'] == team, yr]
        return standing.iloc[0]
    except (IndexError, KeyError):
        return 21.0


def get_home_hist(row: pd.Series, standings: pd.DataFrame) -> float:
    try:
        yr = str(int(row['Year']) - 1)
        team = row['HomeTeam']
        standing = standings.loc[standings['Team'] == team, yr]
        return standing.iloc[0]
    except (IndexError, KeyError):
        return 21.0


def get_year(row):
    """Function to be used with EPL Standings to track previous year's
    EPL Standing for each team"""
    month = row.split('/')[1]
    yr = row.split('/')[2]
    if len(yr) > 2:
        yr = yr[-2:]
    # if the month is between Aug-Dec, return same year
    if int(month) >= 8:
        return '20' + yr
    # Otherwise the season is recorded as previous year
    else:
        yr = '20' + str(int(yr) - 1)
        if len(yr) == 3:
            yr = yr[:2] + '0' + yr[-1]
        return yr


def process_data(data_path: str, standings_path: str,
                 feature_percentile: int):
    data = pd.read_csv(data_path)
    standings = pd.read_csv(standings_path)
    standings = update_standings(standings=standings)

    # remove extra column
    data.drop(columns=['Unnamed: 0'], inplace=True)

    # update data to include Result column
    data['Result'] = data.apply(get_result, axis=1)
    y = data['Result']

    # update data to include HomeWins and AwayWins
    data['HomeWins'] = data.apply(get_home_wins, axis=1)
    data['AwayWins'] = data.apply(get_away_wins, axis=1)
    data['Year'] = data['Date'].apply(get_year)

    # update data to include previous year rankings
    data['HomeStanding'] = data.apply(lambda row:
                                      get_home_hist(row, standings=standings),
                                      axis=1)
    data['AwayStanding'] = data.apply(lambda row:
                                      get_away_hist(row, standings=standings),
                                      axis=1)

    # drop columns that are too cardinal (*FormPtsStr)
    data = data.drop(columns=['HTFormPtsStr', 'ATFormPtsStr', 'Result'])
    # drop columns that can indicate the result
    data = data.drop(columns=['FTHG', 'FTAG', 'FTR'])

    # encode categorical columns
    proc_data, encoders = update_cols(data)

    feature_selector = SelectPercentile(score_func=f_classif,
                                        percentile=feature_percentile)

    proc_data = feature_selector.fit_transform(proc_data, y)

    feats = feature_selector.feature_names_in_
    out_feats = feature_selector.get_feature_names_out()
    with open('features.txt', 'w') as f:
        for feat in feats:
            f.write(f"{feat}\n")
    with open('selected_feats.txt', 'w') as f:
        for feat in out_feats:
            f.write(f"{feat}\n")

    encoders['feature_selector'] = feature_selector

    proc_data = pd.DataFrame(proc_data, columns=out_feats)

    proc_data['Result'] = y

    train_data, test_data = train_test_split(proc_data, test_size=0.2,
                                             shuffle=True)

    return train_data, test_data, standings, encoders


def save_data(train: pd.DataFrame, train_path: str,
              test: pd.DataFrame, test_path: str,
              standing_df: pd.DataFrame, standings_new: str,
              pipe: Pipeline, pipe_name: str):
    train.to_csv(train_path)
    test.to_csv(test_path)
    standing_df.to_csv(standings_new)

    # save pipe
    with open(pipe_name, 'wb') as file:
        pickle.dump(pipe, file)


if __name__ == '__main__':

    params = yaml.safe_load(open("params.yaml"))["features"]
    data_path = params['data_path']
    chi2pct = params['chi2percentile']
    standings_path = params['standings_path']

    train_df, test_df, standings, encoders = process_data(
        data_path=data_path, standings_path=standings_path,
        feature_percentile=chi2pct)

    save_data(train_df, 'data/processed_train.csv',
              test_df, 'data/processed_test.csv',
              standings, 'data/processed_standings.csv',
              encoders, 'data/pipeline.pkl')
