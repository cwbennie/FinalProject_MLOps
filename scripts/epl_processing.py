import pandas as pd
import numpy as np
import pickle
import yaml
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split
from sklearn import set_config


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


def update_pipeline(train_data: pd.DataFrame, test_data: pd.DataFrame,
                    train_y: pd.DataFrame, chi2pct: int = 75) -> \
                    Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a pipeline to transform necessary columns in the dataframe"""
    # set config of Pipeline to pass column names through
    # set_config(transform_output='pandas')
    # Pipeline object to transform numerical columns
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )
    # Pipeline object to transform categorical columns
    cat_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore',
                                         drop='if_binary')),
               ('selector', SelectPercentile(chi2, percentile=chi2pct))]
    )
    # Column Transformer and Full Pipeline
    column_transform = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,
             make_column_selector(dtype_include=['float', 'int'])),
            ('cat', cat_transformer,
             make_column_selector(dtype_include=['object']))
        ],
        verbose_feature_names_out=False
    )

    processor = Pipeline(
        steps=[('processor', column_transform)],
    )

    processor.fit(train_data, train_y)
    cols = processor.get_feature_names_out()

    train_data = processor.transform(train_data)
    test_data = processor.transform(test_data)

    train_data = pd.DataFrame.sparse.from_spmatrix(train_data, columns=cols)
    test_data = pd.DataFrame.sparse.from_spmatrix(test_data, columns=cols)

    return train_data, test_data


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


def get_match_history(row: pd.Series, standings: pd.DataFrame,
                      home: bool = True) -> float:
    col = 'HomeTeam' if home else 'AwayTeam'
    try:
        yr = str(int(row['Year']) - 1)
        team = row[col]
        standing = standings.loc[standings['Team'] == team, yr]
        return standing.iloc[0]
    except (IndexError, KeyError):
        return 21.0


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


def create_new_columns(epl_data: pd.DataFrame, standings: pd.DataFrame) \
      -> pd.DataFrame:
    # update data to include Result column
    epl_data['Result'] = epl_data.apply(get_result, axis=1)

    # update data to include HomeWins and AwayWins
    epl_data['HomeWins'] = epl_data.apply(get_home_wins, axis=1)
    epl_data['AwayWins'] = epl_data.apply(get_away_wins, axis=1)
    epl_data['Year'] = epl_data['Date'].apply(get_year)

    # update data to include previous year rankings
    epl_data['HomeStanding'] = epl_data.apply(
        lambda row: get_match_history(row, standings=standings,
                                      home=True), axis=1)
    epl_data['AwayStanding'] = epl_data.apply(
        lambda row: get_match_history(row, standings=standings,
                                      home=False), axis=1)

    return epl_data


def update_targets(train_target: pd.DataFrame,
                   test_target: pd.DataFrame) -> pd.DataFrame:
    label_enc = LabelEncoder()
    train_target.reset_index(inplace=True)
    train_target.drop(columns='index', inplace=True)
    test_target.reset_index(inplace=True)
    test_target.drop(columns='index', inplace=True)

    train_target['Result'] = label_enc.fit_transform(train_target['Result'])
    test_target['Result'] = label_enc.transform(test_target['Result'])

    return train_target['Result'], test_target['Result']


def process_data(epl_data: pd.DataFrame, standings_data: pd.DataFrame,
                 feature_percentile: int):
    # update the standings to reflect accurate data for all years
    standings = update_standings(standings=standings_data)

    epl_data = create_new_columns(epl_data=epl_data, standings=standings)

    # save 'Result' column to be used as y variable
    y = epl_data[['Result']]

    # drop columns that are too cardinal (*FormPtsStr)
    epl_data = epl_data.drop(
        columns=['HTFormPtsStr', 'ATFormPtsStr'])
    # drop date columns
    epl_data = epl_data.drop(columns=['Year', 'Date'])
    # drop columns that can indicate the result
    epl_data = epl_data.drop(columns=['FTHG', 'FTAG', 'FTR', 'Result'])

    # train-test split on the data
    train_data, test_data, train_y, test_y = train_test_split(epl_data, y,
                                                              test_size=0.2,
                                                              shuffle=True)

    # normalize and encode columns columns
    train_data, test_data = update_pipeline(train_data=train_data,
                                            test_data=test_data,
                                            train_y=train_y,
                                            chi2pct=feature_percentile)

    # update and encode target columns
    train_y, test_y = update_targets(train_y, test_y)

    train_data['Result'] = train_y
    test_data['Result'] = test_y

    return train_data, test_data, standings


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

# if __name__ == '__main__':

#     train_df, test_df, standings = process_data(
#         epl_data=data_path, standings_path=standings_path,
#         feature_percentile=chi2pct)

#     save_data(train_df, 'data/processed_train.csv',
#               test_df, 'data/processed_test.csv',
#               standings, 'data/processed_standings.csv',
#               encoders, 'data/pipeline.pkl')
