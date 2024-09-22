from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split


def get_result(row: pd.DataFrame) -> str:
    """
    Determine the result of a football match.

    Parameters:
    -----------
    row : pd.DataFrame
        A row of the dataframe containing match data including
          FTHG (Full-time Home Goals) and FTAG (Full-time Away Goals).

    Returns:
    --------
    str
        'H' if home team wins, 'A' if away team wins, 'D' if
        the match is a draw.
    """
    if row['FTHG'] > row['FTAG']:
        return 'H'
    elif row['FTHG'] < row['FTAG']:
        return 'A'
    else:
        return 'D'


def get_home_wins(row: pd.DataFrame) -> int:
    """
    Calculate the number of home wins in the last five matches.

    Parameters:
    -----------
    row : pd.DataFrame
        A row of the dataframe containing the last five home match results.

    Returns:
    --------
    int
        Number of home wins in the last five matches.
    """
    cols = ['HM1', 'HM2', 'HM3', 'HM4', 'HM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def get_away_wins(row: pd.DataFrame) -> int:
    """
    Calculate the number of away wins in the last five matches.

    Parameters:
    -----------
    row : pd.DataFrame
        A row of the dataframe containing the last five away match results.

    Returns:
    --------
    int
        Number of away wins in the last five matches.
    """
    cols = ['AM1', 'AM2', 'AM3', 'AM4', 'AM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def update_pipeline(train_data: pd.DataFrame, test_data: pd.DataFrame,
                    train_y: pd.DataFrame, chi2pct: int = 75) -> \
                    Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create and apply a preprocessing pipeline that scales numerical features,
    encodes categorical features, and selects features based on
    chi-squared tests.

    Parameters:
    -----------
    train_data : pd.DataFrame
        The training data.
    test_data : pd.DataFrame
        The testing data.
    train_y : pd.DataFrame
        The target variable for the training data.
    chi2pct : int, default=75
        Percentile of features to keep based on the chi-squared test.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Transformed training and testing data as dataframes.
    """
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
    """
    Update standings to include historical standings for teams not present
    in the current standings data.

    Parameters:
    -----------
    standings : pd.DataFrame
        The current standings dataframe.

    Returns:
    --------
    pd.DataFrame
        Updated standings dataframe with historical data filled in.
    """
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
    """
    Retrieve the standing of a team from the previous year.

    Parameters:
    -----------
    row : pd.Series
        A row of match data.
    standings : pd.DataFrame
        DataFrame containing the historical standings of teams.
    home : bool, default=True
        Whether to retrieve the home team standing. Set to False for away team.

    Returns:
    --------
    float
        The team's standing from the previous year. If not found, return 21.0.
    """
    col = 'HomeTeam' if home else 'AwayTeam'
    try:
        yr = str(int(row['Year']) - 1)
        team = row[col]
        standing = standings.loc[standings['Team'] == team, yr]
        return standing.iloc[0]
    except (IndexError, KeyError):
        return 21.0


def get_year(row):
    """
    Determine the year based on the match date.

    Parameters:
    -----------
    row : str
        A string representing the date of the match in the format 'DD/MM/YY'.

    Returns:
    --------
    str
        The year in which the match took place, based on the month and year.
    """
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
    """
    Create new columns for EPL match data, including results,
    home and away wins, and previous standings.

    Parameters:
    -----------
    epl_data : pd.DataFrame
        The EPL match data.
    standings : pd.DataFrame
        The standings data for teams.

    Returns:
    --------
    pd.DataFrame
        The updated EPL data with new columns.
    """
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
    """
    Update and encode target columns for training and testing sets.

    Parameters:
    -----------
    train_target : pd.DataFrame
        The target variable for the training data.
    test_target : pd.DataFrame
        The target variable for the testing data.

    Returns:
    --------
    pd.DataFrame
        Encoded training and testing target variables.
    """
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
    """
    Process EPL match data by adding new columns, normalizing features,
    and preparing target variables.

    Parameters:
    -----------
    epl_data : pd.DataFrame
        The EPL match data.
    standings_data : pd.DataFrame
        DataFrame containing historical standings.
    feature_percentile : int
        Percentile of features to keep based on the chi-squared test.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The processed training data, testing data, and updated standings.
    """
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
