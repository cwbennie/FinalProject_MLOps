import pandas as pd
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.model_selection import train_test_split


def get_result(row):
    """Function to return the result of the match"""
    if row['FTHG'] > row['FTAG']:
        return 'H'
    elif row['FTHG'] < row['FTAG']:
        return 'A'
    else:
        return 'D'


def get_home_wins(row):
    cols = ['HM1', 'HM2', 'HM3', 'HM4', 'HM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def get_away_wins(row):
    cols = ['AM1', 'AM2', 'AM3', 'AM4', 'AM5']
    wins = [1 for el in cols if row[el] == 'W']
    return len(wins)


def update_cols(data):
    # label encoding for categorical features
    cols = [col for col in data.columns if data[col].dtype == 'object']
    le_dict = dict()

    for col in cols:
        label_enc = LabelEncoder()
        data.loc[:, col] = label_enc.fit_transform(data[col])
        # final_epl = final_epl.astype({col: int})
        le_dict[col] = label_enc

    return data, le_dict


def process_data(data_path, feature_percentile):
    data = pd.read_csv(data_path)

    # remove extra column
    data.drop(columns=['Unnamed: 0'], inplace=True)

    # update data to include Result column
    data['Result'] = data.apply(get_result, axis=1)
    y = data['Result']

    # update data to include HomeWins and AwayWins
    data['HomeWins'] = data.apply(get_home_wins, axis=1)
    data['AwayWins'] = data.apply(get_away_wins, axis=1)

    # drop columns that are too cardinal (*FormPtsStr)
    data = data.drop(columns=['HTFormPtsStr', 'ATFormPtsStr', 'Result'])

    # encode categorical columns
    proc_data, encoders = update_cols(data)

    feature_selector = SelectPercentile(score_func=f_classif,
                                        percentile=feature_percentile)

    proc_data = feature_selector.fit_transform(proc_data, y)
    encoders['feature_selector'] = feature_selector

    proc_data = pd.DataFrame(proc_data)

    proc_data['Result'] = y

    train_data, test_data = train_test_split(proc_data, test_size=0.2,
                                             shuffle=True)

    return train_data, test_data, encoders


def save_data(train: pd.DataFrame, train_path: str,
              test: pd.DataFrame, test_path: str,
              pipe: Pipeline, pipe_name: str):
    train.to_csv(train_path)
    test.to_csv(test_path)

    # save pipe
    with open(pipe_name, 'wb') as file:
        pickle.dump(pipe, file)


if __name__ == '__main__':

    params = yaml.safe_load(open("params.yaml"))["features"]
    data_path = params['data_path']
    chi2pct = params['chi2percentile']

    train_df, test_df, encoders = process_data(data_path, chi2pct)
    save_data(train_df, 'data/processed_train.csv',
              test_df, 'data/processed_test.csv',
              encoders, 'data/pipeline.pkl')
