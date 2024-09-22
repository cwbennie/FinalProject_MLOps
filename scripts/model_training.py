import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Tuple
from pathlib import Path


def strat_kfold(model, X: pd.DataFrame, y: pd.DataFrame,
                num_splits: int = 5, random_state: int = 42) \
                    -> Tuple[float, float]:
    accs = list()
    f1_scores = list()
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True,
                          random_state=random_state)
    for _, (train_ind, val_ind) in enumerate(skf.split(X, y)):
        # fit model and make predictions
        model.fit(X.iloc[train_ind], y.iloc[train_ind])
        y_preds = model.predict(X.iloc[val_ind])

        # get accuracy for kth-fold
        accs.append(accuracy_score(y.iloc[val_ind], y_preds))

        # get F1 score for kth-fold
        f1_scores.append(np.mean(f1_score(y.iloc[val_ind], y_preds,
                                          average='weighted')))

    return np.mean(accs), np.mean(f1_scores)


def update_params(params: dict) -> Tuple[dict, dict]:
    """Function to update parameter dictionary for hyperopt
    objective function"""
    del params['exp_name']
    non_model_params = dict()
    nmp_names = ['type', 'train_data', 'y',
                 'random_state', 'num_splits']
    for key in nmp_names:
        non_model_params[key] = params[key]
        del params[key]

    return params, non_model_params


def mlflow_obj(params: dict) -> dict:
    with mlflow.start_run() as run:
        params, id_params = update_params(params=params)

        # instantiate the models and perform k-fold CV
        if id_params['type'] == 'decision_tree':
            clf = DecisionTreeClassifier(**params)
        elif id_params['type'] == 'random_forest':
            clf = RandomForestClassifier(**params)
        elif id_params['type'] == 'xgboost':
            clf = XGBClassifier(**params)
        else:
            return 0
        acc, f1 = strat_kfold(model=clf, X=id_params['train_data'],
                              y=id_params['y'],
                              num_splits=id_params['num_splits'],
                              random_state=id_params['random_state'])

        # log metrics and parameters
        mlflow.set_tag("Model", id_params['type'])
        mlflow.log_params(params)
        mlflow.log_param('kfold_random_state', 42)
        mlflow.log_metric("validation_acc", acc)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_artifacts('save_data/')

        # log models
        if id_params['type'] == 'xgboost':
            mlflow.xgboost.log_model(clf, artifact_path='better_models')
        else:
            mlflow.sklearn.log_model(clf, artifact_path='better_models')

        run_id = run.info.run_id

        mlflow.end_run()

    return {'loss': -acc, 'status': STATUS_OK, 'run_id': run_id}


def hp_tuning(exp_name: str, train_data: pd.DataFrame,
              train_y: pd.DataFrame, num_splits: int = 5,
              random_state: int = 42) -> Tuple[dict, str]:
    search_space = hp.choice('classifier_type', [
        {
            'type': 'decision_tree',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice(
                'dtree_max_depth', [None,
                                    hp.randint('dtree_max_depth_int', 1, 10)]),
            'min_samples_split': hp.randint('dtree_min_samples_split', 2, 10),
            'train_data': train_data, 'y': train_y, 'num_splits': num_splits,
            'random_state': random_state,
            'exp_name': exp_name
            },
        {
            'type': 'random_forest',
            'n_estimators': hp.randint('rf_n_estimators', 20, 200),
            'max_features': hp.randint('rf_max_features', 2, 9),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'train_data': train_data, 'y': train_y, 'num_splits': num_splits,
            'random_state': random_state,
            'exp_name': exp_name
            },
        {
            'type': 'xgboost',
            'max_depth': hp.randint('xgb_max_depth', 2, 20),
            'eta': hp.choice('xgb_eta', np.arange(0.1, 0.51, 0.1)),
            'subsample': hp.choice('xgb_subsample', np.arange(0.3, 0.81, 0.1)),
            'colsample_bynode': hp.choice('xgb_colsample_bynode',
                                          np.arange(0.3, 0.81, 0.1)),
            'n_estimators': hp.choice('xgb_n_estimators',
                                      np.arange(100, 301, 50)),
            'objective': 'multi:softmax',
            'train_data': train_data, 'y': train_y, 'num_splits': num_splits,
            'random_state': random_state,
            'exp_name': exp_name
            }])

    algo = tpe.suggest
    trials = Trials()
    best_result = fmin(fn=mlflow_obj, space=search_space,
                       algo=algo, max_evals=32, trials=trials)

    best_run = min(trials.results, key=lambda x: x['loss'])

    classifier_types = {0: 'decision_tree', 1: 'random_forest',
                        2: 'xgboost'}
    c_key = best_result['classifier_type']
    best_result['classifier_type'] = classifier_types[c_key]

    best_run_id = best_run['run_id']

    return best_result, best_run_id
