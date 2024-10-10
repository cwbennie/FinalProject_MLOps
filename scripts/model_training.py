from typing import Tuple, List
import os
import pickle
import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold, \
    RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def strat_kfold(model, X: pd.DataFrame, y: pd.DataFrame,
                num_splits: int = 5, random_state: int = 42) \
                    -> Tuple[float, float]:
    """
    Perform stratified k-fold cross-validation on the given model.

    Parameters:
    -----------
    model : estimator object
        The machine learning model to be evaluated
        (e.g., DecisionTreeClassifier).
    X : pd.DataFrame
        The feature data used for training and validation.
    y : pd.DataFrame
        The target labels.
    num_splits : int, default=5
        The number of splits for cross-validation.
    random_state : int, default=42
        Random seed for shuffling the data.

    Returns:
    --------
    Tuple[float, float]
        A tuple containing the mean accuracy and mean F1-score
        across all folds.
    """
    # create lists to track accuracy and f1 scores
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
    """
    Splits the parameter dictionary into model-specific and
    non-model parameters.

    Parameters:
    -----------
    params : dict
        The parameter dictionary that includes both model and
        non-model parameters.

    Returns:
    --------
    Tuple[dict, dict]
        A tuple containing two dictionaries:
        - The updated model parameter dictionary.
        - The non-model parameter dictionary (e.g., train data, labels, etc.).
    """
    # create new dictionary to track non-model parameters for training loops
    non_model_params = dict()
    nmp_names = ['type', 'train_data', 'y',
                 'random_state', 'num_splits']
    for key in nmp_names:
        non_model_params[key] = params[key]
        del params[key]

    return params, non_model_params


def mlflow_obj(params: dict) -> dict:
    """
    Train a model using the given parameters and log the results to MLflow.

    Parameters:
    -----------
    params : dict
        The hyperparameter dictionary used to configure and train the model.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'loss': The negative accuracy for minimization by Hyperopt.
        - 'status': The status of the run (e.g., STATUS_OK).
        - 'run_id': The MLflow run ID associated with the experiment.
    """
    with mlflow.start_run() as run:
        # update parameter dictionary to use in training
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
        os.makedirs('save_data', exist_ok=True)
        with open('save_data/x_train.pkl', 'wb') as fname:
            pickle.dump(id_params['train_data'], fname)

        mlflow.log_artifact('save_data/x_train.pkl')

        # log models
        if id_params['type'] == 'xgboost':
            mlflow.xgboost.log_model(clf, artifact_path='better_models')
        else:
            mlflow.sklearn.log_model(clf, artifact_path='better_models')

        # retrieve run_id to be used in logging best model
        run_id = run.info.run_id

        mlflow.end_run()

    return {'loss': -acc, 'status': STATUS_OK, 'run_id': run_id}


def hp_tuning(train_data: pd.DataFrame, train_y: pd.DataFrame,
              num_splits: int = 5, random_state: int = 42) -> Tuple[dict, str]:
    """
    Perform hyperparameter tuning using Hyperopt and log experiments in MLflow.

    Parameters:
    -----------
    exp_name : str
        The experiment name for MLflow.
    train_data : pd.DataFrame
        The feature data used for training.
    train_y : pd.DataFrame
        The target labels for training.
    num_splits : int, default=5
        The number of splits for cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[dict, str]
        - A dictionary with the best hyperparameters found by Hyperopt.
        - The MLflow run ID for the best run.
    """
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
            },
        {
            'type': 'random_forest',
            'n_estimators': hp.randint('rf_n_estimators', 20, 500),
            'max_features': hp.randint('rf_max_features', 2, 9),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'train_data': train_data, 'y': train_y, 'num_splits': num_splits,
            'random_state': random_state,
            },
        {
            'type': 'xgboost',
            'max_depth': hp.randint('xgb_max_depth', 2, 20),
            'eta': hp.choice('xgb_eta', np.arange(0.1, 0.51, 0.1)),
            'subsample': hp.choice('xgb_subsample', np.arange(0.3, 0.81, 0.1)),
            'colsample_bynode': hp.choice('xgb_colsample_bynode',
                                          np.arange(0.3, 0.81, 0.1)),
            'n_estimators': hp.choice('xgb_n_estimators',
                                      np.arange(100, 1001, 50)),
            'objective': 'multi:softmax',
            'train_data': train_data, 'y': train_y, 'num_splits': num_splits,
            'random_state': random_state,
            }])

    algo = tpe.suggest
    trials = Trials()
    best_result = fmin(fn=mlflow_obj, space=search_space,
                       algo=algo, max_evals=32, trials=trials)

    # access the best run to get the run_id
    best_run = min(trials.results, key=lambda x: x['loss'])
    best_run_id = best_run['run_id']

    # update result dictionary to include name of classifier
    classifier_types = {0: 'decision_tree', 1: 'random_forest',
                        2: 'xgboost'}
    c_key = best_result['classifier_type']
    best_result['classifier_type'] = classifier_types[c_key]

    return best_result, best_run_id


def get_param_grids(random_state: int = 42) -> List[Tuple[object, dict]]:

    classifiers = [DecisionTreeClassifier(), RandomForestClassifier(),
                   XGBClassifier()]
    clf_names = ['decision_tree', 'random_forest', 'xgboost']
    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None] + list(range(1, 11)),
        'min_samples_split': range(2, 10),
        'random_state': [random_state]
    }
    rf_params = {
        'criterion': ['gini', 'entropy'],
        'max_features': range(2, 10),
        'n_estimators': range(20, 501, 40),
        'max_depth': range(2, 10),
        'random_state': [random_state]
    }
    xgb_params = {
        'objective': ['multi:softmax'],
        'max_depth': range(2, 10),
        'eta': np.arange(0.1, 0.51, 0.1),
        'colsample_bynode': np.arange(0.3, 0.81, 0.1),
        'n_estimators': range(100, 1001, 100),
        'random_state': [random_state]
    }
    param_grids = [dt_params, rf_params, xgb_params]
    return list(zip(classifiers, param_grids, clf_names))


def search_cv(train_data: pd.DataFrame, train_y: pd.DataFrame,
              clf_info: tuple, scoring: dict, grid: bool = True,
              num_splits: int = 5, random_state: int = 42) -> Tuple[dict, str]:
    # warnings.filterwarnings("ignore")
    clf, clf_grid, clf_name = clf_info
    if grid:
        search = GridSearchCV(estimator=clf, param_grid=clf_grid,
                              cv=num_splits, n_jobs=2,
                              scoring=scoring, refit='accuracy')
    else:
        search = RandomizedSearchCV(estimator=clf,
                                    param_distributions=clf_grid,
                                    scoring=scoring, n_jobs=2,
                                    cv=num_splits, random_state=random_state,
                                    refit='accuracy', n_iter=5)

    train_x, val_x, tmp_y, val_y = train_test_split(train_data, train_y,
                                                    test_size=0.2,
                                                    random_state=random_state)

    with mlflow.start_run() as run:
        # fit grid search
        search.fit(train_x, tmp_y)
        # log parameters
        mlflow.set_tag("Model", clf_name)
        mlflow.log_params(search.best_params_)
        mlflow.log_param('kfold_random_state', 42)

        # determine best model and calculate validation metrics
        best_mod = search.best_estimator_
        y_preds = best_mod.predict(val_x)
        val_acc = accuracy_score(val_y, y_preds)
        mod_f1 = f1_score(val_y, y_preds, average='weighted')

        # log metrics
        mlflow.log_metric("validation_acc", val_acc)
        mlflow.log_metric('f1_score', mod_f1)

        os.makedirs('save_data', exist_ok=True)

        with open('save_data/x_train.pkl', 'wb') as fname:
            pickle.dump(train_data, fname)

        mlflow.log_artifact('save_data/x_train.pkl')

        # log models
        if clf_name == 'xgboost':
            mlflow.xgboost.log_model(best_mod, artifact_path='better_models')
        else:
            mlflow.sklearn.log_model(best_mod, artifact_path='better_models')

        # retrieve run_id to be used in logging best model
        run_id = run.info.run_id
        mlflow.end_run()

    mod_info = {'type': clf_name,
                'accuracy': val_acc,
                'f1_score': mod_f1,
                'model': best_mod,
                'run_id': run_id}

    return mod_info
