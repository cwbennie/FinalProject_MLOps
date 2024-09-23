from typing import Tuple
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
