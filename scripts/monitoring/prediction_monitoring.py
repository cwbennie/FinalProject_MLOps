from typing import Tuple
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_predicted_probabilities(
        input_data: pd.DataFrame,
        run_id: str = "d4f8d43642bd4396b27f3f8d972c3fd4",
        experiment_name: str = "final_training",
        mlflow_uri: str = "https://mlops-680-370980073666.us-west2.run.app",
        output_path: str = "predicted_probability_distribution.png"):
    """
    Load an XGBoost model from MLFlow and plot the predicted
        probability distributions.

    Args:
        input_data (pd.DataFrame): DataFrame containing the input
            features for prediction.
        run_id (str): MLFlow Run ID of the trained model.
        experiment_name (str): MLFlow experiment name where the model
            is stored.
        mlflow_uri (str): The MLFlow server URI.
        output_path (str): Path to save the plot of predicted probability
            distributions.

    Returns:
        None
    """
    # Set up MLFlow environment variables
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Construct the model URI using the run ID
    model_uri = f"runs:/{run_id}/better_models"

    # Load the model using MLFlow's pyfunc loader
    print(f"Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # Predict the probability distribution
    probs = model.predict_proba(input_data)
    plt.figure(figsize=(12, 6))

    # Define colors and labels for each class
    colors = ['blue', 'orange', 'green']
    class_labels = ['Away Win Probability', 'Draw Probability',
                    'Home Win Probability']

    # Plot the histogram for each class
    for i in range(probs.shape[1]):
        plt.hist(probs[:, i], bins=30, alpha=0.6,
                 label=class_labels[i], color=colors[i])

    # Plot formatting
    plt.title('Predicted Probability Distributions')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot
    plt.savefig(output_path)
    print(f"Probability distribution plot saved to {output_path}")


def load_data(train_path: str, test_path: str)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to return a tuple of dataframes
    """

    train_data = pd.read_csv(train_path, index_col=0)
    train_y = train_data['Result']
    train_data.drop(columns=['Result'], inplace=True)
    test_data = pd.read_csv(test_path, index_col=0)
    test_y = test_data['Result']
    test_data.drop(columns=['Result'], inplace=True)

    return train_data, test_data, train_y, test_y


if __name__ == '__main__':
    train_data, test_data, _, _ = load_data('../../data/processed_train.csv',
                                            '../../data/processed_test.csv')
    train_output = './plots/train_prediction_probabilities.png'
    test_output = './plots/test_prediction_probabilities.png'
    plot_predicted_probabilities(train_data, output_path=train_output)
    plot_predicted_probabilities(test_data, output_path=test_output)
