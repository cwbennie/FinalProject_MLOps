from typing import Tuple
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def kl_divergence(train_data: pd.DataFrame, test_data: pd.DataFrame,
                  target_col: str, output_file: str, bins: int = 10):
    """
    Calculate the KL Divergence between two columns of variables.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The test data.
        target_col (str): The target column for calculating KL Divergence.
        bins (int): Number of bins to use for creating histograms
            of the columns.
        output_file (str): The file path to save the plot showing the results
            of the KL Divergence Test

    Returns:
        float: KL Divergence value between the two columns.
    """
    train_col = train_data[target_col]
    test_col = test_data[target_col]
    # Calculate histograms for both columns (P and Q distributions)
    train_hist, bin_edges = np.histogram(train_col, bins=bins,
                                     density=True)
    test_hist, _ = np.histogram(test_col, bins=bin_edges,
                             density=True)

    # Create probability distributions
    p_prob = train_hist / np.sum(train_hist)
    q_prob = test_hist / np.sum(test_hist)

    # Add a small constant to prevent division by zero or log(0)
    epsilon = 1e-10
    p_prob += epsilon
    q_prob += epsilon

    # Calculate KL Divergence using scipy's entropy function
    kl_div = stats.entropy(p_prob, q_prob)

    # Plotting the histograms and distributions
    plt.figure(figsize=(10, 6))
    plt.hist(train_col, bins=bins, alpha=0.5, label='Training Data',
             density=True, color='blue')
    plt.hist(test_col, bins=bins, alpha=0.5, label='Test Data',
             density=True, color='orange')

    # Plot the probability density functions
    plt.plot(bin_edges[1:], p_prob, label='Training Distribution',
             marker='o', linestyle='--', color='blue')
    plt.plot(bin_edges[1:], q_prob, label='Test Distribution',
             marker='o', linestyle='--', color='orange')

    # Adding labels and legend
    plt.title(f'KL Divergence for {target_col}: {kl_div:.4f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='best')

    # Save the plot to the specified file path
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

    return kl_div


def load_data(train_path: str, test_path: str)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to return a tuple of dataframes
    """

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = load_data('../../data/processed_train.csv',
                                      '../../data/processed_test.csv')

    kl_divergence(train_data, test_data,
                  'HTGS', '../monitoring/home_team_goals_kl_div.png')
    kl_divergence(train_data, test_data,
                  'ATGC', '../monitoring/away_team_goals_kl_div.png')
