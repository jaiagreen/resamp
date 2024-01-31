### ACTIVE FINAL SCRIPT
##FINAL SCRIPTS VERSION - 4.0 (UPDATED BOOTSRATPPING FUNCTION), OVERRIDE EXPECTED VALUE
## ALL MODIFICATION ACTIVE

import pandas as pd
import numpy as np
from scipy.stats import chi2
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(input_data):
    # Function to read data from either a file path or DataFrame
    if isinstance(input_data, pd.DataFrame):
        data = input_data.copy()
    else:
        _, file_extension = os.path.splitext(input_data)
        if file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(input_data, index_col=0)
        elif file_extension == '.csv':
            data = pd.read_csv(input_data, index_col=0)
        else:
            raise ValueError("Unsupported file type")
    return data.apply(pd.to_numeric, errors='coerce')


def compare_dimensions(observed, expected):
    if observed.shape != expected.shape:
        raise ValueError("Dimensions of observed and expected data do not match")

def calculate_chi_squared(observed, expected):
    chi_squared = ((observed - expected) ** 2 / expected).sum().sum()
    return chi_squared

# #Calculating the expected Value 
def calculate_expected(observed):
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum().sum()
    expected = np.outer(row_sums, col_sums) / total
    return expected


def calculate_chi_abs(observed, expected=None):
    try:
        # Check if the input data is DataFrame and convert to numpy arrays if necessary
        if isinstance(observed, pd.DataFrame):
            observed = observed.values
        if expected is None:
            expected = calculate_expected(observed)
        elif isinstance(expected, pd.DataFrame):
            expected = expected.values

        # Calculate the chi absolute statistic
        chi_abs = (np.abs(observed - expected) / expected).sum().sum()
        return chi_abs
    except Exception as e:
        logging.error("Error calculating p-value: ", exc_info=True)
        return None

def chi_abs_stat(observed_data, expected_data=None):
    try:
        # If expected data is not provided, calculate it
        if expected_data is None:
            expected_data = calculate_expected(observed_data)

        # Ensure dimensions match
        if observed_data.shape != expected_data.shape:
            raise ValueError("Dimensions of observed and expected data do not match")

        chi_abs = calculate_chi_abs(observed_data, expected_data)
        return chi_abs
    except Exception as e:
        logging.error("Error calculating chi absolute: ", exc_info=True)
        return None


def calculate_p_value(chi_squared, dof):
    p_value = chi2.sf(chi_squared, dof)
    return p_value


def chi_squared_stat(observed_data, expected_data):
    try:
        observed = read_data(observed_data)
        expected = read_data(expected_data)
        compare_dimensions(observed, expected)
        chi_squared = calculate_chi_squared(observed, expected)
        return chi_squared
    except Exception as e:
        logging.error("Error calculating chi-squared: ", exc_info=True)
        return None


def p_value_stat(observed_data, expected_data):
    try:
        observed = read_data(observed_data)
        expected = read_data(expected_data)
        compare_dimensions(observed, expected)
        chi_squared = calculate_chi_squared(observed, expected)
        dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        return calculate_p_value(chi_squared, dof)
    except Exception as e:
        logging.error("Error calculating p-value: ", exc_info=True)
        return None


# Example usage (assuming observed_data and expected_data are numpy arrays):
# p_value_mean_ratio = calculate_p_value_mean_ratio(observed_data, expected_data, num_simulations=1000, with_replacement=True, sample_size=None)
# print(f"P-value (Mean Ratio): {p_value_mean_ratio}")
        

def convert_df_to_numpy(df_observed, df_expected):
    """
    Converts DataFrame data to numpy arrays for use in other functions, 
    particularly for bootstrapping.

    Parameters:
        df_observed (pd.DataFrame): DataFrame containing observed data.
        df_expected (pd.DataFrame): DataFrame containing expected data.

    Returns:
        tuple: A tuple of numpy arrays (observed_array, expected_array).
    """
    observed_array = df_observed.values
    expected_array = df_expected.values
    return observed_array, expected_array       


def bootstrap_chi_abs(observed_data, num_simulations=10000, with_replacement=True):
    
    """
    Generates a bootstrap distribution of the chi absolute statistic for an n*n contingency table.

    Parameters:
        observed_data (np.array or pd.DataFrame): n*n contingency table with observed frequencies.
        num_simulations (int): Number of bootstrap samples to generate.
        with_replacement (bool): Indicates whether sampling should be with replacement."""
    if isinstance(observed_data, pd.DataFrame):
        observed_data = observed_data.values

    total_rows, total_columns = observed_data.shape
    expected_data = calculate_expected(observed_data)

    results = np.zeros(num_simulations)
    total_counts_per_column = observed_data.sum(axis=0)

    # Create a pooled data array combining all categories across rows and columns
    pooled_data = np.concatenate([np.repeat(row, sum(observed_data[row, :])) for row in range(total_rows)])

    for i in range(num_simulations):
        sim = np.zeros_like(observed_data)

        for col in range(total_columns):
            column_sample = np.random.choice(pooled_data, total_counts_per_column[col], replace=with_replacement)

            for row in range(total_rows):
                # Count occurrences of each category in the column sample
                sim[row, col] = np.sum(column_sample == row)
        #print(sim)
        # Calculate the chi absolute statistic for the simulated data
        chi_abs = (np.abs(sim - expected_data) / expected_data).sum().sum()
        results[i] = chi_abs

    return results


def calculate_p_value_bootstrap(chi_abs_observed, simulated_chi_abs, two_tailed=False):
    """
    Calculates the p-value for the chi absolute statistic using bootstrap methods.

    Parameters:
        chi_abs_observed (float): The observed chi absolute statistic.
        simulated_chi_abs (np.array): The array of chi absolute statistics from bootstrap samples.
        two_tailed (bool): If True, perform a two-tailed test. Defaults to False (one-tailed test).

    Returns:
        float: The p-value.
    """
    try:
        if two_tailed:
            # For a two-tailed test, consider both tails of the distribution
            tail_proportion = np.mean(simulated_chi_abs >= chi_abs_observed)
            p_value = 2 * min(tail_proportion, 1 - tail_proportion)
        else:
            # For a one-tailed test, only consider the tail of interest
            p_value = np.mean(simulated_chi_abs >= chi_abs_observed)
        
        return p_value
    except Exception as e:
        logging.error("Error in calculating p-value: ", exc_info=True)
        return None



def plot_chi_abs_distribution(simulated_data, observed_data, p_value):
    """
    Plots the distribution of simulated chi absolute values and the observed chi absolute value.

    Parameters:
        simulated_data (np.array): Array of chi absolute statistics from bootstrap samples.
        observed_data (pd.DataFrame): DataFrame containing the observed frequencies.
        p_value (float): The calculated p-value for the chi absolute statistic.
    """
    # Calculate the observed chi absolute value
    chi_abs_observed = calculate_chi_abs(observed_data)
    #chi_abs_observed = observed_data

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_data, color="skyblue", kde=True, stat="density", linewidth=0)
    plt.axvline(chi_abs_observed, color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Distribution of Simulated Chi Absolute Values\nObserved Chi Absolute Value and P-Value: {p_value:.3f}')
    plt.xlabel('Chi Absolute Value')
    plt.ylabel('Density')
    plt.legend(['Simulated Chi Absolute','Observed Chi Absolute'])
    plt.show()



