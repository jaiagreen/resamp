### ACTIVE FINAL SCRIPT
##FINAL SCRIPTS VERSION - 4.0 (UPDATED BOOTSRATPPING FUNCTION), OVERRIDE EXPECTED VALUE
## ALL MODIFICATION ACTIVE

# LS 40 resampling library to complement/combine with Hypothesize

### ACTIVE FINAL SCRIPT

## NAME: resamp.py
## VERSION - 1.5
## STATUS: ALL MODIFICATION ACTIVE AND LIVE IN PYPI (PIP INSTALL STATISTICS_LIBRARY)
## DATE: 9 MARCH 2024 
## AUTHOR: VISHANTH HARI RAJ
## SUPERVISOR: JANE

import pandas as pd
import numpy as np
from scipy.stats import chi2
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt


##MAD FUNCTION Mean-Absolute-Deviation

def mean_absolute_deviation(data):
    """
    Calculate the Mean Absolute Deviation (MAD) of a 1D dataset.

    The function accepts either a 1D array-like, a filename of a CSV or an Excel file.
    In case of a file, the function expects it to contain a single column of numbers.

    Parameters:
    data (array-like or str): A 1D array or list containing the dataset, or a filename.

    Returns:
    float: The Mean Absolute Deviation of the dataset.

    Example:
    >>> mean_absolute_deviation([1, 2, 3, 4, 5])
    1.2

    >>> mean_absolute_deviation('data.csv') # Assuming 'data.csv' contains a single column of numbers
    # Returns the MAD of the numbers in 'data.csv'
    """
#     try:
#         # Check if data is a filename
#         if isinstance(data, str):
#             if data.endswith('.csv'):
#                 data = pd.read_csv(data).squeeze()  # Assuming a single column
#             elif data.endswith(('.xls', '.xlsx')):
#                 data = pd.read_excel(data).squeeze()  # Assuming a single column
#             else:
#                 raise ValueError("File format not supported. Please use CSV or Excel files.")
        
        # Calculate MAD
#2-D array (work column by column)
    try:
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.mean(deviations)
        return mad
    except Exception as e:
        return f"An error occurred: {e}"

# test_data = [15, 26, 30, 40, 55]
# mad = mean_absolute_deviation(test_data)
# print(mad)

###############################################################################################################################

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


# def calculate_p_value_bootstrap(observed_data, simulated_data, two_tailed=False):
#     """
#     Calculates the p-value for the chi absolute statistic using bootstrap methods.

#     Parameters:
#         observed_data(np.array): The observed chi absolute statistic.
#         simulated_data (np.array): The array of chi absolute statistics from bootstrap samples.
#         two_tailed (bool): If True, perform a two-tailed test. Defaults to False (one-tailed test).

#     Returns:
#         float: The p-value.
#     """
#     try:
#         if two_tailed:
#             # For a two-tailed test, consider both tails of the distribution
#             tail_proportion = np.mean(simulated_data >= observed_data)
#             p_value = 2 * min(tail_proportion, 1 - tail_proportion)
#         else:
#             # For a one-tailed test, only consider the tail of interest
#             p_value = np.mean(simulated_data >= observed_data)
        
#         return p_value
#     except Exception as e:
#         logging.error("Error in calculating p-value: ", exc_info=True)
#         return None

#Updated Fuction:
def calculate_p_value_bootstrap(observed_data, simulated_data, two_tailed=False):
    """
    Calculates the p-value for the chi absolute statistic using bootstrap methods, 
    determining first if the observed statistic lies on the left or right side of the distribution's mean.

    Parameters:
        observed_data (float): The observed chi absolute statistic.
        simulated_data (np.array): The array of chi absolute statistics from bootstrap samples.
        two_tailed (bool): If True, perform a two-tailed test. Defaults to False (one-tailed test).

    Returns:
        float: The p-value.
    """
    try:
        # Determine the side of the distribution where the observed data lies
        mean_simulated_data = np.mean(simulated_data)
        is_right_side = observed_data > mean_simulated_data
        
        if two_tailed:
            if is_right_side:
                # For a two-tailed test, consider both tails of the distribution (right side logic)
                tail_proportion = np.mean(simulated_data >= observed_data)
            else:
                # For a two-tailed test, consider both tails of the distribution (left side logic)
                tail_proportion = np.mean(simulated_data <= observed_data)
            p_value = tail_proportion
        else:
            if is_right_side:
                # For a one-tailed test, only consider the tail of interest (right side logic)
                p_value = np.mean(simulated_data >= observed_data)
            else:
                # For a one-tailed test, only consider the tail of interest (left side logic)
                p_value = np.mean(simulated_data <= observed_data)
        
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


##############################################################################################################################  
    
#CALCULATION OF RELATIVE RISK, (SIM DATA VS OBSERVED DATA) CONFIDENCE INTERVAL, PLOT GRAPH
# Relative Risk of two treatments
# ProbCalculation
# Confidence Interval
# Plotting the Graph

# VERSION - 1 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_relative_risk_two_treatments(observed_data, event_row_index, treatment1_index, treatment2_index):
    """
    Calculate the relative risk of an event between two specific treatments.

    Parameters:
        observed_data (np.array): The observed data as a 2D array.
        event_row_index (int): The row index for the event.
        treatment1_index (int): The column index for the first treatment.
        treatment2_index (int): The column index for the second treatment.

    Returns:
        float: The relative risk of the event between the two treatments.
    """
    total_treatment1 = observed_data[:, treatment1_index].sum()
    total_treatment2 = observed_data[:, treatment2_index].sum()
    
    prob_event_treatment1 = observed_data[event_row_index, treatment1_index] / total_treatment1
    prob_event_treatment2 = observed_data[event_row_index, treatment2_index] / total_treatment2
    relative_risk = prob_event_treatment1 / prob_event_treatment2
    return relative_risk


def resample_and_calculate_rr(observed_data, event_row_index, reference_treatment_index=0, num_simulations=10000):
    # Extract the dimensions of the observed_data array
    total_rows, total_columns = observed_data.shape
    
    # Calculate the total counts for each column (treatment group)
    total_counts_per_column = observed_data.sum(axis=0)
    
    # Initialize an array to store simulated RR values
    simulated_rr = np.zeros(num_simulations)

    # Loop through the specified number of simulations
    for i in range(num_simulations):
        # Create an array to simulate a resampled dataset, initialized with zeros
        simulated_sample = np.zeros_like(observed_data)
        
        # Iterate over columns (treatment groups)
        for col in range(total_columns):
            # Generate a random sample of 0s and 1s based on the proportions in the observed data
            column_sample = np.random.choice([0, 1], size=total_counts_per_column[col], replace=True,
                                             p=[observed_data[1, col] / total_counts_per_column[col], 
                                                observed_data[0, col] / total_counts_per_column[col]])
            
            # Count the occurrences of 0s and 1s in the resampled column
            simulated_sample[1, col] = np.sum(column_sample == 0)
            simulated_sample[0, col] = np.sum(column_sample == 1)

        # Calculate the probability of the event for the reference treatment group
        prob_event_treatment1 = simulated_sample[event_row_index, reference_treatment_index] / total_counts_per_column[reference_treatment_index]
        
        # Calculate the probability of the event for the other treatment group
        prob_event_other_treatment = simulated_sample[event_row_index, 1 - reference_treatment_index] / total_counts_per_column[1 - reference_treatment_index]
        
        # Calculate the Risk Ratio (RR) for the current simulated dataset and store it
        simulated_rr[i] = prob_event_treatment1 / prob_event_other_treatment

    # Return an array of simulated RR values
    return simulated_rr


# Calculate the 95% confidence interval
def calculate_confidence_interval(simulated_rr, percentile=99):
    """
    Calculate the confidence interval for the relative risk based on the distribution of simulated relative risks.

    Parameters:
        simulated_rr (np.array): Array of simulated relative risks.
        percentile (float, optional): The percentile for the confidence interval. Defaults to 95.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    lower_percentile = (100 - percentile) / 2
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(simulated_rr, lower_percentile)
    upper_bound = np.percentile(simulated_rr, upper_percentile)
    return lower_bound, upper_bound

def calculate_probabilities_for_each_treatment(observed_data, event_row_index):
    """
    Calculate the probabilities of an event for each treatment.

    Parameters:
        observed_data (np.array or pd.DataFrame): The observed data as a 2D array or DataFrame.
        event_row_index (int): The row index for the event.

    Returns:
        dict: Probabilities of the event for each treatment.
    """
    if isinstance(observed_data, pd.DataFrame):
        observed_data = observed_data.to_numpy()

    total_columns = observed_data.shape[1]
    total_counts_per_treatment = observed_data.sum(axis=0)
    probabilities = {}

    for col in range(total_columns):
        prob_event_treatment = observed_data[event_row_index, col] / total_counts_per_treatment[col]
        probabilities[f"Treatment_{col+1}"] = prob_event_treatment

    return probabilities

def plot_relative_risk_distribution(simulated_rr, observed_rr):
    """
    Plots the distribution of simulated relative risks with observed relative risk and confidence intervals.

    Parameters:
        simulated_rr (np.array): Array of simulated relative risks.
        observed_rr (float): The observed relative risk.
    """
    # Sort the simulated results
    simulated_rr.sort()
    
    # Calculate the 2.5th and 97.5th percentiles for the confidence interval
    Mlower = simulated_rr[int(len(simulated_rr) * 0.025)]
    Mupper = simulated_rr[int(len(simulated_rr) * 0.975)]

    # Calculate the log-based lower and upper bounds
    lowerbound = np.exp(2 * np.log(observed_rr) - np.log(Mupper))
    upperbound = np.exp(2 * np.log(observed_rr) - np.log(Mlower))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_rr, kde=True, color="skyblue")
    plt.axvline(observed_rr, color="red", linestyle='dashed', linewidth=2)
    plt.axvline(lowerbound, color="yellow", linestyle='dashed', linewidth=2)
    plt.axvline(upperbound, color="yellow", linestyle='dashed', linewidth=2)
    plt.title('Distribution of Simulated Relative Risks with Confidence Interval')
    plt.xlabel('Relative Risk')
    plt.ylabel('Frequency')
    plt.legend(['Simulated' , 'Observed Relative Risk', 'Bound'])
    plt.show()

    print("Lower Bound:", lowerbound)
    print("Upper Bound:", upperbound)

# Example usage with simulated relative risks and an observed relative risk
# Replace `simulated_rr` with your array of simulated relative risks
# Replace `observed_rr` with your observed relative risk
# plot_relative_risk_distribution(simulated_rr, observed_rr)


###############################################################################################################################
#Correlation Resampling 

##FINAL CODE FOR CORRELATION 

import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate the p-value based on simulated data and observed correlation
def calculate_p_value(sims, corr_obs, two_tailed=False):
    if corr_obs > 0:
        if two_tailed:
            p_value = (np.sum(sims >= corr_obs) + np.sum(sims <= -corr_obs)) / len(sims)
        else:
            # For one-tailed, we assume a positive relationship.
            p_value = np.sum(sims >= corr_obs) / len(sims)
    else:
        if two_tailed:
            p_value = (np.sum(sims <= corr_obs) + np.sum(sims >= -corr_obs)) / len(sims)
        else:
            # For one-tailed, we assume a negative relationship.
            p_value = np.sum(sims <= corr_obs) / len(sims)
    return p_value

def plot_null_distribution(sims, corr_obs, two_tailed=False):
    """
    Plot the null distribution of simulated correlation coefficients and the observed correlation.
    
    Parameters:
    - sims (np.array): Simulated correlation coefficients.
    - corr_obs (float): Observed correlation coefficient.
    - two_tailed (bool, optional): Indicates if the test is two-tailed (default is False).
    """
    try:
        p = sns.displot(sims, kde=False)
        p.set(xlabel="Pearson's Correlation Coefficient", ylabel="Count", title="Null Distribution")
        plt.axvline(corr_obs, color='red', label=f'Observed Correlation: {corr_obs:.3f}')
        if two_tailed:
            plt.axvline(-corr_obs, color='red', label=f'Negative Observed Correlation: {-corr_obs:.3f}')
        plt.legend(loc='upper right')
        plt.show()
    except Exception as e:
        print(f"Error plotting null distribution: {e}")

def permute_correlation(x, y, num_simulations=10000):
    """
    Generate simulated correlation coefficients by permuting one variable and calculating Pearson's correlation.
    
    Parameters:
    - x (np.array): Values of variable 1.
    - y (np.array): Values of variable 2.
    - num_simulations (int, optional): Number of permutations to perform (default 10000).
    
    Returns:
    - np.array: Simulated correlation coefficients.
    """
    try:
        x = np.asarray(x)
        y = np.asarray(y)
        simulated_correlations = np.zeros(num_simulations)
        for i in range(num_simulations):
            permuted_x = np.random.permutation(x)
            simulated_correlations[i] = pearsonr(permuted_x, y)[0]
        return simulated_correlations
    except Exception as e:
        print(f"Error generating simulated correlations: {e}")
        return np.array([])  # Return an empty array in case of error

def compute_correlation_ci(x, y, num_simulations=10000, confidence_interval=0.95):
    """
    Compute the confidence interval around the observed correlation by resampling the dataset
    and plotting the distribution of correlation coefficients from the resampled datasets with error handling.
    
    Parameters:
    - x (np.array): Values of variable 1.
    - y (np.array): Values of variable 2.
    - num_simulations (int, optional): Number of bootstrap samples to generate (default 10000).
    - confidence_interval (float, optional): Confidence level for the interval (default 0.95).
    """
    try:
        observed_correlation = pearsonr(x, y)[0]
        simulated_correlations = []
        
        for _ in range(num_simulations):
            indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
            resampled_x = x[indices]
            resampled_y = y[indices]
            resample_correlation = pearsonr(resampled_x, resampled_y)[0]
            simulated_correlations.append(resample_correlation)
        
        simulated_correlations = np.array(simulated_correlations)
        lower_bound = np.percentile(simulated_correlations, (1 - confidence_interval) / 2 * 100)
        upper_bound = np.percentile(simulated_correlations, (1 + confidence_interval) / 2 * 100)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(simulated_correlations, kde=True, color="blue", stat="density", linewidth=0)
        plt.axvline(observed_correlation, color='red', linestyle='--', label='Observed Correlation')
        plt.axvline(lower_bound, color='green', linestyle='-', label=f'{confidence_interval*100:.0f}% CI Lower Bound')
        plt.axvline(upper_bound, color='green', linestyle='-', label=f'{confidence_interval*100:.0f}% CI Upper Bound')
        plt.title('Distribution of Simulated Correlations with Confidence Interval')
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        print("Observed Correlation:", observed_correlation)
        print(f"{confidence_interval*100:.0f}% Confidence Interval: ({lower_bound:.3f}, {upper_bound:.3f})")
    except Exception as e:
        print(f"An error occurred: {e}")


###############################################################################################################################

# 1-D Confidence Interval Calculation 
# Additional this function also do the same job : def calculate_confidence_interval(simulated_rr, percentile=95):

import numpy as np

def cal_ci_onedim(data, confidence_level=99):
    """
    Calculate a custom confidence interval for a 1-D array based on the specified confidence level.
    
    Parameters:
        data (np.array): The 1-D array of resampled values or any numeric data.
        confidence_level (float): The confidence level expressed as a percentage. Defaults to 99.
    
    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Ensure data is a numpy array for efficient operations
    data = np.array(data)
    
    # Sort the data
    data.sort()
    
    # Calculate the positions for the lower and upper bounds
    total_elements = len(data)
    lower_pos = int(((100 - confidence_level) / 2) * total_elements / 100)
    upper_pos = int(total_elements - lower_pos - 1)  # Adjust by 1 for zero-based indexing
    
    # Extract the values at the calculated positions
    lower_value = data[lower_pos]
    upper_value = data[upper_pos]
    
    return lower_value, upper_value


#########################################################################################################################################

import numpy as np
import pandas as pd
from scipy.stats import linregress

def bootstrap_confidence_interval(x, y, n_bootstrap=1000, confidence_level=99, return_type='both'):
    slopes = []
    intercepts = []
    n = len(x)
    alpha = 100 - confidence_level
    percentile_lower = alpha / 2
    percentile_upper = 100 - alpha / 2
    
    # Resampling and fitting
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(range(n), size=n, replace=True)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
        
        # Using scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_sample, y_sample)
        slopes.append(slope)
        intercepts.append(intercept)
    
    # Preparing the output
    result = {}
    if return_type in ('slope', 'both'):
        slope_lower_bound = np.percentile(slopes, percentile_lower)
        slope_upper_bound = np.percentile(slopes, percentile_upper)
        result['slope'] = (slope_lower_bound, slope_upper_bound)
    
    if return_type in ('intercept', 'both'):
        intercept_lower_bound = np.percentile(intercepts, percentile_lower)
        intercept_upper_bound = np.percentile(intercepts, percentile_upper)
        result['intercept'] = (intercept_lower_bound, intercept_upper_bound)
    
    return result

# Example usage
# Assuming 'brainhead' is a DataFrame with 'Head Size' and 'Brain Weight' columns
# brainhead = pd.read_csv('path_to_your_data.csv')
# x = brainhead['Head Size'].values
# y = brainhead['Brain Weight'].values
# results = bootstrap_confidence_interval(x, y, return_type='both')  # Can be 'slope', 'intercept', or 'both'
# print(f"Results: {results}")

###############################################################################################################################

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_bootstrap_lines(x, y, n_bootstrap=1000, original_slope=2, original_intercept=0):
    slopes = []
    intercepts = []
    
    # Bootstrap resampling and linear regression fitting
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(range(len(x)), size=len(x), replace=True)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
        model = LinearRegression().fit(x_sample.reshape(-1, 1), y_sample)
        slopes.append(model.coef_[0])
        intercepts.append(model.intercept_)
    
    # Plotting data points
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label='Data Points')
    
    # Plotting lines for a subset of bootstrap samples
    for i in range(min(100, len(slopes))):  # Plot lines for 100 bootstrap samples
        y_pred = slopes[i] * x + intercepts[i]
        plt.plot(x, y_pred, color='grey', alpha=0.2, linewidth=1)
    
    # Plotting original line for comparison
    y_original = original_slope * x + original_intercept
    plt.plot(x, y_original, color='red', label='Original Line', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bootstrap Regression Lines')
    plt.legend()
    plt.show()


###################################################################################################################################

##Power Analysis

import numpy as np

def calculate_statistic(data, stat_type='median'):
    """Calculate either the mean or median of the data, based on stat_type."""
    return np.mean(data) if stat_type == 'mean' else np.median(data)

def power_analysis(obs_diff, group1, group2, num_simulations=1000, alpha=0.01, power_threshold=0.8, factor_limit=10, measure='median', verbose=False):
        """
    Perform a power analysis using resampling methods to determine the sample size required to achieve a desired power level.
    The function stops increasing the sample size once the factor limit is reached.
    
    Parameters:
    obs_diff -- the observed difference in medians or means between the two groups, depending on 'measure'
    group1 -- data for group 1
    group2 -- data for group 2
    num_simulations -- number of simulations to perform
    alpha -- significance level
    power_threshold -- desired power level to achieve
    factor_limit -- the maximum factor by which to increase the sample size
    measure -- 'median' or 'mean', the statistical measure to use for comparison
    verbose -- if True, print intermediate results
    
    Returns:
    required_sample_sizes -- a tuple of the required sample sizes for group 1 and group 2 to achieve the desired power
    achieved_power -- the power that was achieved with the returned sample sizes
    """
    factor = 1
    achieved_power = 0

    while achieved_power < power_threshold and factor <= factor_limit:
        sample_size_group1 = len(group1) * factor
        sample_size_group2 = len(group2) * factor
        pvals = []

        for _ in range(num_simulations):
            # Simulate resampling from each group according to the current factor
            sim_group1 = np.random.choice(group1, sample_size_group1, replace=True)
            sim_group2 = np.random.choice(group2, sample_size_group2, replace=True)
            phantom_diff = calculate_statistic(sim_group1, measure) - calculate_statistic(sim_group2, measure)

            # Generate the null distribution
            pooled = np.concatenate([group1, group2])
            null_diffs = np.zeros(num_simulations)
            for j in range(num_simulations):
                # Resample from the pooled data to simulate the null hypothesis
                #null_resample = np.random.choice(pooled, sample_size_group1 + sample_size_group2, replace=True)
                null_group1 = np.random.choice(pooled,len(sim_group1)*factor)
                null_group2 = np.random.choice(pooled,len(sim_group1)*factor)
                null_diffs[j] = calculate_statistic(null_group1, measure) - calculate_statistic(null_group2, measure)

            # Calculate the p-value for this simulation
            pval = (np.sum(null_diffs >= abs(phantom_diff)) + np.sum(null_diffs <= -abs(phantom_diff))) / num_simulations
            pvals.append(pval)

        # Calculate the overall power based on the simulations
        achieved_power = np.mean(np.array(pvals) < alpha)
        
        if verbose:
            print(f"Iteration with factor {factor}: Sample size group 1: {sample_size_group1}, Sample size group 2: {sample_size_group2}, Achieved power: {achieved_power}")

        factor += 1

    required_sample_sizes = (sample_size_group1, sample_size_group2)
    return required_sample_sizes, achieved_power



# Placeholder for actual calls with real data:
# required_sample_sizes, achieved_power = power_analysis(
#     obs_diff, plant_eph, plant_perm,
#     num_simulations=1000, alpha=0.01, power_threshold=0.8, factor_limit=10,
#     measure='mean', verbose=True
# )





