# LS 40 resampling library to complement/combine with Hypothesize

### ACTIVE FINAL SCRIPT

## VERSION - 1.7.10
## DATE: 6 MAY 2025
## AUTHOR: VISHANTH HARI RAJ, JANE SHEVTSOV, KRISTIN MCCULLY
## SUPERVISOR: JANE SHEVTSOV


import pandas as pd
import numpy as np
from scipy.stats import chi2
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# CORE FUNCTIONS

#This should be the canonical function for p-values in resamp
def p_value_resampled(observed_stat, simulated_stats, two_tailed=True):
    """
    Calculates the p-value for a statistic using bootstrap methods,
    determining first if the observed statistic lies on the left or right side of the distribution's mean.

    Parameters:
        observed_stat (float): The observed statistic.
        simulated_stats (np.array): The array of resampled statistics.
        two_tailed (bool): If True, perform a two-tailed test; otherwise, do one-tailed. Defaults to True.

    Returns:
        p (float): The p-value.
    """
    try:
        # Determine the side of the distribution where the observed data lies
        mean_simulated_stats = np.mean(simulated_stats)
        is_right_side = observed_stat > mean_simulated_stats

        if two_tailed:
            if is_right_side:
                # For a two-tailed test, consider both tails of the distribution (right side logic)
                tail_proportion = np.mean(simulated_stats >= observed_stat) + np.mean(simulated_stats <= -observed_stat)
            else:
                # For a two-tailed test, consider both tails of the distribution (left side logic)
                tail_proportion = np.mean(simulated_stats <= observed_stat) + np.mean(simulated_stats >= -observed_stat) # FROM KRISTIN: Added - to second calculation
            p_value = tail_proportion
        else:
            if is_right_side:
                # For a one-tailed test, only consider the tail of interest (right side logic)
                p_value = np.mean(simulated_stats >= observed_stat)
            else:
                # For a one-tailed test, only consider the tail of interest (left side logic)
                p_value = np.mean(simulated_stats <= observed_stat)
        return p_value
    except Exception as e:
        logging.error("Error in calculating p-value: ", exc_info=True)
        return None


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


#Median Absolute Deviation
def median_absolute_deviation(data):
    """
    Calculate the Median Absolute Deviation (MAD) of a 1D dataset.

    The function accepts either a 1D array-like, a filename of a CSV or an Excel file.
    In case of a file, the function expects it to contain a single column of numbers.

    Parameters:
    data (array-like or str): A 1D array or list containing the dataset, or a filename.

    Returns:
    float: The Median Absolute Deviation of the dataset.

    Example:
    >>> median_absolute_deviation([1, 2, 3, 4, 5])
    1.2

    >>> median_absolute_deviation('data.csv') # Assuming 'data.csv' contains a single column of numbers
    # Returns the MAD of the numbers in 'data.csv'
    """
    # Calculate MAD
    #2-D array (work column by column)
    try:
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.median(deviations)
        return mad
    except Exception as e:
        return f"An error occurred: {e}"


def CI_percentile_to_pivotal(Mobs, CIpercentile):
    """
    Convert percentile confidence interval to pivotal confidence interval.

    Parameters:
        Mobs (float): Measured quantity you want to put a CI on (mean, median, etc.)
        CIpercentile (list or Numpy array): Percentile CI

    Returns:
        Confidence interval (Numpy array)
    """
    return np.array([2*Mobs-CIpercentile[1], 2*Mobs-CIpercentile[0]])



###############################################################################################################################
# 1-Sample Tests

def resample_one_group_count(box, sample_stat, sample_size, count_what="A", two_tailed=True, sims=10000, proportion=False, return_resamples=False):
    """
    Calculate a p-value for a count or proportion. Used to compare a sample to a population/base rate.

    Parameters:
        box (np array or list): Box model representing population
        sample_stat (float): Statistic for the sample
        sample_size (int): Number of individuals (or sites, etc.) in sample
        count_what (char): What character in the box model should be counted. Default "A".
        two_tailed (bool): Whether to compute a two-tailed p-value. If False, do one-tailed.
        sims (int): How many simulations to run. Default 10,000
        proportion (bool): Calculate count or proportion (default count)
        return_resamples (bool): Whether to return resampling results used to generate p-value. Primarily for pedagogical purposes.

    Returns:
        p-value
        resampling data (if desired)
    """

    dataArr = np.array(box)  #Converts box model to NumPy array

    #Resampling loop
    resampleArr = np.zeros(sims)  #Preallocates array to store resampling results
    for i in range(sims):
        p_sample = np.random.choice(dataArr, sample_size, replace=True)  #Samples from the box model (with replacement)
        p_count = np.sum(p_sample == count_what)
        if proportion:
            p_count = p_count/sample_size
        resampleArr[i] = p_count

    #Compute p-value
    if proportion:
        observed = np.sum(dataArr == count_what)
    else:
        observed = np.mean(dataArr == count_what)
    p = p_value_resampled(observed_stat = sample_stat, simulated_stats = resampleArr, two_tailed=two_tailed)

    #Return results
    if return_resamples:
        return p, resampleArr
    else:
        return p


def confidence_interval_count(box, sample_size, confidence_level=99, count_what="A", sims=10000, pivotal=True, proportion=False, return_resamples=False):
    """
    Calculate percentile confidence interval for a count or proportion.

    Parameters:
        box (np array or list): Box model representing population
        sample_size (int): Number of individuals (or sites, etc.) in sample
        confidence_level (float): The level of confidence interval you want (95%, 99%, etc.) This needs to be a number between 0 and 100.
        count_what (char): What character in the box model should be counted. Default "A".
        sims (int): How many simulations to run. Default 10,000
        pivotal (bool): Whether to compute a pivotal confidence interval (default True). If False, percentile will be used.
        proportion (bool): Calculate count or proportion (default count)
        return_resamples (bool): Whether to return resampling results used to generate p-value. Primarily for pedagogical purposes.

    Returns:
        confidence interval (as numpy array)
        resampling data (if desired)
    """

    dataArr = np.array(box)  #Converts box model to Numpy array

    if pivotal==True and proportion==False:  #Percentile CIs don't use Mobs
        Mobs = np.sum(dataArr==count_what)
    elif pivotal==True and proportion==True:
        Mobs = np.mean(dataArr==count_what)

    #Resampling loop
    resampleArr = np.zeros(sims)
    for i in range(sims):
        p_sample = np.random.choice(dataArr, sample_size, replace=True)  #Samples from the box model (with replacement)
        p_count = np.sum(p_sample == count_what)
        if proportion:  #Convert to proportions if desired
            p_count = p_count/sample_size
        resampleArr[i] = p_count

    #Compute confidence interval
    CIpercentile = np.percentile(resampleArr, sorted([(100-confidence_level)/2, 100-(100-confidence_level)/2]))
    if pivotal:
        CIpivotal = np.array([2*Mobs-CIpercentile[1], 2*Mobs-CIpercentile[0]])
        CI = CIpivotal
    else:
        CI = CIpercentile

    #Return results
    if return_resamples:
        return CI, resampleArr
    else:
        return CI


def confidence_interval_one_sample(data, measure_function, confidence_level=99, sims=10000, pivotal=True, return_resamples=False):
    """
    Calculates a confidence interval for a measure on a 1-D array based on the specified confidence level.

    Parameters:
        data (list or np.array): 1-D array or list of numeric data
        measure function (function name): A function that returns your measure, such as np.median or a custom function
        confidence_level (float): The confidence level expressed as a percentage. Defaults to 99.
        sims (int): How many simulations to run. Default 10,000
        pivotal (bool): Whether to compute a pivotal confidence interval (default True). If False, percentile will be used.
        return_resamples (bool): Whether to return resampling results used to generate p-value. Primarily for pedagogical purposes.

    Returns:
        confidence interval (as numpy array)
        resampling data (if desired)
    """
    # Ensure data is a numpy array for efficient operations
    dataArr = np.array(data)

    #Get sample measure
    Mobs = measure_function(dataArr)

    #Resampling loop
    resampleArr = np.zeros(sims)
    for i in range(sims):
        p_sample = np.random.choice(dataArr, len(dataArr), replace=True)  #Samples from the data (with replacement)
        p_measure = measure_function(p_sample)
        resampleArr[i] = p_measure

    #Compute confidence interval
    CIpercentile = np.percentile(resampleArr, sorted([(100-confidence_level)/2, 100-(100-confidence_level)/2]))
    if pivotal:
        CIpivotal = np.array([2*Mobs-CIpercentile[1], 2*Mobs-CIpercentile[0]])
        CI = CIpivotal
    else:
        CI = CIpercentile

    #Return results
    if return_resamples:
        return CI, resampleArr
    else:
        return CI


################################################################################################################################
#TWO-GROUP COMPARISONS

def two_group_comparison(group1, group2, measure=np.median, comparison="difference", boxes=1, two_tailed=True, num_simulations=10000, return_resamples=False):
    """
    Compares two groups of continuous data

    Inputs:
        group1, group2 (list or array): data in list or 1-D array format
        measure (function): function used to summarize data. Default: np.median
        comparison ("difference" or "ratio"): how measures should be compared. Default: "difference"
        boxes (1 or 2): specifies whether to perform a one-box or two-box test. Default: 1
        two-tailed (bool): whether to compute a two-tailed p-value. Default: True
        num_simulations (int): number of bootstrap simulations to run. Default: 10,000
        return_resamples (bool): whether to return simulated results. Default: False

    Output:
        pval (float): Simulated p-value
        resamples (numpy array) if desired

    """

    if comparison == "difference":
        stat = measure(group1) - measure(group2)
    elif comparison == "ratio":
        stat = measure(group1)/measure(group2)
    else:
        raise ValueError("Comparison must be difference or ratio.")

    size1 = len(group1)
    size2 = len(group2)

    # Generate the null distribution
    ps_stats = np.zeros(num_simulations)

    if boxes==1:
        pooled = np.concatenate([group1, group2])
        for i in range(num_simulations):
            # Resample from the pooled data to simulate the null hypothesis
            ps_group1 = np.random.choice(pooled,size1)
            ps_group2 = np.random.choice(pooled,size2)
            if comparison == "difference":
                ps_stats[i] = measure(ps_group1) - measure(ps_group2)
            elif comparison == "ratio":
                ps_stats[i] = measure(ps_group1)/measure(ps_group2)
    elif boxes==2:
        #Decenter and keep separate
        arr1 = np.array(group1) - measure(group1)
        arr2 = np.array(group2) - measure(group2)
        for i in range(num_simulations):
            # Resample from the decentered data to simulate the null hypothesis
            ps_group1 = np.random.choice(arr1,size1)
            ps_group2 = np.random.choice(arr2,size2)
            if comparison == "difference":
                ps_stats[i] = measure(ps_group1) - measure(ps_group2)
            elif comparison == "ratio":
                ps_stats[i] = measure(ps_group1)/measure(ps_group2)

    # Calculate the p-value
    pval = p_value_resampled(stat, ps_stats, two_tailed=two_tailed)
    if return_resamples == False:
        return pval
    else:
        return pval, ps_stats


def two_group_diff_CI(group1, group2, measure=np.median, confidence_level=99, CItype="pivotal", num_simulations=10000, return_resamples=False):
    """
    Generates confidence interval for two groups of data

    Inputs:
        group1, group2 (list or array): data in list or 1-D array format
        measure (function): function used to summarize data. Default: np.median
        confidence_level (float): confidence level for CI. alpha=100-confidence_level. Default:99
        CItype (string): specifies whether to compute pivotal or percentile CIs. Default: "pivotal"
        num_simulations (int): number of bootstrap simulations to run. Default: 10,000
        return_resamples (bool): whether to return simulated results. Default: False

    Output:
        CI (numpy array): confidence interval
        resamples (numpy array) if desired
    """
    Mobs = measure(group1) - measure(group2)

    size1 = len(group1)
    size2 = len(group2)

    # Generate the resampling distribution
    ps_stats = np.zeros(num_simulations)

    for i in range(num_simulations):
        # Resample from the data
        ps_group1 = np.random.choice(group1,size1)
        ps_group2 = np.random.choice(group2,size2)
        ps_stats[i] = measure(ps_group1) - measure(ps_group2)

    #Compute confidence interval
    CIpercentile = np.percentile(ps_stats, sorted([(100-confidence_level)/2, 100-(100-confidence_level)/2]))
    if CItype == "pivotal":
        CIpivotal = np.array([2*Mobs-CIpercentile[1], 2*Mobs-CIpercentile[0]])
        CI = CIpivotal
    else:
        CI = CIpercentile

    #Return results
    if return_resamples:
        return CI, ps_stats
    else:
        return CI



def paired_plot(data, group_labels=["", ""], line_color="gray", point_color="black"):
    """
    Plots connected dot plots for 2 groups of paired data and lines

    Inputs:
        data: two-column array or data frame of paired data points
        group_labels: list of what each group should be labeled (default unlabeled)
        line_color: color of connecting lines (default gray)
        point_color: color of points (default black)

    Output:
        ax: connected dot plot

    """

    dataArr = np.array(data)
    fig, ax = plt.subplots(figsize=(4, 3))

    x1=0.8
    x2=1.2
    n = dataArr.shape[0]
    for i in range(n):
        ax.plot([x1, x2], [dataArr[i,0], dataArr[i,1]], color=line_color)

        # Plot the points
        ax.scatter(n*[x1-0.01], dataArr[:,0], color=point_color, s=25, label=group_labels[0])
        ax.scatter(n*[x2+0.01], dataArr[:,1], color=point_color, s=25, label=group_labels[1])

    # Fix the axes and labels
    ax.set_xticks([x1, x2])
    _ = ax.set_xticklabels(group_labels, fontsize='x-large')

    return ax


def paired_sample_pvalue(deltas, measure_function, sims=10000, return_resamples=False):
    """
    Computes a p-value for paired data

    Inputs:
        deltas (list or 1-D array): differences between paired measurements
        measure_function (function): function that computed measure of central tendency for the deltas. Typically np.mean or np.median.
        sims (int): How many simulations to run. Default 10,000
        return_resamples (bool): Whether to return resampling results used to generate p-value. Primarily for pedagogical purposes.

    Outputs:
        p-value (two-tailed)
        resamples (numpy array) if desired
    """
    Mobs = measure_function(deltas)

    p_diffs_arr=np.zeros(sims)
    for i in range(sims):
        ones_arr=np.random.choice([1,-1], len(deltas))  #Randomly make each delta + or -
        p_diffs=deltas*ones_arr
        p_diffs_arr[i]=measure_function(p_diffs)

    pval=p_value_resampled(Mobs, p_diffs_arr, two_tailed=True)
    if return_resamples == True:
        return pval, p_diffs_arr
    else:
        return pval



###############################################################################################################################
#CATEGORICAL DATA
def compare_dimensions(observed, expected):
    if observed.shape != expected.shape:
        raise ValueError("Dimensions of observed and expected data do not match")

def calculate_chi_squared(observed, expected):
    chi_squared = ((observed - expected) ** 2 / expected).sum().sum()
    return chi_squared

#Calculating the expected Value
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
    """
    Compute chi-abs statistic

    Input:
        observed data (array or data frame): Observed counts
        expected data (array or data frame, optional): Expected counts. If not provided,
        the function will calculate an expected table that assumes equal probabilities across columns.

    Output:
        chi-abs statistic
    """

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


def resample_chi_abs(observed_data, sims=10000, with_replacement=True):
    """
    Generates a bootstrap distribution of the chi absolute statistic for an n*n contingency table.

    Parameters:
        observed_data (np.array or pd.DataFrame): n*n contingency table with observed frequencies with treatments in columns and outcomes in rows.
        num_simulations (int): Number of bootstrap samples to generate.
        with_replacement (bool): Indicates whether sampling should be with replacement.
     """

    if isinstance(observed_data, pd.DataFrame):
        observed_data = observed_data.values

    total_rows, total_columns = observed_data.shape
    expected_data = calculate_expected(observed_data) # Requires treatments in columns and outcomes in rows - TODO - allow input of expected table that is set up differently?

    results = np.zeros(sims)
    total_counts_per_column = observed_data.sum(axis=0)

    # Create a pooled data array combining all categories across rows and columns
    pooled_data = np.concatenate([np.repeat(row, sum(observed_data[row, :])) for row in range(total_rows)])

    for i in range(sims):
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


def relative_risk(observed_data, event_row_index, treatment1_index, treatment2_index):
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


def resample_relative_risk(observed_data, event_row_index, baseline_index=0, sims=10000):
    """
    Resamples relative risk from observed data

    Inputs:
        observed_data (array or data frame): 2x2 table of counts
        event_row_index (int): which row contains event of interest
        baseline_index (int): which column is the baseline for comparison
        sims (int): number of resampling simulations to run

    Output:
        array of relative risk values
    """

    # Extract the dimensions of the observed_data array
    total_rows, total_columns = observed_data.shape
    if total_rows > 2 or total_columns>2:
        raise ValueError ("This function only works for 2x2 arrays")

    # Calculate the total counts for each column (treatment group)
    total_counts_per_column = observed_data.sum(axis=0)

    # Initialize an array to store simulated RR values
    simulated_rr = np.zeros(sims)

    # Loop through the specified number of simulations
    for i in range(sims):
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

        # Calculate the probability of the event for the baseline group
        prob_event_baseline = simulated_sample[event_row_index, baseline_index] / total_counts_per_column[baseline_index]

        # Calculate the probability of the event for the other treatment group
        prob_event_other = simulated_sample[event_row_index, 1 - baseline_index] / total_counts_per_column[1 - baseline_index]

        # Calculate the probability of the event for the reference treatment group
        prob_event_treatment1 = simulated_sample[event_row_index, baseline_index] / total_counts_per_column[baseline_index]

        # Calculate the probability of the event for the other treatment group
        prob_event_other_treatment = simulated_sample[event_row_index, 1 - baseline_index] / total_counts_per_column[1 - baseline_index]

        # Calculate the Risk Ratio (RR) for the current simulated dataset and store it
        simulated_rr[i] = prob_event_other / prob_event_baseline

    # Return an array of simulated RR values
    return simulated_rr


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


def relative_risk_ci(simulated_rr_array, observed_rr, confidence_level=99, pivotal=True):
    """
    Compute confidence interval from simulated relative risk values, using exponential adjustment for pivotal CIs.

    Parameters:
        simulated_rr_array (np.array): Array of simulated relative risks
        observed_rr (float): The observed relative risk
        confidence_level (float): The level of confidence interval you want (95%, 99%, etc.) This needs to be a number between 0 and 100.
        pivotal (bool): Whether to compute a pivotal confidence interval (default True). If False, percentile will be used.
    """

    CIpercentile = np.percentile(simulated_rr_array, sorted([(100-confidence_level)/2, 100-(100-confidence_level)/2]))

    if pivotal:
        Mlower = CIpercentile[0]
        Mupper = CIpercentile[1]
        lowerbound = np.exp(2 * np.log(observed_rr) - np.log(Mupper))
        upperbound = np.exp(2 * np.log(observed_rr) - np.log(Mlower))
        CIpivotal = np.array([lowerbound, upperbound])
        CI = CIpivotal
    else:
        CI = CIpercentile

    return CI



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
#Correlation and Regression

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


def permute_correlation(x, y, sims=10000):
    """
    Generate simulated correlation coefficients by permuting one variable and calculating Pearson's correlation.

    Parameters:
    - x (np.array): Values of variable 1.
    - y (np.array): Values of variable 2.
    - sims (int, optional): Number of permutations to perform (default 10000).
    - num_simulations (int, optional): Number of permutations to perform (default 10000).

    Returns:
    - np.array: Simulated correlation coefficients.
    """
    try:
        x = np.asarray(x)
        y = np.asarray(y)
        simulated_correlations = np.zeros(sims)
        for i in range(sims):
            permuted_x = np.random.permutation(x)
            simulated_correlations[i] = pearsonr(permuted_x, y)[0]
        return simulated_correlations
    except Exception as e:
        print(f"Error generating simulated correlations: {e}")
        return np.array([])  # Return an empty array in case of error


def compute_correlation_ci(x, y, sims=10000, confidence_level=99, pivotal=True):
    """
    Compute the confidence interval around the observed correlation by resampling.

    Parameters:
    - x (np.array): Values of variable 1.
    - y (np.array): Values of variable 2.
    - sims (int, optional): Number of bootstrap samples to generate (default: 10000).
    - confidence_level (float, optional): Confidence level for the interval (default: 99).
    - pivotal (bool, optional): Whether to return a pivotal confidence interval (default: True).
    """
    try:
        Mobs = pearsonr(x, y)[0]
        simulated_correlations = np.zeros(sims)
        observed_correlation = pearsonr(x, y)[0]

        for i in range(sims):
            indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
            resampled_x = x[indices]
            resampled_y = y[indices]
            resample_correlation = pearsonr(resampled_x, resampled_y)[0]
            simulated_correlations[i]=resample_correlation

        CIpercentile = np.percentile(simulated_correlations, sorted([(100-confidence_level)/2, 100-(100-confidence_level)/2]))
        if pivotal:
            CIpivotal = np.array([2*Mobs-CIpercentile[1], 2*Mobs-CIpercentile[0]])
            CI = CIpivotal
        else:
            CI = CIpercentile

        return CI

    except Exception as e:
        print(f"An error occurred: {e}")



def regression_ci(x, y, sims=10000, confidence_level=99, return_type='both', pivotal=True):
    """
    Computes a confidence interval for a linear regression (slope, intercept, or both) via resampling.

    Inputs:
        *x and y: lists or 1-D arrays of data
        *sims (int): number of resampling runs (default: 10,000)
        *confidence_level (float): desired confidence level (default: 99)
        *return_type (string): whether to return CIs for "slope", "intercept", or "both" (default: "both")
        *pivotal (bool): whether to compute a pivotal CI (default: True)

    Output:
        *dict of slope and/or intercept CIs (as tuples)
    """
    from scipy.stats import linregress

    slopes = []
    intercepts = []
    n = len(x)
    alpha = 100 - confidence_level
    percentile_lower = alpha / 2
    percentile_upper = 100 - alpha / 2

    # Resampling and fitting
    for i in range(sims):
        sample_indices = np.random.choice(range(n), size=n, replace=True)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]

        # Using scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_sample, y_sample)
        slopes.append(slope)
        intercepts.append(intercept)

    # Preparing the output
    result = {}
    if pivotal:
        obs_slope, obs_intercept, *others = linregress(x,y)

    if return_type in ('slope', 'both'):
        slope_lower_bound = np.percentile(slopes, percentile_lower)
        slope_upper_bound = np.percentile(slopes, percentile_upper)
        result['slope'] = (slope_lower_bound, slope_upper_bound)
        if pivotal:
            result['slope'] = CI_percentile_to_pivotal(obs_slope, result['slope'])

    if return_type in ('intercept', 'both'):
        intercept_lower_bound = np.percentile(intercepts, percentile_lower)
        intercept_upper_bound = np.percentile(intercepts, percentile_upper)
        result['intercept'] = (intercept_lower_bound, intercept_upper_bound)
        if pivotal:
            result['intercept'] = CI_percentile_to_pivotal(obs_slope, result['intercept'])

    return result

# Example usage
# Assuming 'brainhead' is a DataFrame with 'Head Size' and 'Brain Weight' columns
# brainhead = pd.read_csv('path_to_your_data.csv')
# x = brainhead['Head Size'].values
# y = brainhead['Brain Weight'].values
# results = bootstrap_confidence_interval(x, y, return_type='both')  # Can be 'slope', 'intercept', or 'both'
# print(f"Results: {results}")

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
