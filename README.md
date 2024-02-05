# Statistics_library

Library for calculating Chi abs, resampling, Bootstrapping -- LS 40 @UCLA Life science course


Statistics Library Documentation
Author: Vishanth Hari Raj
GitHub Repository: https://github.com/vishanth10/statistics_library.git
 
Function Documentation
1. read_data
The read_data function reads data from either a DataFrame or a file path. If the input is a DataFrame, it makes a copy. If it's a file path, it checks the file extension (.xls, .xlsx, .csv) and reads the data accordingly. It then applies pandas' to_numeric function to convert the data to numeric types, with non-numeric values coerced to NaN.
Usage Example:
data = read_data('path_to_file.csv') # For CSV files
data = read_data('path_to_file.xlsx') # For Excel files
data = read_data(dataframe) # For existing DataFrame
 
2. compare_dimensions
This function compares the dimensions of the observed and expected datasets. If the dimensions don't match, it raises a ValueError. This ensures that further statistical calculations are valid.
Usage Example:
compare_dimensions(observed_data, expected_data) # Checks if dimensions of observed and expected data match
 
3. calculate_chi_squared
Calculates the chi-squared statistic from observed and expected datasets. This is done by summing the squared differences between observed and expected values, divided by the expected values.
Usage Example:
chi_squared = calculate_chi_squared(observed_data, expected_data) # Returns the chi-squared statistic
 
4. calculate_expected
Computes expected frequencies for a contingency table based on the observed data. It uses row and column sums to calculate these frequencies, assuming independence between rows and columns.
Usage Example:
expected_data = calculate_expected(observed_data) # Returns the expected frequencies
 
5. calculate_chi_abs
This function calculates the chi absolute statistic for observed and expected data. If expected data is not provided, it calculates the expected data based on the observed data.
Usage Example:
chi_abs = calculate_chi_abs(observed_data, expected_data) # Calculates the chi absolute statistic
 
6. chi_abs_stat
A wrapper function for calculate_chi_abs. It allows the user to provide only observed data, with an option to provide expected data. It handles dimension comparison and error logging.
Usage Example:
chi_abs = chi_abs_stat(observed_data, expected_data) # Calculates chi absolute statistic with optional expected data
 
7. calculate_p_value
Calculates the p-value from a chi-squared statistic and degrees of freedom using the chi-squared survival function.
Usage Example:
p_value = calculate_p_value(chi_squared, dof) # Returns the p-value
 
8. chi_squared_stat
Reads observed and expected data, checks their dimensions, calculates the chi-squared statistic, and then computes the p-value.
Usage Example:
chi_squared_value = chi_squared_stat(observed_data, expected_data) # Calculates chi-squared statistic
 
9. p_value_stat
Reads observed and expected data, checks dimensions, calculates the chi-squared statistic, and then computes the p-value.
Usage Example:
p_value = p_value_stat(observed_data, expected_data) # Calculates the p-value for chi-squared statistic
 
10. convert_df_to_numpy
Converts DataFrame data to numpy arrays for use in functions, particularly for bootstrapping. It returns a tuple of numpy arrays for observed and expected data.
Usage Example:
observed_array, expected_array = convert_df_to_numpy(df_observed, df_expected) # Converts DataFrame to numpy arrays
 
11. bootstrap_chi_abs_distribution
Generates a bootstrap distribution of the chi absolute statistic for an n*n contingency table. Simulates new datasets and calculates the chi absolute for each, returning an array of these statistics.
Usage Example:
simulated_chi_abs = bootstrap_chi_abs_distribution(observed_data) # Returns array of simulated chi absolute statistics
 
12. calculate_p_value_bootstrap
Calculates the p-value for the chi absolute statistic using bootstrap methods. Compares the observed chi absolute statistic against the distribution of simulated chi absolute values to compute the p-value.
Usage Example:
p_value = calculate_p_value_bootstrap(observed_chi_abs, simulated_chi_abs) # Calculates p-value using bootstrap method
 
13. plot_chi_abs_distribution
Plots the distribution of simulated chi absolute values along with the observed chi absolute value. Shows the calculated p-value, providing a visual representation of the statistical analysis.
Usage Example:
plot_chi_abs_distribution(simulated_data, observed_data, p_value) # Plots distribution of chi absolute values
 


Extended:- Relative Risk Analysis Documentation
This documentation provides an overview and usage guide for a set of Python functions designed to calculate the relative risk between two treatments, resample data for statistical analysis, calculate confidence intervals, and plot the distribution of relative risks. These functions are intended for statistical analysis in lifesciences research or any field requiring comparative risk assessment.
 
1. calculate_relative_risk_two_treatments
Calculates the relative risk (RR) of an event occurring between two treatments. RR is a measure of the strength of association between an exposure and an outcome.
Parameters:
- observed_data: 2D array of observed data.
- event_row_index: Index of the row corresponding to the event.
- treatment1_index: Column index for the first treatment.
- treatment2_index: Column index for the second treatment.
Logic:
The function sums the total occurrences for each treatment and calculates the probability of the event for each treatment. The relative risk is then the ratio of these probabilities.
Usage Example:
relative_risk = calculate_relative_risk_two_treatments(observed_data, 0, 1, 2)
 
2. resample_and_calculate_rr
Performs resampling to calculate a distribution of relative risks through bootstrapping.
Parameters:
- observed_data: 2D array of observed data.
- event_row_index: Index of the event row.
- reference_treatment_index: Index for the reference treatment column (default is 0).
- num_simulations: Number of bootstrap simulations to perform.
Logic:
Generates simulated datasets by resampling from the observed data, maintaining the original proportions. Calculates the relative risk for each simulated dataset to create a distribution of relative risks.
Usage Example:
simulated_rr = resample_and_calculate_rr(observed_data, 0, num_simulations=10000)
 
3. calculate_confidence_interval
Calculates the 95% confidence interval for the relative risk from a distribution of simulated relative risks.
Parameters:
- simulated_rr (np.array): Array of simulated relative risks.
- percentile (float, optional): The percentile for the confidence interval, defaulting to 95%.
Logic:
The function calculates the lower and upper bounds of the confidence interval based on the specified percentile using numpy's percentile function. This provides an estimate of the interval within which the true relative risk is expected to lie with 95% certainty.
Usage Example:
lower_bound, upper_bound = calculate_confidence_interval(simulated_rr, 95)
 
4. calculate_probabilities_for_each_treatment
Calculates the probability of an event occurring for each treatment group.
Parameters:
- observed_data (np.array or pd.DataFrame): The observed data.
- event_row_index (int): The row index for the event.
Logic:
For each treatment group, the function calculates the total counts and then determines the probability of the event. These probabilities are stored in a dictionary, providing a quick reference for each treatment's event probability.
Usage Example:
probabilities = calculate_probabilities_for_each_treatment(observed_data, 0)
 
5. plot_relative_risk_distribution
Plots the distribution of simulated relative risks against the observed relative risk, including confidence intervals.
Parameters:
- simulated_rr (np.array): Array of simulated relative risks.
- observed_rr (float): The observed relative risk.
Logic:
This function visualizes the distribution of simulated relative risks, highlighting the observed relative risk and marking the confidence intervals. It uses seaborn and matplotlib for plotting.
Usage Example:
plot_relative_risk_distribution(simulated_rr, observed_rr)


APPENDIX - I

bootstrap_chi_abs_distribution :
The bootstrap_chi_abs_distribution function is designed for generating a bootstrap distribution of the chi absolute statistic for an n*n contingency table. This is crucial in statistical analysis, especially when assessing the significance of observed frequencies against expected frequencies in categorical data.
Parameters:
- observed_data: Observed frequencies in an n*n contingency table, either a numpy array or pandas DataFrame.
- num_simulations: Number of bootstrap samples to generate.
- with_replacement: Boolean indicating whether sampling should be with replacement.
Process:
1. Convert DataFrame to Numpy Array: If observed_data is a DataFrame, it's converted to a numpy array.
2. Determine Dimensions and Calculate Expected Frequencies: Calculates total rows, total columns, and expected frequencies.
3. Initialize Results Array: An array to store chi absolute statistics for each simulation.
4. Calculate Total Counts Per Column: Computes the total count of observations per column.
5. Create Pooled Data Array: Concatenates all categories for resampling.
6. Bootstrap Simulation Loop: Iterates num_simulations times, creating and analyzing simulated datasets.
7. Calculate Chi Absolute Statistic: Computes chi absolute statistic for each simulated dataset.
8. Return Results: Returns the array containing chi absolute statistics from all simulations.
Usage:
This function is used in statistical hypothesis testing to assess the significance of the observed data. By comparing the observed chi absolute statistic to a distribution generated through bootstrapping, one can infer the likelihood of observing such a statistic under the null hypothesis.



APPENDIX II
 

