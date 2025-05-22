# Darbro Statistical Library

Welcome to the Darbro Statistical Library!

This library is designed as an educational tool for individuals studying the proper application and use of common statistical methods. Our goal is to provide clear implementations of these methods along with detailed documentation to help users understand the underlying mathematics and assumptions.

Currently, the library offers functionalities for:
*   Multiple Linear Regression
*   Independent Two-Sample t-tests

We encourage you to explore the functions, delve into their implementations, and use this library as a companion in your statistical learning journey.


## Multiple Linear Regression (`Darbro` Class)

The `Darbro` class provides an implementation of multiple linear regression. This statistical method models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

**Mathematical Background:**

The formula for multiple linear regression is:

$Y = X\beta + \epsilon$

Where:
- $Y$ is the dependent variable (vector of observed values).
- $X$ is the matrix of independent variables (each row is an observation, each column is a variable, includes a column of ones for the intercept).
- $\beta$ is the vector of coefficients to be estimated.
- $\epsilon$ is the vector of random errors or residuals.

The `Darbro` class estimates the $\beta$ coefficients using the Ordinary Least Squares (OLS) method:

$\hat{\beta} = (X^T X)^{-1} X^T Y$

**Usage:**

The class is initialized with a Pandas DataFrame. The first column of the DataFrame is assumed to be the dependent variable (`y`), and the remaining columns are the independent variables (`X`).

**Key Methods:**

*   `Darbro(dataframe)`: Constructor.
    *   `dataframe`: A Pandas DataFrame where the first column is the dependent variable and subsequent columns are independent variables.
*   `Darbro.read_csv(csv_path, y_column, predictor_columns)`: A static method to read data from a CSV file and format it into the required DataFrame structure.
    *   `csv_path`: Path to the CSV file.
    *   `y_column`: Name of the column to be used as the dependent variable.
    *   `predictor_columns`: A list of column names to be used as independent variables.
*   `calculate_analytical_information()`: Calculates and stores various regression statistics like residuals, SSE, MSE, and the variance-covariance matrix of coefficients. This method should be called before accessing these attributes.
*   `predict(X_new)`: Predicts the `y` value for a new set of independent variables.
    *   `X_new`: A NumPy array containing the new independent variable values (should not include the intercept term).

**Example:**

```python
import pandas as pd
import numpy as np
from darbro import Darbro

# Sample data (replace with your actual data loading)
# For example, using the provided bodyfat.csv:
# First, ensure 'bodyfat.csv' is in your working directory.
# df = Darbro.read_csv('bodyfat.csv', y_column='bodyfat', 
# predictor_columns=['triceps', 'thigh', 'midarm'])

# Or, creating a dummy DataFrame for illustration:
data = {
    'Y_values': [10, 12, 15, 18, 20],
    'X1_values': [1, 2, 3, 4, 5],
    'X2_values': [2, 3, 2, 5, 4]
}
df = pd.DataFrame(data)

# Initialize Darbro with the DataFrame
reg_model = Darbro(df)

# The coefficients are calculated upon initialization
print(f"Coefficients: {reg_model.coefficients}")

# To get other analytical info like MSE, residuals:
reg_model.calculate_analytical_information()
print(f"MSE: {reg_model.mse}")
# print(f"Residuals: {reg_model.residuals}")

# Predict a new value
# Example: new_X_values = np.array([6, 6]) # For X1=6, X2=6
# prediction = reg_model.predict(new_X_values)
# print(f"Prediction for {new_X_values}: {prediction}")
```


## Independent Two-Sample t-test (`two_sample_t_test` function)

The `two_sample_t_test` function is used to determine if there is a statistically significant difference between the means of two independent groups.

**Mathematical Background:**

The function can perform both Student's t-test (assuming equal variances) and Welch's t-test (not assuming equal variances).

**1. Student's t-test (Equal Variances Assumed - `equal_var=True`)**

The t-statistic is calculated as:

$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$

where:
- $\bar{X}_1, \bar{X}_2$ are the sample means of group 1 and group 2.
- $n_1, n_2$ are the sample sizes of group 1 and group 2.
- $s_1^2, s_2^2$ are the sample variances of group 1 and group 2.
- $s_p$ is the pooled standard deviation, calculated as:
  $s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

The degrees of freedom ($df$) for Student's t-test are:
$df = n_1+n_2-2$

**2. Welch's t-test (Unequal Variances Assumed - `equal_var=False`)**

The t-statistic is calculated as:

$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$

The degrees of freedom ($df$) for Welch's t-test are approximated using the Welch-Satterthwaite equation:

$df \approx \frac{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$

**P-value Calculation:**

For both tests, the two-tailed p-value is determined using the t-distribution with the calculated $t$-statistic and degrees of freedom ($df$). This library calculates the p-value by implementing the cumulative distribution function (CDF) of the t-distribution using the regularized incomplete beta function, avoiding direct reliance on `scipy.stats`.

**Function Parameters:**

*   `sample1` (array-like): Data for the first group.
*   `sample2` (array-like): Data for the second group.
*   `equal_var` (bool, optional): If `True` (default), performs Student's t-test. If `False`, performs Welch's t-test.

**Returns:**

*   `t_statistic` (float): The calculated t-statistic.
*   `p_value` (float): The two-tailed p-value.

**Usage Examples:**

```python
import numpy as np
from darbro import two_sample_t_test # Assuming darbro.py contains the function

# Example Data
group1 = np.array([20, 22, 19, 20, 21, 20, 18, 25])
group2_equal_var = np.array([24, 25, 23, 26, 25, 24, 23, 28]) # Similar variance to group1
group2_unequal_var = np.array([28, 35, 32, 30, 33, 29, 37, 34]) # Different variance

# Student's t-test (assuming equal variances)
t_stat_student, p_value_student = two_sample_t_test(group1, group2_equal_var, equal_var=True)
print(f"Student's t-test: t-statistic = {t_stat_student:.4f}, p-value = {p_value_student:.4f}")

# Welch's t-test (assuming unequal variances)
t_stat_welch, p_value_welch = two_sample_t_test(group1, group2_unequal_var, equal_var=False)
print(f"Welch's t-test: t-statistic = {t_stat_welch:.4f}, p-value = {p_value_welch:.4f}")

# Example where means might be similar
group3 = np.array([20.5, 21.5, 19.5, 20.5, 21.0, 20.5, 18.5, 24.5])
t_stat_similar, p_value_similar = two_sample_t_test(group1, group3, equal_var=True)
print(f"Student's t-test (similar means): t-statistic = {t_stat_similar:.4f}, p-value = {p_value_similar:.4f}")
```


## Dependencies and Setup

This library requires the following Python packages:

*   **NumPy**: For numerical operations, especially array manipulations.
*   **Pandas**: For data manipulation, particularly for the `Darbro` regression class which uses DataFrames.

You can install these dependencies using pip:

```bash
pip install numpy pandas
```

**Note:** The unit tests (`tests.py`) also use `scipy` for comparison to validate the results of the custom `two_sample_t_test` implementation. If you intend to run the tests and verify against SciPy, you can install it via:
```bash
pip install scipy
```
However, `scipy` is *not* required for the core functionality of the `darbro` library itself.
