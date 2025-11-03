# Darbro

A Python library for exploring linear regression and multiple linear regression using matrix algebra and ordinary least squares (OLS) estimation.

## Overview

Darbro is a lightweight educational library designed to help understand the mathematical foundations of linear regression. It implements regression analysis using matrix operations, providing insight into the underlying statistical calculations.

## Features

- **Simple and Multiple Linear Regression**: Supports both single and multiple predictor variables
- **Matrix-Based Calculations**: Uses numpy for efficient matrix operations
- **Comprehensive Statistics**: Calculates coefficients, residuals, MSE, SSE, hat matrix, and variance-covariance matrix
- **Hypothesis Testing**: t-statistics, p-values for coefficients, and F-test for overall model significance
- **Model Fit Metrics**: R² and adjusted R² for evaluating model performance
- **Statistical Summary**: Formatted regression results table similar to R or statsmodels
- **Prediction**: Make predictions on new data points
- **Confidence Intervals**: Confidence intervals for mean response and prediction intervals for individual observations
- **CSV Integration**: Easy data loading from CSV files
- **Educational Focus**: Clear implementation showing the mathematical operations

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy
```

Then clone or download the repository and import the `Darbro` class.

## Quick Start

```python
import numpy as np
from darbro import Darbro

# Load data from CSV
df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])

# Create Darbro instance
model = Darbro(df)

# Calculate analytical information
model.calculate_analytical_information()

# View coefficients
print("Coefficients:", model.coefficients)

# Make a prediction
new_data = np.array([25.0, 50.0, 28.0])
prediction = model.predict(new_data)
print("Prediction:", prediction)
```

## API Documentation

### Class: `Darbro`

#### Constructor

```python
Darbro(dataframe)
```

**Parameters:**
- `dataframe` (pandas.DataFrame): A DataFrame where the first column is the dependent variable (y) and remaining columns are predictor variables (X)

**Automatically Calculated:**
- `coefficients`: Regression coefficients (β) including intercept
- `X`: Design matrix with intercept column added
- `y`: Dependent variable vector

**After calling `calculate_analytical_information()`:**
All statistical measures and hypothesis tests are computed automatically.

**Example:**
```python
import pandas as pd
from darbro import Darbro

# Create DataFrame (first column is y, rest are X variables)
data = pd.DataFrame({
    'y': [1, 2, 3, 4, 5],
    'x1': [2, 4, 6, 8, 10],
    'x2': [1, 3, 5, 7, 9]
})

model = Darbro(data)
```

#### Attributes

After calling `calculate_analytical_information()`, the following attributes are available:

**Basic Statistics:**
- **`coefficients`** (numpy.ndarray): Regression coefficients [β₀, β₁, β₂, ...]
- **`residuals`** (numpy.ndarray): Residuals (y - ŷ)
- **`fitted`** (numpy.ndarray): Fitted values (ŷ)
- **`hat_matrix`** (numpy.ndarray): Hat matrix (H = X(X'X)⁻¹X')
- **`mse`** (float): Mean Squared Error (SSE / (n - p))
- **`sse`** (float): Sum of Squared Errors
- **`variance_covariance`** (numpy.ndarray): Variance-covariance matrix of coefficients

**Hypothesis Testing:**
- **`t_statistics`** (numpy.ndarray): t-statistics for each coefficient
- **`p_values`** (numpy.ndarray): Two-tailed p-values for each coefficient
- **`standard_errors`** (numpy.ndarray): Standard errors of coefficients
- **`f_statistic`** (float): F-statistic for overall model significance
- **`f_p_value`** (float): p-value for F-statistic

**Model Fit:**
- **`r_squared`** (float): R² (coefficient of determination)
- **`adjusted_r_squared`** (float): Adjusted R² (penalized for number of predictors)

#### Methods

##### `calculate_coefficients()`

Calculates regression coefficients using the OLS formula: **β = (X'X)⁻¹X'y**

**Returns:** numpy.ndarray of coefficients

**Called automatically in constructor.**

##### `calculate_hat_matrix()`

Calculates the hat matrix: **H = X(X'X)⁻¹X'**

**Returns:** numpy.ndarray (hat matrix)

The hat matrix projects y onto the fitted values: ŷ = Hy

##### `calculate_analytical_information()`

Calculates all analytical statistics including:
- Hat matrix
- Residuals and fitted values
- Sum of Squared Errors (SSE) and Mean Squared Error (MSE)
- Variance-covariance matrix
- R² and adjusted R²
- t-statistics, p-values, and standard errors for coefficients
- F-statistic and p-value for overall model significance

**Must be called before accessing statistical measures.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()

print("R²:", model.r_squared)
print("F-statistic:", model.f_statistic)
print("P-values:", model.p_values)
```

##### `get_summary()`

Returns a formatted summary of regression results similar to R or statsmodels output.

**Returns:** str (formatted summary table)

**Must be called after `calculate_analytical_information()`.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()
print(model.get_summary())
```

**Output:**
```
======================================================================
                     Regression Results
======================================================================
R-squared:           0.8014
Adjusted R-squared:  0.7641
F-statistic:         21.5157
Prob (F-statistic):  7.3433e-06
Mean Squared Error:  6.1503
Sum Squared Error:   98.4049
No. Observations:    20
Df Residuals:        16
Df Model:            3
======================================================================

Coefficients:
----------------------------------------------------------------------
Variable                Coef      Std Err          t      P>|t|
----------------------------------------------------------------------
Intercept           117.0847      99.7824      1.173     0.2578
triceps               4.3341       3.0155      1.437     0.1699
thigh                -2.8568       2.5820     -1.106     0.2849
midarm               -2.1861       1.5955     -1.370     0.1896
======================================================================
```

##### `predict(X_new)`

Makes a prediction for new data.

**Parameters:**
- `X_new` (numpy.ndarray): New predictor values (without intercept)

**Returns:** float (predicted value)

**Example:**
```python
# For a model with 3 predictors
prediction = model.predict(np.array([25.0, 50.0, 28.0]))
```

##### `confidence_interval_mean(X_new, alpha=0.05)`

Calculates a confidence interval for the mean response E[Y|X].

This interval estimates where the average value of Y is likely to fall for a given X, with a specified confidence level.

**Parameters:**
- `X_new` (numpy.ndarray): New predictor values (without intercept)
- `alpha` (float): Significance level (default=0.05 for 95% confidence)

**Returns:** tuple (lower_bound, point_estimate, upper_bound)

**Must be called after `calculate_analytical_information()`.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()

# Get 95% confidence interval for mean response
lower, prediction, upper = model.confidence_interval_mean(np.array([25.0, 50.0, 28.0]))
print(f"95% CI for mean: [{lower:.2f}, {upper:.2f}]")
```

##### `prediction_interval(X_new, alpha=0.05)`

Calculates a prediction interval for a new individual observation Y|X.

This interval estimates where a single new observation of Y is likely to fall for a given X. It is wider than the confidence interval for the mean because it accounts for both the uncertainty in estimating the mean and the random variation of individual observations.

**Parameters:**
- `X_new` (numpy.ndarray): New predictor values (without intercept)
- `alpha` (float): Significance level (default=0.05 for 95% confidence)

**Returns:** tuple (lower_bound, point_estimate, upper_bound)

**Must be called after `calculate_analytical_information()`.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()

# Get 95% prediction interval for new observation
lower, prediction, upper = model.prediction_interval(np.array([25.0, 50.0, 28.0]))
print(f"95% PI for new observation: [{lower:.2f}, {upper:.2f}]")
```

##### `predict_with_intervals(X_new, alpha=0.05)`

Makes a prediction with both confidence and prediction intervals in one call.

**Parameters:**
- `X_new` (numpy.ndarray): New predictor values (without intercept)
- `alpha` (float): Significance level (default=0.05 for 95% confidence)

**Returns:** dict containing:
- `prediction`: Point estimate
- `confidence_interval`: Tuple (lower, upper) for mean response
- `prediction_interval`: Tuple (lower, upper) for new observation
- `confidence_level`: Confidence level as percentage

**Must be called after `calculate_analytical_information()`.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()

results = model.predict_with_intervals(np.array([25.0, 50.0, 28.0]))
print(f"Prediction: {results['prediction']:.2f}")
print(f"95% CI: {results['confidence_interval']}")
print(f"95% PI: {results['prediction_interval']}")
```

##### `read_csv(csv_path, y_column, predictor_columns)` [Static Method]

Reads a CSV file and extracts specified columns.

**Parameters:**
- `csv_path` (str): Path to CSV file
- `y_column` (str): Name of dependent variable column
- `predictor_columns` (list): List of predictor variable column names

**Returns:** pandas.DataFrame with y as first column, predictors as remaining columns

**Example:**
```python
df = Darbro.read_csv('data.csv', 'price', ['sqft', 'bedrooms', 'age'])
```

## Detailed Examples

### Example 1: Simple Linear Regression

```python
import pandas as pd
import numpy as np
from darbro import Darbro

# Create simple dataset
data = pd.DataFrame({
    'sales': [10, 15, 20, 25, 30],
    'advertising': [1, 2, 3, 4, 5]
})

# Fit model
model = Darbro(data)
model.calculate_analytical_information()

# Display results
print("Intercept (β₀):", model.coefficients[0])
print("Slope (β₁):", model.coefficients[1])
print("R²:", model.r_squared)
print("F-statistic p-value:", model.f_p_value)

# Make predictions
new_advertising = np.array([6.0])
predicted_sales = model.predict(new_advertising)
print(f"Predicted sales for advertising=6: {predicted_sales}")
```

### Example 2: Multiple Linear Regression (Body Fat Dataset)

```python
import numpy as np
from darbro import Darbro

# Load body fat data
df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])

# Create model
model = Darbro(df)
model.calculate_analytical_information()

# Display comprehensive results
print("=== Regression Coefficients ===")
print(f"Intercept: {model.coefficients[0]:.3f}")
print(f"Triceps: {model.coefficients[1]:.3f}")
print(f"Thigh: {model.coefficients[2]:.3f}")
print(f"Midarm: {model.coefficients[3]:.3f}")

print("\n=== Model Statistics ===")
print(f"Sum of Squared Errors (SSE): {model.sse:.3f}")
print(f"Mean Squared Error (MSE): {model.mse:.3f}")
print(f"Root MSE: {np.sqrt(model.mse):.3f}")

print("\n=== Hypothesis Tests ===")
print(f"R²: {model.r_squared:.4f}")
print(f"Adjusted R²: {model.adjusted_r_squared:.4f}")
print(f"F-statistic: {model.f_statistic:.3f}")
print(f"F p-value: {model.f_p_value:.6f}")

print("\n=== Coefficient Tests ===")
var_names = ['Intercept', 'Triceps', 'Thigh', 'Midarm']
for i, name in enumerate(var_names):
    print(f"{name}: coef={model.coefficients[i]:.3f}, "
          f"SE={model.standard_errors[i]:.3f}, "
          f"t={model.t_statistics[i]:.3f}, "
          f"p={model.p_values[i]:.4f}")

# Make prediction for new individual
new_person = np.array([25.0, 50.0, 28.0])  # triceps, thigh, midarm
predicted_bodyfat = model.predict(new_person)
print(f"\n=== Prediction ===")
print(f"Predicted body fat: {predicted_bodyfat:.2f}%")

# Examine residuals
print("\n=== Residual Analysis ===")
print(f"Sum of residuals: {np.sum(model.residuals):.6f} (should be ~0)")
print(f"Min residual: {np.min(model.residuals):.3f}")
print(f"Max residual: {np.max(model.residuals):.3f}")
```

### Example 3: Understanding the Hat Matrix

```python
import numpy as np
from darbro import Darbro

df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
model = Darbro(df)
model.calculate_analytical_information()

# The hat matrix "puts the hat on y" to get ŷ
print("Hat matrix shape:", model.hat_matrix.shape)
print("Hat matrix diagonal (leverage values):")
print(np.diag(model.hat_matrix))

# Verify: H * y = ŷ
fitted_via_hat = model.hat_matrix @ model.y.values
print("\nVerification - Fitted values match:")
print(np.allclose(fitted_via_hat, model.fitted))
```

### Example 4: Variance-Covariance Matrix

```python
from darbro import Darbro
import numpy as np

df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
model = Darbro(df)
model.calculate_analytical_information()

print("Variance-Covariance Matrix:")
print(model.variance_covariance)

print("\nStandard Errors of Coefficients:")
std_errors = np.sqrt(np.diag(model.variance_covariance))
for i, se in enumerate(std_errors):
    var_name = ['Intercept', 'Triceps', 'Thigh', 'Midarm'][i] if i < 4 else f'β{i}'
    print(f"{var_name}: {se:.4f}")

print("\nCorrelation between coefficient estimates:")
corr_matrix = np.corrcoef(model.variance_covariance)
print(corr_matrix)
```

### Example 5: Confidence and Prediction Intervals

```python
from darbro import Darbro
import numpy as np

# Load data and fit model
df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
model = Darbro(df)
model.calculate_analytical_information()

# New observation to predict
new_person = np.array([25.0, 50.0, 28.0])  # triceps, thigh, midarm

# Method 1: Get both intervals at once
results = model.predict_with_intervals(new_person, alpha=0.05)
print("=== Prediction with Intervals ===")
print(f"Point Estimate: {results['prediction']:.2f}%")
print(f"95% Confidence Interval for mean: [{results['confidence_interval'][0]:.2f}, {results['confidence_interval'][1]:.2f}]")
print(f"95% Prediction Interval for individual: [{results['prediction_interval'][0]:.2f}, {results['prediction_interval'][1]:.2f}]")

# Method 2: Get intervals separately
print("\n=== Individual Method Calls ===")

# Confidence interval for the mean response
ci_lower, prediction, ci_upper = model.confidence_interval_mean(new_person)
print(f"\nConfidence Interval (Mean Response):")
print(f"  We are 95% confident that the average body fat percentage")
print(f"  for people with these measurements is between {ci_lower:.2f}% and {ci_upper:.2f}%")

# Prediction interval for a new observation
pi_lower, prediction, pi_upper = model.prediction_interval(new_person)
print(f"\nPrediction Interval (New Observation):")
print(f"  We are 95% confident that a single person with these measurements")
print(f"  will have a body fat percentage between {pi_lower:.2f}% and {pi_upper:.2f}%")

# Different confidence levels
print("\n=== Different Confidence Levels ===")
for conf_level in [0.90, 0.95, 0.99]:
    alpha = 1 - conf_level
    ci_lower, pred, ci_upper = model.confidence_interval_mean(new_person, alpha=alpha)
    pi_lower, pred, pi_upper = model.prediction_interval(new_person, alpha=alpha)
    print(f"\n{int(conf_level*100)}% Intervals:")
    print(f"  Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Prediction Interval: [{pi_lower:.2f}, {pi_upper:.2f}]")
```

**Key Differences:**
- **Confidence Interval**: Estimates the average response for a given X (narrower)
- **Prediction Interval**: Estimates a single new observation for a given X (wider)
- The prediction interval is always wider because it accounts for individual variation

## Mathematical Background

### Ordinary Least Squares (OLS)

The regression model is: **y = Xβ + ε**

Where:
- y is the n×1 vector of dependent variable observations
- X is the n×p design matrix (including intercept column)
- β is the p×1 vector of coefficients
- ε is the n×1 vector of errors

### Key Formulas

1. **Coefficients**: β = (X'X)⁻¹X'y
2. **Fitted Values**: ŷ = Xβ = Hy, where H is the hat matrix
3. **Hat Matrix**: H = X(X'X)⁻¹X'
4. **Residuals**: e = y - ŷ
5. **Sum of Squared Errors**: SSE = Σ(yᵢ - ŷᵢ)² = e'e
6. **Mean Squared Error**: MSE = SSE / (n - p)
7. **Variance-Covariance Matrix**: Var(β) = MSE × (X'X)⁻¹

## Use Cases

- **Educational**: Understanding the matrix algebra behind regression
- **Statistical Learning**: Exploring relationships between variables
- **Prediction**: Forecasting continuous outcomes
- **Feature Analysis**: Understanding which predictors are important
- **Research**: Quick regression analysis for small to medium datasets

## Limitations

- No regularization (Ridge, Lasso)
- No residual diagnostic plots
- Assumes no severe multicollinearity issues
- No automatic handling of categorical variables
- Requires manual data preprocessing

## Data Format

Input DataFrame must have:
- **First column**: Dependent variable (y)
- **Remaining columns**: Predictor variables (X₁, X₂, ..., Xₚ)

The intercept is added automatically.

## Contributing

This is an educational library. Suggestions for improvements are welcome!

## Future Enhancements

Planned features for future versions:
- ~~Hypothesis testing (t-statistics, p-values, F-test)~~ ✅ Implemented
- ~~R² and adjusted R²~~ ✅ Implemented
- ~~Confidence intervals for predictions~~ ✅ Implemented
- Residual diagnostics and plots
- Support for polynomial regression
- Cross-validation utilities
- Logistic regression
- Other statistical tests and methods

## License

Open source - free to use and modify for educational purposes.

## Example Dataset

The included `bodyfat.csv` contains data on body fat percentage and body measurements (triceps, thigh, midarm skinfold measurements) for 20 subjects, commonly used in regression textbooks.

## References

- Linear regression theory and OLS estimation
- Matrix formulation of regression analysis
- Statistical inference in linear models

---

**Darbro** - Explore the mathematics of linear regression!
