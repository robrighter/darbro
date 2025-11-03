# Darbro

A Python library for exploring linear regression and multiple linear regression using matrix algebra and ordinary least squares (OLS) estimation.

## Overview

Darbro is a lightweight educational library designed to help understand the mathematical foundations of linear regression. It implements regression analysis using matrix operations, providing insight into the underlying statistical calculations.

## Features

- **Simple and Multiple Linear Regression**: Supports both single and multiple predictor variables
- **Matrix-Based Calculations**: Uses numpy for efficient matrix operations
- **Comprehensive Statistics**: Calculates coefficients, residuals, MSE, SSE, hat matrix, and variance-covariance matrix
- **Prediction**: Make predictions on new data points
- **CSV Integration**: Easy data loading from CSV files
- **Educational Focus**: Clear implementation showing the mathematical operations

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy pandas
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

- **`coefficients`** (numpy.ndarray): Regression coefficients [β₀, β₁, β₂, ...]
- **`residuals`** (numpy.ndarray): Residuals (y - ŷ)
- **`fitted`** (numpy.ndarray): Fitted values (ŷ)
- **`hat_matrix`** (numpy.ndarray): Hat matrix (H = X(X'X)⁻¹X')
- **`mse`** (float): Mean Squared Error (SSE / (n - p))
- **`sse`** (float): Sum of Squared Errors
- **`variance_covariance`** (numpy.ndarray): Variance-covariance matrix of coefficients

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
- Residuals
- Fitted values
- Sum of Squared Errors (SSE)
- Mean Squared Error (MSE)
- Variance-covariance matrix

**Must be called before accessing residuals, MSE, SSE, etc.**

**Example:**
```python
model = Darbro(df)
model.calculate_analytical_information()

print("MSE:", model.mse)
print("SSE:", model.sse)
print("Residuals:", model.residuals)
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
print("Mean Squared Error:", model.mse)
print("R² could be calculated from SSE and SST")

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

print("\n=== Standard Errors ===")
std_errors = np.sqrt(np.diag(model.variance_covariance))
for i, se in enumerate(std_errors):
    print(f"SE(β{i}): {se:.3f}")

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

- No built-in hypothesis testing (t-tests, F-tests)
- No R² calculation (can be computed manually)
- No regularization (Ridge, Lasso)
- Assumes no multicollinearity issues
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
- Hypothesis testing (t-statistics, p-values, F-test)
- R² and adjusted R²
- Confidence intervals for predictions
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
