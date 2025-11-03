import pandas as pd
import numpy as np
from scipy import stats


class Darbro:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.y = dataframe.iloc[:, 0]
        self.X = dataframe.iloc[:, 1:]
        self.X = np.insert(self.X.values, 0, 1, axis=1)  # Adding intercept
        self.coefficients = self.calculate_coefficients()
        self.residuals = None
        self.fitted = None
        self.hat_matrix = None
        self.mse = None
        self.sse = None
        self.variance_covariance = None
        # New attributes for hypothesis testing and R²
        self.t_statistics = None
        self.p_values = None
        self.f_statistic = None
        self.f_p_value = None
        self.r_squared = None
        self.adjusted_r_squared = None
        self.standard_errors = None

    def calculate_hat_matrix(self):
        X = self.X
        return X @ np.linalg.inv(X.T @ X) @ X.T

    def calculate_coefficients(self):
        X = self.X
        y = self.y.values
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def calculate_analytical_information(self):
        self.hat_matrix = self.calculate_hat_matrix()
        self.residuals = self.y.values - self.X @ self.coefficients
        self.fitted = self.X @ self.coefficients
        self.sse = np.sum(self.residuals ** 2)
        self.mse = self.sse / (len(self.y) - len(self.X[0]))
        self.variance_covariance = self.mse * np.linalg.inv(self.X.T @ self.X)
        # Calculate additional statistics
        self._calculate_r_squared()
        self._calculate_hypothesis_tests()

    def _calculate_r_squared(self):
        """
        Calculate R² and adjusted R² values.

        R² = 1 - (SSE / SST), where SST is the total sum of squares
        Adjusted R² adjusts for the number of predictors
        """
        # Total sum of squares (SST)
        y_mean = np.mean(self.y.values)
        sst = np.sum((self.y.values - y_mean) ** 2)

        # R-squared
        self.r_squared = 1 - (self.sse / sst)

        # Adjusted R-squared
        n = len(self.y)
        p = self.X.shape[1]  # Number of parameters including intercept
        self.adjusted_r_squared = 1 - ((1 - self.r_squared) * (n - 1) / (n - p))

    def _calculate_hypothesis_tests(self):
        """
        Calculate hypothesis testing statistics:
        - t-statistics for each coefficient
        - p-values for each coefficient
        - F-statistic for overall model significance
        - p-value for F-statistic
        """
        # Standard errors of coefficients
        self.standard_errors = np.sqrt(np.diag(self.variance_covariance))

        # t-statistics: t = coefficient / standard_error
        self.t_statistics = self.coefficients / self.standard_errors

        # Degrees of freedom for t-test
        n = len(self.y)
        p = self.X.shape[1]
        df_residual = n - p

        # p-values for two-tailed t-test
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), df_residual))

        # F-statistic for overall model significance
        # F = (SSR / (p-1)) / (SSE / (n-p))
        # where SSR = SST - SSE (regression sum of squares)
        y_mean = np.mean(self.y.values)
        sst = np.sum((self.y.values - y_mean) ** 2)
        ssr = sst - self.sse

        df_model = p - 1  # degrees of freedom for model (excluding intercept)

        if df_model > 0:  # Only calculate F-statistic if there are predictors
            self.f_statistic = (ssr / df_model) / (self.sse / df_residual)
            self.f_p_value = 1 - stats.f.cdf(self.f_statistic, df_model, df_residual)
        else:
            self.f_statistic = None
            self.f_p_value = None

    def get_summary(self):
        """
        Returns a formatted summary of the regression results including:
        - Coefficients with standard errors, t-statistics, and p-values
        - R², adjusted R²
        - F-statistic and its p-value
        - Model diagnostics (MSE, SSE)

        Returns:
            str: Formatted summary string
        """
        if self.t_statistics is None or self.r_squared is None:
            raise ValueError("Must call calculate_analytical_information() first")

        summary = []
        summary.append("=" * 70)
        summary.append("                     Regression Results")
        summary.append("=" * 70)

        # Model statistics
        summary.append(f"R-squared:           {self.r_squared:.4f}")
        summary.append(f"Adjusted R-squared:  {self.adjusted_r_squared:.4f}")
        summary.append(f"F-statistic:         {self.f_statistic:.4f}" if self.f_statistic else "F-statistic:         N/A")
        summary.append(f"Prob (F-statistic):  {self.f_p_value:.4e}" if self.f_p_value else "Prob (F-statistic):  N/A")
        summary.append(f"Mean Squared Error:  {self.mse:.4f}")
        summary.append(f"Sum Squared Error:   {self.sse:.4f}")
        n = len(self.y)
        p = self.X.shape[1]
        summary.append(f"No. Observations:    {n}")
        summary.append(f"Df Residuals:        {n - p}")
        summary.append(f"Df Model:            {p - 1}")
        summary.append("=" * 70)

        # Coefficients table
        summary.append("")
        summary.append("Coefficients:")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<15} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}")
        summary.append("-" * 70)

        # Get variable names from dataframe
        var_names = ['Intercept'] + list(self.dataframe.columns[1:])

        for i in range(len(self.coefficients)):
            var_name = var_names[i] if i < len(var_names) else f'x{i}'
            summary.append(
                f"{var_name:<15} {self.coefficients[i]:>12.4f} {self.standard_errors[i]:>12.4f} "
                f"{self.t_statistics[i]:>10.3f} {self.p_values[i]:>10.4f}"
            )

        summary.append("=" * 70)

        return "\n".join(summary)

    def predict(self, X_new):
        X_new = np.insert(X_new, 0, 1)  # Adding intercept
        return np.dot(X_new, self.coefficients)

    @staticmethod
    def read_csv(csv_path, y_column, predictor_columns):
        df = pd.read_csv(csv_path)
        return df[[y_column] + predictor_columns]

# Example usage
# darbro = Darbro(Darbro.read_csv('data.csv', 'bodyfat', ['triceps', 'thigh', 'midarm']))
# prediction = darbro.predict(np.array([value_X1, value_X2, value_X3]))
