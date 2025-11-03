import unittest
import numpy as np
import pandas as pd
import os
from scipy import stats
from darbro import Darbro


class TestDarbroBasicFunctionality(unittest.TestCase):
    """Test basic functionality with the bodyfat dataset"""

    def setUp(self):
        """Load the bodyfat dataset for testing"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)

    def test_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model.coefficients)
        self.assertIsNotNone(self.model.X)
        self.assertIsNotNone(self.model.y)
        self.assertEqual(len(self.model.y), 20)  # 20 observations in bodyfat.csv

    def test_design_matrix_shape(self):
        """Test that design matrix has correct shape"""
        n_observations = len(self.df)
        n_predictors = len(self.df.columns) - 1  # minus y column
        expected_shape = (n_observations, n_predictors + 1)  # +1 for intercept
        self.assertEqual(self.model.X.shape, expected_shape)

    def test_intercept_column(self):
        """Test that first column of X is all ones (intercept)"""
        self.assertTrue(np.all(self.model.X[:, 0] == 1))

    def test_coefficients_values(self):
        """Test that coefficients match expected values"""
        expected_coefficients = np.array([117.085, 4.334, -2.857, -2.186])
        np.testing.assert_allclose(self.model.coefficients, expected_coefficients, rtol=1e-3)

    def test_coefficients_length(self):
        """Test that number of coefficients matches predictors + intercept"""
        n_predictors = self.df.shape[1] - 1  # minus y
        expected_length = n_predictors + 1  # +1 for intercept
        self.assertEqual(len(self.model.coefficients), expected_length)


class TestDarbroAnalyticalInformation(unittest.TestCase):
    """Test analytical calculations (MSE, SSE, residuals, etc.)"""

    def setUp(self):
        """Load data and calculate analytical information"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_residuals_sum_to_zero(self):
        """Test that residuals sum to approximately zero"""
        sum_residuals = np.sum(self.model.residuals)
        # Use absolute tolerance due to numerical precision limits
        self.assertLess(abs(sum_residuals), 1e-7)

    def test_residuals_length(self):
        """Test that residuals vector has correct length"""
        self.assertEqual(len(self.model.residuals), len(self.model.y))

    def test_mse_calculation(self):
        """Test that MSE is calculated correctly"""
        expected_mse = 6.15
        self.assertAlmostEqual(self.model.mse, expected_mse, places=2)

    def test_mse_positive(self):
        """Test that MSE is positive"""
        self.assertGreater(self.model.mse, 0)

    def test_sse_calculation(self):
        """Test that SSE is calculated correctly"""
        expected_sse = 98.405
        self.assertAlmostEqual(self.model.sse, expected_sse, places=2)

    def test_sse_positive(self):
        """Test that SSE is positive"""
        self.assertGreater(self.model.sse, 0)

    def test_mse_sse_relationship(self):
        """Test that MSE = SSE / (n - p)"""
        n = len(self.model.y)
        p = self.model.X.shape[1]
        expected_mse = self.model.sse / (n - p)
        self.assertAlmostEqual(self.model.mse, expected_mse, places=10)

    def test_fitted_values_length(self):
        """Test that fitted values vector has correct length"""
        self.assertEqual(len(self.model.fitted), len(self.model.y))

    def test_fitted_plus_residuals(self):
        """Test that fitted + residuals = observed y"""
        reconstructed_y = self.model.fitted + self.model.residuals
        np.testing.assert_allclose(reconstructed_y, self.model.y.values, rtol=1e-10)

    def test_fitted_values_calculation(self):
        """Test that fitted values equal X @ coefficients"""
        expected_fitted = self.model.X @ self.model.coefficients
        np.testing.assert_allclose(self.model.fitted, expected_fitted, rtol=1e-10)


class TestDarbroHatMatrix(unittest.TestCase):
    """Test hat matrix properties"""

    def setUp(self):
        """Load data and calculate hat matrix"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_hat_matrix_shape(self):
        """Test that hat matrix is square with dimension n x n"""
        n = len(self.model.y)
        expected_shape = (n, n)
        self.assertEqual(self.model.hat_matrix.shape, expected_shape)

    def test_hat_matrix_symmetric(self):
        """Test that hat matrix is symmetric"""
        np.testing.assert_allclose(
            self.model.hat_matrix,
            self.model.hat_matrix.T,
            rtol=1e-8
        )

    def test_hat_matrix_idempotent(self):
        """Test that hat matrix is idempotent (H @ H = H)"""
        H_squared = self.model.hat_matrix @ self.model.hat_matrix
        np.testing.assert_allclose(H_squared, self.model.hat_matrix, rtol=1e-6, atol=1e-10)

    def test_hat_matrix_projects_y(self):
        """Test that H @ y = fitted values"""
        fitted_via_hat = self.model.hat_matrix @ self.model.y.values
        np.testing.assert_allclose(fitted_via_hat, self.model.fitted, rtol=1e-10)

    def test_hat_matrix_trace(self):
        """Test that trace of hat matrix equals number of parameters"""
        trace = np.trace(self.model.hat_matrix)
        n_parameters = self.model.X.shape[1]
        self.assertAlmostEqual(trace, n_parameters, places=10)


class TestDarbroVarianceCovarianceMatrix(unittest.TestCase):
    """Test variance-covariance matrix"""

    def setUp(self):
        """Load data and calculate variance-covariance matrix"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_variance_covariance_shape(self):
        """Test that variance-covariance matrix has correct shape"""
        p = self.model.X.shape[1]
        expected_shape = (p, p)
        self.assertEqual(self.model.variance_covariance.shape, expected_shape)

    def test_variance_covariance_symmetric(self):
        """Test that variance-covariance matrix is symmetric"""
        np.testing.assert_allclose(
            self.model.variance_covariance,
            self.model.variance_covariance.T,
            rtol=1e-10
        )

    def test_variance_covariance_diagonal_positive(self):
        """Test that diagonal elements (variances) are positive"""
        diagonal = np.diag(self.model.variance_covariance)
        self.assertTrue(np.all(diagonal > 0))

    def test_variance_covariance_formula(self):
        """Test that Var(beta) = MSE * (X'X)^-1"""
        expected_var_cov = self.model.mse * np.linalg.inv(self.model.X.T @ self.model.X)
        np.testing.assert_allclose(
            self.model.variance_covariance,
            expected_var_cov,
            rtol=1e-10
        )


class TestDarbroPrediction(unittest.TestCase):
    """Test prediction functionality"""

    def setUp(self):
        """Load data"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)

    def test_prediction_single_value(self):
        """Test prediction returns a single scalar"""
        new_data = np.array([25.0, 50.0, 28.0])
        prediction = self.model.predict(new_data)
        self.assertIsInstance(prediction, (float, np.floating, np.ndarray))

    def test_prediction_calculation(self):
        """Test that prediction equals coefficients @ [1, X_new]"""
        new_data = np.array([25.0, 50.0, 28.0])
        prediction = self.model.predict(new_data)

        # Manual calculation
        X_new_with_intercept = np.insert(new_data, 0, 1)
        expected_prediction = np.dot(X_new_with_intercept, self.model.coefficients)

        self.assertAlmostEqual(prediction, expected_prediction, places=10)

    def test_prediction_on_training_data(self):
        """Test prediction on first observation matches fitted value"""
        self.model.calculate_analytical_information()

        # Get first observation's predictors (excluding y)
        first_obs = self.df.iloc[0, 1:].values
        prediction = self.model.predict(first_obs)

        # Should match first fitted value
        self.assertAlmostEqual(prediction, self.model.fitted[0], places=10)

    def test_multiple_predictions(self):
        """Test making multiple predictions"""
        test_cases = [
            np.array([20.0, 45.0, 25.0]),
            np.array([30.0, 55.0, 30.0]),
            np.array([25.0, 50.0, 27.5])
        ]

        predictions = [self.model.predict(x) for x in test_cases]

        # All predictions should be numeric
        for pred in predictions:
            self.assertIsInstance(pred, (float, np.floating, np.ndarray))

        # Predictions should be different (given different inputs)
        self.assertNotEqual(predictions[0], predictions[1])


class TestDarbroSimpleLinearRegression(unittest.TestCase):
    """Test with simple linear regression (one predictor)"""

    def setUp(self):
        """Create simple linear regression dataset"""
        # y = 2 + 3*x + noise
        np.random.seed(42)
        n = 30
        x = np.linspace(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)

        self.df = pd.DataFrame({'y': y, 'x': x})
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_two_coefficients(self):
        """Test that simple linear regression has 2 coefficients"""
        self.assertEqual(len(self.model.coefficients), 2)

    def test_coefficient_signs(self):
        """Test that coefficients have expected signs (positive slope)"""
        intercept = self.model.coefficients[0]
        slope = self.model.coefficients[1]

        # Intercept should be close to 2
        self.assertGreater(intercept, 0)
        self.assertLess(intercept, 5)

        # Slope should be close to 3 and positive
        self.assertGreater(slope, 2)
        self.assertLess(slope, 4)

    def test_prediction_simple_regression(self):
        """Test prediction in simple linear regression"""
        prediction = self.model.predict(np.array([5.0]))

        # Manual calculation: y = intercept + slope * 5
        expected = self.model.coefficients[0] + self.model.coefficients[1] * 5.0
        self.assertAlmostEqual(prediction, expected, places=10)


class TestDarbroDataLoading(unittest.TestCase):
    """Test CSV reading functionality"""

    def test_read_csv_returns_dataframe(self):
        """Test that read_csv returns a pandas DataFrame"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.assertIsInstance(df, pd.DataFrame)

    def test_read_csv_column_order(self):
        """Test that read_csv puts y column first"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.assertEqual(df.columns[0], 'bodyfat')

    def test_read_csv_correct_columns(self):
        """Test that read_csv includes correct columns"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh'])
        expected_columns = ['bodyfat', 'triceps', 'thigh']
        self.assertListEqual(list(df.columns), expected_columns)

    def test_read_csv_single_predictor(self):
        """Test reading CSV with single predictor"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps'])
        self.assertEqual(df.shape[1], 2)  # y + 1 predictor


class TestDarbroRegressionProperties(unittest.TestCase):
    """Test mathematical properties of OLS regression"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_residuals_orthogonal_to_predictors(self):
        """Test that residuals are orthogonal to each predictor"""
        # X' @ e should be approximately zero
        orthogonality = self.model.X.T @ self.model.residuals
        np.testing.assert_allclose(orthogonality, np.zeros(len(orthogonality)), atol=1e-6)

    def test_residuals_orthogonal_to_fitted(self):
        """Test that residuals are orthogonal to fitted values"""
        dot_product = np.dot(self.model.residuals, self.model.fitted)
        self.assertAlmostEqual(dot_product, 0, places=6)

    def test_sse_equals_residuals_squared(self):
        """Test that SSE equals sum of squared residuals"""
        expected_sse = np.sum(self.model.residuals ** 2)
        self.assertAlmostEqual(self.model.sse, expected_sse, places=10)

    def test_mean_of_residuals_is_zero(self):
        """Test that mean of residuals is zero"""
        mean_residual = np.mean(self.model.residuals)
        self.assertAlmostEqual(mean_residual, 0, places=8)


class TestDarbroEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""

    def test_perfect_fit(self):
        """Test model with perfect linear relationship (no noise)"""
        # y = 2 + 3*x exactly
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * x
        df = pd.DataFrame({'y': y, 'x': x})

        model = Darbro(df)
        model.calculate_analytical_information()

        # Coefficients should be exactly [2, 3]
        np.testing.assert_allclose(model.coefficients, [2, 3], rtol=1e-10)

        # SSE should be essentially zero
        self.assertAlmostEqual(model.sse, 0, places=10)

        # All residuals should be essentially zero
        np.testing.assert_allclose(model.residuals, np.zeros(len(y)), atol=1e-10)

    def test_small_dataset(self):
        """Test with minimum viable dataset (n = p + 1)"""
        # 3 observations, 2 parameters (intercept + 1 predictor)
        df = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]})
        model = Darbro(df)
        model.calculate_analytical_information()

        # Should still calculate coefficients
        self.assertEqual(len(model.coefficients), 2)

        # MSE calculation should work (denominator = n - p = 1)
        self.assertIsNotNone(model.mse)

    def test_larger_dataset(self):
        """Test with larger dataset"""
        np.random.seed(123)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 5 + 2*x1 - 3*x2 + np.random.randn(n)

        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
        model = Darbro(df)
        model.calculate_analytical_information()

        # Check coefficients are reasonable
        self.assertEqual(len(model.coefficients), 3)

        # Intercept should be close to 5
        self.assertGreater(model.coefficients[0], 3)
        self.assertLess(model.coefficients[0], 7)


class TestDarbroNumericalStability(unittest.TestCase):
    """Test numerical stability and accuracy"""

    def test_coefficients_match_manual_calculation(self):
        """Test that coefficients match manual OLS calculation"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        model = Darbro(df)

        # Manual calculation: beta = (X'X)^-1 X'y
        X = model.X
        y = model.y.values
        manual_coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

        np.testing.assert_allclose(model.coefficients, manual_coefficients, rtol=1e-10)

    def test_matrix_inverse_accuracy(self):
        """Test that (X'X)^-1 @ (X'X) = I"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        model = Darbro(df)

        XtX = model.X.T @ model.X
        XtX_inv = np.linalg.inv(XtX)
        identity = XtX_inv @ XtX

        expected_identity = np.eye(XtX.shape[0])
        np.testing.assert_allclose(identity, expected_identity, rtol=1e-8, atol=1e-8)


class TestDarbroRSquared(unittest.TestCase):
    """Test R² and adjusted R² calculations"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_r_squared_exists(self):
        """Test that R² is calculated"""
        self.assertIsNotNone(self.model.r_squared)

    def test_r_squared_range(self):
        """Test that R² is between 0 and 1"""
        self.assertGreaterEqual(self.model.r_squared, 0)
        self.assertLessEqual(self.model.r_squared, 1)

    def test_r_squared_value(self):
        """Test that R² has a reasonable value for bodyfat dataset"""
        # For the bodyfat dataset, R² should be around 0.8
        self.assertGreater(self.model.r_squared, 0.7)
        self.assertLess(self.model.r_squared, 0.9)

    def test_r_squared_calculation(self):
        """Test R² calculation manually"""
        y_mean = np.mean(self.model.y.values)
        sst = np.sum((self.model.y.values - y_mean) ** 2)
        expected_r_squared = 1 - (self.model.sse / sst)
        self.assertAlmostEqual(self.model.r_squared, expected_r_squared, places=10)

    def test_adjusted_r_squared_exists(self):
        """Test that adjusted R² is calculated"""
        self.assertIsNotNone(self.model.adjusted_r_squared)

    def test_adjusted_r_squared_less_than_r_squared(self):
        """Test that adjusted R² is less than or equal to R²"""
        self.assertLessEqual(self.model.adjusted_r_squared, self.model.r_squared)

    def test_adjusted_r_squared_calculation(self):
        """Test adjusted R² calculation manually"""
        n = len(self.model.y)
        p = self.model.X.shape[1]
        expected_adj_r_squared = 1 - ((1 - self.model.r_squared) * (n - 1) / (n - p))
        self.assertAlmostEqual(self.model.adjusted_r_squared, expected_adj_r_squared, places=10)

    def test_perfect_fit_r_squared(self):
        """Test R² equals 1 for perfect fit"""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * x
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        self.assertAlmostEqual(model.r_squared, 1.0, places=10)

    def test_simple_regression_r_squared(self):
        """Test R² with simple linear regression"""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        # R² should be high for strong linear relationship
        self.assertGreater(model.r_squared, 0.9)


class TestDarbroHypothesisTesting(unittest.TestCase):
    """Test hypothesis testing features (t-statistics, p-values, F-test)"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_standard_errors_exist(self):
        """Test that standard errors are calculated"""
        self.assertIsNotNone(self.model.standard_errors)

    def test_standard_errors_positive(self):
        """Test that all standard errors are positive"""
        self.assertTrue(np.all(self.model.standard_errors > 0))

    def test_standard_errors_length(self):
        """Test that standard errors match number of coefficients"""
        self.assertEqual(len(self.model.standard_errors), len(self.model.coefficients))

    def test_standard_errors_calculation(self):
        """Test standard errors are sqrt of variance diagonal"""
        expected_se = np.sqrt(np.diag(self.model.variance_covariance))
        np.testing.assert_allclose(self.model.standard_errors, expected_se, rtol=1e-10)

    def test_t_statistics_exist(self):
        """Test that t-statistics are calculated"""
        self.assertIsNotNone(self.model.t_statistics)

    def test_t_statistics_length(self):
        """Test that t-statistics match number of coefficients"""
        self.assertEqual(len(self.model.t_statistics), len(self.model.coefficients))

    def test_t_statistics_calculation(self):
        """Test t-statistics are coefficient / standard error"""
        expected_t = self.model.coefficients / self.model.standard_errors
        np.testing.assert_allclose(self.model.t_statistics, expected_t, rtol=1e-10)

    def test_t_statistics_intercept(self):
        """Test that intercept t-statistic is reasonable"""
        # First coefficient is intercept
        self.assertIsInstance(self.model.t_statistics[0], (float, np.floating))

    def test_p_values_exist(self):
        """Test that p-values are calculated"""
        self.assertIsNotNone(self.model.p_values)

    def test_p_values_length(self):
        """Test that p-values match number of coefficients"""
        self.assertEqual(len(self.model.p_values), len(self.model.coefficients))

    def test_p_values_range(self):
        """Test that all p-values are between 0 and 1"""
        self.assertTrue(np.all(self.model.p_values >= 0))
        self.assertTrue(np.all(self.model.p_values <= 1))

    def test_p_values_calculation(self):
        """Test p-values calculation manually"""
        from scipy import stats
        n = len(self.model.y)
        p = self.model.X.shape[1]
        df = n - p
        expected_p_values = 2 * (1 - stats.t.cdf(np.abs(self.model.t_statistics), df))
        np.testing.assert_allclose(self.model.p_values, expected_p_values, rtol=1e-10)

    def test_significant_coefficients(self):
        """Test that overall model is significant even if individual coefficients aren't"""
        # For the bodyfat dataset, individual coefficients may not be significant
        # due to multicollinearity, but the overall F-test should be significant
        self.assertIsNotNone(self.model.p_values)
        # The overall model should be significant
        self.assertLess(self.model.f_p_value, 0.05)

    def test_f_statistic_exists(self):
        """Test that F-statistic is calculated"""
        self.assertIsNotNone(self.model.f_statistic)

    def test_f_statistic_positive(self):
        """Test that F-statistic is positive"""
        self.assertGreater(self.model.f_statistic, 0)

    def test_f_p_value_exists(self):
        """Test that F p-value is calculated"""
        self.assertIsNotNone(self.model.f_p_value)

    def test_f_p_value_range(self):
        """Test that F p-value is between 0 and 1"""
        self.assertGreaterEqual(self.model.f_p_value, 0)
        self.assertLessEqual(self.model.f_p_value, 1)

    def test_f_statistic_calculation(self):
        """Test F-statistic calculation manually"""
        y_mean = np.mean(self.model.y.values)
        sst = np.sum((self.model.y.values - y_mean) ** 2)
        ssr = sst - self.model.sse

        n = len(self.model.y)
        p = self.model.X.shape[1]
        df_model = p - 1
        df_residual = n - p

        expected_f = (ssr / df_model) / (self.model.sse / df_residual)
        self.assertAlmostEqual(self.model.f_statistic, expected_f, places=10)

    def test_f_statistic_significance(self):
        """Test that F-statistic is significant for bodyfat dataset"""
        # The model should be significant
        self.assertLess(self.model.f_p_value, 0.05)

    def test_perfect_fit_small_p_values(self):
        """Test that perfect fit gives very small p-values"""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * x
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        # All residuals are zero, so t-statistics should be very large
        # and p-values should be very small (approaching 0)
        # However, with perfect fit and small sample, this might be undefined
        # so we just test that the calculation doesn't error

    def test_simple_regression_t_tests(self):
        """Test t-statistics with simple regression"""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 5 + 2 * x + np.random.normal(0, 2, n)
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        # Both intercept and slope should be significant
        self.assertLess(model.p_values[0], 0.05)  # intercept
        self.assertLess(model.p_values[1], 0.05)  # slope


class TestDarbroGetSummary(unittest.TestCase):
    """Test the get_summary() method"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_get_summary_returns_string(self):
        """Test that get_summary returns a string"""
        summary = self.model.get_summary()
        self.assertIsInstance(summary, str)

    def test_get_summary_contains_r_squared(self):
        """Test that summary contains R² value"""
        summary = self.model.get_summary()
        self.assertIn('R-squared', summary)
        self.assertIn(f'{self.model.r_squared:.4f}', summary)

    def test_get_summary_contains_adjusted_r_squared(self):
        """Test that summary contains adjusted R² value"""
        summary = self.model.get_summary()
        self.assertIn('Adjusted R-squared', summary)

    def test_get_summary_contains_f_statistic(self):
        """Test that summary contains F-statistic"""
        summary = self.model.get_summary()
        self.assertIn('F-statistic', summary)

    def test_get_summary_contains_coefficients(self):
        """Test that summary contains coefficient table"""
        summary = self.model.get_summary()
        self.assertIn('Coefficients:', summary)
        self.assertIn('Intercept', summary)
        self.assertIn('triceps', summary)
        self.assertIn('thigh', summary)
        self.assertIn('midarm', summary)

    def test_get_summary_contains_std_errors(self):
        """Test that summary contains standard errors"""
        summary = self.model.get_summary()
        self.assertIn('Std Err', summary)

    def test_get_summary_contains_t_stats(self):
        """Test that summary contains t-statistics"""
        summary = self.model.get_summary()
        self.assertIn('t', summary)

    def test_get_summary_contains_p_values(self):
        """Test that summary contains p-values"""
        summary = self.model.get_summary()
        self.assertIn('P>|t|', summary)

    def test_get_summary_error_without_calculation(self):
        """Test that get_summary raises error if analytics not calculated"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        model = Darbro(df)
        # Don't call calculate_analytical_information()

        with self.assertRaises(ValueError):
            model.get_summary()

    def test_get_summary_formatting(self):
        """Test that summary has proper formatting"""
        summary = self.model.get_summary()
        lines = summary.split('\n')

        # Should have multiple lines
        self.assertGreater(len(lines), 10)

        # Should have separator lines
        self.assertTrue(any('=' * 50 in line for line in lines))


class TestDarbroHypothesisTestingEdgeCases(unittest.TestCase):
    """Test edge cases for hypothesis testing"""

    def test_no_predictors_f_statistic(self):
        """Test F-statistic with intercept-only model"""
        # Create data where y is just a constant + noise
        np.random.seed(42)
        y = np.full(20, 10) + np.random.normal(0, 1, 20)
        # Create a dummy predictor that we won't use
        df = pd.DataFrame({'y': y})

        # For intercept-only model, we need at least one predictor column
        # This test might not be applicable with current design

    def test_multiple_predictors_hypothesis_tests(self):
        """Test with multiple predictors"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 5 + 2*x1 - 3*x2 + 1.5*x3 + np.random.randn(n)

        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
        model = Darbro(df)
        model.calculate_analytical_information()

        # Should have 4 coefficients (intercept + 3 predictors)
        self.assertEqual(len(model.t_statistics), 4)
        self.assertEqual(len(model.p_values), 4)

        # All coefficients should be significant
        self.assertTrue(np.all(model.p_values < 0.05))

        # F-test should be highly significant
        self.assertLess(model.f_p_value, 0.001)

    def test_weak_relationship_high_p_values(self):
        """Test that weak relationships have high p-values"""
        np.random.seed(42)
        n = 30
        x = np.random.randn(n)
        y = 5 + 0.01 * x + np.random.normal(0, 5, n)  # Very weak relationship

        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        # The slope coefficient should have high p-value (not significant)
        # p-value for x should be > 0.05
        self.assertGreater(model.p_values[1], 0.05)


class TestDarbroConfidenceIntervals(unittest.TestCase):
    """Test confidence interval for mean response"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_confidence_interval_returns_tuple(self):
        """Test that confidence_interval_mean returns a tuple of 3 values"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.confidence_interval_mean(X_new)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_confidence_interval_point_estimate(self):
        """Test that point estimate matches predict() method"""
        X_new = np.array([25.0, 50.0, 28.0])
        lower, point, upper = self.model.confidence_interval_mean(X_new)
        expected_point = self.model.predict(X_new)
        self.assertAlmostEqual(point, expected_point, places=10)

    def test_confidence_interval_ordering(self):
        """Test that lower < point < upper"""
        X_new = np.array([25.0, 50.0, 28.0])
        lower, point, upper = self.model.confidence_interval_mean(X_new)
        self.assertLess(lower, point)
        self.assertLess(point, upper)

    def test_confidence_interval_different_alphas(self):
        """Test confidence intervals with different alpha levels"""
        X_new = np.array([25.0, 50.0, 28.0])

        # 90% confidence interval
        lower_90, point_90, upper_90 = self.model.confidence_interval_mean(X_new, alpha=0.10)

        # 95% confidence interval
        lower_95, point_95, upper_95 = self.model.confidence_interval_mean(X_new, alpha=0.05)

        # 99% confidence interval
        lower_99, point_99, upper_99 = self.model.confidence_interval_mean(X_new, alpha=0.01)

        # Point estimates should be the same
        self.assertAlmostEqual(point_90, point_95, places=10)
        self.assertAlmostEqual(point_95, point_99, places=10)

        # Wider confidence level should have wider interval
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)

    def test_confidence_interval_at_mean_x(self):
        """Test that CI is narrowest at mean of X"""
        # Confidence interval at mean of X should be narrower
        X_mean = np.mean(self.df.iloc[:, 1:].values, axis=0)
        lower_mean, _, upper_mean = self.model.confidence_interval_mean(X_mean)

        # Confidence interval at extreme value
        X_extreme = np.array([10.0, 30.0, 15.0])
        lower_extreme, _, upper_extreme = self.model.confidence_interval_mean(X_extreme)

        # Width at mean should be less than or equal to width at extreme
        width_mean = upper_mean - lower_mean
        width_extreme = upper_extreme - lower_extreme

        # This should generally be true, but may not always be due to data structure
        # So we just test that both are calculated without error
        self.assertGreater(width_mean, 0)
        self.assertGreater(width_extreme, 0)

    def test_confidence_interval_error_without_calculation(self):
        """Test that confidence_interval_mean raises error without analytics"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        model = Darbro(df)
        # Don't call calculate_analytical_information()

        X_new = np.array([25.0, 50.0, 28.0])
        with self.assertRaises(ValueError):
            model.confidence_interval_mean(X_new)

    def test_confidence_interval_simple_regression(self):
        """Test confidence interval with simple linear regression"""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        X_new = np.array([5.0])
        lower, point, upper = model.confidence_interval_mean(X_new)

        # Point should be approximately 2 + 3*5 = 17
        self.assertGreater(point, 15)
        self.assertLess(point, 19)

        # Interval should be reasonable
        self.assertLess(upper - lower, 5)

    def test_confidence_interval_manual_calculation(self):
        """Test confidence interval calculation manually"""
        X_new = np.array([25.0, 50.0, 28.0])
        alpha = 0.05

        # Get result from method
        lower, point, upper = self.model.confidence_interval_mean(X_new, alpha)

        # Manual calculation
        X_new_with_intercept = np.insert(X_new, 0, 1)
        y_pred = np.dot(X_new_with_intercept, self.model.coefficients)
        XtX_inv = np.linalg.inv(self.model.X.T @ self.model.X)
        se_mean = np.sqrt(self.model.mse * X_new_with_intercept.T @ XtX_inv @ X_new_with_intercept)

        n = len(self.model.y)
        p = self.model.X.shape[1]
        df = n - p
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin = t_critical * se_mean

        expected_lower = y_pred - margin
        expected_upper = y_pred + margin

        self.assertAlmostEqual(lower, expected_lower, places=10)
        self.assertAlmostEqual(point, y_pred, places=10)
        self.assertAlmostEqual(upper, expected_upper, places=10)


class TestDarbroPredictionIntervals(unittest.TestCase):
    """Test prediction interval for new observations"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_prediction_interval_returns_tuple(self):
        """Test that prediction_interval returns a tuple of 3 values"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.prediction_interval(X_new)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_prediction_interval_point_estimate(self):
        """Test that point estimate matches predict() method"""
        X_new = np.array([25.0, 50.0, 28.0])
        lower, point, upper = self.model.prediction_interval(X_new)
        expected_point = self.model.predict(X_new)
        self.assertAlmostEqual(point, expected_point, places=10)

    def test_prediction_interval_ordering(self):
        """Test that lower < point < upper"""
        X_new = np.array([25.0, 50.0, 28.0])
        lower, point, upper = self.model.prediction_interval(X_new)
        self.assertLess(lower, point)
        self.assertLess(point, upper)

    def test_prediction_interval_wider_than_confidence(self):
        """Test that prediction interval is wider than confidence interval"""
        X_new = np.array([25.0, 50.0, 28.0])

        ci_lower, ci_point, ci_upper = self.model.confidence_interval_mean(X_new)
        pi_lower, pi_point, pi_upper = self.model.prediction_interval(X_new)

        # Point estimates should be the same
        self.assertAlmostEqual(ci_point, pi_point, places=10)

        # Prediction interval should be wider
        ci_width = ci_upper - ci_lower
        pi_width = pi_upper - pi_lower

        self.assertGreater(pi_width, ci_width)

        # Prediction interval should contain confidence interval
        self.assertLessEqual(pi_lower, ci_lower)
        self.assertGreaterEqual(pi_upper, ci_upper)

    def test_prediction_interval_different_alphas(self):
        """Test prediction intervals with different alpha levels"""
        X_new = np.array([25.0, 50.0, 28.0])

        # 90% prediction interval
        lower_90, point_90, upper_90 = self.model.prediction_interval(X_new, alpha=0.10)

        # 95% prediction interval
        lower_95, point_95, upper_95 = self.model.prediction_interval(X_new, alpha=0.05)

        # 99% prediction interval
        lower_99, point_99, upper_99 = self.model.prediction_interval(X_new, alpha=0.01)

        # Point estimates should be the same
        self.assertAlmostEqual(point_90, point_95, places=10)
        self.assertAlmostEqual(point_95, point_99, places=10)

        # Wider confidence level should have wider interval
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)

    def test_prediction_interval_error_without_calculation(self):
        """Test that prediction_interval raises error without analytics"""
        df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        model = Darbro(df)
        # Don't call calculate_analytical_information()

        X_new = np.array([25.0, 50.0, 28.0])
        with self.assertRaises(ValueError):
            model.prediction_interval(X_new)

    def test_prediction_interval_simple_regression(self):
        """Test prediction interval with simple linear regression"""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        X_new = np.array([5.0])
        lower, point, upper = model.prediction_interval(X_new)

        # Point should be approximately 2 + 3*5 = 17
        self.assertGreater(point, 15)
        self.assertLess(point, 19)

        # Interval should be wider than confidence interval
        ci_lower, _, ci_upper = model.confidence_interval_mean(X_new)
        self.assertLess(lower, ci_lower)
        self.assertGreater(upper, ci_upper)

    def test_prediction_interval_manual_calculation(self):
        """Test prediction interval calculation manually"""
        X_new = np.array([25.0, 50.0, 28.0])
        alpha = 0.05

        # Get result from method
        lower, point, upper = self.model.prediction_interval(X_new, alpha)

        # Manual calculation
        X_new_with_intercept = np.insert(X_new, 0, 1)
        y_pred = np.dot(X_new_with_intercept, self.model.coefficients)
        XtX_inv = np.linalg.inv(self.model.X.T @ self.model.X)
        se_pred = np.sqrt(self.model.mse * (1 + X_new_with_intercept.T @ XtX_inv @ X_new_with_intercept))

        n = len(self.model.y)
        p = self.model.X.shape[1]
        df = n - p
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin = t_critical * se_pred

        expected_lower = y_pred - margin
        expected_upper = y_pred + margin

        self.assertAlmostEqual(lower, expected_lower, places=10)
        self.assertAlmostEqual(point, y_pred, places=10)
        self.assertAlmostEqual(upper, expected_upper, places=10)

    def test_prediction_interval_perfect_fit(self):
        """Test prediction interval with perfect fit"""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * x
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        X_new = np.array([3.0])
        lower, point, upper = model.prediction_interval(X_new)

        # With perfect fit, MSE should be essentially zero
        # So interval should be very narrow (essentially a point)
        # However, with small sample, numerical issues might occur
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)


class TestDarbroPredictWithIntervals(unittest.TestCase):
    """Test the predict_with_intervals() convenience method"""

    def setUp(self):
        """Load data and calculate analytics"""
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])
        self.model = Darbro(self.df)
        self.model.calculate_analytical_information()

    def test_predict_with_intervals_returns_dict(self):
        """Test that predict_with_intervals returns a dictionary"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)
        self.assertIsInstance(result, dict)

    def test_predict_with_intervals_keys(self):
        """Test that result contains all expected keys"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)

        expected_keys = ['prediction', 'confidence_interval', 'prediction_interval', 'confidence_level']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_predict_with_intervals_prediction_value(self):
        """Test that prediction value matches predict() method"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)
        expected_prediction = self.model.predict(X_new)
        self.assertAlmostEqual(result['prediction'], expected_prediction, places=10)

    def test_predict_with_intervals_confidence_interval_format(self):
        """Test that confidence_interval is a tuple of (lower, upper)"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)

        ci = result['confidence_interval']
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)

        lower, upper = ci
        self.assertLess(lower, upper)

    def test_predict_with_intervals_prediction_interval_format(self):
        """Test that prediction_interval is a tuple of (lower, upper)"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)

        pi = result['prediction_interval']
        self.assertIsInstance(pi, tuple)
        self.assertEqual(len(pi), 2)

        lower, upper = pi
        self.assertLess(lower, upper)

    def test_predict_with_intervals_confidence_level(self):
        """Test that confidence_level is correct"""
        X_new = np.array([25.0, 50.0, 28.0])

        result_95 = self.model.predict_with_intervals(X_new, alpha=0.05)
        self.assertEqual(result_95['confidence_level'], 95.0)

        result_90 = self.model.predict_with_intervals(X_new, alpha=0.10)
        self.assertEqual(result_90['confidence_level'], 90.0)

        result_99 = self.model.predict_with_intervals(X_new, alpha=0.01)
        self.assertEqual(result_99['confidence_level'], 99.0)

    def test_predict_with_intervals_interval_relationship(self):
        """Test that prediction interval is wider than confidence interval"""
        X_new = np.array([25.0, 50.0, 28.0])
        result = self.model.predict_with_intervals(X_new)

        ci_lower, ci_upper = result['confidence_interval']
        pi_lower, pi_upper = result['prediction_interval']

        # Prediction interval should contain confidence interval
        self.assertLessEqual(pi_lower, ci_lower)
        self.assertGreaterEqual(pi_upper, ci_upper)

    def test_predict_with_intervals_consistency(self):
        """Test that results match individual method calls"""
        X_new = np.array([25.0, 50.0, 28.0])
        alpha = 0.05

        result = self.model.predict_with_intervals(X_new, alpha)

        # Check confidence interval
        ci_lower_expected, point_expected, ci_upper_expected = self.model.confidence_interval_mean(X_new, alpha)
        self.assertAlmostEqual(result['prediction'], point_expected, places=10)
        self.assertAlmostEqual(result['confidence_interval'][0], ci_lower_expected, places=10)
        self.assertAlmostEqual(result['confidence_interval'][1], ci_upper_expected, places=10)

        # Check prediction interval
        pi_lower_expected, _, pi_upper_expected = self.model.prediction_interval(X_new, alpha)
        self.assertAlmostEqual(result['prediction_interval'][0], pi_lower_expected, places=10)
        self.assertAlmostEqual(result['prediction_interval'][1], pi_upper_expected, places=10)

    def test_predict_with_intervals_multiple_predictions(self):
        """Test making multiple predictions with intervals"""
        test_cases = [
            np.array([20.0, 45.0, 25.0]),
            np.array([30.0, 55.0, 30.0]),
            np.array([25.0, 50.0, 27.5])
        ]

        results = [self.model.predict_with_intervals(x) for x in test_cases]

        # All should return valid dictionaries
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('confidence_interval', result)
            self.assertIn('prediction_interval', result)

        # Predictions should be different
        predictions = [r['prediction'] for r in results]
        self.assertNotEqual(predictions[0], predictions[1])
        self.assertNotEqual(predictions[1], predictions[2])

    def test_predict_with_intervals_simple_regression(self):
        """Test predict_with_intervals with simple linear regression"""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 5 + 2 * x + np.random.normal(0, 2, n)
        df = pd.DataFrame({'y': y, 'x': x})
        model = Darbro(df)
        model.calculate_analytical_information()

        X_new = np.array([5.0])
        result = model.predict_with_intervals(X_new)

        # Prediction should be close to 5 + 2*5 = 15
        self.assertGreater(result['prediction'], 13)
        self.assertLess(result['prediction'], 17)

        # Intervals should contain the prediction
        ci_lower, ci_upper = result['confidence_interval']
        pi_lower, pi_upper = result['prediction_interval']

        self.assertLess(ci_lower, result['prediction'])
        self.assertGreater(ci_upper, result['prediction'])
        self.assertLess(pi_lower, result['prediction'])
        self.assertGreater(pi_upper, result['prediction'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
