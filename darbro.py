import pandas as pd
import numpy as np


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
