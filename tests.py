import unittest
import numpy as np
import pandas as pd
from darbro import Darbro  # Replace 'your_module_name' with the actual module name

class TestDarbro(unittest.TestCase):

    def setUp(self):
        # Load the dataset for testing
        self.df = Darbro.read_csv('bodyfat.csv', 'bodyfat', ['triceps', 'thigh', 'midarm'])


    def test_coefficients(self):
        darbro = Darbro(self.df)
        expected_coefficients = np.array([117.085, 4.334, -2.857, -2.186])
        np.testing.assert_allclose(darbro.coefficients, expected_coefficients, rtol=1e-3)

    def test_residuals(self):
        darbro = Darbro(self.df)
        darbro.calculate_analytical_information()
        # Test if the residuals sum to zero
        sum_residuals = np.sum(darbro.residuals)
        self.assertAlmostEqual(sum_residuals, 0, delta=1e-3)

    def test_mse(self):
        darbro = Darbro(self.df)
        darbro.calculate_analytical_information()

        # Test if the MSE calculation is correct
        expected_mse = 6.15
        self.assertAlmostEqual(darbro.mse, expected_mse, delta=1e-3)

if __name__ == '__main__':
    unittest.main()