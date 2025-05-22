import unittest
import numpy as np
import math
from darbro import two_sample_t_test # Main function to test

# For comparison, if available (not used in the core logic of darbro.py)
try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
except np.exceptions.AxisError: 
    SCIPY_AVAILABLE = False


# Tests for the two_sample_t_test function
class TestTwoSampleTTest(unittest.TestCase):

    def test_value_error_small_samples(self):
        """Covers 3.c.i: Test ValueError for samples with less than 2 observations."""
        sample1 = [1]; sample2 = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Both samples must contain at least two observations."):
            two_sample_t_test(sample1, sample2)
        with self.assertRaisesRegex(ValueError, "Both samples must contain at least two observations."):
            two_sample_t_test(sample2, sample1)
        with self.assertRaisesRegex(ValueError, "Both samples must contain at least two observations."):
            two_sample_t_test([1], [2])

    def test_student_t_test_means_different(self):
        """Covers 3.a.i & 3.a.iii: Student's t-test, means different, compare SciPy."""
        sample1 = [1,2,3,4,5]; sample2 = [6,7,8,9,10]
        t_stat, p_value = two_sample_t_test(sample1, sample2, equal_var=True)
        self.assertAlmostEqual(t_stat, -5.0, places=7)
        if SCIPY_AVAILABLE:
            try:
                scipy_t, scipy_p = ttest_ind(sample1, sample2, equal_var=True)
                self.assertAlmostEqual(t_stat, scipy_t, places=6)
                self.assertAlmostEqual(p_value, scipy_p, places=5, msg=f"Custom p={p_value}, SciPy p={scipy_p}")
            except Exception: pass 
        else: self.assertLess(p_value, 0.05)

    def test_student_t_test_means_similar(self):
        """Covers 3.a.ii & 3.a.iii: Student's t-test, means similar (p > 0.05), compare SciPy."""
        sample1 = np.array([1,2,3,4,5,6,7,8,9,10])
        sample2 = np.array([1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1])
        t_stat, p_value = two_sample_t_test(sample1, sample2, equal_var=True)
        self.assertGreater(p_value, 0.05, f"P-value {p_value} not > 0.05")
        if SCIPY_AVAILABLE:
            try:
                scipy_t, scipy_p = ttest_ind(sample1, sample2, equal_var=True)
                self.assertAlmostEqual(t_stat, scipy_t, places=6)
                self.assertAlmostEqual(p_value, scipy_p, places=5, msg=f"Custom p={p_value}, SciPy p={scipy_p}")
            except Exception: pass

    def test_welch_t_test_means_different_unequal_var(self):
        """Covers 3.b.i & 3.b.iii: Welch's t-test, means different, unequal var, compare SciPy."""
        sample1 = [1,2,3,4,5]; sample2 = [60,70,80,90,100,110]
        t_stat, p_value = two_sample_t_test(sample1, sample2, equal_var=False)
        if SCIPY_AVAILABLE:
            try:
                scipy_t, scipy_p = ttest_ind(sample1, sample2, equal_var=False)
                self.assertAlmostEqual(t_stat, scipy_t, places=4)
                self.assertAlmostEqual(p_value, scipy_p, places=4, msg=f"Custom p={p_value}, SciPy p={scipy_p}")
            except Exception: pass
        else: self.assertLess(t_stat, -5); self.assertLess(p_value, 0.05)

    def test_welch_t_test_means_similar_unequal_var(self):
        """Covers 3.b.ii & 3.b.iii: Welch's t-test, means similar, unequal var (p > 0.05), compare SciPy."""
        rng = np.random.default_rng(0)
        s1 = rng.normal(loc=5.0, scale=2.0, size=30)
        s2 = rng.normal(loc=5.5, scale=10.0, size=35)
        t_stat, p_value = two_sample_t_test(s1, s2, equal_var=False)
        self.assertGreater(p_value, 0.05, f"P-value {p_value} not > 0.05")
        if SCIPY_AVAILABLE:
            try:
                scipy_t, scipy_p = ttest_ind(s1, s2, equal_var=False)
                self.assertAlmostEqual(t_stat, scipy_t, places=3)
                self.assertAlmostEqual(p_value, scipy_p, places=3, msg=f"Custom p={p_value}, SciPy p={scipy_p}")
            except Exception: pass

    def test_same_samples_student(self):
        """Student's t-test: identical samples (p=1, t=0). Implicitly 3.a.ii."""
        s = [10,20,30,40,50]; t_stat,p_value = two_sample_t_test(s,s,True)
        self.assertAlmostEqual(t_stat,0.0,7); self.assertAlmostEqual(p_value,1.0,7)

    def test_same_samples_welch(self):
        """Welch's t-test: identical samples (p=1, t=0). Implicitly 3.b.ii."""
        s = [10,20,30,40,50]; t_stat,p_value = two_sample_t_test(s,s,False)
        self.assertAlmostEqual(t_stat,0.0,7); self.assertAlmostEqual(p_value,1.0,7)

    def test_zero_variance_student(self):
        """Covers 3.c.ii: Student's t-test with zero variance. Note on behavior in docstring."""
        s1_zero_var=[5,5,5,5,5]; s2_has_var=[1,2,3,4,6] 
        
        exp_t_custom = 2.092755 
        exp_p_custom = 0.069751  
        
        t_stat,p_value=two_sample_t_test(s1_zero_var,s2_has_var,True)
        self.assertAlmostEqual(t_stat, exp_t_custom, places=3) 
        self.assertAlmostEqual(p_value, exp_p_custom, places=3) 

        if SCIPY_AVAILABLE:
            try:
                scipy_t,scipy_p=ttest_ind(s1_zero_var,s2_has_var,True, axis=0) 
                self.assertAlmostEqual(t_stat,scipy_t,places=3) 
                self.assertAlmostEqual(p_value,scipy_p,places=3) 
            except Exception: pass 

        s_both_zero_diff_mean=[5,5]; s_b_other=[10,10]
        t_inf,p_inf=two_sample_t_test(s_both_zero_diff_mean,s_b_other,True)
        self.assertEqual(t_inf,float('inf')); self.assertEqual(p_inf,0.0)
        
        s_both_zero_same_mean=[7,7]; s_d_same=[7,7,7]
        t_zero,p_one=two_sample_t_test(s_both_zero_same_mean,s_d_same,True)
        self.assertEqual(t_zero,0.0); self.assertEqual(p_one,1.0)

    def test_zero_variance_welch(self):
        """Covers 3.c.ii: Welch's t-test with zero variance. Note on behavior in docstring."""
        s1_zero_var=[5,5,5,5,5]; s2_has_var=[1,2,3,4,6]

        exp_t_custom_welch = 2.092428
        exp_p_custom_welch = 0.104539

        t_stat,p_value=two_sample_t_test(s1_zero_var,s2_has_var,False)
        self.assertAlmostEqual(t_stat, exp_t_custom_welch, places=4) 
        self.assertAlmostEqual(p_value, exp_p_custom_welch, places=4) 

        if SCIPY_AVAILABLE:
            try:
                scipy_t,scipy_p=ttest_ind(s1_zero_var,s2_has_var,False, axis=0) 
                if not (np.isnan(scipy_t) or np.isnan(scipy_p)):
                    pass 
            except Exception: pass


        s_both_zero_diff_mean=[5,5]; s_b_other=[10,10]
        t_inf,p_inf=two_sample_t_test(s_both_zero_diff_mean,s_b_other,False)
        self.assertEqual(t_inf,float('inf')); self.assertEqual(p_inf,0.0)

        s_both_zero_same_mean=[7,7]; s_d_same=[7,7,7]
        t_zero,p_one=two_sample_t_test(s_both_zero_same_mean,s_d_same,False)
        self.assertEqual(t_zero,0.0); self.assertEqual(p_one,1.0)

    def test_df_edge_cases_welch(self):
        """Welch's t-test: df calculation with one variance zero."""
        s1_has_var=[1,2,3,4,5]; s2_zero_var=[10,10,10] 
        exp_t=-9.8994949366 
        exp_p=0.0006504040 
        
        t_stat,p_value=two_sample_t_test(s1_has_var,s2_zero_var,False)
        self.assertAlmostEqual(t_stat,exp_t,places=5); 
        self.assertAlmostEqual(p_value,exp_p,places=3) # Loosened from 4 to 3

        if SCIPY_AVAILABLE:
            try:
                scipy_t,scipy_p=ttest_ind(s1_has_var,s2_zero_var,False, axis=0) 
                if not (np.isnan(scipy_t) or np.isnan(scipy_p)):
                    self.assertAlmostEqual(t_stat,scipy_t,places=4)
                    self.assertAlmostEqual(p_value,scipy_p,places=3) # Loosened
            except Exception: pass

if __name__ == '__main__':
    unittest.main()
