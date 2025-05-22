import numpy as np
import math

_BETAINC_MAXITER = 200
_BETAINC_EPS = 1e-14 # Slightly relaxed from 1e-16 for stability vs convergence speed trade-off
_FPMIN = 1.0e-30 # Smallest representable positive number

def _beta_cf(x, a, b):
    """
    Evaluates the continued fraction for the incomplete beta function I_x(a,b)
    using a method based on Numerical Recipes `betacf` (Lentz's method).
    This function computes 'F' such that I_x(a,b) = [x^a * (1-x)^b / (a * B(a,b))] * F.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    # Initialize
    c = 1.0
    # First term for d using NR's specific formulation for this CF
    d_val = 1.0 - qab * x / qap
    if abs(d_val) < _FPMIN:
        d_val = _FPMIN
    d_val = 1.0 / d_val
    h = d_val

    for m in range(1, _BETAINC_MAXITER + 1):
        m2 = 2 * m

        # Even term (numerator_term for N_2m)
        # aa = m*(b-m)*x / ( (qam+m2)*(a+m2) ) -> NR formula
        # My j (iterator for N_j) even: j=2m'. m is m'.
        # numerator_term = (m' * (b - m') * x) / ((a + 2*m' - 1) * (a + 2*m'))
        # This is N_{2m'}
        aa = m * (b - m) * x / ((qam + m2) * (a + m2 - 2.0)) # Corrected qam+m2 to a+m2-2 (as qam = a-1, qam+m2 = a-1+2m)
                                                          # (a+m2) in NR is (a+2m)
                                                          # Denom: (a+2m-1)*(a+2m)
        # Denominators in NR betacf for even term: (a-1+2m)(a+2m)
        # My N_2m: (m * (b-m) * x) / ((a + 2*m - 1) * (a + 2*m))
        # NR: aa = m*(b-m)*x / ( (a-1+2*m)*(a+2*m) )
        numerator_even = m * (b - m) * x / ((qam + m2) * (a + m2))


        d_val = 1.0 + numerator_even * d_val
        if abs(d_val) < _FPMIN:
            d_val = _FPMIN
        c = 1.0 + numerator_even / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d_val = 1.0 / d_val
        h = h * d_val * c

        # Odd term (numerator_term for N_2m+1)
        # aa = -(a+m)*(qab+m)*x / ( (a+m2)*(qap+m2) ) -> NR formula
        # My j (iterator for N_j) odd: j=2m'+1. m is m'.
        # numerator_term = -((a+m')*(a+b+m')*x) / ((a+2m')*(a+2m'+1))
        # This is N_{2m'+1}
        # NR: aa = -(a+m)*(a+b+m)*x / ( (a+2*m)*(a+1+2*m) )
        numerator_odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))

        d_val = 1.0 + numerator_odd * d_val
        if abs(d_val) < _FPMIN:
            d_val = _FPMIN
        c = 1.0 + numerator_odd / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d_val = 1.0 / d_val
        delta = d_val * c
        h = h * delta

        if abs(delta - 1.0) < _BETAINC_EPS:
            return h
            
    # print(f"Warning: Continued fraction for incomplete beta did not converge for x={x}, a={a}, b={b} after {_BETAINC_MAXITER} iterations. Current h={h}")
    return h


def _regularized_incomplete_beta(x, a, b):
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in the interval [0, 1] for regularized incomplete beta function.")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive for regularized incomplete beta function.")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Symmetry property: I_x(a,b) = 1 - I_{1-x}(b,a)
    # Use this if x is large, as the continued fraction converges faster for smaller x.
    # (a+1)/(a+b+2) is the point where x is roughly "halfway" in some sense for beta dist.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    log_beta_complete = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    
    # Prefix for I_x(a,b) = [x^a * (1-x)^b / (a * B(a,b))] * F
    # F is computed by _beta_cf
    # Handle cases where x is very small or 1-x is very small to avoid log(0)
    if x == 0.0: log_x = -float('inf') # Should be caught by x==0 earlier
    else: log_x = math.log(x)

    if (1.0-x) == 0.0: log_1_minus_x = -float('inf') # Should be caught by x==1 earlier
    else: log_1_minus_x = math.log(1.0 - x)

    # Potential underflow/overflow if a or b are large, making x^a or (1-x)^b huge/tiny.
    # The lgamma terms help manage large a,b for B(a,b).
    # log_prefix_val = a * log_x + b * log_1_minus_x - math.log(a) - log_beta_complete
    # prefix = math.exp(log_prefix_val)
    
    # Numerical Recipes formulation: I_x(a,b) = exp(a*log(x) + b*log(1-x) - log(B(a,b))) * CF_val / a
    # This means CF_val from NR's betacf is F from DLMF.
    # So my _beta_cf should return F.
    # Then prefix is: x^a * (1-x)^b / (a * B(a,b))
    
    # If _beta_cf returns h_NR (which is F_DLMF)
    # Then result is exp(a*logx + b*log(1-x) - lgamma(a) - lgamma(b) + lgamma(a+b)) * h_NR / a
    # This is (x^a * (1-x)^b / B(a,b)) * h_NR / a
    # = (x^a * (1-x)^b) / (a * gamma(a)gamma(b)/gamma(a+b)) * h_NR
    
    # Current _beta_cf is intended to return F (the h_NR from NR's betacf).
    # So the formula for I_x(a,b) should be:
    # ( exp(a*log(x) + b*log(1-x) - log_beta_complete) / a ) * F_returned_by_beta_cf
    
    front_factor = math.exp(a * log_x + b * log_1_minus_x - log_beta_complete)
    
    cf_value = _beta_cf(x, a, b) # This should be F_DLMF = h_NR

    # Check for a=0 case, though earlier check a > 0 should prevent it.
    if a == 0: # Should not happen
        return 0.0 # Or raise error, as B(a,b) would be problematic.

    return front_factor * cf_value / a


def two_sample_t_test(sample1, sample2, equal_var=True):
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Both samples must contain at least two observations.")

    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    s1 = np.std(sample1, ddof=1)
    s2 = np.std(sample2, ddof=1)

    if equal_var:
        s_p_num = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2)
        s_p_den = (n1 + n2 - 2)
        if s_p_den <= 0:
             raise ValueError("Degrees of freedom for Student's t-test must be positive.")
        s_p = math.sqrt(s_p_num / s_p_den) if s_p_den > 0 else 0.0
        
        t_stat_den = (s_p * math.sqrt(1/n1 + 1/n2)) if s_p > 0 else 0.0
        if t_stat_den == 0:
            t_stat = float('inf') if mean1 != mean2 else 0.0
        else:
            t_stat = (mean1 - mean2) / t_stat_den
        df = n1 + n2 - 2
    else:
        s1_sq_n1 = s1**2 / n1 if n1 > 0 else 0.0
        s2_sq_n2 = s2**2 / n2 if n2 > 0 else 0.0
        
        t_stat_den_welch = math.sqrt(s1_sq_n1 + s2_sq_n2)
        if t_stat_den_welch == 0:
             t_stat = float('inf') if mean1 != mean2 else 0.0
        else:
            t_stat = (mean1 - mean2) / t_stat_den_welch
        
        if n1 <= 1 or n2 <= 1: 
             raise ValueError("Degrees of freedom calculation requires n1 > 1 and n2 > 1 for Welch's t-test.")

        df_num = (s1_sq_n1 + s2_sq_n2)**2
        df_den_term1 = (s1_sq_n1**2) / (n1-1) if (n1-1) > 0 else 0.0
        df_den_term2 = (s2_sq_n2**2) / (n2-1) if (n2-1) > 0 else 0.0
        df_den = df_den_term1 + df_den_term2
        
        if df_den == 0: 
            if mean1 == mean2: return 0.0, 1.0
            else: return t_stat, 0.0 # (inf, 0) or (-inf,0)
        df = df_num / df_den

    if df <= 0:
        # This case should ideally be handled by specific scenarios like df_den == 0 above
        # Or indicates an issue if df becomes non-positive unexpectedly.
        raise ValueError("Degrees of freedom must be positive.")

    if t_stat == 0.0: return 0.0, 1.0
    if t_stat == float('inf') or t_stat == float('-inf'): return t_stat, 0.0

    t_abs = abs(t_stat)
    x_val = df / (df + t_abs**2)
    alpha_param = df / 2.0
    beta_param = 0.5
    
    if alpha_param <=0 or beta_param <=0:
        raise ValueError("Parameters a and b for regularized incomplete beta function must be positive.")
    
    try:
        # P-value = I_x(x_val, alpha_param, beta_param)
        p_value = _regularized_incomplete_beta(x_val, alpha_param, beta_param)
    except Exception as e:
        # print(f"Error in _regularized_incomplete_beta for t={t_stat}, df={df}, x={x_val}, a={alpha_param}, b={beta_param}: {e}")
        raise e

    return t_stat, p_value
