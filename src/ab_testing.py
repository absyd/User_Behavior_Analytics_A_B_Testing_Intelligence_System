"""
A/B Testing Module for User Behavior Analytics & A/B Testing Intelligence System

This module provides comprehensive A/B testing functionality including sample size calculation,
hypothesis testing, statistical power analysis, and result interpretation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ABTestEngine:
    """
    A/B Testing engine providing methods for experimental design, statistical testing,
    and result analysis.
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize A/B Testing Engine.
        
        Args:
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.power = power
        self.z_alpha = stats.norm.ppf(1 - alpha/2)
        self.z_beta = stats.norm.ppf(power)
    
    def calculate_sample_size(self, baseline_rate: float, 
                            expected_lift: float, 
                            test_type: str = 'two_sided') -> Dict:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Current conversion rate (0-1)
            expected_lift: Expected relative lift (e.g., 0.1 for 10% lift)
            test_type: 'one_sided' or 'two_sided'
            
        Returns:
            Dictionary containing sample size requirements
        """
        if not 0 < baseline_rate < 1:
            raise ValueError("Baseline rate must be between 0 and 1")
        if expected_lift <= 0:
            raise ValueError("Expected lift must be positive")
        
        # Calculate treatment rate
        treatment_rate = baseline_rate * (1 + expected_lift)
        
        # Pooled proportion
        pooled_p = (baseline_rate + treatment_rate) / 2
        
        # Adjust z-score based on test type
        if test_type == 'one_sided':
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = self.z_alpha
        
        # Sample size formula for proportion test
        n_per_group = ((z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                       self.z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + 
                                           treatment_rate * (1 - treatment_rate))) ** 2 / 
                      (treatment_rate - baseline_rate) ** 2)
        
        # Round up to nearest integer
        n_per_group = int(np.ceil(n_per_group))
        total_sample = 2 * n_per_group
        
        return {
            'sample_per_group': n_per_group,
            'total_sample': total_sample,
            'baseline_rate': baseline_rate,
            'treatment_rate': treatment_rate,
            'expected_lift': expected_lift,
            'test_type': test_type,
            'alpha': self.alpha,
            'power': self.power
        }
    
    def calculate_sample_size_continuous(self, baseline_mean: float, 
                                       baseline_std: float,
                                       expected_lift: float,
                                       test_type: str = 'two_sided') -> Dict:
        """
        Calculate required sample size for continuous metrics.
        
        Args:
            baseline_mean: Current mean value
            baseline_std: Current standard deviation
            expected_lift: Expected relative lift (e.g., 0.1 for 10% lift)
            test_type: 'one_sided' or 'two_sided'
            
        Returns:
            Dictionary containing sample size requirements
        """
        if baseline_std <= 0:
            raise ValueError("Standard deviation must be positive")
        if expected_lift <= 0:
            raise ValueError("Expected lift must be positive")
        
        # Expected treatment mean
        treatment_mean = baseline_mean * (1 + expected_lift)
        effect_size = (treatment_mean - baseline_mean) / baseline_std
        
        # Adjust z-score based on test type
        if test_type == 'one_sided':
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = self.z_alpha
        
        # Sample size formula for continuous test
        n_per_group = 2 * ((z_alpha + self.z_beta) / effect_size) ** 2
        n_per_group = int(np.ceil(n_per_group))
        total_sample = 2 * n_per_group
        
        return {
            'sample_per_group': n_per_group,
            'total_sample': total_sample,
            'baseline_mean': baseline_mean,
            'treatment_mean': treatment_mean,
            'effect_size': effect_size,
            'expected_lift': expected_lift,
            'test_type': test_type,
            'alpha': self.alpha,
            'power': self.power
        }
    
    def proportion_test(self, control_conversions: int, control_size: int,
                       treatment_conversions: int, treatment_size: int,
                       test_type: str = 'two_sided') -> Dict:
        """
        Perform A/B test for conversion rates.
        
        Args:
            control_conversions: Number of conversions in control group
            control_size: Total users in control group
            treatment_conversions: Number of conversions in treatment group
            treatment_size: Total users in treatment group
            test_type: 'one_sided' or 'two_sided'
            
        Returns:
            Dictionary containing test results
        """
        if control_size <= 0 or treatment_size <= 0:
            raise ValueError("Group sizes must be positive")
        if control_conversions > control_size or treatment_conversions > treatment_size:
            raise ValueError("Conversions cannot exceed group sizes")
        
        # Calculate conversion rates
        control_rate = control_conversions / control_size
        treatment_rate = treatment_conversions / treatment_size
        
        # Calculate pooled proportion
        pooled_p = (control_conversions + treatment_conversions) / (control_size + treatment_size)
        
        # Calculate standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_size + 1/treatment_size))
        
        # Calculate z-statistic
        if se == 0:
            z_stat = 0
        else:
            z_stat = (treatment_rate - control_rate) / se
        
        # Calculate p-value
        if test_type == 'one_sided':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Calculate confidence interval
        if test_type == 'one_sided':
            z_critical = stats.norm.ppf(1 - self.alpha)
        else:
            z_critical = self.z_alpha
        
        diff_se = np.sqrt(control_rate * (1 - control_rate) / control_size + 
                         treatment_rate * (1 - treatment_rate) / treatment_size)
        
        ci_lower = (treatment_rate - control_rate) - z_critical * diff_se
        ci_upper = (treatment_rate - control_rate) + z_critical * diff_se
        
        # Calculate relative lift
        if control_rate > 0:
            relative_lift = (treatment_rate - control_rate) / control_rate
        else:
            relative_lift = np.inf
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_difference': treatment_rate - control_rate,
            'relative_lift': relative_lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'control_size': control_size,
            'treatment_size': treatment_size,
            'test_type': test_type,
            'alpha': self.alpha
        }
    
    def continuous_test(self, control_data: Union[pd.Series, np.ndarray],
                       treatment_data: Union[pd.Series, np.ndarray],
                       test_type: str = 'two_sided') -> Dict:
        """
        Perform A/B test for continuous metrics.
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            test_type: 'one_sided' or 'two_sided'
            
        Returns:
            Dictionary containing test results
        """
        if isinstance(control_data, pd.Series):
            control_data = control_data.dropna()
        else:
            control_data = control_data[~np.isnan(control_data)]
            
        if isinstance(treatment_data, pd.Series):
            treatment_data = treatment_data.dropna()
        else:
            treatment_data = treatment_data[~np.isnan(treatment_data)]
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            raise ValueError("Data arrays cannot be empty")
        
        # Calculate statistics
        control_mean = np.mean(control_data)
        control_std = np.std(control_data, ddof=1)
        control_n = len(control_data)
        
        treatment_mean = np.mean(treatment_data)
        treatment_std = np.std(treatment_data, ddof=1)
        treatment_n = len(treatment_data)
        
        # Perform t-test
        if test_type == 'one_sided':
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data, 
                                            equal_var=False, alternative='greater')
        else:
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data, 
                                            equal_var=False)
        
        # Calculate confidence interval
        if test_type == 'one_sided':
            t_critical = stats.t.ppf(1 - self.alpha, df=min(control_n, treatment_n) - 1)
        else:
            t_critical = stats.t.ppf(1 - self.alpha/2, df=min(control_n, treatment_n) - 1)
        
        # Standard error of difference
        se_diff = np.sqrt(control_std**2/control_n + treatment_std**2/treatment_n)
        
        diff = treatment_mean - control_mean
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        # Calculate relative lift
        if control_mean != 0:
            relative_lift = diff / control_mean
        else:
            relative_lift = np.inf
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((control_n - 1) * control_std**2 + 
                             (treatment_n - 1) * treatment_std**2) / 
                            (control_n + treatment_n - 2))
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'absolute_difference': diff,
            'relative_lift': relative_lift,
            'effect_size': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'control_size': control_n,
            'treatment_size': treatment_n,
            'test_type': test_type,
            'alpha': self.alpha
        }
    
    def power_analysis(self, effect_size: float, sample_size: int,
                      test_type: str = 'two_sided') -> Dict:
        """
        Calculate statistical power for given effect size and sample size.
        
        Args:
            effect_size: Effect size (Cohen's d for continuous, difference in proportions for binary)
            sample_size: Sample size per group
            test_type: 'one_sided' or 'two_sided'
            
        Returns:
            Dictionary containing power analysis results
        """
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")
        
        # Adjust z-score based on test type
        if test_type == 'one_sided':
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = self.z_alpha
        
        # Calculate power
        z_power = (effect_size * np.sqrt(sample_size / 2)) - z_alpha
        power = stats.norm.cdf(z_power)
        
        return {
            'effect_size': effect_size,
            'sample_size_per_group': sample_size,
            'power': power,
            'test_type': test_type,
            'alpha': self.alpha
        }
    
    def sequential_testing(self, control_conversions: List[int], 
                          control_sizes: List[int],
                          treatment_conversions: List[int], 
                          treatment_sizes: List[int],
                          alpha_spent: List[float] = None) -> Dict:
        """
        Perform sequential A/B testing with interim analyses.
        
        Args:
            control_conversions: Cumulative conversions at each analysis
            control_sizes: Cumulative sample sizes at each analysis
            treatment_conversions: Cumulative conversions at each analysis
            treatment_sizes: Cumulative sample sizes at each analysis
            alpha_spent: Alpha spending function values (optional)
            
        Returns:
            Dictionary containing sequential test results
        """
        if len(control_conversions) != len(treatment_conversions):
            raise ValueError("Control and treatment arrays must have same length")
        
        n_analyses = len(control_conversions)
        
        # Default alpha spending (O'Brien-Fleming)
        if alpha_spent is None:
            alpha_spent = [2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha/4) / np.sqrt(i))) 
                          for i in range(1, n_analyses + 1)]
        
        results = []
        for i in range(n_analyses):
            # Perform test at this analysis
            test_result = self.proportion_test(
                control_conversions[i], control_sizes[i],
                treatment_conversions[i], treatment_sizes[i],
                test_type='two_sided'
            )
            
            # Check if significant at this interim analysis
            is_significant_interim = test_result['p_value'] < alpha_spent[i]
            
            results.append({
                'analysis_number': i + 1,
                'sample_size': control_sizes[i] + treatment_sizes[i],
                'p_value': test_result['p_value'],
                'alpha_spent': alpha_spent[i],
                'is_significant': is_significant_interim,
                'control_rate': test_result['control_rate'],
                'treatment_rate': test_result['treatment_rate'],
                'relative_lift': test_result['relative_lift']
            })
        
        # Check if test should be stopped early
        stop_early = any(result['is_significant'] for result in results)
        stopping_analysis = next((i for i, result in enumerate(results) 
                                 if result['is_significant']), None)
        
        return {
            'interim_results': results,
            'stop_early': stop_early,
            'stopping_analysis': stopping_analysis + 1 if stopping_analysis is not None else None,
            'final_alpha_spent': sum(alpha_spent),
            'total_analyses': n_analyses
        }


def create_ab_test_sample_data(n_control: int = 1000, n_treatment: int = 1000,
                              baseline_rate: float = 0.1, lift: float = 0.15,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Create sample A/B test data for demonstration.
    
    Args:
        n_control: Sample size for control group
        n_treatment: Sample size for treatment group
        baseline_rate: Baseline conversion rate
        lift: Relative lift for treatment group
        random_state: Random seed
        
    Returns:
        DataFrame with A/B test data
    """
    np.random.seed(random_state)
    
    # Generate control group
    control_conversions = np.random.binomial(1, baseline_rate, n_control)
    control_data = pd.DataFrame({
        'user_id': range(1, n_control + 1),
        'group': 'control',
        'converted': control_conversions,
        'revenue': np.random.lognormal(3, 1, n_control) * control_conversions
    })
    
    # Generate treatment group
    treatment_rate = baseline_rate * (1 + lift)
    treatment_conversions = np.random.binomial(1, treatment_rate, n_treatment)
    treatment_data = pd.DataFrame({
        'user_id': range(n_control + 1, n_control + n_treatment + 1),
        'group': 'treatment',
        'converted': treatment_conversions,
        'revenue': np.random.lognormal(3, 1, n_treatment) * treatment_conversions
    })
    
    # Combine data
    ab_data = pd.concat([control_data, treatment_data], ignore_index=True)
    
    return ab_data


if __name__ == "__main__":
    # Demonstration
    engine = ABTestEngine()
    
    print("=== A/B Testing Engine Demo ===")
    
    # Sample size calculation
    sample_size = engine.calculate_sample_size(baseline_rate=0.1, expected_lift=0.15)
    print(f"\nSample Size Calculation:")
    print(f"  Required per group: {sample_size['sample_per_group']}")
    print(f"  Total required: {sample_size['total_sample']}")
    
    # Create sample data
    ab_data = create_ab_test_sample_data(1000, 1000, 0.1, 0.15)
    
    # Proportion test
    control_conv = ab_data[ab_data['group'] == 'control']['converted'].sum()
    control_size = len(ab_data[ab_data['group'] == 'control'])
    treatment_conv = ab_data[ab_data['group'] == 'treatment']['converted'].sum()
    treatment_size = len(ab_data[ab_data['group'] == 'treatment'])
    
    prop_test = engine.proportion_test(control_conv, control_size, 
                                     treatment_conv, treatment_size)
    print(f"\nProportion Test Results:")
    print(f"  Control rate: {prop_test['control_rate']:.3f}")
    print(f"  Treatment rate: {prop_test['treatment_rate']:.3f}")
    print(f"  Relative lift: {prop_test['relative_lift']:.3f}")
    print(f"  P-value: {prop_test['p_value']:.4f}")
    print(f"  Significant: {prop_test['is_significant']}")
    
    # Continuous test
    control_revenue = ab_data[ab_data['group'] == 'control']['revenue']
    treatment_revenue = ab_data[ab_data['group'] == 'treatment']['revenue']
    
    cont_test = engine.continuous_test(control_revenue, treatment_revenue)
    print(f"\nContinuous Test Results (Revenue):")
    print(f"  Control mean: ${cont_test['control_mean']:.2f}")
    print(f"  Treatment mean: ${cont_test['treatment_mean']:.2f}")
    print(f"  Relative lift: {cont_test['relative_lift']:.3f}")
    print(f"  Effect size: {cont_test['effect_size']:.3f}")
    print(f"  P-value: {cont_test['p_value']:.4f}")
    print(f"  Significant: {cont_test['is_significant']}")
    
    # Power analysis
    power_analysis = engine.power_analysis(effect_size=0.2, sample_size=1000)
    print(f"\nPower Analysis:")
    print(f"  Effect size: {power_analysis['effect_size']}")
    print(f"  Sample size per group: {power_analysis['sample_size_per_group']}")
    print(f"  Power: {power_analysis['power']:.3f}")
