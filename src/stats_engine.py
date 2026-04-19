"""
Statistical Engine for User Behavior Analytics & A/B Testing Intelligence System

This module provides core statistical functions for data analysis, hypothesis testing,
and statistical inference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class StatisticalEngine:
    """
    Core statistical engine providing methods for descriptive statistics,
    hypothesis testing, and statistical inference.
    """
    
    def __init__(self):
        self.alpha = 0.05  # Default significance level
    
    def set_significance_level(self, alpha: float):
        """Set the significance level for hypothesis tests."""
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
    
    def descriptive_statistics(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            data: Input data as pandas Series or numpy array
            
        Returns:
            Dictionary containing descriptive statistics
        """
        if isinstance(data, pd.Series):
            data = data.dropna()
        else:
            data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise ValueError("Data array is empty after removing NaN values")
        
        stats_dict = {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'mode': stats.mode(data, keepdims=True)[0][0] if len(stats.mode(data, keepdims=True)[0]) > 0 else None,
            'std': np.std(data, ddof=1),
            'var': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.inf
        }
        
        return stats_dict
    
    def confidence_interval(self, data: Union[pd.Series, np.ndarray], 
                          confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data: Input data
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if isinstance(data, pd.Series):
            data = data.dropna()
        else:
            data = data[~np.isnan(data)]
        
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 data points for confidence interval")
        
        mean = np.mean(data)
        sem = stats.sem(data)
        alpha = 1 - confidence
        
        if n >= 30:
            # Use z-distribution for large samples
            critical_value = stats.norm.ppf(1 - alpha/2)
        else:
            # Use t-distribution for small samples
            critical_value = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin_error = critical_value * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def normality_test(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Test for normality using Shapiro-Wilk and Kolmogorov-Smirnov tests.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary containing test results
        """
        if isinstance(data, pd.Series):
            data = data.dropna()
        else:
            data = data[~np.isnan(data)]
        
        if len(data) < 3:
            raise ValueError("Need at least 3 data points for normality testing")
        
        # Shapiro-Wilk test (recommended for n < 5000)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            shapiro_result = {
                'test': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.alpha
            }
        else:
            shapiro_result = None
        
        # Kolmogorov-Smirnov test
        # Standardize the data
        standardized_data = (data - np.mean(data)) / np.std(data, ddof=1)
        ks_stat, ks_p = stats.kstest(standardized_data, 'norm')
        ks_result = {
            'test': 'Kolmogorov-Smirnov',
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > self.alpha
        }
        
        return {
            'shapiro_wilk': shapiro_result,
            'kolmogorov_smirnov': ks_result,
            'sample_size': len(data)
        }
    
    def correlation_analysis(self, x: Union[pd.Series, np.ndarray], 
                           y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Perform correlation analysis between two variables.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Dictionary containing correlation results
        """
        if isinstance(x, pd.Series):
            x = x.dropna()
        else:
            x = x[~np.isnan(x)]
            
        if isinstance(y, pd.Series):
            y = y.dropna()
        else:
            y = y[~np.isnan(y)]
        
        if len(x) != len(y):
            raise ValueError("Arrays must have the same length")
        
        if len(x) < 3:
            raise ValueError("Need at least 3 data points for correlation analysis")
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(x, y)
        
        return {
            'pearson': {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < self.alpha
            },
            'spearman': {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < self.alpha
            },
            'kendall': {
                'correlation': kendall_tau,
                'p_value': kendall_p,
                'significant': kendall_p < self.alpha
            },
            'sample_size': len(x)
        }
    
    def outlier_detection(self, data: Union[pd.Series, np.ndarray], 
                         method: str = 'iqr') -> Dict:
        """
        Detect outliers using specified method.
        
        Args:
            data: Input data
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dictionary containing outlier information
        """
        if isinstance(data, pd.Series):
            data = data.dropna()
        else:
            data = data[~np.isnan(data)]
        
        if len(data) < 4:
            raise ValueError("Need at least 4 data points for outlier detection")
        
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_indices = np.where(z_scores > 3)[0]
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > 3.5)[0]
            
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        return {
            'method': method,
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': data.iloc[outlier_indices].tolist() if isinstance(data, pd.Series) else data[outlier_indices].tolist(),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100
        }


def create_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create sample data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(random_state)
    
    data = {
        'user_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 10, n_samples),
        'sessions': np.random.poisson(5, n_samples),
        'conversion_rate': np.random.beta(2, 5, n_samples),
        'revenue': np.random.lognormal(3, 1, n_samples),
        'group': np.random.choice(['control', 'treatment'], n_samples, p=[0.5, 0.5])
    }
    
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 18, 80)
    df['conversion_rate'] = np.clip(df['conversion_rate'], 0, 1)
    
    return df


if __name__ == "__main__":
    # Demonstration
    engine = StatisticalEngine()
    
    # Create sample data
    sample_data = create_sample_data(1000)
    
    # Example usage
    print("=== Statistical Engine Demo ===")
    
    # Descriptive statistics
    age_stats = engine.descriptive_statistics(sample_data['age'])
    print(f"\nAge Statistics:")
    for key, value in age_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Confidence interval
    ci_lower, ci_upper = engine.confidence_interval(sample_data['revenue'])
    print(f"\nRevenue 95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
    
    # Normality test
    normality = engine.normality_test(sample_data['age'])
    print(f"\nNormality Test for Age:")
    if normality['shapiro_wilk']:
        print(f"  Shapiro-Wilk p-value: {normality['shapiro_wilk']['p_value']:.4f}")
    print(f"  KS p-value: {normality['kolmogorov_smirnov']['p_value']:.4f}")
    
    # Correlation analysis
    correlation = engine.correlation_analysis(sample_data['sessions'], sample_data['revenue'])
    print(f"\nCorrelation between Sessions and Revenue:")
    print(f"  Pearson r: {correlation['pearson']['correlation']:.4f} (p={correlation['pearson']['p_value']:.4f})")
    
    # Outlier detection
    outliers = engine.outlier_detection(sample_data['revenue'], method='iqr')
    print(f"\nRevenue Outliers (IQR method): {outliers['outlier_count']} ({outliers['outlier_percentage']:.1f}%)")
