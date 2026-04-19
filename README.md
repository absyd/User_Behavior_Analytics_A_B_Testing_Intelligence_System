# User Behavior Analytics & A/B Testing Intelligence System

A comprehensive Python-based system for analyzing user behavior, conducting A/B tests, and deriving actionable insights from user data.

## 🏗️ Project Architecture

```
project/
│
├── data/                   # Dataset storage
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Core modules
│   ├── stats_engine.py     # Statistical computing engine
│   ├── ab_testing.py       # A/B testing framework
│   └── feature_analysis.py # Feature analysis & user segmentation
├── dashboard/              # Streamlit dashboard
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🚀 Features

### Statistical Engine (`src/stats_engine.py`)
- **Descriptive Statistics**: Comprehensive statistical summaries
- **Confidence Intervals**: Calculate confidence intervals for means
- **Normality Testing**: Shapiro-Wilk and Kolmogorov-Smirnov tests
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
- **Outlier Detection**: IQR, Z-score, and Modified Z-score methods

### A/B Testing Engine (`src/ab_testing.py`)
- **Sample Size Calculation**: For both binary and continuous metrics
- **Hypothesis Testing**: Proportion and continuous metric tests
- **Power Analysis**: Calculate statistical power for given effect sizes
- **Sequential Testing**: Interim analysis with alpha spending
- **Result Interpretation**: Statistical significance and practical significance

### Feature Analysis (`src/feature_analysis.py`)
- **Feature Importance**: Random Forest-based importance ranking
- **User Segmentation**: K-means clustering with silhouette analysis
- **Behavioral Pattern Analysis**: Time-based and event-based patterns
- **Cohort Analysis**: User retention and cohort tracking
- **Correlation Analysis**: Feature correlation matrices
- **PCA Analysis**: Dimensionality reduction and variance explanation

## 📊 Streamlit Dashboard

The dashboard provides an interactive interface for:
- Real-time statistical analysis
- A/B test design and results visualization
- User behavior pattern exploration
- Feature importance and correlation analysis

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd User_Behavior_Analytics_&_A:B_Testing_Intelligence_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

## 📚 Usage Examples

### Basic Statistical Analysis

```python
from src.stats_engine import StatisticalEngine, create_sample_data

# Initialize engine
engine = StatisticalEngine()

# Create sample data
data = create_sample_data(1000)

# Calculate descriptive statistics
stats = engine.descriptive_statistics(data['age'])
print(f"Mean age: {stats['mean']:.2f}")
print(f"Std deviation: {stats['std']:.2f}")

# Confidence interval
ci_lower, ci_upper = engine.confidence_interval(data['revenue'])
print(f"95% CI for revenue: ({ci_lower:.2f}, {ci_upper:.2f})")
```

### A/B Testing

```python
from src.ab_testing import ABTestEngine, create_ab_test_sample_data

# Initialize A/B testing engine
ab_engine = ABTestEngine()

# Calculate required sample size
sample_size = ab_engine.calculate_sample_size(
    baseline_rate=0.1, 
    expected_lift=0.15
)
print(f"Required sample size per group: {sample_size['sample_per_group']}")

# Create test data
test_data = create_ab_test_sample_data(1000, 1000, 0.1, 0.15)

# Perform proportion test
control_conv = test_data[test_data['group'] == 'control']['converted'].sum()
treatment_conv = test_data[test_data['group'] == 'treatment']['converted'].sum()

results = ab_engine.proportion_test(
    control_conv, 1000,
    treatment_conv, 1000
)

print(f"Relative lift: {results['relative_lift']:.3f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['is_significant']}")
```

### Feature Analysis

```python
from src.feature_analysis import FeatureAnalyzer

# Initialize analyzer
analyzer = FeatureAnalyzer()

# Feature importance analysis
importance = analyzer.feature_importance_analysis(
    data, 
    target_column='group',
    task_type='classification'
)

print("Top features:")
for _, row in importance['feature_importance'].head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# User segmentation
segmentation = analyzer.user_segmentation(
    data,
    feature_columns=['age', 'sessions', 'conversion_rate', 'revenue'],
    n_clusters=4
)

print(f"Silhouette score: {segmentation['silhouette_score']:.3f}")
```

## 📈 Key Metrics & KPIs

### Conversion Metrics
- **Conversion Rate**: Percentage of users completing desired actions
- **Revenue Per User (RPU)**: Average revenue generated per user
- **Average Order Value (AOV)**: Average transaction value

### Engagement Metrics
- **Session Duration**: Average time spent per session
- **Page Views**: Number of pages viewed per session
- **Bounce Rate**: Percentage of single-page sessions

### Retention Metrics
- **User Retention**: Percentage of users returning over time
- **Cohort Analysis**: Retention rates by user cohorts
- **Churn Rate**: Percentage of users lost over time

## 🔧 Configuration

### Statistical Parameters
- **Significance Level (α)**: Default 0.05
- **Statistical Power**: Default 0.8
- **Confidence Level**: Default 95%

### A/B Testing Parameters
- **Minimum Detectable Effect**: Configurable based on business needs
- **Test Duration**: Calculated based on sample size requirements
- **Sequential Testing**: Configurable interim analyses

## 📊 Data Requirements

### Input Data Format
- **User Data**: CSV/Excel files with user-level metrics
- **Event Data**: Timestamped user interaction logs
- **A/B Test Data**: Group assignments and outcome metrics

### Data Quality
- **Missing Values**: Handled through imputation strategies
- **Outliers**: Detected and handled using multiple methods
- **Data Types**: Automatic type detection and conversion

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run individual module tests:
```bash
python -m pytest tests/test_stats_engine.py
python -m pytest tests/test_ab_testing.py
python -m pytest tests/test_feature_analysis.py
```

## 📝 Documentation

- **API Documentation**: Detailed function documentation in source code
- **Examples**: Jupyter notebooks with practical examples
- **Best Practices**: Guidelines for statistical analysis and A/B testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SciPy**: For statistical computing capabilities
- **Scikit-learn**: For machine learning algorithms
- **Pandas**: For data manipulation and analysis
- **Streamlit**: For interactive dashboard creation
- **NumPy**: For numerical computing

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation for common questions
- Review the examples for implementation guidance

## 🔮 Future Enhancements

- **Machine Learning Models**: Predictive analytics for user behavior
- **Advanced Segmentation**: Hierarchical clustering and behavioral segmentation
- **Real-time Analytics**: Stream processing for real-time insights
- **Multi-armed Bandits**: Adaptive experimentation framework
- **Bayesian A/B Testing**: Bayesian approach to hypothesis testing

---

**Note**: This system is designed for educational and research purposes. Always validate results with domain experts and consider business context when making decisions based on statistical analysis.
