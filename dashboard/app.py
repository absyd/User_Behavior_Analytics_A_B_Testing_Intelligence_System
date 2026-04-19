"""
🎯 User Behavior Analytics & A/B Testing Intelligence System Dashboard
A comprehensive Streamlit dashboard for analyzing user behavior, conducting A/B tests,
and deriving actionable insights from user data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="User Behavior Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.stats_engine import StatisticalEngine
from src.ab_testing import ABTestEngine
from src.feature_analysis import FeatureAnalyzer

# Initialize engines
@st.cache_resource
def initialize_engines():
    """Initialize all analysis engines."""
    return {
        'stats': StatisticalEngine(),
        'ab_test': ABTestEngine(),
        'feature': FeatureAnalyzer()
    }

# Load and cache data
@st.cache_data
def load_data():
    """Load the e-commerce dataset."""
    try:
        # Try to load the fixed CSV file
        df = pd.read_csv('Ecommerce_Consumer_Behavior_Analysis_Data_fixed.csv')
        
        # Data preprocessing
        df['Purchase_Amount_Float'] = df['Purchase_Amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Convert boolean columns
        bool_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main dashboard function."""
    engines = initialize_engines()
    df = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">📊 User Behavior Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Page",
        [
            "📈 Overview & Summary",
            "📊 Distribution Analysis", 
            "📉 Statistical Metrics",
            "🎯 A/B Testing",
            "🔍 Outlier Detection",
            "👥 Customer Segmentation",
            "📱 Feature Analysis"
        ]
    )
    
    # Data overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Overview")
    st.sidebar.write(f"📊 Total Records: {len(df):,}")
    st.sidebar.write(f"👥 Unique Customers: {df['Customer_ID'].nunique():,}")
    st.sidebar.write(f"💰 Total Revenue: ${df['Purchase_Amount_Float'].sum():,.2f}")
    
    if page == "📈 Overview & Summary":
        overview_page(df, engines)
    elif page == "📊 Distribution Analysis":
        distribution_page(df, engines)
    elif page == "📉 Statistical Metrics":
        metrics_page(df, engines)
    elif page == "🎯 A/B Testing":
        ab_testing_page(df, engines)
    elif page == "🔍 Outlier Detection":
        outlier_page(df, engines)
    elif page == "👥 Customer Segmentation":
        segmentation_page(df, engines)
    elif page == "📱 Feature Analysis":
        feature_page(df, engines)

def overview_page(df, engines):
    """Overview and summary page."""
    st.markdown("## 📈 Overview & Summary")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_purchase = df['Purchase_Amount_Float'].mean()
        st.metric("💰 Avg Purchase", f"${avg_purchase:.2f}")
    
    with col2:
        total_customers = df['Customer_ID'].nunique()
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col3:
        conversion_rate = (df['Discount_Used'].sum() / len(df)) * 100
        st.metric("🎯 Discount Usage", f"{conversion_rate:.1f}%")
    
    with col4:
        loyalty_rate = (df['Customer_Loyalty_Program_Member'].sum() / len(df)) * 100
        st.metric("⭐ Loyalty Rate", f"{loyalty_rate:.1f}%")
    
    # Revenue by Category
    st.markdown("### 💰 Revenue by Category")
    category_revenue = df.groupby('Purchase_Category')['Purchase_Amount_Float'].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=category_revenue.values,
        y=category_revenue.index,
        orientation='h',
        title="Revenue by Product Category",
        labels={'x': 'Total Revenue ($)', 'y': 'Category'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Age Distribution")
        fig_age = px.histogram(df, x='Age', nbins=20, title="Customer Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Income Level Distribution")
        income_counts = df['Income_Level'].value_counts()
        fig_income = px.pie(
            values=income_counts.values,
            names=income_counts.index,
            title="Income Level Distribution"
        )
        st.plotly_chart(fig_income, use_container_width=True)

def distribution_page(df, engines):
    """Distribution analysis page."""
    st.markdown("## 📊 Distribution Analysis")
    
    # Select variable for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Select Variable for Distribution Analysis", numeric_cols)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Histogram")
        fig_hist = px.histogram(
            df, 
            x=selected_var,
            nbins=30,
            title=f"Distribution of {selected_var}",
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Box Plot")
        fig_box = px.box(
            df, 
            y=selected_var,
            title=f"Box Plot of {selected_var}"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical measures
    st.markdown("### 📋 Statistical Measures")
    stats_dict = engines['stats'].descriptive_statistics(df[selected_var])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean", f"{stats_dict['mean']:.2f}")
        st.metric("Median", f"{stats_dict['median']:.2f}")
        st.metric("Mode", f"{stats_dict['mode']:.2f}" if stats_dict['mode'] else "N/A")
    
    with col2:
        st.metric("Std Dev", f"{stats_dict['std']:.2f}")
        st.metric("Variance", f"{stats_dict['var']:.2f}")
        st.metric("Range", f"{stats_dict['range']:.2f}")
    
    with col3:
        st.metric("Skewness", f"{stats_dict['skewness']:.3f}")
        st.metric("Kurtosis", f"{stats_dict['kurtosis']:.3f}")
        st.metric("CV", f"{stats_dict['cv']:.3f}")
    
    # Normality test
    st.markdown("### 🧪 Normality Test")
    normality = engines['stats'].normality_test(df[selected_var])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if normality['shapiro_wilk']:
            st.write("**Shapiro-Wilk Test:**")
            st.write(f"- Statistic: {normality['shapiro_wilk']['statistic']:.4f}")
            st.write(f"- P-value: {normality['shapiro_wilk']['p_value']:.4f}")
            st.write(f"- Normal: {'✅ Yes' if normality['shapiro_wilk']['is_normal'] else '❌ No'}")
    
    with col2:
        st.write("**Kolmogorov-Smirnov Test:**")
        st.write(f"- Statistic: {normality['kolmogorov_smirnov']['statistic']:.4f}")
        st.write(f"- P-value: {normality['kolmogorov_smirnov']['p_value']:.4f}")
        st.write(f"- Normal: {'✅ Yes' if normality['kolmogorov_smirnov']['is_normal'] else '❌ No'}")

def metrics_page(df, engines):
    """Statistical metrics page."""
    st.markdown("## 📉 Statistical Metrics")
    
    # Select variables for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Select Variable 1", numeric_cols, key="var1")
    with col2:
        var2 = st.selectbox("Select Variable 2", numeric_cols, key="var2")
    
    # Correlation analysis
    if var1 != var2:
        correlation = engines['stats'].correlation_analysis(df[var1], df[var2])
        
        st.markdown("### 🔗 Correlation Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pearson r", f"{correlation['pearson']['correlation']:.3f}")
            st.write(f"P-value: {correlation['pearson']['p_value']:.4f}")
        
        with col2:
            st.metric("Spearman ρ", f"{correlation['spearman']['correlation']:.3f}")
            st.write(f"P-value: {correlation['spearman']['p_value']:.4f}")
        
        with col3:
            st.metric("Kendall τ", f"{correlation['kendall']['correlation']:.3f}")
            st.write(f"P-value: {correlation['kendall']['p_value']:.4f}")
        
        # Scatter plot
        fig_scatter = px.scatter(
            df, 
            x=var1, 
            y=var2,
            title=f"Scatter Plot: {var1} vs {var2}",
            trendline="ols"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Confidence intervals
    st.markdown("### 📏 Confidence Intervals")
    selected_var_ci = st.selectbox("Select Variable for CI", numeric_cols, key="ci_var")
    
    confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
    
    ci_lower, ci_upper = engines['stats'].confidence_interval(df[selected_var_ci], confidence_level)
    
    st.write(f"**{int(confidence_level*100)}% Confidence Interval for {selected_var_ci}:**")
    st.write(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Visual representation of CI
    mean_val = df[selected_var_ci].mean()
    fig_ci = go.Figure()
    
    fig_ci.add_trace(go.Bar(
        x=['Mean'],
        y=[mean_val],
        name='Mean',
        marker_color='blue'
    ))
    
    fig_ci.add_trace(go.Bar(
        x=['CI Lower', 'CI Upper'],
        y=[ci_lower, ci_upper],
        name='Confidence Interval',
        marker_color='lightblue'
    ))
    
    fig_ci.update_layout(title=f"Confidence Interval for {selected_var_ci}")
    st.plotly_chart(fig_ci, use_container_width=True)

def ab_testing_page(df, engines):
    """A/B testing page."""
    st.markdown("## 🎯 A/B Testing Analysis")
    
    # Select test type
    test_type = st.selectbox(
        "Select A/B Test Type",
        ["Discount Impact", "Loyalty Program Impact", "Custom Test"]
    )
    
    if test_type == "Discount Impact":
        control = df[df['Discount_Used'] == False]['Purchase_Amount_Float']
        treatment = df[df['Discount_Used'] == True]['Purchase_Amount_Float']
        test_name = "Discount Usage"
        
    elif test_type == "Loyalty Program Impact":
        control = df[df['Customer_Loyalty_Program_Member'] == False]['Purchase_Amount_Float']
        treatment = df[df['Customer_Loyalty_Program_Member'] == True]['Purchase_Amount_Float']
        test_name = "Loyalty Membership"
        
    else:  # Custom Test
        st.markdown("### Custom A/B Test Setup")
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox("Select Group Column", df.columns.tolist())
            group_values = df[group_col].unique()
            control_value = st.selectbox("Control Group Value", group_values)
        
        with col2:
            metric_col = st.selectbox("Select Metric Column", 
                                    df.select_dtypes(include=[np.number]).columns.tolist())
            treatment_value = st.selectbox("Treatment Group Value", 
                                         [v for v in group_values if v != control_value])
        
        control = df[df[group_col] == control_value][metric_col]
        treatment = df[df[group_col] == treatment_value][metric_col]
        test_name = f"{group_col}: {treatment_value} vs {control_value}"
    
    # Perform A/B test
    if len(control) > 0 and len(treatment) > 0:
        ab_results = engines['ab_test'].continuous_test(control, treatment)
        
        # Results overview
        st.markdown(f"### 📊 {test_name} - Test Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Control Mean", f"${ab_results['control_mean']:.2f}")
            st.metric("Control Size", f"{ab_results['control_size']:,}")
        
        with col2:
            st.metric("Treatment Mean", f"${ab_results['treatment_mean']:.2f}")
            st.metric("Treatment Size", f"{ab_results['treatment_size']:,}")
        
        with col3:
            st.metric("Difference", f"${ab_results['absolute_difference']:.2f}")
            st.metric("Relative Lift", f"{ab_results['relative_lift']*100:.1f}%")
        
        with col4:
            st.metric("P-value", f"{ab_results['p_value']:.4f}")
            significance = "✅ Significant" if ab_results['is_significant'] else "❌ Not Significant"
            st.metric("Result", significance)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot comparison
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=control, name='Control'))
            fig_box.add_trace(go.Box(y=treatment, name='Treatment'))
            fig_box.update_layout(title=f"Distribution Comparison: {test_name}")
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Confidence interval
            ci_lower, ci_upper = ab_results['confidence_interval']
            fig_ci = go.Figure()
            
            fig_ci.add_trace(go.Bar(
                x=['Mean Difference'],
                y=[ab_results['absolute_difference']],
                name='Mean Difference',
                marker_color='blue'
            ))
            
            fig_ci.add_trace(go.Bar(
                x=['CI Lower', 'CI Upper'],
                y=[ci_lower, ci_upper],
                name='95% CI',
                marker_color='lightblue'
            ))
            
            fig_ci.update_layout(title="Mean Difference with Confidence Interval")
            st.plotly_chart(fig_ci, use_container_width=True)
        
        # Statistical interpretation
        st.markdown("### 📋 Statistical Interpretation")
        
        if ab_results['is_significant']:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("✅ **Statistically Significant Result**")
            st.write(f"- P-value ({ab_results['p_value']:.4f}) < 0.05")
            st.write(f"- Effect size (Cohen's d): {ab_results['effect_size']:.3f}")
            
            if abs(ab_results['effect_size']) < 0.2:
                st.write("- Effect size: Small")
            elif abs(ab_results['effect_size']) < 0.5:
                st.write("- Effect size: Medium")
            else:
                st.write("- Effect size: Large")
            
            if ab_results['absolute_difference'] > 0:
                st.write(f"- Treatment group performs **better** by ${ab_results['absolute_difference']:.2f}")
            else:
                st.write(f"- Control group performs **better** by ${abs(ab_results['absolute_difference']):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("❌ **No Statistically Significant Difference**")
            st.write(f"- P-value ({ab_results['p_value']:.4f}) ≥ 0.05")
            st.write("- Cannot reject the null hypothesis")
            st.markdown('</div>', unsafe_allow_html=True)

def outlier_page(df, engines):
    """Outlier detection page."""
    st.markdown("## 🔍 Outlier Detection")
    
    # Select variable and method
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Select Variable for Outlier Detection", numeric_cols)
    
    method = st.selectbox(
        "Select Detection Method",
        ["IQR Method", "Z-Score Method", "Modified Z-Score Method"]
    )
    
    method_map = {
        "IQR Method": "iqr",
        "Z-Score Method": "zscore", 
        "Modified Z-Score Method": "modified_zscore"
    }
    
    # Detect outliers
    outliers = engines['stats'].outlier_detection(df[selected_var], method_map[method])
    
    # Display results
    st.markdown("### 📊 Outlier Detection Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Outliers", outliers['outlier_count'])
        st.metric("Outlier %", f"{outliers['outlier_percentage']:.1f}%")
    
    with col2:
        st.metric("Method", method)
        st.metric("Sample Size", len(df))
    
    with col3:
        if outliers['outlier_count'] > 0:
            min_outlier = min(outliers['outlier_values'])
            max_outlier = max(outliers['outlier_values'])
            st.metric("Min Outlier", f"{min_outlier:.2f}")
            st.metric("Max Outlier", f"{max_outlier:.2f}")
    
    # Visualization
    fig = go.Figure()
    
    # Add normal points
    normal_mask = ~df[selected_var].isin(outliers['outlier_values'])
    fig.add_trace(go.Scatter(
        x=df[normal_mask].index,
        y=df[normal_mask][selected_var],
        mode='markers',
        name='Normal Points',
        marker=dict(color='blue', size=6)
    ))
    
    # Add outliers
    if outliers['outlier_count'] > 0:
        outlier_indices = df[df[selected_var].isin(outliers['outlier_values'])].index
        fig.add_trace(go.Scatter(
            x=outlier_indices,
            y=df.loc[outlier_indices, selected_var],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f"Outlier Detection: {selected_var} ({method})",
        xaxis_title="Index",
        yaxis_title=selected_var
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot
    fig_box = px.box(df, y=selected_var, title=f"Box Plot: {selected_var}")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Outlier details
    if outliers['outlier_count'] > 0:
        st.markdown("### 📋 Outlier Details")
        outlier_df = df[df[selected_var].isin(outliers['outlier_values'])]
        st.write(f"Showing {len(outlier_df)} outliers:")
        st.dataframe(outlier_df.head(10))

def segmentation_page(df, engines):
    """Customer segmentation page."""
    st.markdown("## 👥 Customer Segmentation")
    
    # Select features for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select Features for Segmentation",
        numeric_cols,
        default=['Age', 'Purchase_Amount_Float', 'Frequency_of_Purchase']
    )
    
    if len(selected_features) >= 2:
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
        
        # Perform segmentation
        segmentation = engines['feature'].user_segmentation(
            df, selected_features, n_clusters
        )
        
        # Display results
        st.markdown("### 📊 Segmentation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Clusters", n_clusters)
            st.metric("Silhouette Score", f"{segmentation['silhouette_score']:.3f}")
        
        with col2:
            # Cluster sizes
            cluster_sizes = [cluster['size'] for cluster in segmentation['cluster_stats']]
            fig_pie = px.pie(
                values=cluster_sizes,
                names=[f"Cluster {i}" for i in range(n_clusters)],
                title="Cluster Sizes"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Cluster statistics
        st.markdown("### 📋 Cluster Statistics")
        cluster_stats_df = pd.DataFrame(segmentation['cluster_stats'])
        st.dataframe(cluster_stats_df)
        
        # Visualization
        if len(selected_features) >= 2:
            # Create scatter plot of first two features
            df_with_clusters = segmentation['data_with_clusters'].copy()
            df_with_clusters['Cluster'] = df_with_clusters['cluster'].astype(str)
            
            fig_scatter = px.scatter(
                df_with_clusters,
                x=selected_features[0],
                y=selected_features[1],
                color='Cluster',
                title=f"Customer Segments: {selected_features[0]} vs {selected_features[1]}",
                hover_data=selected_features
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def feature_page(df, engines):
    """Feature analysis page."""
    st.markdown("## 📱 Feature Analysis")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance Analysis")
    
    # Select target variable
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = st.selectbox("Select Target Variable", numeric_cols)
    
    task_type = st.selectbox("Task Type", ["Classification", "Regression"])
    
    if st.button("Run Feature Importance Analysis"):
        with st.spinner("Analyzing feature importance..."):
            importance = engines['feature'].feature_importance_analysis(
                df, target_col, task_type.lower()
            )
            
            # Display top features
            top_features = importance['feature_importance'].head(10)
            
            fig = px.bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                title=f"Top 10 Feature Importance for {target_col}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("### 📊 Feature Importance Table")
            st.dataframe(top_features)
    
    # Correlation matrix
    st.markdown("### 🔗 Correlation Matrix")
    
    selected_corr_features = st.multiselect(
        "Select Features for Correlation Matrix",
        numeric_cols,
        default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
    )
    
    if len(selected_corr_features) >= 2:
        correlation = engines['feature'].feature_correlation_analysis(
            df, selected_corr_features
        )
        
        # Create heatmap
        corr_matrix = correlation['correlation_matrix']
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Strong correlations
        if correlation['n_strong_correlations'] > 0:
            st.markdown("### ⚠️ Strong Correlations")
            strong_corr_df = pd.DataFrame(correlation['strong_correlations'])
            st.dataframe(strong_corr_df)

if __name__ == "__main__":
    main()
