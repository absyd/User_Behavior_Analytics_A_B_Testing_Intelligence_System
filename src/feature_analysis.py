"""
Feature Analysis Module for User Behavior Analytics & A/B Testing Intelligence System

This module provides comprehensive feature analysis capabilities including feature importance,
segmentation analysis, behavioral patterns, and predictive modeling insights.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score, classification_report, regression_metrics
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Feature analysis engine providing methods for feature importance,
    user segmentation, behavioral analysis, and predictive insights.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Feature Analyzer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess_features(self, df: pd.DataFrame, 
                          target_column: str = None,
                          categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Preprocess features for analysis.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (optional)
            categorical_columns: List of categorical column names
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Handle categorical variables
        if categorical_columns:
            for col in categorical_columns:
                if col in df_processed.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(
                        df_processed[col].astype(str))
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != target_column:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        return df_processed
    
    def feature_importance_analysis(self, df: pd.DataFrame, 
                                  target_column: str,
                                  task_type: str = 'classification',
                                  n_features: int = None) -> Dict:
        """
        Analyze feature importance using Random Forest.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            task_type: 'classification' or 'regression'
            n_features: Number of top features to return
            
        Returns:
            Dictionary containing feature importance results
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        X_processed = self.preprocess_features(X, categorical_columns=categorical_columns)
        
        # Remove any remaining non-numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        if X_processed.shape[1] == 0:
            raise ValueError("No numeric features available for analysis")
        
        # Train Random Forest model
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        model.fit(X_processed, y)
        
        # Get feature importance
        importance_scores = model.feature_importances_
        feature_names = X_processed.columns
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        if n_features:
            feature_importance_df = feature_importance_df.head(n_features)
        
        return {
            'feature_importance': feature_importance_df,
            'model_type': task_type,
            'n_features': len(feature_importance_df),
            'top_feature': feature_importance_df.iloc[0]['feature'] if len(feature_importance_df) > 0 else None
        }
    
    def user_segmentation(self, df: pd.DataFrame, 
                         feature_columns: List[str],
                         n_clusters: int = 5,
                         method: str = 'kmeans') -> Dict:
        """
        Perform user segmentation analysis.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns for clustering
            n_clusters: Number of clusters
            method: Clustering method ('kmeans')
            
        Returns:
            Dictionary containing segmentation results
        """
        # Prepare data
        X = df[feature_columns].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        X_processed = self.preprocess_features(X, categorical_columns=categorical_columns)
        
        # Remove any remaining non-numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        if X_processed.shape[1] == 0:
            raise ValueError("No numeric features available for clustering")
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = clusterer.fit_predict(X_scaled)
        else:
            raise ValueError("Only 'kmeans' clustering method is supported")
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Add cluster labels to original data
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            stats_dict = {
                'cluster_id': i,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Add mean values for numeric features
            for col in feature_columns:
                if col in cluster_data.columns and cluster_data[col].dtype in ['int64', 'float64']:
                    stats_dict[f'{col}_mean'] = cluster_data[col].mean()
            
            cluster_stats.append(stats_dict)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'method': method,
            'data_with_clusters': df_with_clusters
        }
    
    def behavioral_pattern_analysis(self, df: pd.DataFrame,
                                  user_column: str,
                                  time_column: str,
                                  event_column: str,
                                  value_column: str = None) -> Dict:
        """
        Analyze behavioral patterns in user data.
        
        Args:
            df: Input DataFrame
            user_column: User identifier column
            time_column: Timestamp column
            event_column: Event/action column
            value_column: Value/metric column (optional)
            
        Returns:
            Dictionary containing behavioral analysis results
        """
        # Convert time column to datetime
        df_processed = df.copy()
        df_processed[time_column] = pd.to_datetime(df_processed[time_column])
        
        # Sort by user and time
        df_processed = df_processed.sort_values([user_column, time_column])
        
        # Calculate user-level metrics
        user_metrics = df_processed.groupby(user_column).agg({
            event_column: 'count',
            time_column: ['min', 'max']
        }).round(2)
        
        user_metrics.columns = ['event_count', 'first_seen', 'last_seen']
        user_metrics['session_duration'] = (user_metrics['last_seen'] - user_metrics['first_seen']).dt.total_seconds() / 3600  # hours
        user_metrics['events_per_hour'] = user_metrics['event_count'] / user_metrics['session_duration'].replace(0, np.nan)
        
        # Event frequency analysis
        event_frequency = df_processed[event_column].value_counts().reset_index()
        event_frequency.columns = ['event', 'frequency']
        event_frequency['percentage'] = event_frequency['frequency'] / len(df_processed) * 100
        
        # Time-based patterns
        df_processed['hour'] = df_processed[time_column].dt.hour
        df_processed['day_of_week'] = df_processed[time_column].dt.dayofweek
        df_processed['date'] = df_processed[time_column].dt.date
        
        hourly_pattern = df_processed.groupby('hour').size().reset_index()
        hourly_pattern.columns = ['hour', 'event_count']
        
        daily_pattern = df_processed.groupby('day_of_week').size().reset_index()
        daily_pattern.columns = ['day_of_week', 'event_count']
        
        # User engagement segments
        user_metrics['engagement_segment'] = pd.cut(
            user_metrics['event_count'],
            bins=[0, 1, 5, 20, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        engagement_summary = user_metrics['engagement_segment'].value_counts().reset_index()
        engagement_summary.columns = ['segment', 'user_count']
        engagement_summary['percentage'] = engagement_summary['user_count'] / len(user_metrics) * 100
        
        return {
            'user_metrics': user_metrics.reset_index(),
            'event_frequency': event_frequency,
            'hourly_pattern': hourly_pattern,
            'daily_pattern': daily_pattern,
            'engagement_segments': engagement_summary,
            'total_users': len(user_metrics),
            'total_events': len(df_processed),
            'avg_events_per_user': user_metrics['event_count'].mean(),
            'avg_session_duration_hours': user_metrics['session_duration'].mean()
        }
    
    def cohort_analysis(self, df: pd.DataFrame,
                       user_column: str,
                       time_column: str,
                       event_column: str = None) -> Dict:
        """
        Perform cohort analysis for user retention.
        
        Args:
            df: Input DataFrame
            user_column: User identifier column
            time_column: Timestamp column
            event_column: Event column (optional)
            
        Returns:
            Dictionary containing cohort analysis results
        """
        # Prepare data
        df_processed = df.copy()
        df_processed[time_column] = pd.to_datetime(df_processed[time_column])
        
        # Get first activity date for each user
        user_first_activity = df_processed.groupby(user_column)[time_column].min().reset_index()
        user_first_activity.columns = [user_column, 'first_activity_date']
        
        # Merge back to main data
        df_processed = df_processed.merge(user_first_activity, on=user_column)
        
        # Calculate cohort period (in months)
        df_processed['cohort_month'] = df_processed['first_activity_date'].dt.to_period('M')
        df_processed['activity_month'] = df_processed[time_column].dt.to_period('M')
        df_processed['period_number'] = (df_processed['activity_month'] - df_processed['cohort_month']).apply(lambda x: x.n + 1)
        
        # Create cohort table
        cohort_data = df_processed.groupby(['cohort_month', 'period_number'])[user_column].nunique().reset_index()
        cohort_sizes = df_processed.groupby('cohort_month')[user_column].nunique().reset_index()
        cohort_sizes.columns = ['cohort_month', 'cohort_size']
        
        # Merge cohort sizes
        cohort_data = cohort_data.merge(cohort_sizes, on='cohort_month')
        cohort_data['percentage'] = cohort_data[user_column] / cohort_data['cohort_size'] * 100
        
        # Pivot for heatmap
        cohort_table = cohort_data.pivot(index='cohort_month', 
                                        columns='period_number', 
                                        values='percentage')
        
        return {
            'cohort_table': cohort_table,
            'cohort_sizes': cohort_sizes,
            'raw_cohort_data': cohort_data,
            'total_cohorts': len(cohort_sizes)
        }
    
    def feature_correlation_analysis(self, df: pd.DataFrame,
                                   feature_columns: List[str] = None,
                                   method: str = 'pearson') -> Dict:
        """
        Analyze correlations between features.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns (optional)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation analysis results
        """
        # Select numeric columns
        if feature_columns:
            df_numeric = df[feature_columns].select_dtypes(include=[np.number])
        else:
            df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df_numeric.corr(method=method)
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'method': method,
            'n_features': len(corr_matrix.columns),
            'n_strong_correlations': len(strong_correlations)
        }
    
    def pca_analysis(self, df: pd.DataFrame,
                    feature_columns: List[str],
                    n_components: int = None) -> Dict:
        """
        Perform Principal Component Analysis.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            n_components: Number of components to keep
            
        Returns:
            Dictionary containing PCA results
        """
        # Prepare data
        X = df[feature_columns].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        X_processed = self.preprocess_features(X, categorical_columns=categorical_columns)
        
        # Remove any remaining non-numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        if X_processed.shape[1] == 0:
            raise ValueError("No numeric features available for PCA")
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Determine number of components
        if n_components is None:
            n_components = min(len(feature_columns), X_scaled.shape[0])
        
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create results DataFrame
        pca_results = pd.DataFrame(
            X_pca[:, :min(5, n_components)],
            columns=[f'PC{i+1}' for i in range(min(5, n_components))]
        )
        
        # Feature loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X_processed.columns
        )
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        return {
            'pca_results': pca_results,
            'loadings': loadings,
            'explained_variance_ratio': explained_variance,
            'cumulative_variance_ratio': cumulative_variance,
            'n_components': n_components,
            'total_variance_explained': cumulative_variance[-1]
        }


def create_behavioral_sample_data(n_users: int = 1000, n_events: int = 10000,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Create sample behavioral data for demonstration.
    
    Args:
        n_users: Number of users
        n_events: Number of events
        random_state: Random seed
        
    Returns:
        DataFrame with behavioral data
    """
    np.random.seed(random_state)
    
    # Generate user data
    user_ids = np.random.choice(range(1, n_users + 1), n_events)
    
    # Generate timestamps over the past 30 days
    start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    timestamps = [start_date + pd.Timedelta(hours=np.random.randint(0, 30*24)) 
                 for _ in range(n_events)]
    
    # Generate event types
    event_types = np.random.choice(['login', 'view_page', 'click_button', 'purchase', 'logout'],
                                 n_events, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Generate values (e.g., purchase amounts)
    values = np.where(event_types == 'purchase', 
                     np.random.lognormal(3, 1, n_events), 0)
    
    # Create DataFrame
    behavioral_data = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'event_type': event_types,
        'value': values
    })
    
    return behavioral_data


if __name__ == "__main__":
    # Demonstration
    analyzer = FeatureAnalyzer()
    
    print("=== Feature Analysis Demo ===")
    
    # Create sample data
    from .stats_engine import create_sample_data
    sample_data = create_sample_data(1000)
    behavioral_data = create_behavioral_sample_data(500, 5000)
    
    # Feature importance analysis
    importance = analyzer.feature_importance_analysis(
        sample_data, target_column='group', task_type='classification')
    print(f"\nFeature Importance Analysis:")
    print(f"  Top 3 features:")
    for i, row in importance['feature_importance'].head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # User segmentation
    segmentation = analyzer.user_segmentation(
        sample_data, 
        feature_columns=['age', 'sessions', 'conversion_rate', 'revenue'],
        n_clusters=4
    )
    print(f"\nUser Segmentation:")
    print(f"  Number of clusters: {segmentation['n_clusters']}")
    print(f"  Silhouette score: {segmentation['silhouette_score']:.3f}")
    print(f"  Cluster sizes:")
    for cluster in segmentation['cluster_stats']:
        print(f"    Cluster {cluster['cluster_id']}: {cluster['size']} users ({cluster['percentage']:.1f}%)")
    
    # Behavioral pattern analysis
    behavior = analyzer.behavioral_pattern_analysis(
        behavioral_data,
        user_column='user_id',
        time_column='timestamp',
        event_column='event_type',
        value_column='value'
    )
    print(f"\nBehavioral Pattern Analysis:")
    print(f"  Total users: {behavior['total_users']}")
    print(f"  Total events: {behavior['total_events']}")
    print(f"  Avg events per user: {behavior['avg_events_per_user']:.2f}")
    print(f"  Top 3 events:")
    for i, row in behavior['event_frequency'].head(3).iterrows():
        print(f"    {row['event']}: {row['frequency']} ({row['percentage']:.1f}%)")
    
    # Correlation analysis
    correlations = analyzer.feature_correlation_analysis(
        sample_data,
        feature_columns=['age', 'sessions', 'conversion_rate', 'revenue']
    )
    print(f"\nCorrelation Analysis:")
    print(f"  Strong correlations found: {correlations['n_strong_correlations']}")
    for corr in correlations['strong_correlations'][:3]:
        print(f"    {corr['feature_1']} - {corr['feature_2']}: {corr['correlation']:.3f}")
    
    # PCA analysis
    pca = analyzer.pca_analysis(
        sample_data,
        feature_columns=['age', 'sessions', 'conversion_rate', 'revenue'],
        n_components=2
    )
    print(f"\nPCA Analysis:")
    print(f"  Variance explained by PC1: {pca['explained_variance_ratio'][0]:.3f}")
    print(f"  Variance explained by PC2: {pca['explained_variance_ratio'][1]:.3f}")
    print(f"  Total variance explained: {pca['total_variance_explained']:.3f}")
