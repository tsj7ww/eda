import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import networkx as nx
from collections import defaultdict
import warnings

# Setup logging
logger = logging.getLogger(__name__)

# Import custom utilities if needed
try:
    from utils import timer, ErrorHandler
except ImportError:
    # Simplified versions if utils module is not available
    from contextlib import contextmanager
    import time

    @contextmanager
    def timer(name=None, logger=None):
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        message = f"Time elapsed for {name or 'operation'}: {elapsed_time:.2f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)
            
    class ErrorHandler:
        @staticmethod
        def log_exception(logger, exception, message="An error occurred", include_traceback=True):
            logger.error(f"{message}: {str(exception)}")
            if include_traceback:
                logger.error("Traceback:", exc_info=True)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering operations."""
    # Base column configuration (matching with preprocessing module)
    time_column: str = 'Time'
    amount_column: str = 'Amount'
    target_column: str = 'Class'
    feature_prefix: str = 'V'
    
    # Time-based feature generation
    create_time_features: bool = True
    time_windows: List[int] = field(default_factory=lambda: [300, 3600, 86400])  # 5min, 1hr, 1day in seconds
    time_cyclical_features: bool = True  # Create sin/cos features for time of day, day of week, etc.
    
    # Amount-based feature generation
    create_amount_features: bool = True
    amount_bins: int = 10  # Number of bins for amount discretization
    log_amount: bool = True  # Create log-transformed amount feature
    
    # PCA-based feature generation
    create_pca_aggregates: bool = True  # Create aggregate features from PCA components
    create_pca_interactions: bool = False  # Create interaction features between PCA components
    max_interactions: int = 10  # Maximum number of interaction features to create
    
    # Statistical features
    create_statistical_features: bool = True  # Create statistical features (mean, std, etc.)
    
    # Entity-based features
    entity_columns: List[str] = field(default_factory=list)  # Columns identifying entities (e.g., customer_id, merchant_id)
    create_entity_features: bool = False  # Create features based on entity behavior
    
    # Network-based features
    create_network_features: bool = False  # Create features based on transaction networks
    
    # Feature selection
    perform_feature_selection: bool = False  # Whether to perform feature selection
    feature_selection_method: str = 'mutual_info'  # 'mutual_info' or 'f_classif'
    feature_selection_k: int = 50  # Number of features to select
    
    # Dimensionality reduction
    perform_dimensionality_reduction: bool = False  # Whether to perform dimensionality reduction
    dimensionality_reduction_method: str = 'pca'  # Currently only 'pca' supported
    dimensionality_reduction_n: int = 10  # Number of components to reduce to
    
    # Random state for reproducibility
    random_state: int = 42


class FraudFeatureEngineer:
    """
    Feature engineering for fraud detection.
    
    This class creates advanced features for fraud detection models, including:
    - Time-based features (time windows, cyclical time features)
    - Amount-based features (binning, log transformation)
    - PCA-based features (aggregates, interactions)
    - Statistical features
    - Entity-based features (customer, merchant behavior)
    - Network-based features (transaction networks)
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration for feature engineering
        """
        self.config = config or FeatureEngineeringConfig()
        
        # Initialize components
        self.pca_columns = None
        self.derived_columns = []
        self.has_been_fit = False
        
        # Feature selection components
        self.feature_selector = None
        self.selected_features = None
        
        # Dimensionality reduction components
        self.dim_reducer = None
        
        # Components for feature generation
        self.amount_bins = None
        self.amount_bin_labels = None
    
    def _detect_columns(self, df: pd.DataFrame) -> None:
        """
        Detect columns in the dataframe.
        
        Args:
            df: Input dataframe
        """
        # Detect PCA feature columns
        self.pca_columns = [col for col in df.columns if col.startswith(self.config.feature_prefix)]
        
        # Check that required columns exist
        required_columns = [self.config.time_column, self.config.amount_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            missing_cols_str = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {missing_cols_str}")
        
        # Log detected columns
        logger.info(f"Detected {len(self.pca_columns)} PCA feature columns")
        
        # Check entity columns if specified
        if self.config.create_entity_features and self.config.entity_columns:
            missing_entity_columns = [col for col in self.config.entity_columns if col not in df.columns]
            if missing_entity_columns:
                missing_cols_str = ", ".join(missing_entity_columns)
                logger.warning(f"Missing entity columns: {missing_cols_str}. Entity features will be limited.")
    
    def _create_time_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional time features
        """
        result_df = df.copy()
        time_col = self.config.time_column
        
        if time_col not in result_df.columns:
            logger.warning(f"Time column '{time_col}' not found. Skipping time feature creation.")
            return result_df
        
        # Assume time column is seconds since some reference point
        # Check if time is in reasonable range for UNIX timestamp
        time_values = result_df[time_col].values
        
        # Create time recency feature
        result_df['TransactionRecency'] = time_values.max() - time_values
        
        # Time of day, assuming seconds in a day (0-86400)
        seconds_in_day = 24 * 60 * 60
        result_df['TimeOfDay'] = time_values % seconds_in_day
        
        # Create hour of day (0-23)
        result_df['HourOfDay'] = (result_df['TimeOfDay'] / 3600).astype(int)
        
        # Create cyclical time features if configured
        if self.config.time_cyclical_features:
            # Hour of day cyclical features (sin/cos transformation prevents discontinuity at day boundaries)
            hours = result_df['HourOfDay']
            result_df['HourOfDay_sin'] = np.sin(2 * np.pi * hours / 24)
            result_df['HourOfDay_cos'] = np.cos(2 * np.pi * hours / 24)
            
            # If time column appears to be UNIX timestamp (seconds since 1970-01-01)
            # We can extract more calendar features
            if time_values.min() > 946684800:  # 2000-01-01 in UNIX time
                try:
                    # Convert to datetime
                    timestamps = pd.to_datetime(time_values, unit='s')
                    
                    # Day of week features
                    result_df['DayOfWeek'] = timestamps.dt.dayofweek
                    
                    # Day of week cyclical features
                    days = result_df['DayOfWeek']
                    result_df['DayOfWeek_sin'] = np.sin(2 * np.pi * days / 7)
                    result_df['DayOfWeek_cos'] = np.cos(2 * np.pi * days / 7)
                    
                    # Is weekend feature
                    result_df['IsWeekend'] = (result_df['DayOfWeek'] >= 5).astype(int)
                    
                    # Month cyclical features
                    months = timestamps.dt.month - 1  # 0-11
                    result_df['Month_sin'] = np.sin(2 * np.pi * months / 12)
                    result_df['Month_cos'] = np.cos(2 * np.pi * months / 12)
                    
                except Exception as e:
                    logger.warning(f"Failed to create calendar features: {str(e)}")
        
        # Create velocity features for time windows
        if self.config.time_windows:
            for window in self.config.time_windows:
                window_name = f"{window}s"
                
                # Calculate number of transactions in window
                result_df[f'TransactionCount_{window_name}'] = 0
                
                # This is computationally expensive, so only do it during fit
                if fit:
                    # Sort by time for accurate window calculations
                    sorted_df = result_df.sort_values(by=time_col)
                    time_values = sorted_df[time_col].values
                    
                    # For each transaction, count how many occurred in the previous window
                    for i in range(len(sorted_df)):
                        current_time = time_values[i]
                        window_start = current_time - window
                        
                        # Count transactions in window
                        count = ((time_values >= window_start) & (time_values < current_time)).sum()
                        
                        # Update the count in the original position
                        result_df.loc[sorted_df.index[i], f'TransactionCount_{window_name}'] = count
                    
                    # Create velocity feature (transactions per second)
                    result_df[f'TransactionVelocity_{window_name}'] = result_df[f'TransactionCount_{window_name}'] / window
        
        return result_df
    
    def _create_amount_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create amount-based features.
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional amount features
        """
        result_df = df.copy()
        amount_col = self.config.amount_column
        
        if amount_col not in result_df.columns:
            logger.warning(f"Amount column '{amount_col}' not found. Skipping amount feature creation.")
            return result_df
        
        # Get amount values
        amount_values = result_df[amount_col].values
        
        # Create log-transformed amount if configured
        if self.config.log_amount:
            # Add a small constant to handle zero amounts
            result_df['LogAmount'] = np.log1p(amount_values)
        
        # Create amount deviation from mean
        mean = amount_values.mean()
        std = amount_values.std()
        
        result_df['AmountZScore'] = (amount_values - mean) / std
        result_df['AmountDeviation'] = amount_values - mean
        
        # Flag for unusual amounts (more than 3 standard deviations from mean)
        result_df['UnusualAmount'] = (np.abs(result_df['AmountZScore']) > 3).astype(int)
        
        # Round amount flags (fraudsters often use round numbers)
        result_df['IsRoundAmount'] = ((amount_values % 10 == 0) | 
                                      (amount_values % 100 == 0) | 
                                      (amount_values % 1000 == 0)).astype(int)
        
        # Create amount bins if configured
        if self.config.amount_bins > 0:
            if fit or self.amount_bins is None:
                # Create bins during fit
                self.amount_bins = pd.qcut(
                    amount_values, 
                    q=self.config.amount_bins, 
                    labels=False, 
                    duplicates='drop',
                    retbins=True
                )[1]
                
                # Create bin labels
                self.amount_bin_labels = [f"bin_{i}" for i in range(len(self.amount_bins)-1)]
                
            # Apply binning
            try:
                # Use pandas cut with pre-computed bins
                result_df['AmountBin'] = pd.cut(
                    amount_values,
                    bins=self.amount_bins,
                    labels=self.amount_bin_labels,
                    include_lowest=True
                )
                
                # Convert to categorical and then to integer codes
                result_df['AmountBin'] = result_df['AmountBin'].astype('category').cat.codes
                
                # Handle potential NaN values
                result_df['AmountBin'] = result_df['AmountBin'].fillna(-1).astype(int)
                
            except Exception as e:
                logger.warning(f"Failed to create amount bins: {str(e)}")
                # Fallback: use equal-width bins
                result_df['AmountBin'] = pd.cut(
                    amount_values, 
                    bins=self.config.amount_bins, 
                    labels=False
                ).fillna(-1).astype(int)
        
        return result_df
    
    def _create_pca_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create PCA-based features.
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional PCA-based features
        """
        result_df = df.copy()
        
        if not self.pca_columns or len(self.pca_columns) == 0:
            logger.warning("No PCA columns found. Skipping PCA feature creation.")
            return result_df
        
        # Get PCA features
        X_pca = result_df[self.pca_columns].values
        
        # Create aggregate features if configured
        if self.config.create_pca_aggregates:
            # Magnitude of PCA vector (L2 norm)
            result_df['PCA_Magnitude'] = np.sqrt(np.sum(X_pca**2, axis=1))
            
            # Count of extreme values (features with unusual values)
            result_df['ExtremeFeatureCount'] = np.sum(np.abs(X_pca) > 3, axis=1)
            
            # Calculate skewness and kurtosis of PCA features per transaction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Calculate skewness per row (transaction)
                result_df['PCA_Skewness'] = np.apply_along_axis(
                    lambda x: skew(x) if len(set(x)) > 1 else 0, 
                    axis=1, 
                    arr=X_pca
                )
                
                # Calculate kurtosis per row (transaction)
                result_df['PCA_Kurtosis'] = np.apply_along_axis(
                    lambda x: kurtosis(x) if len(set(x)) > 1 else 0, 
                    axis=1, 
                    arr=X_pca
                )
        
        # Create interaction features if configured
        if self.config.create_pca_interactions and len(self.pca_columns) >= 2:
            # Limit number of interaction features to avoid explosion
            max_interactions = min(self.config.max_interactions, len(self.pca_columns) * (len(self.pca_columns) - 1) // 2)
            
            # If there are too many possible interactions, select important ones or random subset
            if len(self.pca_columns) > 10:
                # Simple heuristic: use first few columns which often contain more information in PCA
                selected_cols = self.pca_columns[:5]
                interaction_cols = []
                
                for i, col1 in enumerate(selected_cols):
                    for col2 in selected_cols[i+1:]:
                        interaction_cols.append((col1, col2))
                        
                        if len(interaction_cols) >= max_interactions:
                            break
                    
                    if len(interaction_cols) >= max_interactions:
                        break
            else:
                # For fewer columns, create all pairwise interactions
                interaction_cols = []
                for i, col1 in enumerate(self.pca_columns):
                    for col2 in self.pca_columns[i+1:]:
                        interaction_cols.append((col1, col2))
                        
                        if len(interaction_cols) >= max_interactions:
                            break
                    
                    if len(interaction_cols) >= max_interactions:
                        break
            
            # Create interaction features
            for i, (col1, col2) in enumerate(interaction_cols):
                result_df[f'Interaction_{i+1}'] = result_df[col1] * result_df[col2]
        
        return result_df
    
    def _create_statistical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create statistical features based on feature distributions.
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional statistical features
        """
        result_df = df.copy()
        
        if not self.pca_columns or len(self.pca_columns) == 0:
            logger.warning("No PCA columns found. Skipping statistical feature creation.")
            return result_df
        
        # Calculate average absolute deviations from zero
        # For PCA features, zero represents the "average" behavior
        result_df['AvgAbsDeviation'] = np.mean(np.abs(result_df[self.pca_columns].values), axis=1)
        
        # Calculate maximum absolute deviation
        result_df['MaxAbsDeviation'] = np.max(np.abs(result_df[self.pca_columns].values), axis=1)
        
        # Count positive and negative values in PCA features
        # Imbalance might indicate unusual behavior
        positive_counts = (result_df[self.pca_columns] > 0).sum(axis=1)
        result_df['PositiveFeatureCount'] = positive_counts
        result_df['NegativeFeatureCount'] = len(self.pca_columns) - positive_counts
        
        # Calculate feature balance/ratio
        # This measures if the PCA features have an unusual balance of positive/negative components
        # Avoid division by zero
        result_df['FeatureBalanceRatio'] = positive_counts / (len(self.pca_columns) - positive_counts + 1e-10)
        
        return result_df
    
    def _create_entity_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create entity-based features (customer, merchant behavior).
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional entity-based features
        """
        result_df = df.copy()
        
        # Skip if no entity columns are configured or available
        if not self.config.entity_columns:
            logger.warning("No entity columns configured. Skipping entity feature creation.")
            return result_df
        
        # Check which entity columns are available
        available_entities = [col for col in self.config.entity_columns if col in result_df.columns]
        
        if not available_entities:
            logger.warning("No entity columns found in data. Skipping entity feature creation.")
            return result_df
        
        # For each entity type, create behavioral features
        for entity_col in available_entities:
            # Group by entity and calculate statistics
            entity_groups = result_df.groupby(entity_col)
            
            # Transaction count per entity
            entity_counts = entity_groups.size().to_dict()
            result_df[f'{entity_col}_TransactionCount'] = result_df[entity_col].map(entity_counts)
            
            # Amount statistics per entity
            if self.config.amount_column in result_df.columns:
                # Calculate mean amount per entity
                entity_mean_amount = entity_groups[self.config.amount_column].mean().to_dict()
                result_df[f'{entity_col}_MeanAmount'] = result_df[entity_col].map(entity_mean_amount)
                
                # Calculate amount relative to entity's mean (detects unusual amounts for this entity)
                result_df[f'{entity_col}_AmountDeviation'] = (
                    result_df[self.config.amount_column] - result_df[f'{entity_col}_MeanAmount']
                )
                
                # Calculate standardized amount within entity
                entity_std_amount = entity_groups[self.config.amount_column].std().fillna(1).to_dict()
                result_df[f'{entity_col}_AmountZScore'] = (
                    result_df[f'{entity_col}_AmountDeviation'] / 
                    result_df[entity_col].map(entity_std_amount).replace(0, 1)  # Avoid division by zero
                )
            
            # Time-based features per entity if time column exists
            if self.config.time_column in result_df.columns:
                # Calculate time since entity's first transaction
                entity_first_time = entity_groups[self.config.time_column].min().to_dict()
                result_df[f'{entity_col}_AccountAge'] = (
                    result_df[self.config.time_column] - result_df[entity_col].map(entity_first_time)
                )
                
                # Calculate time since entity's last transaction (transaction velocity)
                # This requires sorting and is more complex, so only do during fit
                if fit:
                    # Initialize with a large value
                    result_df[f'{entity_col}_TimeSinceLast'] = float('inf')
                    
                    # Sort by entity and time
                    sorted_df = result_df.sort_values([entity_col, self.config.time_column])
                    
                    # Group by entity and calculate time difference
                    for entity, group in sorted_df.groupby(entity_col):
                        if len(group) <= 1:
                            continue
                            
                        # Calculate time differences
                        time_diffs = np.diff(group[self.config.time_column].values)
                        
                        # Pad with infinity for the first transaction
                        time_diffs = np.insert(time_diffs, 0, float('inf'))
                        
                        # Update values in the original dataframe
                        result_df.loc[group.index, f'{entity_col}_TimeSinceLast'] = time_diffs
                    
                    # Replace infinity with a large value
                    result_df[f'{entity_col}_TimeSinceLast'] = result_df[f'{entity_col}_TimeSinceLast'].replace(
                        float('inf'), result_df[f'{entity_col}_TimeSinceLast'].replace(float('inf'), 0).max() * 2
                    )
        
        return result_df
    
    def _create_network_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create network-based features using transaction graphs.
        
        Args:
            df: Input dataframe
            fit: Whether this is a fit operation
            
        Returns:
            DataFrame with additional network features
        """
        result_df = df.copy()
        
        # Skip if network features are not enabled
        if not self.config.create_network_features:
            return result_df
        
        # Need at least a source and target entity for network analysis
        entity_cols = [col for col in self.config.entity_columns if col in result_df.columns]
        
        if len(entity_cols) < 2:
            logger.warning("Need at least 2 entity columns for network features. Skipping.")
            return result_df
        
        # For simplicity, use the first two entity columns as source and target
        source_col = entity_cols[0]
        target_col = entity_cols[1]
        
        # Create a directed graph from transactions
        G = nx.DiGraph()
        
        # Add nodes and edges from transactions
        for _, row in result_df.iterrows():
            source = row[source_col]
            target = row[target_col]
            
            # Add nodes if they don't exist
            if source not in G:
                G.add_node(source, type='source')
            
            if target not in G:
                G.add_node(target, type='target')
            
            # Add or update edge
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)
        
        # Calculate network metrics
        try:
            # Out-degree (number of unique targets for each source)
            source_outdegree = dict(G.out_degree())
            result_df[f'{source_col}_OutDegree'] = result_df[source_col].map(source_outdegree).fillna(0)
            
            # In-degree (number of unique sources for each target)
            target_indegree = dict(G.in_degree())
            result_df[f'{target_col}_InDegree'] = result_df[target_col].map(target_indegree).fillna(0)
            
            # Transaction count per edge (source-target pair)
            edge_weights = {(s, t): d['weight'] for s, t, d in G.edges(data=True)}
            result_df['EdgeWeight'] = result_df.apply(
                lambda row: edge_weights.get((row[source_col], row[target_col]), 0), 
                axis=1
            )
            
            # PageRank of entities (identifies important nodes in the network)
            try:
                # Calculate PageRank
                pagerank = nx.pagerank(G, weight='weight')
                
                # Map PageRank values to entities
                result_df[f'{source_col}_PageRank'] = result_df[source_col].map(pagerank).fillna(0)
                result_df[f'{target_col}_PageRank'] = result_df[target_col].map(pagerank).fillna(0)
            except Exception as e:
                logger.warning(f"Failed to calculate PageRank: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Error creating network features: {str(e)}")
        
        return result_df
    
    def _perform_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature selection to reduce dimensionality.
        
        Args:
            df: Input dataframe with features and target
            
        Returns:
            DataFrame with selected features
        """
        # Feature selection requires the target column
        if self.config.target_column not in df.columns:
            logger.warning(f"Target column '{self.config.target_column}' not found. Skipping feature selection.")
            return df
        
        # Identify feature columns (exclude target and any other special columns)
        exclude_cols = [self.config.target_column]
        if self.config.entity_columns:
            exclude_cols.extend(self.config.entity_columns)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No feature columns found. Skipping feature selection.")
            return df
        
        # Get features and target
        X = df[feature_cols].values
        y = df[self.config.target_column].values
        
        # Create feature selector if not already created
        if self.feature_selector is None:
            # Choose scoring function based on configuration
            if self.config.feature_selection_method == 'mutual_info':
                score_func = mutual_info_classif
            else:  # default to f_classif
                score_func = f_classif
            
            # Create selector
            k = min(self.config.feature_selection_k, len(feature_cols))
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            
            # Fit the selector
            self.feature_selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            
            # Get selected feature names
            self.selected_features = [feature_cols[i] for i in selected_indices]
            
            logger.info(f"Selected {len(self.selected_features)} features: {', '.join(self.selected_features[:5])}...")
        
        # Return dataframe with selected features and target
        result_cols = self.selected_features + [self.config.target_column]
        if self.config.entity_columns:
            result_cols.extend([col for col in self.config.entity_columns if col in df.columns])
        
        return df[result_cols]
    
    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously fit feature selection.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with selected features
        """
        if self.selected_features is None:
            logger.warning("No feature selection has been performed. Returning original dataframe.")
            return df
        
        # Check that all selected features are in the dataframe
        missing_features = [col for col in self.selected_features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing selected features: {', '.join(missing_features)}. Feature selection may be incorrect.")
        
        # Get columns to keep in the result
        result_cols = [col for col in self.selected_features if col in df.columns]
        
        # Add target column if it exists
        if self.config.target_column in df.columns:
            result_cols.append(self.config.target_column)
        
        # Add entity columns if they exist
        if self.config.entity_columns:
            result_cols.extend([col for col in self.config.entity_columns if col in df.columns])
        
        return df[result_cols]
    
    def _perform_dimensionality_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform dimensionality reduction on the features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with reduced dimensionality
        """
        # Identify feature columns (exclude target and any other special columns)
        exclude_cols = []
        if self.config.target_column in df.columns:
            exclude_cols.append(self.config.target_column)
        
        if self.config.entity_columns:
            exclude_cols.extend([col for col in self.config.entity_columns if col in df.columns])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No feature columns found. Skipping dimensionality reduction.")
            return df
        
        # Get features
        X = df[feature_cols].values
        
        # Create dimensionality reducer if not already created
        if self.dim_reducer is None:
            if self.config.dimensionality_reduction_method == 'pca':
                # Cap the number of components
                n_components = min(self.config.dimensionality_reduction_n, len(feature_cols), X.shape[0])
                
                # Create and fit PCA
                self.dim_reducer = PCA(n_components=n_components, random_state=self.config.random_state)
                self.dim_reducer.fit(X)
                
                logger.info(f"PCA explained variance: {sum(self.dim_reducer.explained_variance_ratio_):.4f}")
            else:
                logger.warning(f"Unsupported dimensionality reduction method: {self.config.dimensionality_reduction_method}")
                return df
        
        # Apply dimensionality reduction
        X_reduced = self.dim_reducer.transform(X)
        
        # Create dataframe with reduced features
        reduced_cols = [f'Component_{i+1}' for i in range(X_reduced.shape[1])]
        reduced_df = pd.DataFrame(X_reduced, columns=reduced_cols, index=df.index)
        
        # Add back target and entity columns
        for col in exclude_cols:
            if col in df.columns:
                reduced_df[col] = df[col].values
        
        return reduced_df
    
    def _apply_dimensionality_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously fit dimensionality reduction.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with reduced dimensionality
        """
        if self.dim_reducer is None:
            logger.warning("No dimensionality reduction has been performed. Returning original dataframe.")
            return df
        
        # Identify feature columns (exclude target and any other special columns)
        exclude_cols = []
        if self.config.target_column in df.columns:
            exclude_cols.append(self.config.target_column)
        
        if self.config.entity_columns:
            exclude_cols.extend([col for col in self.config.entity_columns if col in df.columns])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get features
        X = df[feature_cols].values
        
        # Apply dimensionality reduction
        try:
            X_reduced = self.dim_reducer.transform(X)
            
            # Create dataframe with reduced features
            reduced_cols = [f'Component_{i+1}' for i in range(X_reduced.shape[1])]
            reduced_df = pd.DataFrame(X_reduced, columns=reduced_cols, index=df.index)
            
            # Add back target and entity columns
            for col in exclude_cols:
                if col in df.columns:
                    reduced_df[col] = df[col].values
            
            return reduced_df
            
        except Exception as e:
            logger.error(f"Failed to apply dimensionality reduction: {str(e)}")
            return df

# Example usage and utility functions
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample data with PCA features V1-V28, Amount, Time, and Class
    np.random.seed(42)
    n_samples = 1000
    n_features = 28  # V1 to V28
    
    # Create dataset with PCA features
    X = np.random.randn(n_samples, n_features)
    
    # Add Amount feature (transaction amount)
    amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
    
    # Add Time column (seconds elapsed)
    time_col = np.arange(n_samples) * 100  # 100 seconds between transactions
    
    # Create imbalanced target (1% fraud)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    # Add some entity IDs for network features
    customer_ids = np.random.randint(1, 100, size=n_samples)
    merchant_ids = np.random.randint(1, 50, size=n_samples)
    
    # Create dataframe with proper column names
    feature_cols = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['Amount'] = amount
    df['Time'] = time_col
    df['Class'] = y
    df['CustomerID'] = customer_ids
    df['MerchantID'] = merchant_ids
    
    print(f"Original dataframe shape: {df.shape}")
    
    # Configure feature engineering
    config = FeatureEngineeringConfig(
        create_time_features=True,
        create_amount_features=True,
        create_pca_aggregates=True,
        create_statistical_features=True,
        entity_columns=['CustomerID', 'MerchantID'],
        create_entity_features=True,
        create_network_features=True
    )
    
    # Create feature engineer
    engineer = FraudFeatureEngineer(config)
    
    # Apply feature engineering
    enriched_df = engineer.fit_transform(df)
    
    print(f"Enriched dataframe shape: {enriched_df.shape}")
    print(f"Added {len(engineer.derived_columns)} new features")
    
    # Print sample of new features
    print("\nSample of new features:")
    new_features = list(set(enriched_df.columns) - set(df.columns))
    sample_features = sorted(new_features)[:10]  # First 10 new features
    
    for feature in sample_features:
        print(f"{feature}: {enriched_df[feature].describe()}")
    
    # Test transform on new data
    print("\nTesting transform on new data:")
    df_test = df.copy()
    enriched_test = engineer.transform(df_test)
    print(f"Test dataframe shape after transform: {enriched_test.shape}")
    
    # Check feature importance using a simple model
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        print("\nCalculating feature importance:")
        
        # Select numeric columns only
        numeric_cols = enriched_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'Class']
        
        # Train a simple model
        X = enriched_df[feature_cols].values
        y = enriched_df['Class'].values
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        print("Top 20 most important features:")
        for i in range(min(20, len(feature_cols))):
            idx = indices[i]
            print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
        
    except ImportError:
        print("RandomForestClassifier not available. Skipping feature importance calculation.")