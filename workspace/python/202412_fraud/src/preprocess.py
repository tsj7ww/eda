import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations."""
    # Column naming
    time_column: str = 'Time'
    amount_column: str = 'Amount'
    target_column: str = 'Class'
    feature_prefix: str = 'V'
    
    # Scaling methods
    pca_scaling: str = 'standard'  # 'standard', 'robust', 'none'
    amount_scaling: str = 'robust'  # 'standard', 'robust', 'log', 'none'
    time_scaling: str = 'standard'  # 'standard', 'robust', 'none'
    
    # Sampling configuration
    sampling_method: str = 'none'  # 'smote', 'undersampling', 'none'
    sampling_ratio: float = 0.1  # Target ratio for minority class
    
    # Train/validation split
    validation_size: float = 0.2
    stratify_split: bool = True
    random_state: int = 42
    
    # Missing values handling
    handle_missing: str = 'impute'  # 'impute', 'drop', 'none'
    
    # Outlier handling
    outlier_handling: str = 'none'  # 'clip', 'remove', 'none'
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection


class FraudDataPreprocessor:
    """
    Handles preprocessing of fraud detection data, including:
    - Feature extraction
    - Scaling
    - Handling missing values
    - Outlier treatment
    - Data balancing
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration object for preprocessing operations
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize scalers
        self.pca_scaler = None
        self.amount_scaler = None 
        self.time_scaler = None
        
        # Initialize detected feature columns
        self.pca_columns = None
        self.has_been_fit = False
    
    def _detect_columns(self, df: pd.DataFrame) -> None:
        """
        Detect column types in the dataframe.
        
        Args:
            df: Input dataframe
        """
        # Detect PCA feature columns
        self.pca_columns = [col for col in df.columns if col.startswith(self.config.feature_prefix)]
        
        # Check that required columns exist
        required_columns = [self.config.amount_column, self.config.time_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Log detected columns
        logger.info(f"Detected {len(self.pca_columns)} PCA feature columns")
    
    def _initialize_scalers(self) -> None:
        """Initialize the appropriate scalers based on configuration."""
        # PCA features scaler
        if self.config.pca_scaling == 'standard':
            self.pca_scaler = StandardScaler()
        elif self.config.pca_scaling == 'robust':
            self.pca_scaler = RobustScaler()
        
        # Amount scaler
        if self.config.amount_scaling == 'standard':
            self.amount_scaler = StandardScaler()
        elif self.config.amount_scaling == 'robust':
            self.amount_scaler = RobustScaler()
        elif self.config.amount_scaling == 'log':
            self.amount_scaler = LogTransformer()
        
        # Time scaler
        if self.config.time_scaling == 'standard':
            self.time_scaler = StandardScaler()
        elif self.config.time_scaling == 'robust':
            self.time_scaler = RobustScaler()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to configuration.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with missing values handled
        """
        if self.config.handle_missing == 'none':
            return df
            
        # Count missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            
            if self.config.handle_missing == 'drop':
                # Drop rows with any missing values
                df = df.dropna()
                logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
            elif self.config.handle_missing == 'impute':
                # Impute missing values - for PCA features use mean
                if self.pca_columns:
                    df[self.pca_columns] = df[self.pca_columns].fillna(df[self.pca_columns].mean())
                
                # For amount, use median
                if self.config.amount_column in df.columns:
                    df[self.config.amount_column] = df[self.config.amount_column].fillna(
                        df[self.config.amount_column].median()
                    )
                
                # For time, forward fill or use min
                if self.config.time_column in df.columns:
                    df[self.config.time_column] = df[self.config.time_column].fillna(method='ffill')
                    # If still has missing (e.g., first row), use min
                    df[self.config.time_column] = df[self.config.time_column].fillna(df[self.config.time_column].min())
                
                logger.info("Imputed missing values")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers according to configuration.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with outliers handled
        """
        if self.config.outlier_handling == 'none':
            return df
        
        # We'll focus on the amount column, which typically has outliers
        if self.config.amount_column in df.columns:
            amount = df[self.config.amount_column]
            
            # Detect outliers - values more than threshold standard deviations from mean
            mean = amount.mean()
            std = amount.std()
            threshold = self.config.outlier_threshold
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            outlier_mask = (amount < lower_bound) | (amount > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {self.config.amount_column}")
                
                if self.config.outlier_handling == 'clip':
                    # Clip outliers to the bounds
                    df[self.config.amount_column] = amount.clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Clipped {outlier_count} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
                elif self.config.outlier_handling == 'remove':
                    # Remove outliers
                    df = df[~outlier_mask]
                    logger.info(f"Removed {outlier_count} outliers. New shape: {df.shape}")
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame, fit: bool = False) -> Dict[str, np.ndarray]:
        """
        Apply scaling to features based on configuration.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scalers (True) or just transform (False)
            
        Returns:
            Dictionary with scaled feature arrays
        """
        result = {}
        
        # Scale PCA features
        if self.pca_columns and self.pca_scaler:
            X_pca = df[self.pca_columns].values
            if fit:
                result['X_pca'] = self.pca_scaler.fit_transform(X_pca)
            else:
                result['X_pca'] = self.pca_scaler.transform(X_pca)
        else:
            # If no scaling is applied, just return the values
            result['X_pca'] = df[self.pca_columns].values if self.pca_columns else np.array([])
        
        # Scale amount
        if self.config.amount_column in df.columns and self.amount_scaler:
            X_amount = df[[self.config.amount_column]].values
            if fit:
                result['X_amount'] = self.amount_scaler.fit_transform(X_amount)
            else:
                result['X_amount'] = self.amount_scaler.transform(X_amount)
        else:
            # If no scaling is applied, just return the values
            result['X_amount'] = df[[self.config.amount_column]].values if self.config.amount_column in df.columns else np.array([])
        
        # Scale time
        if self.config.time_column in df.columns and self.time_scaler:
            X_time = df[[self.config.time_column]].values
            if fit:
                result['X_time'] = self.time_scaler.fit_transform(X_time)
            else:
                result['X_time'] = self.time_scaler.transform(X_time)
        else:
            # If no scaling is applied, just return the values
            result['X_time'] = df[[self.config.time_column]].values if self.config.time_column in df.columns else np.array([])
        
        return result
    
    def _apply_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sampling to handle class imbalance.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (resampled_features, resampled_targets)
        """
        if self.config.sampling_method == 'none':
            return X, y
        
        # Apply SMOTE oversampling
        if self.config.sampling_method == 'smote':
            logger.info(f"Applying SMOTE with sampling ratio {self.config.sampling_ratio}")
            sampler = SMOTE(
                sampling_strategy=self.config.sampling_ratio,
                random_state=self.config.random_state
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
            logger.info(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
            
            return X_resampled, y_resampled
        
        # Apply random undersampling
        if self.config.sampling_method == 'undersampling':
            logger.info(f"Applying undersampling with sampling ratio {self.config.sampling_ratio}")
            sampler = RandomUnderSampler(
                sampling_strategy=self.config.sampling_ratio,
                random_state=self.config.random_state
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
            logger.info(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
            
            return X_resampled, y_resampled
        
        return X, y
    
    def _combine_features(self, scaled_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple feature arrays into a single array.
        
        Args:
            scaled_features: Dictionary with feature arrays
            
        Returns:
            Combined feature array
        """
        feature_arrays = []
        
        # Add features in a consistent order
        if 'X_pca' in scaled_features and scaled_features['X_pca'].size > 0:
            feature_arrays.append(scaled_features['X_pca'])
        
        if 'X_amount' in scaled_features and scaled_features['X_amount'].size > 0:
            feature_arrays.append(scaled_features['X_amount'])
        
        if 'X_time' in scaled_features and scaled_features['X_time'].size > 0:
            feature_arrays.append(scaled_features['X_time'])
        
        if 'X_derived' in scaled_features and scaled_features['X_derived'].size > 0:
            feature_arrays.append(scaled_features['X_derived'])
        
        # Combine all features horizontally
        if feature_arrays:
            return np.hstack(feature_arrays)
        else:
            return np.array([])
    
    def fit_transform(self, df: pd.DataFrame, return_separate_features: bool = False) -> Union[
        Tuple[np.ndarray, Optional[np.ndarray]],
        Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]
    ]:
        """
        Fit the preprocessor to the data and transform it.
        
        Args:
            df: Input dataframe with features and optionally the target
            return_separate_features: Whether to return features separately or combined
            
        Returns:
            If return_separate_features is True:
                Tuple of (feature_dict, target_array)
            Else:
                Tuple of (combined_feature_array, target_array)
            
            Note: target_array will be None if target column is not in the dataframe
        """
        # Detect columns
        self._detect_columns(df)
        
        # Initialize scalers
        self._initialize_scalers()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Apply scaling
        scaled_features = self._apply_scaling(df, fit=True)
        
        # Extract target if available
        y = None
        if self.config.target_column in df.columns:
            y = df[self.config.target_column].values
        
        # Apply sampling if target is available
        if y is not None and self.config.sampling_method != 'none':
            # Combine features for sampling
            X_combined = self._combine_features(scaled_features)
            
            # Apply sampling
            X_resampled, y_resampled = self._apply_sampling(X_combined, y)
            
            # If we need to return separate features, we need to split them back
            if return_separate_features:
                # This is tricky - we need to split the resampled features back
                # For simplicity, let's just return combined features in this case
                logger.warning("Returning combined features when sampling is applied with return_separate_features=True")
                return {'X_combined': X_resampled}, y_resampled
            else:
                return X_resampled, y_resampled
        
        # If no sampling or no target, return either separate or combined features
        self.has_been_fit = True
        
        if return_separate_features:
            return scaled_features, y
        else:
            return self._combine_features(scaled_features), y
    
    def transform(self, df: pd.DataFrame, return_separate_features: bool = False) -> Union[
        np.ndarray,
        Dict[str, np.ndarray]
    ]:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            df: Input dataframe with features
            return_separate_features: Whether to return features separately or combined
            
        Returns:
            If return_separate_features is True:
                Dictionary with feature arrays
            Else:
                Combined feature array
        """
        if not self.has_been_fit:
            raise ValueError("Preprocessor has not been fit. Call fit_transform first.")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Apply scaling without fitting
        scaled_features = self._apply_scaling(df, fit=False)
        
        if return_separate_features:
            return scaled_features
        else:
            return self._combine_features(scaled_features)
    
    def prepare_train_val_split(self, df: pd.DataFrame, return_separate_features: bool = False) -> Dict[str, Any]:
        """
        Prepare training and validation datasets with preprocessing.
        
        Args:
            df: Input dataframe with features and target
            return_separate_features: Whether to return features separately or combined
            
        Returns:
            Dictionary with train/validation splits
        """
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in dataframe")
        
        # Process the full dataset
        if return_separate_features:
            features, y = self.fit_transform(df, return_separate_features=True)
        else:
            X, y = self.fit_transform(df, return_separate_features=False)
        
        # Create train/validation split
        if self.config.stratify_split:
            if return_separate_features:
                # For separate features, we need to split each feature set
                train_idx, val_idx = train_test_split(
                    np.arange(len(y)),
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state,
                    stratify=y
                )
                
                # Split each feature array by index
                train_features = {}
                val_features = {}
                
                for key, feature_array in features.items():
                    if feature_array.size > 0:
                        train_features[key] = feature_array[train_idx]
                        val_features[key] = feature_array[val_idx]
                
                # Split target
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                return {
                    'train_features': train_features,
                    'train_target': y_train,
                    'val_features': val_features,
                    'val_target': y_val
                }
            else:
                # For combined features, standard train_test_split works
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state,
                    stratify=y
                )
                
                return {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val
                }
        else:
            # Non-stratified split
            if return_separate_features:
                # For separate features, we need to split each feature set
                train_idx, val_idx = train_test_split(
                    np.arange(len(y)),
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state
                )
                
                # Split each feature array by index
                train_features = {}
                val_features = {}
                
                for key, feature_array in features.items():
                    if feature_array.size > 0:
                        train_features[key] = feature_array[train_idx]
                        val_features[key] = feature_array[val_idx]
                
                # Split target
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                return {
                    'train_features': train_features,
                    'train_target': y_train,
                    'val_features': val_features,
                    'val_target': y_val
                }
            else:
                # For combined features, standard train_test_split works
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state
                )
                
                return {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val
                }


# Custom transformer for log transformation
class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to data, handling zeros appropriately."""
    
    def __init__(self, offset: float = 1.0):
        """
        Initialize with offset for log transformation.
        
        Args:
            offset: Value to add before taking log to handle zeros/small values
        """
        self.offset = offset
    
    def fit(self, X, y=None):
        """Fit method (no-op for log transformation)."""
        return self
    
    def transform(self, X):
        """Apply log transformation."""
        return np.log1p(X + self.offset - 1.0)  # ensures log(1) = 0 when X = 0
    
    def inverse_transform(self, X):
        """Reverse the log transformation."""
        return np.expm1(X) - self.offset + 1.0


# Example usage
if __name__ == "__main__":
    # Sample data with PCA features V1-V28, Amount, Time, and Class
    np.random.seed(42)
    n_samples = 1000
    n_features = 28  # V1 to V28
    
    # Create dataset with PCA features
    X = np.random.randn(n_samples, n_features)
    
    # Add Amount feature
    amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
    
    # Add Time column
    time_col = np.arange(n_samples)
    
    # Create imbalanced target (1% fraud)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    # Create dataframe
    feature_cols = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['Amount'] = amount
    df['Time'] = time_col
    df['Class'] = y
    
    # Configure preprocessor
    config = PreprocessingConfig(
        sampling_method='smote',
        sampling_ratio=0.5,  # 1:2 ratio of fraud to non-fraud
        outlier_handling='clip'
    )
    
    preprocessor = FraudDataPreprocessor(config)
    
    # Process and split data
    split_data = preprocessor.prepare_train_val_split(df)
    
    # Print results
    print(f"Training data shape: {split_data['X_train'].shape}")
    print(f"Validation data shape: {split_data['X_val'].shape}")
    print(f"Training class distribution: {np.bincount(split_data['y_train'].astype(int))}")
    print(f"Validation class distribution: {np.bincount(split_data['y_val'].astype(int))}")