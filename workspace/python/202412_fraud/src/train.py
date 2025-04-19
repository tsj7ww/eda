import os
import time
import logging
import json
import pickle
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# Setup logging
logger = logging.getLogger(__name__)

# Import utilities if available
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


class BaseModelTrainer(ABC):
    """
    Abstract base class for fraud detection model trainers.
    
    This base class defines the common interface for all model trainers
    and implements shared functionality.
    """
    
    def __init__(self, config: ModelTrainingConfig = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or ModelTrainingConfig()
        
        # Ensure model directory exists
        if self.config.model_dir:
            os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.threshold = 0.5  # Default threshold, will be optimized if configured
        self.feature_importance = None
        self.optimization_history = []
        self.best_params = {}
        self.has_been_fit = False
    
    @abstractmethod
    def _build_model(self, params: Dict[str, Any] = None) -> Any:
        """
        Build and return a model instance with the given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: Optional[np.ndarray] = None, 
                     y_val: Optional[np.ndarray] = None) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def _optimize_hyperparams(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Best hyperparameters
        """
        pass
    
    @abstractmethod
    def _extract_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from the model.
        
        Args:
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importances
        """
        pass
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Find the optimal threshold for classification based on criterion.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import precision_recall_curve, confusion_matrix
        
        criterion = self.config.threshold_criterion
        
        # Get precision and recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        if criterion == 'f1':
            # Calculate F1 score for each threshold
            f1_scores = []
            for i in range(len(precision)):
                if precision[i] + recall[i] > 0:  # Avoid division by zero
                    f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                else:
                    f1 = 0
                f1_scores.append(f1)
                
            # Find threshold that maximizes F1 score
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])  # Exclude the last value
                best_threshold = thresholds[best_idx]
                logger.info(f"Optimized threshold (F1 criterion): {best_threshold:.4f}")
                return best_threshold
                
        elif criterion == 'precision_recall_balance':
            # Find threshold where precision and recall are closest
            differences = np.abs(precision - recall)
            best_idx = np.argmin(differences[:-1])  # Exclude the last value
            best_threshold = thresholds[best_idx]
            logger.info(f"Optimized threshold (Precision-Recall balance): {best_threshold:.4f}")
            return best_threshold
            
        elif criterion == 'cost':
            # Define a cost function (example: 10*FP + 500*FN)
            costs = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                # Cost function: each FP costs $10, each FN costs $500
                cost = 10 * fp + 500 * fn
                costs.append(cost)
                
            best_idx = np.argmin(costs)
            best_threshold = thresholds[best_idx]
            logger.info(f"Optimized threshold (Cost criterion): {best_threshold:.4f}")
            return best_threshold
            
        # Default
        return 0.5
    
    def _apply_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sampling method to handle class imbalance.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (resampled_features, resampled_labels)
        """
        if self.config.sampling_method == 'none':
            return X, y
        
        if self.config.sampling_method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                
                logger.info(f"Applying SMOTE with ratio {self.config.sampling_ratio}")
                sampler = SMOTE(
                    sampling_strategy=self.config.sampling_ratio,
                    random_state=self.config.random_state
                )
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
                logger.info(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
                
                return X_resampled, y_resampled
                
            except ImportError:
                logger.warning("SMOTE not available. Skipping sampling.")
                return X, y
                
        elif self.config.sampling_method == 'undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler
                
                logger.info(f"Applying undersampling with ratio {self.config.sampling_ratio}")
                sampler = RandomUnderSampler(
                    sampling_strategy=self.config.sampling_ratio,
                    random_state=self.config.random_state
                )
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
                logger.info(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
                
                return X_resampled, y_resampled
                
            except ImportError:
                logger.warning("RandomUnderSampler not available. Skipping sampling.")
                return X, y
        
        return X, y
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None,
            feature_names: Optional[List[str]] = None) -> 'BaseModelTrainer':
        """
        Fit the model to the data.
        
        Args:
            X: Training features or DataFrame with features and target
            y: Training labels (optional if X is DataFrame with target column)
            feature_names: Names of features (optional if X is DataFrame)
            
        Returns:
            Trained model trainer instance
        """
        start_time = time.time()
        
        # Extract X and y if X is DataFrame
        if isinstance(X, pd.DataFrame):
            # Extract feature names if not provided
            if feature_names is None:
                if y is None and self.config.target_column in X.columns:
                    # If y is not provided, extract from DataFrame
                    feature_names = [col for col in X.columns if col != self.config.target_column]
                else:
                    # If y is provided, use all columns as features
                    feature_names = X.columns.tolist()
            
            # Extract target from DataFrame if not provided
            if y is None and self.config.target_column in X.columns:
                y = X[self.config.target_column].values
                X = X[feature_names].values
            else:
                X = X.values
        
        # Convert y to numpy array if it's a Series
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}")
        
        # Set default feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Starting model training with {X.shape[0]} samples and {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Apply hyperparameter optimization if configured
        if self.config.perform_optimization:
            with timer("Hyperparameter optimization", logger):
                try:
                    self.best_params = self._optimize_hyperparams(X, y)
                    logger.info(f"Best parameters: {self.best_params}")
                except Exception as e:
                    ErrorHandler.log_exception(
                        logger, 
                        e, 
                        message="Hyperparameter optimization failed, using default parameters",
                        include_context={"X_shape": str(X.shape), "y_shape": str(y.shape)}
                    )
                    # Use default parameters
                    if self.config.model_type == 'gbm':
                        self.best_params = self.config.gbm_params
                    elif self.config.model_type == 'deeplearning':
                        self.best_params = self.config.dl_params
                    else:
                        self.best_params = {}
        else:
            # Use configured default parameters
            if self.config.model_type == 'gbm':
                self.best_params = self.config.gbm_params
            elif self.config.model_type == 'deeplearning':
                self.best_params = self.config.dl_params
            else:
                self.best_params = {}
        
        # Create train/validation split if needed for final training
        if not self.config.use_cv:
            from sklearn.model_selection import train_test_split
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=self.config.random_state,
                stratify=y
            )
            
            # Apply sampling to training data if configured
            X_train, y_train = self._apply_sampling(X_train, y_train)
            
            # Train the model
            with timer("Model training", logger):
                self.model = self._train_model(X_train, y_train, X_val, y_val)
        else:
            # Train on the full dataset if using cross-validation for optimization
            # Apply sampling if configured
            X_resampled, y_resampled = self._apply_sampling(X, y)
            
            # Train the model
            with timer("Model training", logger):
                self.model = self._train_model(X_resampled, y_resampled)
        
        # Extract feature importance
        with timer("Feature importance extraction", logger):
            try:
                self.feature_importance = self._extract_feature_importance(feature_names)
            except Exception as e:
                ErrorHandler.log_exception(
                    logger, 
                    e, 
                    message="Feature importance extraction failed",
                    include_context={"feature_names": str(feature_names[:5]) + "..."}
                )
        
        # Optimize threshold if configured
        if self.config.optimize_threshold:
            with timer("Threshold optimization", logger):
                # Get predictions
                y_prob = self.predict_proba(X)
                self.threshold = self._optimize_threshold(y, y_prob)
        
        self.has_been_fit = True
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of fraud probabilities
        """
        if not self.has_been_fit:
            raise ValueError("Model has not been fitted. Call fit first.")
        
        # Convert DataFrame to array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Implementation-specific prediction
        return self._predict_proba_impl(X)
    
    @abstractmethod
    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        """
        Implementation-specific probability prediction.
        
        Args:
            X: Features
            
        Returns:
            Array of fraud probabilities
        """
        pass
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict fraud using the optimized threshold.
        
        Args:
            X: Features
            
        Returns:
            Binary predictions (0/1)
        """
        # Get probabilities
        y_prob = self.predict_proba(X)
        
        # Apply threshold
        return (y_prob >= self.threshold).astype(int)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get predictions
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # Calculate evaluation metrics
        return self._calculate_metrics(y, y_pred, y_prob, self.threshold)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
        """
        Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities
            threshold: Classification threshold used
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import confusion_matrix, precision_recall_curve
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'threshold': threshold
        }
        
        # Add AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Add detection rate and false alarm rate
        metrics['detection_rate'] = metrics['recall']  # Same as recall/TPR
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR
        
        # Add business metrics
        # Assuming each TP saves $200, each FP costs $10, and each FN costs $200
        fraud_savings = tp * 200
        investigation_costs = fp * 10
        missed_fraud_costs = fn * 200
        metrics['net_savings'] = fraud_savings - investigation_costs - missed_fraud_costs
        
        return metrics
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model and components.
        
        Args:
            path: Path to save model (optional, uses config.model_dir if not provided)
            
        Returns:
            Path to saved model
        """
        if not self.has_been_fit:
            raise ValueError("Model has not been trained. Call fit first.")
        
        # Use configured path if not provided
        if path is None:
            path = os.path.join(self.config.model_dir, f"{self.config.model_name}.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Implementation-specific model saving
        model_path = self._save_model_impl(path)
        
        # Save common components
        metadata = {
            'model_type': self.config.model_type,
            'model_implementation': self.config.model_implementation,
            'threshold': self.threshold,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'created_at': time.time(),
        }
        
        with open(path + '.meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance if available
        if self.feature_importance is not None:
            self.feature_importance.to_csv(path + '.feature_importance.csv', index=False)
        
        logger.info(f"Model and metadata saved to {path}")
        return model_path
    
    @abstractmethod
    def _save_model_impl(self, path: str) -> str:
        """
        Implementation-specific model saving.
        
        Args:
            path: Path to save model
            
        Returns:
            Path to saved model
        """
        pass


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
    
    # Create dataframe with proper column names
    feature_cols = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['Amount'] = amount
    df['Time'] = time_col
    df['Class'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {np.bincount(y.astype(int))}")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    
    # Configure GBM model training
    config = ModelTrainingConfig(
        model_type='gbm',
        model_implementation='xgboost',  # Change to 'lightgbm' or 'sklearn_gbm' to test other implementations
        optimization_trials=5,  # Reduced for example
        sampling_method='smote',
        threshold_criterion='f1',
        model_name='fraud_detection_gbm'
    )
    
    # Create trainer
    trainer = GBMModelTrainer(config)
    
    # Train the model
    trainer.fit(train_df)
    
    # Evaluate on test data
    results = trainer.evaluate(test_df, test_df['Class'])
    
    print("\nEvaluation results:")
    for key, value in results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")
    
    # Print confusion matrix
    cm = results['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Print feature importance
    if trainer.feature_importance is not None:
        print("\nTop 10 most important features:")
        print(trainer.feature_importance.head(10))
    
    # Save the model
    model_path = trainer.save()
    print(f"\nModel saved to {model_path}")
    
    # Test loading the model
    loaded_trainer = GBMModelTrainer.load(model_path)
    
    # Evaluate loaded model
    loaded_results = loaded_trainer.evaluate(test_df, test_df['Class'])
    
    print("\nLoaded model evaluation results:")
    print(f"PR-AUC: {loaded_results['pr_auc']:.4f}")
    print(f"ROC-AUC: {loaded_results['roc_auc']:.4f}")
    print(f"F1-Score: {loaded_results['f1_score']:.4f}")
    print(f"Precision: {loaded_results['precision']:.4f}")
    print(f"Recall: {loaded_results['recall']:.4f}")
    print(f"Net Savings: ${loaded_results['net_savings']:.2f}")