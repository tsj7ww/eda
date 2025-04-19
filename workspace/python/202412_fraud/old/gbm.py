import os
import time
import logging

import joblib
from joblib import parallel_backend
from joblib import Memory
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Import the evaluation utilities
from .evaluation import FraudEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GBMFraudDetector')

class GBMFraudDetector:
    """
    A Gradient Boosting Machine-based fraud detector with automated hyperparameter
    optimization using Optuna. Supports multiple GBM implementations and includes
    strategies for handling imbalanced data.
    """
    
    def __init__(
        self,
        time_column='Time',
        amount_column='Amount',
        target_column='Class',
        feature_prefix='V',
        random_state=42,
        model_path='gbm_fraud_model',
        n_jobs=-1,
    ):
        """
        Initialize the GBM fraud detector.
        
        Args:
            time_column: Name of the column containing transaction time
            amount_column: Name of the column containing transaction amount
            target_column: Name of the column with fraud indicator (0/1)
            feature_prefix: Prefix for PCA feature columns (e.g., 'V' for V1-V28)
            random_state: Random seed for reproducibility
            model_path: Path to save model artifacts
        """
        self.time_column = time_column
        self.amount_column = amount_column
        self.target_column = target_column
        self.feature_prefix = feature_prefix
        self.random_state = random_state
        self.model_path = model_path
        self.n_jobs = n_jobs
        
        # Initialize model components
        self.best_model = None
        self.best_params = None
        self.feature_importances = None
        self.study = None
        self.threshold = 0.5  # Default threshold, will be optimized
        self.pca_columns = None
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
    def preprocess_data(self, df, target_included=True):
        """
        Preprocess the dataframe and extract features/target.
        
        Args:
            df: DataFrame with features and target
            target_included: Whether the target column is included in the dataframe
            
        Returns:
            Dict containing preprocessed data
        """
        # Identify PCA feature columns
        self.pca_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
        
        # Extract features
        X = df[self.pca_columns + [self.amount_column, self.time_column]]
        
        # Extract target if included
        y = None
        if target_included:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            y = df[self.target_column]
        
        return {'X': X, 'y': y, 'pca_columns': self.pca_columns}
    
    def _get_preprocessing_pipeline(self, memory=None):
        """
        Create a preprocessing pipeline with appropriate scalers for each feature type.
        
        Returns:
            ColumnTransformer for preprocessing
        """
        # Use RobustScaler for Amount (handles outliers better)
        # Use StandardScaler for PCA features and Time
        preprocessor = ColumnTransformer(
            transformers=[
                ('pca_scaler', StandardScaler(), self.pca_columns),
                ('amount_scaler', RobustScaler(), [self.amount_column]),
                ('time_scaler', StandardScaler(), [self.time_column])
            ],
            n_jobs=self.n_jobs,
        )
        
        return preprocessor
    
    def _get_sampling_method(self, method_name, sampling_ratio=0.1):
        """
        Get the appropriate sampling method for imbalanced data.
        
        Args:
            method_name: Name of sampling method ('smote', 'undersample', 'none')
            sampling_ratio: Ratio of minority to majority class after resampling
            
        Returns:
            Sampling object or None
        """
        if method_name == 'smote':
            return SMOTE(sampling_strategy=sampling_ratio, random_state=self.random_state)
        elif method_name == 'undersample':
            return RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=self.random_state)
        elif method_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown sampling method: {method_name}")
    
    def _create_objective(self, X, y, cv):
        """
        Create an objective function for Optuna optimization.
        
        Args:
            X: Features DataFrame
            y: Target series
            cv: Cross-validation splitter
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Choose GBM implementation
            gbm_implementation = trial.suggest_categorical('gbm_implementation', ['xgboost', 'lightgbm', 'sklearn_gbm'])
            
            # Choose sampling method
            sampling_method = trial.suggest_categorical('sampling_method', ['smote', 'undersample', 'none'])
            
            # Common hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            
            # Only suggest scale_pos_weight when sampling_method is 'none'
            scale_pos_weight = None
            if sampling_method == 'none':
                scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 100.0, log=True)
            
            # Implementation-specific hyperparameters with prefixed parameter names
            if gbm_implementation == 'xgboost':
                learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
                max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
                min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 10)
                
                params = {
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'min_child_weight': min_child_weight,
                    'gamma': trial.suggest_float('xgb_gamma', 1e-8, 1.0, log=True),
                    'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True),
                    'n_estimators': n_estimators,
                    'objective': 'binary:logistic',
                    # 'early_stopping_rounds': 10,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                }
                # Only apply scale_pos_weight when not using sampling
                if sampling_method == 'none':
                    params['scale_pos_weight'] = scale_pos_weight
                
                model = xgb.XGBClassifier(**params)
                
            elif gbm_implementation == 'lightgbm':
                learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True)
                max_depth = trial.suggest_int('lgb_max_depth', 3, 12)
                
                params = {
                    'learning_rate': learning_rate,
                    'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
                    'max_depth': max_depth,
                    'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_float('lgb_min_child_weight', 1e-3, 10.0, log=True),
                    'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
                    'n_estimators': n_estimators,
                    'objective': 'binary',
                    # 'early_stopping_rounds': 10,
                    'random_state': self.random_state,
                    'verbose': -1,
                    'n_jobs': self.n_jobs,
                    'feature_name': 'auto', 
                }
                # Only apply scale_pos_weight when not using sampling
                if sampling_method == 'none':
                    params['scale_pos_weight'] = scale_pos_weight
                    
                model = lgb.LGBMClassifier(**params)
                
            elif gbm_implementation == 'sklearn_gbm':
                params = {
                    'learning_rate': trial.suggest_float('gbm_learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('gbm_max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('gbm_min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('gbm_min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('gbm_subsample', 0.5, 1.0),
                    'max_features': trial.suggest_float('gbm_max_features', 0.5, 1.0),
                    'n_estimators': n_estimators,
                    'random_state': self.random_state,
                    # 'n_jobs': self.n_jobs,
                }
                model = GradientBoostingClassifier(**params)
            else:
                raise ValueError(f'Unknown GBM implementation type: {gbm_implementation}')
                
            # Set up preprocessing
            preprocessor = self._get_preprocessing_pipeline()

            cachedir = mkdtemp()
            memory = Memory(location=cachedir, verbose=0)
            
            # Set up sampling if needed
            if sampling_method != 'none':
                sampling_ratio = trial.suggest_float('sampling_ratio', 0.05, 0.5, log=True)
                sampler = self._get_sampling_method(sampling_method, sampling_ratio)
                
                # Create pipeline with preprocessing, sampling, and model
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('sampler', sampler),
                    ('classifier', model)
                ], memory=memory)
            else:
                # Create pipeline with preprocessing and model only
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ], memory=memory)
                
            # Use cross-validation to evaluate model
            # PR-AUC is a better metric for imbalanced classification than ROC-AUC
            try:
                with parallel_backend('loky', n_jobs=self.n_jobs):
                    scores = cross_val_score(
                        pipeline, X, y, 
                        cv=cv, 
                        scoring='average_precision',
                        n_jobs=self.n_jobs
                    )
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                return float('-inf')
                
        return objective
    
    def optimize(self, df, n_trials=100, cv_folds=5, 
                 optimize_threshold=True, n_parallel=None):
        """
        Run hyperparameter optimization using Optuna.
        
        Args:
            df: DataFrame with features and target
            n_trials: Number of optimization trials to run
            cv_folds: Number of cross-validation folds
            optimize_threshold: Whether to optimize the classification threshold
            
        Returns:
            self with optimized model
        """
        start_time = time.time()
        
        # Set the number of parallel workers if not specified
        if n_parallel is None:
            n_parallel = min(os.cpu_count(), self.n_jobs if self.n_jobs > 0 else os.cpu_count())
        
        # Preprocess data
        data = self.preprocess_data(df)
        X, y = data['X'], data['y']
        
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        minority_class_pct = class_counts.min() / class_counts.sum() * 100
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Minority class percentage: {minority_class_pct:.2f}%")
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Create and run Optuna study
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        
        objective = self._create_objective(X, y, cv)
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel)
        
        # Get best parameters
        self.best_params = self.study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        
        # Build final model with best parameters
        gbm_implementation = self.best_params.pop('gbm_implementation')
        sampling_method = self.best_params.pop('sampling_method')
        
        # Only extract sampling_ratio if it exists
        sampling_ratio = None
        if 'sampling_ratio' in self.best_params:
            sampling_ratio = self.best_params.pop('sampling_ratio')
        
        # Remove scale_pos_weight if it's not applicable
        if 'scale_pos_weight' in self.best_params and sampling_method != 'none':
            self.best_params.pop('scale_pos_weight')
        
        # Create filtered parameters with prefixes removed
        filtered_params = {}
        prefix = gbm_implementation[:3] + '_'  # 'xgb_', 'lgb_', or 'gbm_'
        
        for key, value in self.best_params.items():
            if key.startswith(prefix):
                # Remove the prefix
                filtered_params[key[len(prefix):]] = value
            elif not key.startswith(('xgb_', 'lgb_', 'gbm_')):
                # Keep non-prefixed parameters
                filtered_params[key] = value
        
        # Use filtered parameters for the final model
        self.best_params = filtered_params
        
        # Create model with best parameters
        if gbm_implementation == 'xgboost':
            # Ensure XGBoost-specific settings
            self.best_params['use_label_encoder'] = False
            self.best_params['objective'] = 'binary:logistic'
            model = xgb.XGBClassifier(random_state=self.random_state, **self.best_params)
        elif gbm_implementation == 'lightgbm':
            # Ensure LightGBM-specific settings
            self.best_params['objective'] = 'binary'
            self.best_params['verbose'] = -1
            model = lgb.LGBMClassifier(random_state=self.random_state, **self.best_params)
        else:  # sklearn_gbm
            model = GradientBoostingClassifier(random_state=self.random_state, **self.best_params)
            
        # Create preprocessing pipeline
        preprocessor = self._get_preprocessing_pipeline()
        
        # Create final pipeline based on whether sampling is used
        if sampling_method != 'none' and sampling_ratio is not None:
            sampler = self._get_sampling_method(sampling_method, sampling_ratio)
            self.best_model = ImbPipeline([
                ('preprocessor', preprocessor),
                ('sampler', sampler),
                ('classifier', model)
            ])
        else:
            self.best_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
        # Fit the model on the entire dataset
        logger.info("Fitting final model with best parameters...")
        self.best_model.fit(X, y)
        
        # Extract feature importances
        self._extract_feature_importances(X.columns)
        
        # Optimize threshold if requested
        if optimize_threshold:
            # Use the training data for threshold optimization
            # In a production setting, you might want to use a validation set
            y_prob = self.predict_proba(X)
            self.threshold = FraudEvaluator.find_optimal_threshold(y, y_prob, criterion='f1')
            logger.info(f"Optimized threshold: {self.threshold:.4f}")
            
        logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        return self
    
    def _extract_feature_importances(self, feature_names):
        """
        Extract feature importances from the best model.
        
        Args:
            feature_names: Names of features
        """
        try:
            # Get the classifier from the pipeline
            clf = self.best_model.named_steps['classifier']
            
            # Different GBM implementations store feature importances differently
            if hasattr(clf, 'feature_importances_'):
                # Standard scikit-learn API
                importances = clf.feature_importances_
            elif hasattr(clf, 'get_booster'):
                # XGBoost
                importances = clf.get_booster().get_score(importance_type='gain')
                # Convert to array in the same order as feature_names
                importances = np.array([importances.get(f"f{i}", 0) for i in range(len(feature_names))])
            else:
                logger.warning("Could not extract feature importances")
                return
                
            # Create DataFrame of feature importances
            self.feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
        except Exception as e:
            logger.warning(f"Error extracting feature importances: {str(e)}")
    
    def predict_proba(self, X):
        """
        Predict fraud probabilities.
        
        Args:
            X: Features DataFrame or preprocessed features
            
        Returns:
            Array of fraud probabilities
        """
        if self.best_model is None:
            raise ValueError("Model is not fitted. Call optimize() first.")
            
        # Check if X is a DataFrame and process if needed
        if isinstance(X, pd.DataFrame):
            # Ensure all required columns are present
            required_cols = self.pca_columns + [self.amount_column, self.time_column]
            missing_cols = [col for col in required_cols if col not in X.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Extract features
            X = X[required_cols]
            
        # Get probabilities
        return self.best_model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """
        Predict fraud using the optimized threshold.
        
        Args:
            X: Features DataFrame or preprocessed features
            
        Returns:
            Binary predictions (0/1)
        """
        # Get probabilities and apply threshold
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)
    
    def evaluate(self, df):
        """
        Evaluate the model on test data.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Preprocess data
        data = self.preprocess_data(df)
        X, y = data['X'], data['y']
        
        # Get predictions
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Use the evaluation utility to calculate metrics
        return FraudEvaluator.calculate_metrics(y, y_pred, y_prob, self.threshold)
    
    def plot_evaluation(self, df, figsize=(20, 16)):
        """
        Plot evaluation metrics.
        
        Args:
            df: DataFrame with features and target
            figsize: Figure size
            
        Returns:
            Matplotlib figure with plots
        """
        # Preprocess data
        data = self.preprocess_data(df)
        X, y = data['X'], data['y']
        
        # Get predictions
        y_prob = self.predict_proba(X)
        
        # Use the evaluation utility to create plots
        return FraudEvaluator.plot_metrics(y, y_prob, self.threshold, figsize)
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 10)):
        """
        Plot feature importances.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure with feature importance plot
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not available. Model may not be fitted.")
            
        top_features = self.feature_importances.head(top_n)
        
        return FraudEvaluator.feature_importance_plot(
            top_features['Feature'].values,
            top_features['Importance'].values,
            top_n=top_n,
            figsize=figsize
        )
    
    def plot_optimization_history(self, figsize=(10, 6)):
        """
        Plot the optimization history from Optuna.
        
        Args:
            figsize: Figure size
            
        Returns:
            Optuna visualization figure
        """
        if self.study is None:
            raise ValueError("No optimization study available. Call optimize() first.")
            
        return plot_optimization_history(self.study)
    
    def plot_param_importances(self, figsize=(10, 8)):
        """
        Plot parameter importances from Optuna optimization.
        
        Args:
            figsize: Figure size
            
        Returns:
            Optuna visualization figure
        """
        if self.study is None:
            raise ValueError("No optimization study available. Call optimize() first.")
            
        return plot_param_importances(self.study)
    
    def save_model(self, path=None):
        """
        Save the model and all components.
        
        Args:
            path: Directory path to save the model (defaults to self.model_path)
        """
        if self.best_model is None:
            raise ValueError("Model is not fitted. Call optimize() first.")
            
        if path is None:
            path = self.model_path
            
        # Save model and components
        model_data = {
            'model': self.best_model,
            'feature_importances': self.feature_importances,
            'best_params': self.best_params,
            'threshold': self.threshold,
            'pca_columns': self.pca_columns,
            'time_column': self.time_column,
            'amount_column': self.amount_column,
            'target_column': self.target_column,
            'feature_prefix': self.feature_prefix
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            GBMFraudDetector instance with loaded model
        """
        model_data = joblib.load(path)
        
        # Create a new instance
        detector = cls(
            time_column=model_data['time_column'],
            amount_column=model_data['amount_column'],
            target_column=model_data['target_column'],
            feature_prefix=model_data['feature_prefix'],
            model_path=path
        )
        
        # Restore model components
        detector.best_model = model_data['model']
        detector.feature_importances = model_data['feature_importances']
        detector.best_params = model_data['best_params']
        detector.threshold = model_data['threshold']
        detector.pca_columns = model_data['pca_columns']
        
        logger.info(f"Model loaded from {path}")
        return detector


# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 28  # V1 to V28
    
    # Create dataset with PCA features
    X = np.random.randn(n_samples, n_features)
    
    # Add Amount feature
    amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
    
    # Add Time feature
    time_col = np.arange(n_samples)
    
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
    print(f"Fraud distribution: {df['Class'].value_counts(normalize=True) * 100}")
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    
    # Initialize and optimize GBM detector
    detector = GBMFraudDetector()
    detector.optimize(train_df, n_trials=50)  # Reduced for example
    
    # Evaluate on test data
    results = detector.evaluate(test_df)
    print("\nEvaluation results:")
    print(FraudEvaluator.generate_report(results))
    
    # Plot evaluation metrics
    detector.plot_evaluation(test_df)
    
    # Plot feature importance
    detector.plot_feature_importance()
    
    # Plot optimization history
    detector.plot_optimization_history()
    
    # Save the model
    detector.save_model('gbm_fraud_model.joblib')