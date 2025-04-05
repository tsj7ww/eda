import os
import time
import logging
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from imblearn.over_sampling import SMOTE
from scipy.stats import skew

# Import the evaluation utilities
from .evaluation import FraudEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeepFraudDetector')

class DeepFraudDetector:
    """
    A deep learning-based fraud detector specifically designed for credit card fraud detection
    with PCA features (V1-V28), Amount, Time, and Class columns.
    
    Features:
    - Neural network architecture optimized for imbalanced fraud detection
    - Supports different resampling strategies for imbalanced data
    - Automatic feature scaling
    - Model evaluation focused on fraud detection metrics
    - Feature engineering to create derived features
    """
    
    def __init__(
        self,
        time_column='Time',
        amount_column='Amount',
        target_column='Class',
        feature_prefix='V',
        random_state=42,
        model_path='deep_fraud_model'
    ):
        """
        Initialize the deep learning fraud detector.
        
        Args:
            time_column: Name of the column containing transaction time
            amount_column: Name of the column containing transaction amount
            target_column: Name of the column with fraud indicator (0/1)
            feature_prefix: Prefix for PCA feature columns (e.g., 'V' for V1-V28)
            random_state: Random seed for reproducibility
            model_path: Path to save model checkpoints
        """
        self.time_column = time_column
        self.amount_column = amount_column
        self.target_column = target_column
        self.feature_prefix = feature_prefix
        self.random_state = random_state
        self.model_path = model_path
        
        # Initialize components
        self.model = None
        self.pca_scaler = StandardScaler()
        self.amount_scaler = RobustScaler()  # Better for amount which may have outliers
        self.time_scaler = StandardScaler()
        self.derived_scaler = StandardScaler()  # For derived features
        self.threshold = 0.5  # Default threshold, will be optimized
        self.pca_columns = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def add_derived_features(self, df):
        """
        Add domain-specific derived features that might help with fraud detection.
        
        Args:
            df: DataFrame with original features
            
        Returns:
            DataFrame with additional derived features
        """
        # Create a copy to avoid modifying the original DataFrame
        df_new = df.copy()
        
        # == Time-based features ==
        
        # Transaction recency (how recent was this transaction compared to the latest)
        df_new['TransactionRecency'] = df_new[self.time_column].max() - df_new[self.time_column]
        
        # Time of day (assuming time column represents seconds, normalize to 24h cycle)
        # Many fraud attacks happen during specific times of day
        seconds_in_day = 24 * 60 * 60
        df_new['TimeOfDay'] = df_new[self.time_column] % seconds_in_day
        
        # Is weekend/weekday - if your time is absolute timestamp
        # Requires datetime conversion - example assumes you can convert time to date
        # df_new['IsWeekend'] = pd.to_datetime(df_new[self.time_column], unit='s').dt.dayofweek >= 5
        
        # == Amount-based features ==
        
        # Amount buckets (discretize amounts into meaningful categories)
        # This helps capture non-linear relationships with amount
        try:
            df_new['AmountBucket'] = pd.qcut(
                df_new[self.amount_column], 
                q=10, 
                labels=False, 
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"Could not create AmountBucket: {e}")
            # Fall back to equal-width bins if quantile binning fails
            df_new['AmountBucket'] = pd.cut(
                df_new[self.amount_column], 
                bins=10, 
                labels=False
            )
        
        # Log transformation of amount (fraud patterns often visible in log space)
        # Add small constant to avoid log(0)
        df_new['LogAmount'] = np.log1p(df_new[self.amount_column])
        
        # Unusual amount flag (transactions with amounts that deviate from the norm)
        amount_mean = df_new[self.amount_column].mean()
        amount_std = df_new[self.amount_column].std()
        df_new['AmountZScore'] = (df_new[self.amount_column] - amount_mean) / amount_std
        df_new['UnusualAmount'] = (df_new['AmountZScore'].abs() > 3).astype(int)
        
        # Round numbers flag (fraudsters often use round numbers)
        df_new['IsRoundAmount'] = (
            (df_new[self.amount_column] % 10 == 0) | 
            (df_new[self.amount_column] % 100 == 0) | 
            (df_new[self.amount_column] % 1000 == 0)
        ).astype(int)
        
        # == PCA feature derivations ==
        
        # If we have PCA features, derive aggregate statistics
        if self.pca_columns or [col for col in df.columns if col.startswith(self.feature_prefix)]:
            # Get PCA columns if not already set
            if not self.pca_columns:
                self.pca_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
                
            # Magnitude of PCA vector (overall deviation from mean transaction)
            df_new['PCA_Magnitude'] = np.sqrt(
                np.sum([df_new[col]**2 for col in self.pca_columns], axis=0)
            )
            
            # Count of extreme PCA values (features with unusual values)
            # Assuming PCA features are already centered and scaled
            df_new['ExtremeFeatureCount'] = np.sum(
                [(df_new[col].abs() > 3) for col in self.pca_columns], 
                axis=0
            )
            
            # PCA skewness (transaction with asymmetric feature distribution)
            # This can detect unusual combinations of features
            def calc_skew(row):
                # Return skewness of PCA features for this transaction
                values = [row[col] for col in self.pca_columns]
                if len(set(values)) <= 1:  # If all values are the same
                    return 0
                return skew(values)
            
            try:
                df_new['PCA_Skewness'] = df_new.apply(calc_skew, axis=1)
            except Exception as e:
                logger.warning(f"Could not calculate PCA_Skewness: {e}")
        
        logger.info(f"Added {len(df_new.columns) - len(df.columns)} derived features")
        
        return df_new
    
    def preprocess_data(self, df, fit_scalers=True):
        """
        Preprocess the dataframe by splitting features and scaling.
        
        Args:
            df: DataFrame with features and target
            fit_scalers: Whether to fit the scalers (True for training, False for inference)
            
        Returns:
            Tuple of (scaled_pca_features, scaled_amount, scaled_time, scaled_derived_features, target_if_available)
        """
        # Identify PCA feature columns
        self.pca_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
        
        # Extract core features and target
        X_pca = df[self.pca_columns].values
        X_amount = df[[self.amount_column]].values
        X_time = df[[self.time_column]].values
        
        # Scale features
        if fit_scalers:
            X_pca_scaled = self.pca_scaler.fit_transform(X_pca)
            X_amount_scaled = self.amount_scaler.fit_transform(X_amount)
            X_time_scaled = self.time_scaler.fit_transform(X_time)
        else:
            X_pca_scaled = self.pca_scaler.transform(X_pca)
            X_amount_scaled = self.amount_scaler.transform(X_amount)
            X_time_scaled = self.time_scaler.transform(X_time)
        
        # Extract additional derived features if present
        X_derived_scaled = np.array([])  # Empty array by default
        
        derived_feature_columns = [
            col for col in df.columns 
            if col not in self.pca_columns and 
               col != self.amount_column and 
               col != self.time_column and 
               col != self.target_column
        ]
        
        if derived_feature_columns:
            X_derived = df[derived_feature_columns].values
            
            if fit_scalers:
                X_derived_scaled = self.derived_scaler.fit_transform(X_derived)
            else:
                X_derived_scaled = self.derived_scaler.transform(X_derived)
        
        # Extract target if available
        y = None
        if self.target_column in df.columns:
            y = df[self.target_column].values
        
        return X_pca_scaled, X_amount_scaled, X_time_scaled, X_derived_scaled, y
    
    def apply_sampling(self, X_pca, X_amount, X_time, X_derived, y, sampling_method='smote', sampling_ratio=0.1):
        """
        Apply sampling methods to handle class imbalance.
        
        Args:
            X_pca: PCA features array
            X_amount: Amount feature array
            X_time: Time feature array
            X_derived: Derived features array (can be empty)
            y: Target array
            sampling_method: Method to use ('smote', 'none')
            sampling_ratio: Ratio of minority to majority class after resampling
            
        Returns:
            Resampled versions of inputs
        """
        if sampling_method.lower() == 'none':
            return X_pca, X_amount, X_time, X_derived, y
        
        # Combine features for sampling
        has_derived = X_derived.size > 0
        if has_derived:
            X_combined = np.hstack([X_pca, X_amount, X_time, X_derived])
        else:
            X_combined = np.hstack([X_pca, X_amount, X_time])
        
        # Apply SMOTE
        if sampling_method.lower() == 'smote':
            smote = SMOTE(sampling_strategy=sampling_ratio, random_state=self.random_state)
            X_combined_resampled, y_resampled = smote.fit_resample(X_combined, y)
            
            # Split back into separate feature sets
            X_pca_resampled = X_combined_resampled[:, :X_pca.shape[1]]
            X_amount_resampled = X_combined_resampled[:, X_pca.shape[1]:X_pca.shape[1]+1].reshape(-1, 1)
            X_time_resampled = X_combined_resampled[:, X_pca.shape[1]+1:X_pca.shape[1]+2].reshape(-1, 1)
            
            # Handle derived features if present
            if has_derived:
                X_derived_resampled = X_combined_resampled[:, X_pca.shape[1]+2:]
                return X_pca_resampled, X_amount_resampled, X_time_resampled, X_derived_resampled, y_resampled
            else:
                return X_pca_resampled, X_amount_resampled, X_time_resampled, np.array([]), y_resampled
        
        # Default: return original data
        return X_pca, X_amount, X_time, X_derived, y
    
    def build_model(self, pca_dim=28, derived_dim=0, model_type='multiinput'):
        """
        Build the deep learning model architecture.
        
        Args:
            pca_dim: Dimension of PCA features (default 28 for V1-V28)
            derived_dim: Dimension of derived features (0 if none)
            model_type: Type of model architecture to use
                'multiinput': Separate inputs for PCA, amount, time, and derived features
                'simple': Single input for all features combined
                
        Returns:
            Built Keras model
        """
        if model_type == 'multiinput':
            # Multi-input model with separate processing paths
            
            # PCA features input branch
            pca_input = Input(shape=(pca_dim,), name='pca_input')
            pca_branch = Dense(64, activation='relu')(pca_input)
            pca_branch = BatchNormalization()(pca_branch)
            pca_branch = Dropout(0.3)(pca_branch)
            pca_branch = Dense(32, activation='relu')(pca_branch)
            
            # Amount input branch - single feature but important
            amount_input = Input(shape=(1,), name='amount_input')
            amount_branch = Dense(8, activation='relu')(amount_input)
            amount_branch = BatchNormalization()(amount_branch)
            
            # Time input branch
            time_input = Input(shape=(1,), name='time_input')
            time_branch = Dense(8, activation='relu')(time_input)
            time_branch = BatchNormalization()(time_branch)
            
            # Initialize inputs and combined branches
            inputs = [pca_input, amount_input, time_input]
            branches = [pca_branch, amount_branch, time_branch]
            
            # Derived features branch (if any)
            if derived_dim > 0:
                derived_input = Input(shape=(derived_dim,), name='derived_input')
                derived_branch = Dense(max(16, derived_dim), activation='relu')(derived_input)
                derived_branch = BatchNormalization()(derived_branch)
                derived_branch = Dropout(0.3)(derived_branch)
                derived_branch = Dense(min(16, derived_dim), activation='relu')(derived_branch)
                
                inputs.append(derived_input)
                branches.append(derived_branch)
            
            # Combine all branches
            combined = concatenate(branches)
            
            # Fully connected layers for combined features
            x = Dense(64, activation='relu')(combined)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(16, activation='relu')(x)
            
            # Output layer - fraud probability
            output = Dense(1, activation='sigmoid', name='output')(x)
            
            # Create model with multiple inputs
            model = Model(
                inputs=inputs,
                outputs=output
            )
            
        else:  # 'simple' or any other string defaults to simple model
            # Simple sequential model with single input
            total_dims = pca_dim + 2 + derived_dim  # pca + amount + time + derived
            
            model = Sequential([
                # Input layer
                Dense(128, activation='relu', input_shape=(total_dims,)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Hidden layers
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(32, activation='relu'),
                Dropout(0.3),
                
                Dense(16, activation='relu'),
                
                # Output layer
                Dense(1, activation='sigmoid')
            ])
        
        # Compile model with fraud-detection-appropriate metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(name='pr_auc', curve='PR'),  # Precision-Recall AUC
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def fit(
        self, 
        df, 
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        patience=10,
        sampling_method='smote',
        sampling_ratio=0.1,
        model_type='multiinput',
        class_weight=None,
        use_derived_features=True  # New parameter
    ):
        """
        Fit the model to the training data.
        
        Args:
            df: DataFrame with features and target
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            patience: Patience for early stopping
            sampling_method: Method to handle class imbalance ('smote', 'none')
            sampling_ratio: Ratio of minority to majority class after resampling
            model_type: Model architecture to use
            class_weight: Class weights for imbalanced training
            use_derived_features: Whether to create and use derived features
            
        Returns:
            Training history
        """
        start_time = time.time()
        
        # Create directory for model checkpoints if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Apply feature engineering if requested
        if use_derived_features:
            logger.info("Adding derived features...")
            df = self.add_derived_features(df)
        
        # Preprocess data with the updated function
        X_pca, X_amount, X_time, X_derived, y = self.preprocess_data(df, fit_scalers=True)
        
        if y is None:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Check class distribution
        class_counts = np.bincount(y.astype(int))
        minority_pct = class_counts.min() / class_counts.sum() * 100
        logger.info(f"Class distribution before sampling: {class_counts}")
        logger.info(f"Minority class percentage: {minority_pct:.2f}%")
        
        # Apply sampling if specified (using updated function)
        X_pca, X_amount, X_time, X_derived, y = self.apply_sampling(
            X_pca, X_amount, X_time, X_derived, y, 
            sampling_method=sampling_method,
            sampling_ratio=sampling_ratio
        )
        
        # After sampling class distribution
        resampled_counts = np.bincount(y.astype(int))
        logger.info(f"Class distribution after sampling: {resampled_counts}")
        
        # Prepare train/validation split based on model type
        has_derived_features = X_derived.size > 0
        
        if model_type == 'multiinput':
            # For multi-input model, split each input separately
            # Initialize lists to hold split data components
            splits = [
                train_test_split(
                    X_pca, X_amount, X_time, y, 
                    test_size=validation_split, 
                    random_state=self.random_state,
                    stratify=y  # Stratify to maintain class distribution
                )
            ]
            
            # Unpack the main splits
            X_pca_train, X_pca_val, X_amount_train, X_amount_val, X_time_train, X_time_val, y_train, y_val = splits[0]
            
            # Handle derived features if present
            if has_derived_features:
                # Split derived features
                X_derived_train, X_derived_val = train_test_split(
                    X_derived, 
                    test_size=validation_split, 
                    random_state=self.random_state,
                    stratify=y  # Use same stratification
                )
                
                # Train and validation data for multi-input model with derived features
                train_data = [X_pca_train, X_amount_train, X_time_train, X_derived_train]
                val_data = [X_pca_val, X_amount_val, X_time_val, X_derived_val]
            else:
                # Train and validation data for multi-input model without derived features
                train_data = [X_pca_train, X_amount_train, X_time_train]
                val_data = [X_pca_val, X_amount_val, X_time_val]
                
        else:  # simple model
            # Combine features for simple model
            if has_derived_features:
                X_combined = np.hstack([X_pca, X_amount, X_time, X_derived])
            else:
                X_combined = np.hstack([X_pca, X_amount, X_time])
            
            # Split combined features
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, y,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=y
            )
            
            # Train and validation data for simple model
            train_data = X_train
            val_data = X_val
        
        # Build model if not already built
        if self.model is None:
            pca_dim = X_pca.shape[1]
            derived_dim = X_derived.shape[1] if has_derived_features else 0
            self.build_model(pca_dim=pca_dim, derived_dim=derived_dim, model_type=model_type)
        
        # Calculate class weights if not provided
        if class_weight is None and sampling_method.lower() == 'none':
            # Only use class weights if not using sampling
            n_samples = len(y)
            n_fraud = np.sum(y)
            n_normal = n_samples - n_fraud
            
            class_weight = {
                0: 1.0,
                1: n_normal / n_fraud if n_fraud > 0 else 1.0
            }
            logger.info(f"Using class weights: {class_weight}")
        
        # Set up callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_pr_auc', 
                mode='max',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.keras'),
                monitor='val_pr_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateauing
            ReduceLROnPlateau(
                monitor='val_pr_auc',
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info("Starting model training...")
        history = self.model.fit(
            train_data, y_train,
            validation_data=(val_data, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Optimize threshold on validation data
        self._optimize_threshold(val_data, y_val)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def _optimize_threshold(self, X_val, y_val):
        """
        Find the optimal threshold for classification based on F1 score.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        # Get predictions
        y_pred_prob = self.predict_proba(X_val)
        
        # Use the evaluation utility to find optimal threshold
        self.threshold = FraudEvaluator.find_optimal_threshold(y_val, y_pred_prob, criterion='f1')
        logger.info(f"Optimized threshold: {self.threshold:.4f}")
    
    def predict_proba(self, X):
        """
        Predict fraud probabilities.
        
        Args:
            X: Features (can be DataFrame or preprocessed arrays)
            
        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Process different input types
        if isinstance(X, pd.DataFrame):
            # Preprocess DataFrame with the updated method
            X_pca, X_amount, X_time, X_derived, _ = self.preprocess_data(X, fit_scalers=False)
            
            # Prepare input based on model architecture
            if len(self.model.inputs) > 1:  # Multi-input model
                if len(self.model.inputs) > 3 and X_derived.size > 0:
                    # Model expects derived features
                    model_input = [X_pca, X_amount, X_time, X_derived]
                else:
                    # Model without derived features
                    model_input = [X_pca, X_amount, X_time]
            else:  # Simple model
                # Combine all features
                if X_derived.size > 0:
                    model_input = np.hstack([X_pca, X_amount, X_time, X_derived])
                else:
                    model_input = np.hstack([X_pca, X_amount, X_time])
                    
        elif isinstance(X, list):
            # Already preprocessed multi-input data
            model_input = X
        else:
            # Assume already preprocessed simple model input
            model_input = X
        
        # Get model predictions
        return self.model.predict(model_input, verbose=0).flatten()
    
    def predict(self, X):
        """
        Predict fraud using the optimized threshold.
        
        Args:
            X: Features (can be DataFrame or preprocessed arrays)
            
        Returns:
            Binary predictions (0/1)
        """
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
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Extract features and target with the updated preprocessing function
        X_pca, X_amount, X_time, X_derived, y = self.preprocess_data(df, fit_scalers=False)
        
        if y is None:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Get predictions based on model type
        if len(self.model.inputs) > 1:  # Multi-input model
            # Check if model expects derived features
            if len(self.model.inputs) > 3 and X_derived.size > 0:
                # Model with derived features
                X = [X_pca, X_amount, X_time, X_derived]
            else:
                # Model without derived features
                X = [X_pca, X_amount, X_time]
        else:  # Simple model
            # Combine all features
            if X_derived.size > 0:
                X = np.hstack([X_pca, X_amount, X_time, X_derived])
            else:
                X = np.hstack([X_pca, X_amount, X_time])
                
        # Get probability predictions
        y_pred_prob = self.predict_proba(X)
        
        # Apply threshold for binary predictions
        y_pred = (y_pred_prob >= self.threshold).astype(int)
        
        # Use the evaluation utility to calculate metrics
        return FraudEvaluator.calculate_metrics(y, y_pred, y_pred_prob, self.threshold)
    
    def plot_evaluation(self, df, figsize=(20, 16)):
        """
        Plot evaluation metrics.
        
        Args:
            df: DataFrame with features and target
            figsize: Figure size
            
        Returns:
            Matplotlib figure with plots
        """
        # Extract features and target with updated preprocessing
        X_pca, X_amount, X_time, X_derived, y = self.preprocess_data(df, fit_scalers=False)
        
        if y is None:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Get predictions based on model type
        if len(self.model.inputs) > 1:  # Multi-input model
            # Check if model expects derived features
            if len(self.model.inputs) > 3 and X_derived.size > 0:
                # Model with derived features
                X = [X_pca, X_amount, X_time, X_derived]
            else:
                # Model without derived features
                X = [X_pca, X_amount, X_time]
        else:  # Simple model
            # Combine all features
            if X_derived.size > 0:
                X = np.hstack([X_pca, X_amount, X_time, X_derived])
            else:
                X = np.hstack([X_pca, X_amount, X_time])
                
        y_pred_prob = self.predict_proba(X)
        
        # Use the evaluation utility to create plots
        return FraudEvaluator.plot_metrics(y, y_pred_prob, self.threshold, figsize)
    
    def plot_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Keras training history
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation loss
        axs[0, 0].plot(history.history['loss'])
        axs[0, 0].plot(history.history['val_loss'])
        axs[0, 0].set_title('Model Loss')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation accuracy
        axs[0, 1].plot(history.history['accuracy'])
        axs[0, 1].plot(history.history['val_accuracy'])
        axs[0, 1].set_title('Model Accuracy')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['Train', 'Validation'], loc='lower right')
        
        # Plot training & validation ROC AUC
        axs[1, 0].plot(history.history['auc'])
        axs[1, 0].plot(history.history['val_auc'])
        axs[1, 0].set_title('ROC AUC')
        axs[1, 0].set_ylabel('AUC')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['Train', 'Validation'], loc='lower right')
        
        # Plot training & validation PR AUC
        axs[1, 1].plot(history.history['pr_auc'])
        axs[1, 1].plot(history.history['val_pr_auc'])
        axs[1, 1].set_title('PR AUC')
        axs[1, 1].set_ylabel('PR AUC')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend(['Train', 'Validation'], loc='lower right')
        
        plt.tight_layout()
        return fig
        
    def save_model(self, path=None):
        """
        Save the model and preprocessing components.
        
        Args:
            path: Directory path to save the model (defaults to self.model_path)
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        if path is None:
            path = self.model_path
            
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        self.model.save(os.path.join(path, 'model.keras'))
        
        # Save scalers and other components
        joblib.dump(self.pca_scaler, os.path.join(path, 'pca_scaler.joblib'))
        joblib.dump(self.amount_scaler, os.path.join(path, 'amount_scaler.joblib'))
        joblib.dump(self.time_scaler, os.path.join(path, 'time_scaler.joblib'))
        
        # Save derived feature scaler if it exists
        if hasattr(self, 'derived_scaler'):
            joblib.dump(self.derived_scaler, os.path.join(path, 'derived_scaler.joblib'))
        
        # Save threshold and configuration
        config = {
            'threshold': self.threshold,
            'time_column': self.time_column,
            'amount_column': self.amount_column,
            'target_column': self.target_column,
            'feature_prefix': self.feature_prefix,
            'pca_columns': self.pca_columns,
            'model_type': 'multiinput' if len(self.model.inputs) > 1 else 'simple',
            'has_derived_features': hasattr(self, 'derived_scaler')
        }
        
        joblib.dump(config, os.path.join(path, 'config.joblib'))
        logger.info(f"Model and components saved to {path}")
        
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model and its components.
        
        Args:
            path: Directory path where the model is saved
            
        Returns:
            DeepFraudDetector instance with loaded model
        """
        from tensorflow.keras.models import load_model
        
        # Load configuration
        config = joblib.load(os.path.join(path, 'config.joblib'))
        
        # Create a new instance with loaded configuration
        detector = cls(
            time_column=config['time_column'],
            amount_column=config['amount_column'],
            target_column=config['target_column'],
            feature_prefix=config['feature_prefix'],
            model_path=path
        )
        
        # Load model
        detector.model = load_model(os.path.join(path, 'model.keras'))
        
        # Load scalers
        detector.pca_scaler = joblib.load(os.path.join(path, 'pca_scaler.joblib'))
        detector.amount_scaler = joblib.load(os.path.join(path, 'amount_scaler.joblib'))
        detector.time_scaler = joblib.load(os.path.join(path, 'time_scaler.joblib'))
        
        # Load derived feature scaler if it exists
        if config.get('has_derived_features', False):
            try:
                detector.derived_scaler = joblib.load(os.path.join(path, 'derived_scaler.joblib'))
            except FileNotFoundError:
                logger.warning("Derived feature scaler not found despite configuration indicating its presence")
        
        # Load other attributes
        detector.threshold = config['threshold']
        detector.pca_columns = config['pca_columns']
        
        logger.info(f"Model and components loaded from {path}")
        return detector


# Example usage
if __name__ == "__main__":
    # Sample data with PCA features V1-V28, Amount, Time, and Class
    np.random.seed(42)
    n_samples = 10000
    n_features = 28  # V1 to V28
    
    # Create dataset with PCA features
    X = np.random.randn(n_samples, n_features)
    
    # Add Amount feature (transaction amount)
    amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
    
    # Add Time column (seconds elapsed since first transaction)
    time_col = np.arange(n_samples)
    
    # Create imbalanced target (1% fraud - Class column)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    # Create dataframe with proper column names matching the dataset
    feature_cols = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['Amount'] = amount
    df['Time'] = time_col
    df['Class'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {df['Class'].value_counts(normalize=True) * 100}")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    
    # Initialize the deep learning fraud detector
    detector = DeepFraudDetector(
        time_column='Time',
        amount_column='Amount',
        target_column='Class',
        feature_prefix='V'
    )
    
    # Build and train the model (with fewer epochs for demonstration)
    # Now using the new derived features
    history = detector.fit(
        train_df,
        validation_split=0.2,
        epochs=20,        # Reduced for example
        batch_size=256,
        patience=5,
        sampling_method='smote',
        sampling_ratio=0.1,
        use_derived_features=True  # Enable the derived features
    )
    
    # Plot training history
    detector.plot_history(history)
    
    # Evaluate on test data
    results = detector.evaluate(test_df)
    print("\nEvaluation results:")
    for metric, value in results.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # Plot evaluation metrics
    detector.plot_evaluation(test_df)
    
    # Save the model
    detector.save_model('deep_fraud_model')
    
    # Test loading the model
    loaded_detector = DeepFraudDetector.load_model('deep_fraud_model')
    
    # Make predictions with loaded model
    predictions = loaded_detector.predict(test_df)
    print(f"\nPredictions shape: {predictions.shape}")