# Deep Fraud Detector: In-Depth Technical Analysis

## 1. Neural Network Architecture for Fraud Detection

The `DeepFraudDetector` implements deep learning models specifically designed for credit card fraud detection. Unlike the GBM approach, this system leverages neural networks with specialized architectures to capture complex patterns in transaction data.

### Multi-Input Neural Network Architecture

```python
def build_model(self, pca_dim=28, derived_dim=0, model_type='multiinput'):
    if model_type == 'multiinput':
        # PCA features input branch
        pca_input = Input(shape=(pca_dim,), name='pca_input')
        pca_branch = Dense(64, activation='relu')(pca_input)
        pca_branch = BatchNormalization()(pca_branch)
        pca_branch = Dropout(0.3)(pca_branch)
        pca_branch = Dense(32, activation='relu')(pca_branch)
        
        # Amount input branch
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
            # ...additional layers...
            
            inputs.append(derived_input)
            branches.append(derived_branch)
        
        # Combine all branches
        combined = concatenate(branches)
        
        # Fully connected layers
        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=output)
```

**Technical Details:**

1. **Multi-Branch Design**: 
   - Separate processing paths for different feature types
   - Dedicated neural processing for PCA features, amount, time, and derived features
   - Feature-specific architectures of appropriate complexity

2. **Branch Structure**:
   - **PCA Features Branch**: Larger network (64→32 neurons) with dropout for regularization
   - **Amount Branch**: Smaller network (8 neurons) focused on this critical feature
   - **Time Branch**: Similar treatment to amount feature
   - **Derived Features Branch**: Flexible sizing based on number of derived features

3. **Architectural Components**:
   - **Dense Layers**: Fully connected layers for pattern recognition
   - **BatchNormalization**: Normalizes activations for faster training and stability
   - **Dropout**: Randomly disables neurons to prevent overfitting
   - **Concatenation**: Combines separate feature branches for joint processing

**Pros:**
- Allows different subnetworks to extract patterns from different feature types
- Can handle features of different scales and importance
- More flexible than traditional single-input networks
- Often performs better on tabular data with heterogeneous features

**Cons:**
- More complex to configure and tune
- Requires more hyperparameter choices
- Potentially more prone to overfitting if not properly regularized
- Harder to interpret than simpler models

### Simple Sequential Model Alternative

```python
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
```

**Technical Details:**
- Traditional feedforward network with sequential layers
- Larger initial layer (128 neurons) to process all features together
- Progressive narrowing of layers (128→64→32→16) towards output
- Similar regularization strategy with BatchNormalization and Dropout

**Pros:**
- Simpler implementation and conceptually easier to understand
- Fewer hyperparameters to tune
- Often more stable during training
- May generalize better with limited data

**Cons:**
- Treats all features uniformly, potentially missing feature-specific patterns
- Less flexible for heterogeneous feature handling
- May require more preprocessing to handle different feature scales
- Often lower performance ceiling compared to specialized architectures

### Model Compilation

```python
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
```

**Technical Details:**
- Uses Adam optimizer with conservative initial learning rate (0.001)
- Binary cross-entropy loss appropriate for binary classification
- Monitors multiple metrics during training, particularly PR-AUC
- Includes metrics specifically valuable for imbalanced classification

**Optimizer Characteristics:**
- **Adam**: Adaptive learning rate optimization algorithm
  - Combines benefits of AdaGrad and RMSProp
  - Adjusts learning rates individually for each parameter
  - Includes momentum for faster convergence
  - Well-suited for noisy gradients and non-stationary objectives

## 2. Feature Engineering for Fraud Detection

A key strength of this implementation is extensive domain-specific feature engineering:

```python
def add_derived_features(self, df):
    """
    Add domain-specific derived features that might help with fraud detection.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_new = df.copy()
    
    # == Time-based features ==
    
    # Transaction recency
    df_new['TransactionRecency'] = df_new[self.time_column].max() - df_new[self.time_column]
    
    # Time of day (assuming time column represents seconds, normalize to 24h cycle)
    seconds_in_day = 24 * 60 * 60
    df_new['TimeOfDay'] = df_new[self.time_column] % seconds_in_day
    
    # == Amount-based features ==
    
    # Amount buckets
    try:
        df_new['AmountBucket'] = pd.qcut(
            df_new[self.amount_column], 
            q=10, 
            labels=False, 
            duplicates='drop'
        )
    except ValueError as e:
        # Fall back to equal-width bins if quantile binning fails
        df_new['AmountBucket'] = pd.cut(
            df_new[self.amount_column], 
            bins=10, 
            labels=False
        )
    
    # Log transformation of amount
    df_new['LogAmount'] = np.log1p(df_new[self.amount_column])
    
    # Unusual amount flag
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
    if self.pca_columns or [col for col in df.columns if col.startswith(self.feature_prefix)]:
        # Get PCA columns if not already set
        if not self.pca_columns:
            self.pca_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
            
        # Magnitude of PCA vector
        df_new['PCA_Magnitude'] = np.sqrt(
            np.sum([df_new[col]**2 for col in self.pca_columns], axis=0)
        )
        
        # Count of extreme PCA values
        df_new['ExtremeFeatureCount'] = np.sum(
            [(df_new[col].abs() > 3) for col in self.pca_columns], 
            axis=0
        )
        
        # PCA skewness
        def calc_skew(row):
            values = [row[col] for col in self.pca_columns]
            if len(set(values)) <= 1:  # If all values are the same
                return 0
            return skew(values)
        
        try:
            df_new['PCA_Skewness'] = df_new.apply(calc_skew, axis=1)
        except Exception as e:
            logger.warning(f"Could not calculate PCA_Skewness: {e}")
            
    return df_new
```

**Types of Derived Features:**

1. **Time-Based Features**:
   - **TransactionRecency**: How recent the transaction is compared to latest
   - **TimeOfDay**: Time within 24-hour cycle, capturing diurnal patterns
   - [Comment] Weekend flag noted but commented out due to timestamp conversion requirements

2. **Amount-Based Features**:
   - **AmountBucket**: Discretization of amounts into bins (handles non-linear relationships)
   - **LogAmount**: Log transformation to handle skewed amount distributions
   - **AmountZScore**: Standardized amount values
   - **UnusualAmount**: Flag for statistically anomalous amounts (z-score > 3)
   - **IsRoundAmount**: Flag for suspiciously round amounts (multiples of 10, 100, 1000)

3. **PCA Feature Derivatives**:
   - **PCA_Magnitude**: Euclidean norm of the PCA feature vector (overall deviation)
   - **ExtremeFeatureCount**: Number of PCA features with extreme values
   - **PCA_Skewness**: Asymmetry in the PCA feature distribution

**Value of Feature Engineering in Fraud Detection:**

1. **Domain Knowledge Integration**:
   - Incorporates fraud detection heuristics (e.g., round amounts are suspicious)
   - Creates features based on known fraud patterns
   - Transforms raw data into more informative representations

2. **Statistical Signal Enhancement**:
   - Highlights unusual transaction characteristics
   - Creates meta-features that summarize patterns across many raw features
   - Transforms non-linear relationships into more learnable forms

3. **Performance Impact**:
   - Often more important than model architecture for tabular data
   - Can capture patterns neural networks might struggle to learn directly
   - Makes the learning problem easier by explicit feature creation

**Technical Considerations:**

- **Robust Implementation**: Uses try/except blocks for error handling
- **Performance Optimization**: Vectorized operations where possible
- **Interpretability**: Creates meaningful features with business interpretation
- **Flexibility**: Works with variable datasets through dynamic column detection

## 3. Data Preprocessing and Scaling

The code implements a sophisticated preprocessing pipeline that handles different feature types appropriately:

```python
def preprocess_data(self, df, fit_scalers=True):
    """
    Preprocess the dataframe by splitting features and scaling.
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
```

**Technical Details:**

1. **Feature-Specific Scaling**:
   - **StandardScaler** for PCA features and time
   - **RobustScaler** for amount (handle outliers better)
   - **Separate scaler** for derived features

2. **Preprocessing Architecture**:
   - Features kept separate for multi-input model
   - Maintains feature groups for specialized processing
   - Handles both training (fit_transform) and inference (transform) modes

3. **Dynamic Feature Detection**:
   - Automatically identifies PCA features by prefix
   - Detects derived features by exclusion
   - Adapts to variable input data structures

**Scaling Methods in Detail:**

1. **Standard Scaling (PCA Features)**:
   - Subtracts mean and divides by standard deviation: (x - μ) / σ
   - Results in features with zero mean and unit variance
   - Appropriate for normally distributed features (like PCA components)
   - Makes features comparable in magnitude

2. **Robust Scaling (Amount)**:
   - Uses median and interquartile range: (x - median) / IQR
   - Less influenced by outliers than standard scaling
   - Better for financial data with skewed distributions
   - More appropriate for amount which often has extreme values

3. **Why Different Scalers Matter**:
   - Neural networks are sensitive to feature scales
   - Gradient-based learning works best with well-scaled features
   - Different feature distributions require different scaling approaches
   - Improper scaling can lead to poor convergence or suboptimal solutions

## 4. Handling Imbalanced Data

Fraud detection typically faces severe class imbalance. The code implements SMOTE (Synthetic Minority Over-sampling Technique) to address this:

```python
def apply_sampling(self, X_pca, X_amount, X_time, X_derived, y, sampling_method='smote', sampling_ratio=0.1):
    """
    Apply sampling methods to handle class imbalance.
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
```

**SMOTE Technical Implementation:**

1. **Feature Combination**:
   - Temporarily combines separate feature groups
   - Required because SMOTE operates on a single feature matrix
   - Preserves all feature information during resampling

2. **SMOTE Process**:
   - Creates synthetic fraud examples by interpolating between existing fraud cases
   - Increases minority class proportion to specified ratio (default 0.1 or 10%)
   - Uses k-nearest neighbors to find similar fraud examples for interpolation

3. **Feature Reconstruction**:
   - After resampling, splits combined features back into original groups
   - Preserves multi-input structure needed by the model
   - Carefully handles dimensionality and reshaping

**SMOTE Parameters:**

- **sampling_ratio**: Controls the desired ratio of minority to majority class
  - Default 0.1 means 1 fraud example for every 10 legitimate transactions
  - Higher values create more fraud examples (more balanced)
  - Lower values create fewer fraud examples (less balanced)

- **random_state**: Ensures reproducibility of synthetic examples

**Alternative Approaches in the Code:**

1. **Class Weights**:
   ```python
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
   ```
   
   - Weights loss function to penalize errors on minority class more heavily
   - Used as an alternative when SMOTE is not applied
   - Automatically calculated based on class distribution
   - Preserves original data distribution while addressing imbalance

2. **Threshold Optimization**:
   ```python
   # Optimize threshold on validation data
   self._optimize_threshold(val_data, y_val)
   ```
   
   - Adjusts decision boundary after training
   - Finds optimal threshold based on F1 score
   - Crucial for imbalanced datasets where default 0.5 is rarely optimal

## 5. Training and Optimization System

The model training implementation includes several advanced techniques for optimal performance:

```python
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
    use_derived_features=True
):
    # ...
    
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
    history = self.model.fit(
        train_data, y_train,
        validation_data=(val_data, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
```

**Training Optimization Techniques:**

1. **Callback System**:
   
   a. **Early Stopping**:
   - Monitors validation PR-AUC to detect overfitting
   - Stops training when performance plateaus or degrades
   - Restores best weights automatically
   - Patience parameter allows for temporary performance fluctuations
   
   b. **Model Checkpoint**:
   - Saves the best model based on validation PR-AUC
   - Ensures optimal model is preserved regardless of when it occurs
   - Critical for long training runs
   
   c. **Learning Rate Reduction**:
   - Reduces learning rate when performance plateaus
   - Allows for fine-tuning as training progresses
   - Factor of 0.5 halves the learning rate
   - Minimum learning rate prevents excessive reduction

2. **Training Configuration**:
   
   a. **Batch Size Selection (256)**:
   - Balance between:
     - Training speed (larger batches are faster)
     - Generalization (smaller batches can generalize better)
     - Memory usage (larger batches use more memory)
   - Suitable size for fraud detection with moderate dataset sizes
   
   b. **Epochs and Patience**:
   - Maximum epochs (100) provides upper bound on training duration
   - Patience (10) allows for performance plateaus without premature stopping
   - Combination ensures training continues as long as beneficial

3. **Monitoring Strategy**:
   - **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
     - Most appropriate for imbalanced classification
     - Focuses on model's ability to detect fraud
     - Less influenced by large number of true negatives
   - **Mode**: Maximize (higher values are better)
   - **Verbose Logging**: Provides visibility into training progress

4. **Stratified Validation Split**:
   ```python
   X_pca_train, X_pca_val, X_amount_train, X_amount_val, X_time_train, X_time_val, y_train, y_val = train_test_split(
       X_pca, X_amount, X_time, y, 
       test_size=validation_split, 
       random_state=self.random_state,
       stratify=y  # Stratify to maintain class distribution
   )
   ```
   - Ensures validation set has same class distribution as training
   - Critical for imbalanced datasets to prevent validation instability
   - Maintains consistent evaluation throughout training

## 6. Evaluation and Visualization

The code includes comprehensive evaluation and visualization capabilities:

```python
def plot_history(self, history):
    """
    Plot training history.
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
```

**Evaluation and Visualization Components:**

1. **Training History Visualization**:
   - **Loss Curves**: Shows convergence and potential overfitting
   - **Accuracy Tracking**: General classification performance
   - **ROC AUC**: Discrimination ability across thresholds
   - **PR AUC**: Fraud detection performance focus

2. **Model Evaluation**:
   ```python
   def evaluate(self, df):
       # ...
       # Get probability predictions
       y_pred_prob = self.predict_proba(X)
       
       # Apply threshold for binary predictions
       y_pred = (y_pred_prob >= self.threshold).astype(int)
       
       # Use the evaluation utility to calculate metrics
       return FraudEvaluator.calculate_metrics(y, y_pred, y_pred_prob, self.threshold)
   ```
   - Leverages the FraudEvaluator from previous code
   - Provides consistent metrics across model types
   - Applies optimized threshold for better performance

3. **Performance Visualization**:
   ```python
   def plot_evaluation(self, df, figsize=(20, 16)):
       # ...
       # Use the evaluation utility to create plots
       return FraudEvaluator.plot_metrics(y, y_pred_prob, self.threshold, figsize)
   ```
   - Comprehensive visualization of model performance
   - Reuses FraudEvaluator visualization code
   - Ensures consistent evaluation across model types

**Key Metrics for Neural Network Fraud Detection:**

1. **PR-AUC (Precision-Recall Area Under Curve)**:
   - Primary metric for imbalanced classification
   - Focuses on trade-off between precision and recall
   - Less influenced by large number of true negatives
   - More representative of fraud detection performance

2. **Loss Tracking**:
   - Binary cross-entropy loss monitors training progress
   - Provides gradient information for optimization
   - Can reveal overfitting when validation loss increases

3. **Complementary Metrics**:
   - Accuracy (though less informative for imbalanced data)
   - ROC-AUC (standard discrimination ability metric)
   - Precision (proportion of flagged transactions that are fraudulent)
   - Recall (proportion of fraud that is successfully detected)

## 7. Model Persistence and Deployment

The code implements robust model saving and loading capabilities:

```python
def save_model(self, path=None):
    """
    Save the model and preprocessing components.
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
```

**Model Persistence Components:**

1. **Complete Ecosystem Serialization**:
   - **Neural Network Model**: Saved in Keras format
   - **Preprocessing Components**: All scalers saved with joblib
   - **Configuration**: Parameters and settings saved for reproducibility
   - **Optimized Threshold**: Decision boundary preserved

2. **Modular Storage**:
   - Each component saved as separate file
   - Allows for selective updating if needed
   - Better version control and diff tracking

3. **Configuration Management**:
   - Stores all parameters needed for reconstruction
   - Includes derived model properties (like model type)
   - Records presence of optional components

**Complementary Loading Function:**

```python
@classmethod
def load_model(cls, path):
    """
    Load a saved model and its components.
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
```

## 8. Prediction Pipeline

The prediction functionality handles both raw data and preprocessed inputs:

```python
def predict_proba(self, X):
    """
    Predict fraud probabilities.
    """
    if self.model is None:
        raise ValueError("Model is not fitted. Call fit() first.")
    
    # Process different input types
    if isinstance(X, pd.DataFrame):
        # Preprocess DataFrame
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
```

**Technical Details:**

1. **Input Flexibility**:
   - Handles raw DataFrames, requiring preprocessing
   - Accepts pre-processed feature arrays
   - Automatically detects input type

2. **Model-Aware Processing**:
   - Adapts to model architecture (multi-input vs. simple)
   - Prepares appropriate input structure for each model type
   - Handles presence or absence of derived features

3. **Threshold Application**:
   ```python
   def predict(self, X):
       """
       Predict fraud using the optimized threshold.
       """
       probas = self.predict_proba(X)
       return (probas >= self.threshold).astype(int)
   ```
   - Converts probabilities to binary predictions
   - Uses optimized threshold instead of default 0.5
   - Returns integer class labels (0/1)

**Prediction Process Flow:**

1. Input validation and type detection
2. Preprocessing if needed (scaling, feature derivation)
3. Format adaptation for model architecture
4. Raw probability prediction
5. Threshold application for binary decisions

## 9. Comparison with GBM Approach

The DeepFraudDetector represents a different paradigm from the GBMFraudDetector. Here's how they compare:

### Architecture Differences

**Neural Network Approach (DeepFraudDetector):**
- Multi-branch neural architecture with specialized processing paths
- Deep representation learning through multiple layers
- BatchNormalization and Dropout for regularization
- Concatenation of feature branches for joint processing

**Gradient Boosting Approach (GBMFraudDetector):**
- Ensembles of decision trees built sequentially
- Automatic feature interaction detection
- Multiple GBM implementations with different strengths
- Tree-based learning without explicit feature combinations

### Feature Engineering

**Neural Network Approach:**
- Extensive domain-specific derived features
- Explicit feature creation to help neural networks
- Time and amount transformations
- PCA feature aggregations

**Gradient Boosting Approach:**
- Less emphasis on feature engineering
- GBMs naturally handle interactions and non-linearities
- Relies more on algorithm's ability to discover patterns
- Feature importance extraction for interpretability

### Handling Imbalance

**Neural Network Approach:**
- SMOTE for synthetic minority examples
- Class weighting as alternative
- Threshold optimization
- Custom metrics during training (PR-AUC focus)

**Gradient Boosting Approach:**
- Multiple sampling strategies (SMOTE, undersampling)
- Scale position weight parameter
- Threshold optimization
- Cross-validation with stratification

### Training Optimization

**Neural Network Approach:**
- Adam optimizer with learning rate scheduling
- Early stopping based on validation PR-AUC
- Batch size tuning
- Callbacks for checkpoint saving

**Gradient Boosting Approach:**
- Optuna for hyperparameter optimization
- Tree-specific parameter tuning
- Multiple implementation comparison
- Explicit cross-validation

### Relative Strengths

**Neural Networks Excel At:**
- Learning complex non-linear patterns
- Handling heterogeneous feature types through specialized branches
- Benefiting from explicit feature engineering
- Scaling to very large datasets

**Gradient Boosting Excels At:**
- Better performance with limited data
- Automatic feature interaction detection
- Higher interpretability
- Often requiring less hyperparameter tuning

## 10. Technical Implementation Considerations

### Memory Management

Neural networks can be memory-intensive. The code includes several strategies to manage this:

1. **Batch Processing**:
   - Uses batched training (batch_size=256)
   - Processes data in chunks rather than all at once

2. **Model Size Control**:
   - Modest layer sizes (max 128 neurons)
   - Progressive dimension reduction (128→64→32→16)
   - Separate branches only where beneficial

### Computational Efficiency

Training deep learning models can be computationally expensive:

1. **Early Stopping**:
   - Prevents unnecessary training epochs
   - Monitors validation performance to stop when beneficial

2. **Learning Rate Scheduling**:
   - ReduceLROnPlateau reduces learning rate when progress stalls
   - Allows faster initial learning with refinement later

3. **Efficient Data Preprocessing**:
   - Vectorized operations where possible
   - Avoids redundant calculations

### Error Handling and Logging

The code includes robust error handling:

1. **Graceful Fallbacks**:
   ```python
   try:
       df_new['AmountBucket'] = pd.qcut(df_new[self.amount_column], q=10, labels=False, duplicates='drop')
   except ValueError as e:
       logger.warning(f"Could not create AmountBucket: {e}")
       # Fall back to equal-width bins if quantile binning fails
       df_new['AmountBucket'] = pd.cut(df_new[self.amount_column], bins=10, labels=False)
   ```
   - Handles expected exceptions with fallback strategies
   - Continues processing when possible

2. **Comprehensive Logging**:
   - Records progress, warnings, and errors
   - Provides timing information
   - Documents key decisions and parameters

3. **Input Validation**:
   - Checks for required columns
   - Validates model state before operations
   - Provides clear error messages

## Conclusion

The DeepFraudDetector represents a sophisticated deep learning approach to fraud detection with several key strengths:

1. **Specialized Architecture**: Multi-branch neural network designed specifically for heterogeneous fraud data.

2. **Domain-Specific Features**: Extensive feature engineering incorporating fraud detection domain knowledge.

3. **Imbalance Handling**: Multiple strategies to address the severe class imbalance in fraud detection.

4. **Comprehensive Evaluation**: Focused on metrics appropriate for imbalanced classification.

5. **Flexible Implementation**: Supports different model architectures, feature sets, and sampling approaches.

The implementation demonstrates how deep learning can be effectively applied to tabular data problems like fraud detection, particularly when combined with domain knowledge and appropriate feature engineering.
# Deep Fraud Detector: In-Depth Technical Analysis

## 1. Neural Network Architecture for Fraud Detection

The `DeepFraudDetector` implements deep learning models specifically designed for credit card fraud detection. Unlike the GBM approach, this system leverages neural networks with specialized architectures to capture complex patterns in transaction data.

### Multi-Input Neural Network Architecture

```python
def build_model(self, pca_dim=28, derived_dim=0, model_type='multiinput'):
    if model_type == 'multiinput':
        # PCA features input branch
        pca_input = Input(shape=(pca_dim,), name='pca_input')
        pca_branch = Dense(64, activation='relu')(pca_input)
        pca_branch = BatchNormalization()(pca_branch)
        pca_branch = Dropout(0.3)(pca_branch)
        pca_branch = Dense(32, activation='relu')(pca_branch)
        
        # Amount input branch
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
            # ...additional layers...
            
            inputs.append(derived_input)
            branches.append(derived_branch)
        
        # Combine all branches
        combined = concatenate(branches)
        
        # Fully connected layers
        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=output)
```

**Technical Details:**

1. **Multi-Branch Design**: 
   - Separate processing paths for different feature types
   - Dedicated neural processing for PCA features, amount, time, and derived features
   - Feature-specific architectures of appropriate complexity

2. **Branch Structure**:
   - **PCA Features Branch**: Larger network (64→32 neurons) with dropout for regularization
   - **Amount Branch**: Smaller network (8 neurons) focused on this critical feature
   - **Time Branch**: Similar treatment to amount feature
   - **Derived Features Branch**: Flexible sizing based on number of derived features

3. **Architectural Components**:
   - **Dense Layers**: Fully connected layers for pattern recognition
   - **BatchNormalization**: Normalizes activations for faster training and stability
   - **Dropout**: Randomly disables neurons to prevent overfitting
   - **Concatenation**: Combines separate feature branches for joint processing

**Pros:**
- Allows different subnetworks to extract patterns from different feature types
- Can handle features of different scales and importance
- More flexible than traditional single-input networks
- Often performs better on tabular data with heterogeneous features

**Cons:**
- More complex to configure and tune
- Requires more hyperparameter choices
- Potentially more prone to overfitting if not properly regularized
- Harder to interpret than simpler models

### Simple Sequential Model Alternative

```python
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
```

**Technical Details:**
- Traditional feedforward network with sequential layers
- Larger initial layer (128 neurons) to process all features together
- Progressive narrowing of layers (128→64→32→16) towards output
- Similar regularization strategy with BatchNormalization and Dropout

**Pros:**
- Simpler implementation and conceptually easier to understand
- Fewer hyperparameters to tune
- Often more stable during training
- May generalize better with limited data

**Cons:**
- Treats all features uniformly, potentially missing feature-specific patterns
- Less flexible for heterogeneous feature handling
- May require more preprocessing to handle different feature scales
- Often lower performance ceiling compared to specialized architectures

### Model Compilation

```python
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
```

**Technical Details:**
- Uses Adam optimizer with conservative initial learning rate (0.001)
- Binary cross-entropy loss appropriate for binary classification
- Monitors multiple metrics during training, particularly PR-AUC
- Includes metrics specifically valuable for imbalanced classification

**Optimizer Characteristics:**
- **Adam**: Adaptive learning rate optimization algorithm
  - Combines benefits of AdaGrad and RMSProp
  - Adjusts learning rates individually for each parameter
  - Includes momentum for faster convergence
  - Well-suited for noisy gradients and non-stationary objectives

## 2. Feature Engineering for Fraud Detection

A key strength of this implementation is extensive domain-specific feature engineering:

```python
def add_derived_features(self, df):
    """
    Add domain-specific derived features that might help with fraud detection.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_new = df.copy()
    
    # == Time-based features ==
    
    # Transaction recency
    df_new['TransactionRecency'] = df_new[self.time_column].max() - df_new[self.time_column]
    
    # Time of day (assuming time column represents seconds, normalize to 24h cycle)
    seconds_in_day = 24 * 60 * 60
    df_new['TimeOfDay'] = df_new[self.time_column] % seconds_in_day
    
    # == Amount-based features ==
    
    # Amount buckets
    try:
        df_new['AmountBucket'] = pd.qcut(
            df_new[self.amount_column], 
            q=10, 
            labels=False, 
            duplicates='drop'
        )
    except ValueError as e:
        # Fall back to equal-width bins if quantile binning fails
        df_new['AmountBucket'] = pd.cut(
            df_new[self.amount_column], 
            bins=10, 
            labels=False
        )
    
    # Log transformation of amount
    df_new['LogAmount'] = np.log1p(df_new[self.amount_column])
    
    # Unusual amount flag
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
    if self.pca_columns or [col for col in df.columns if col.startswith(self.feature_prefix)]:
        # Get PCA columns if not already set
        if not self.pca_columns:
            self.pca_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
            
        # Magnitude of PCA vector
        df_new['PCA_Magnitude'] = np.sqrt(
            np.sum([df_new[col]**2 for col in self.pca_columns], axis=0)
        )
        
        # Count of extreme PCA values
        df_new['ExtremeFeatureCount'] = np.sum(
            [(df_new[col].abs() > 3) for col in self.pca_columns], 
            axis=0
        )
        
        # PCA skewness
        def calc_skew(row):
            values = [row[col] for col in self.pca_columns]
            if len(set(values)) <= 1:  # If all values are the same
                return 0
            return skew(values)
        
        try:
            df_new['PCA_Skewness'] = df_new.apply(calc_skew, axis=1)
        except Exception as e:
            logger.warning(f"Could not calculate PCA_Skewness: {e}")
            
    return df_new
```

**Types of Derived Features:**

1. **Time-Based Features**:
   - **TransactionRecency**: How recent the transaction is compared to latest
   - **TimeOfDay**: Time within 24-hour cycle, capturing diurnal patterns
   - [Comment] Weekend flag noted but commented out due to timestamp conversion requirements

2. **Amount-Based Features**:
   - **AmountBucket**: Discretization of amounts into bins (handles non-linear relationships)
   - **LogAmount**: Log transformation to handle skewed amount distributions
   - **AmountZScore**: Standardized amount values
   - **UnusualAmount**: Flag for statistically anomalous amounts (z-score > 3)
   - **IsRoundAmount**: Flag for suspiciously round amounts (multiples of 10, 100, 1000)

3. **PCA Feature Derivatives**:
   - **PCA_Magnitude**: Euclidean norm of the PCA feature vector (overall deviation)
   - **ExtremeFeatureCount**: Number of PCA features with extreme values
   - **PCA_Skewness**: Asymmetry in the PCA feature distribution

**Value of Feature Engineering in Fraud Detection:**

1. **Domain Knowledge Integration**:
   - Incorporates fraud detection heuristics (e.g., round amounts are suspicious)
   - Creates features based on known fraud patterns
   - Transforms raw data into more informative representations

2. **Statistical Signal Enhancement**:
   - Highlights unusual transaction characteristics
   - Creates meta-features that summarize patterns across many raw features
   - Transforms non-linear relationships into more learnable forms

3. **Performance Impact**:
   - Often more important than model architecture for tabular data
   - Can capture patterns neural networks might struggle to learn directly
   - Makes the learning problem easier by explicit feature creation

**Technical Considerations:**

- **Robust Implementation**: Uses try/except blocks for error handling
- **Performance Optimization**: Vectorized operations where possible
- **Interpretability**: Creates meaningful features with business interpretation
- **Flexibility**: Works with variable datasets through dynamic column detection

## 3. Data Preprocessing and Scaling

The code implements a sophisticated preprocessing pipeline that handles different feature types appropriately:

```python
def preprocess_data(self, df, fit_scalers=True):
    """
    Preprocess the dataframe by splitting features and scaling.
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
```

**Technical Details:**

1. **Feature-Specific Scaling**:
   - **StandardScaler** for PCA features and time
   - **RobustScaler** for amount (handle outliers better)
   - **Separate scaler** for derived features

2. **Preprocessing Architecture**:
   - Features kept separate for multi-input model
   - Maintains feature groups for specialized processing
   - Handles both training (fit_transform) and inference (transform) modes

3. **Dynamic Feature Detection**:
   - Automatically identifies PCA features by prefix
   - Detects derived features by exclusion
   - Adapts to variable input data structures

**Scaling Methods in Detail:**

1. **Standard Scaling (PCA Features)**:
   - Subtracts mean and divides by standard deviation: (x - μ) / σ
   - Results in features with zero mean and unit variance
   - Appropriate for normally distributed features (like PCA components)
   - Makes features comparable in magnitude

2. **Robust Scaling (Amount)**:
   - Uses median and interquartile range: (x - median) / IQR
   - Less influenced by outliers than standard scaling
   - Better for financial data with skewed distributions
   - More appropriate for amount which often has extreme values

3. **Why Different Scalers Matter**:
   - Neural networks are sensitive to feature scales
   - Gradient-based learning works best with well-scaled features
   - Different feature distributions require different scaling approaches
   - Improper scaling can lead to poor convergence or suboptimal solutions

## 4. Handling Imbalanced Data

Fraud detection typically faces severe class imbalance. The code implements SMOTE (Synthetic Minority Over-sampling Technique) to address this:

```python
def apply_sampling(self, X_pca, X_amount, X_time, X_derived, y, sampling_method='smote', sampling_ratio=0.1):
    """
    Apply sampling methods to handle class imbalance.
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
```

**SMOTE Technical Implementation:**

1. **Feature Combination**:
   - Temporarily combines separate feature groups
   - Required because SMOTE operates on a single feature matrix
   - Preserves all feature information during resampling

2. **SMOTE Process**:
   - Creates synthetic fraud examples by interpolating between existing fraud cases
   - Increases minority class proportion to specified ratio (default 0.1 or 10%)
   - Uses k-nearest neighbors to find similar fraud examples for interpolation

3. **Feature Reconstruction**:
   - After resampling, splits combined features back into original groups
   - Preserves multi-input structure needed by the model
   - Carefully handles dimensionality and reshaping

**SMOTE Parameters:**

- **sampling_ratio**: Controls the desired ratio of minority to majority class
  - Default 0.1 means 1 fraud example for every 10 legitimate transactions
  - Higher values create more fraud examples (more balanced)
  - Lower values create fewer fraud examples (less balanced)

- **random_state**: Ensures reproducibility of synthetic examples

**Alternative Approaches in the Code:**

1. **Class Weights**:
   ```python
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
   ```
   
   - Weights loss function to penalize errors on minority class more heavily
   - Used as an alternative when SMOTE is not applied
   - Automatically calculated based on class distribution
   - Preserves original data distribution while addressing imbalance

2. **Threshold Optimization**:
   ```python
   # Optimize threshold on validation data
   self._optimize_threshold(val_data, y_val)
   ```
   
   - Adjusts decision boundary after training
   - Finds optimal threshold based on F1 score
   - Crucial for imbalanced datasets where default 0.5 is rarely optimal

## 5. Training and Optimization System

The model training implementation includes several advanced techniques for optimal performance:

```python
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
    use_derived_features=True
):
    # ...
    
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
    history = self.model.fit(
        train_data, y_train,
        validation_data=(val_data, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
```

**Training Optimization Techniques:**

1. **Callback System**:
   
   a. **Early Stopping**:
   - Monitors validation PR-AUC to detect overfitting
   - Stops training when performance plateaus or degrades
   - Restores best weights automatically
   - Patience parameter allows for temporary performance fluctuations
   
   b. **Model Checkpoint**:
   - Saves the best model based on validation PR-AUC
   - Ensures optimal model is preserved regardless of when it occurs
   - Critical for long training runs
   
   c. **Learning Rate Reduction**:
   - Reduces learning rate when performance plateaus
   - Allows for fine-tuning as training progresses
   - Factor of 0.5 halves the learning rate
   - Minimum learning rate prevents excessive reduction

2. **Training Configuration**:
   
   a. **Batch Size Selection (256)**:
   - Balance between:
     - Training speed (larger batches are faster)
     - Generalization (smaller batches can generalize better)
     - Memory usage (larger batches use more memory)
   - Suitable size for fraud detection with moderate dataset sizes
   
   b. **Epochs and Patience**:
   - Maximum epochs (100) provides upper bound on training duration
   - Patience (10) allows for performance plateaus without premature stopping
   - Combination ensures training continues as long as beneficial

3. **Monitoring Strategy**:
   - **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
     - Most appropriate for imbalanced classification
     - Focuses on model's ability to detect fraud
     - Less influenced by large number of true negatives
   - **Mode**: Maximize (higher values are better)
   - **Verbose Logging**: Provides visibility into training progress

4. **Stratified Validation Split**:
   ```python
   X_pca_train, X_pca_val, X_amount_train, X_amount_val, X_time_train, X_time_val, y_train, y_val = train_test_split(
       X_pca, X_amount, X_time, y, 
       test_size=validation_split, 
       random_state=self.random_state,
       stratify=y  # Stratify to maintain class distribution
   )
   ```
   - Ensures validation set has same class distribution as training
   - Critical for imbalanced datasets to prevent validation instability
   - Maintains consistent evaluation throughout training

## 6. Evaluation and Visualization

The code includes comprehensive evaluation and visualization capabilities:

```python
def plot_history(self, history):
    """
    Plot training history.
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
```

**Evaluation and Visualization Components:**

1. **Training History Visualization**:
   - **Loss Curves**: Shows convergence and potential overfitting
   - **Accuracy Tracking**: General classification performance
   - **ROC AUC**: Discrimination ability across thresholds
   - **PR AUC**: Fraud detection performance focus

2. **Model Evaluation**:
   ```python
   def evaluate(self, df):
       # ...
       # Get probability predictions
       y_pred_prob = self.predict_proba(X)
       
       # Apply threshold for binary predictions
       y_pred = (y_pred_prob >= self.threshold).astype(int)
       
       # Use the evaluation utility to calculate metrics
       return FraudEvaluator.calculate_metrics(y, y_pred, y_pred_prob, self.threshold)
   ```
   - Leverages the FraudEvaluator from previous code
   - Provides consistent metrics across model types
   - Applies optimized threshold for better performance

3. **Performance Visualization**:
   ```python
   def plot_evaluation(self, df, figsize=(20, 16)):
       # ...
       # Use the evaluation utility to create plots
       return FraudEvaluator.plot_metrics(y, y_pred_prob, self.threshold, figsize)
   ```
   - Comprehensive visualization of model performance
   - Reuses FraudEvaluator visualization code
   - Ensures consistent evaluation across model types

**Key Metrics for Neural Network Fraud Detection:**

1. **PR-AUC (Precision-Recall Area Under Curve)**:
   - Primary metric for imbalanced classification
   - Focuses on trade-off between precision and recall
   - Less influenced by large number of true negatives
   - More representative of fraud detection performance

2. **Loss Tracking**:
   - Binary cross-entropy loss monitors training progress
   - Provides gradient information for optimization
   - Can reveal overfitting when validation loss increases

3. **Complementary Metrics**:
   - Accuracy (though less informative for imbalanced data)
   - ROC-AUC (standard discrimination ability metric)
   - Precision (proportion of flagged transactions that are fraudulent)
   - Recall (proportion of fraud that is successfully detected)

## 7. Model Persistence and Deployment

The code implements robust model saving and loading capabilities:

```python
def save_model(self, path=None):
    """
    Save the model and preprocessing components.
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
```

**Model Persistence Components:**

1. **Complete Ecosystem Serialization**:
   - **Neural Network Model**: Saved in Keras format
   - **Preprocessing Components**: All scalers saved with joblib
   - **Configuration**: Parameters and settings saved for reproducibility
   - **Optimized Threshold**: Decision boundary preserved

2. **Modular Storage**:
   - Each component saved as separate file
   - Allows for selective updating if needed
   - Better version control and diff tracking

3. **Configuration Management**:
   - Stores all parameters needed for reconstruction
   - Includes derived model properties (like model type)
   - Records presence of optional components

**Complementary Loading Function:**

```python
@classmethod
def load_model(cls, path):
    """
    Load a saved model and its components.
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