import os
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Import custom modules
from .configuration import load_configuration
from .preprocessing import FraudDataPreprocessor, PreprocessingConfig
from .feature_engineering import FraudFeatureEngineer, FeatureEngineeringConfig
from .model_training import GBMModelTrainer, ModelTrainingConfig
from .model_evaluation import FraudModelEvaluator
from .utils import timer, MemoryTracker, ErrorHandler, setup_logging

def run_fraud_detection_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config_file: str = None
) -> Dict[str, Any]:
    """
    Run the complete fraud detection pipeline.
    
    Args:
        train_df: Training DataFrame with features and target
        test_df: Test DataFrame with features and target
        config_file: Path to configuration file (optional)
        
    Returns:
        Dictionary with pipeline results
    """
    start_time = time.time()
    
    # Load configuration
    config_manager = load_configuration(config_file)
    
    # Setup logging
    log_level = config_manager.get("general.log_level", "INFO")
    log_file = config_manager.get("general.log_file", "fraud_detection.log")
    logger = setup_logging(log_level, log_file=log_file)
    
    # Create output directory
    output_dir = config_manager.get("general.output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get target column name
    target_column = config_manager.get("data.target_column", "Class")
    
    # Extract features and target
    train_y = train_df[target_column].values
    test_y = test_df[target_column].values
    
    # Keep original data
    train_features_df = train_df.drop(columns=[target_column])
    test_features_df = test_df.drop(columns=[target_column])
    
    # Track pipeline results
    results = {
        "config": config_manager.get_full_config(),
        "metrics": {},
        "feature_importance": None,
        "paths": {}
    }
    
    # Step 1: Preprocessing
    logger.info("Step 1: Data Preprocessing")
    with timer("Preprocessing", logger):
        try:
            # Create preprocessor with configuration
            preprocessing_config = config_manager.create_module_config("preprocessing")
            preprocessor = FraudDataPreprocessor(preprocessing_config)
            
            # Track memory usage
            MemoryTracker.log_memory_usage(train_df, "Training Data", logger)
            
            # Prepare data with train/validation split
            prep_data = preprocessor.prepare_train_val_split(
                train_df,
                return_separate_features=False
            )
            
            # Get training and validation data
            X_train = prep_data['X_train']
            y_train = prep_data['y_train']
            X_val = prep_data['X_val']
            y_val = prep_data['y_val']
            
            # Process test data
            X_test = preprocessor.transform(test_df)
            
            logger.info(f"Preprocessed shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            results["preprocessing"] = {
                "train_shape": X_train.shape,
                "val_shape": X_val.shape,
                "test_shape": X_test.shape
            }
            
        except Exception as e:
            ErrorHandler.log_exception(
                logger, 
                e, 
                message="Preprocessing failed",
                include_context={"train_shape": str(train_df.shape)}
            )
            raise
    
    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    with timer("Feature Engineering", logger):
        try:
            # Create feature engineer with configuration
            fe_config = config_manager.create_module_config("feature_engineering")
            feature_engineer = FraudFeatureEngineer(fe_config)
            
            # Apply feature engineering
            # Reconstruct DataFrames with original features and target
            train_fe_df = pd.DataFrame(
                X_train, 
                columns=[col for col in train_features_df.columns]
            )
            train_fe_df[target_column] = y_train
            
            # Fit and transform on training data
            train_fe_df = feature_engineer.fit_transform(train_fe_df)
            
            # Transform validation and test data
            val_fe_df = pd.DataFrame(
                X_val,
                columns=[col for col in train_features_df.columns]
            )
            val_fe_df[target_column] = y_val
            val_fe_df = feature_engineer.transform(val_fe_df)
            
            test_fe_df = pd.DataFrame(
                X_test,
                columns=[col for col in test_features_df.columns] if isinstance(X_test, np.ndarray) else test_features_df.columns
            )
            test_fe_df[target_column] = test_y
            test_fe_df = feature_engineer.transform(test_fe_df)
            
            # Extract features and target
            X_train_fe = train_fe_df.drop(columns=[target_column])
            y_train_fe = train_fe_df[target_column]
            
            X_val_fe = val_fe_df.drop(columns=[target_column])
            y_val_fe = val_fe_df[target_column]
            
            X_test_fe = test_fe_df.drop(columns=[target_column])
            y_test_fe = test_fe_df[target_column]
            
            logger.info(f"Feature engineered shapes - Train: {X_train_fe.shape}, Val: {X_val_fe.shape}, Test: {X_test_fe.shape}")
            logger.info(f"Added {len(feature_engineer.derived_columns)} new features")
            
            results["feature_engineering"] = {
                "train_shape": X_train_fe.shape,
                "val_shape": X_val_fe.shape,
                "test_shape": X_test_fe.shape,
                "derived_columns": feature_engineer.derived_columns
            }
            
        except Exception as e:
            ErrorHandler.log_exception(
                logger,
                e,
                message="Feature engineering failed"
            )
            raise
    
    # Step 3: Model Training
    logger.info("Step 3: Model Training")
    with timer("Model Training", logger):
        try:
            # Create model trainer with configuration
            training_config = config_manager.create_module_config("training")
            
            # Set model paths
            model_dir = os.path.join(output_dir, "models")
            training_config.model_dir = model_dir
            
            # Create trainer based on model type
            if training_config.model_type == 'gbm':
                trainer = GBMModelTrainer(training_config)
            else:
                # Default to GBM if other model types not implemented yet
                logger.warning(f"Model type {training_config.model_type} not implemented. Using GBM.")
                training_config.model_type = 'gbm'
                trainer = GBMModelTrainer(training_config)
            
            # Combine train and validation for final training
            # or let the trainer handle validation internally
            if training_config.use_cv:
                # Use cross-validation - train on all data
                combined_X = pd.concat([X_train_fe, X_val_fe])
                combined_y = pd.concat([y_train_fe, y_val_fe])
                
                # Train the model
                trainer.fit(combined_X, combined_y)
            else:
                # Use separate validation set
                trainer.fit(X_train_fe, y_train_fe)
            
            # Save the model
            model_path = trainer.save()
            results["paths"]["model"] = model_path
            
            # Store feature importance
            if trainer.feature_importance is not None:
                results["feature_importance"] = trainer.feature_importance.to_dict()
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            ErrorHandler.log_exception(
                logger,
                e,
                message="Model training failed"
            )
            raise
    
    # Step 4: Model Evaluation
    logger.info("Step 4: Model Evaluation")
    with timer("Model Evaluation", logger):
        try:
            # Get predictions
            y_prob = trainer.predict_proba(X_test_fe)
            y_pred = trainer.predict(X_test_fe)
            
            # Calculate metrics
            threshold = trainer.threshold
            metrics = FraudModelEvaluator.calculate_metrics(
                y_test_fe, y_pred, y_prob, threshold
            )
            
            # Save metrics
            results["metrics"] = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            
            # Generate report
            report_dir = os.path.join(output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            report_path = os.path.join(report_dir, f"{training_config.model_name}_report.txt")
            FraudModelEvaluator.generate_report(metrics, report_path)
            results["paths"]["report"] = report_path
            
            # Create evaluation plots
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Performance plots
            fig = FraudModelEvaluator.plot_metrics(y_test_fe, y_prob, threshold)
            plot_path = os.path.join(plots_dir, f"{training_config.model_name}_performance.png")
            fig.savefig(plot_path)
            plt.close(fig)
            results["paths"]["performance_plot"] = plot_path
            
            # Feature importance plot
            if trainer.feature_importance is not None:
                fig = FraudModelEvaluator.feature_importance_plot(
                    trainer.feature_importance['Feature'].values,
                    trainer.feature_importance['Importance'].values
                )
                plot_path = os.path.join(plots_dir, f"{training_config.model_name}_importance.png")
                fig.savefig(plot_path)
                plt.close(fig)
                results["paths"]["importance_plot"] = plot_path
            
            logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            ErrorHandler.log_exception(
                logger,
                e,
                message="Model evaluation failed"
            )
    
    # Log overall pipeline execution time
    execution_time = time.time() - start_time
    logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
    results["execution_time"] = execution_time
    
    return results


def main():
    """
    Main function to demonstrate the pipeline.
    
    This assumes train_x, train_y, test_x, test_y variables are provided
    or can be loaded from file.
    """
    # Configure logging
    logger = setup_logging("INFO")
    
    # For demonstration: Create synthetic data if not provided
    try:
        # Check if train_x, train_y, test_x, test_y are in global scope
        train_x, train_y, test_x, test_y
    except NameError:
        logger.info("Creating synthetic data for demonstration")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 10000
        n_features = 28  # V1 to V28
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
        time_col = np.arange(n_samples) * 100  # 100 seconds between transactions
        
        # Create target (1% fraud)
        y = np.zeros(n_samples)
        fraud_indices = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
        y[fraud_indices] = 1
        
        # Create DataFrame
        feature_cols = [f'V{i}' for i in range(1, n_features + 1)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['Amount'] = amount
        df['Time'] = time_col
        df['Class'] = y
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    else:
        # Use provided variables
        train_df = pd.DataFrame(train_x)
        test_df = pd.DataFrame(test_x)
        
        # Add target column
        train_df['Class'] = train_y
        test_df['Class'] = test_y
    
    # Run the pipeline
    results = run_fraud_detection_pipeline(train_df, test_df)
    
    # Print results summary
    print("\nPipeline Results Summary:")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    
    print("\nModel Performance Metrics:")
    for metric, value in results['metrics'].items():
        if not isinstance(value, dict) and metric not in ['confusion_matrix']:
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    print("\nOutput Paths:")
    for key, path in results['paths'].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()