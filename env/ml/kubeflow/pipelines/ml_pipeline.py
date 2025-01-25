import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath
import mlflow
from typing import Dict, List

# Data preprocessing component
@func_to_container_op
def preprocess_data(
    data_path: InputPath("CSV"),
    processed_data_path: OutputPath("CSV"),
    params: Dict[str, float]
):
    import pandas as pd
    import mlflow
    
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)
        
        # Read and process data
        data = pd.read_csv(data_path)
        
        # Example preprocessing steps
        data = data.dropna()
        data = data.drop_duplicates()
        
        # Log metrics
        mlflow.log_metric("rows_processed", len(data))
        
        # Save processed data
        data.to_csv(processed_data_path, index=False)

# Model training component
@func_to_container_op
def train_model(
    processed_data_path: InputPath("CSV"),
    model_path: OutputPath("pickle"),
    hyperparameters: Dict[str, float]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import mlflow
    import mlflow.sklearn
    
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        
        # Load data
        data = pd.read_csv(processed_data_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        mlflow.log_metrics({
            "train_accuracy": train_score,
            "test_accuracy": test_score
        })
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

# Model evaluation component
@func_to_container_op
def evaluate_model(
    model_path: InputPath("pickle"),
    test_data_path: InputPath("CSV"),
    metrics_path: OutputPath("JSON")
):
    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import classification_report
    import mlflow
    
    with mlflow.start_run(nested=True):
        # Load model and test data
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics to MLflow
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(report, f)

# Model deployment component
@func_to_container_op
def deploy_model(
    model_path: InputPath("pickle"),
    deployment_path: OutputPath("Text")
):
    import mlflow
    from datetime import datetime
    
    with mlflow.start_run(nested=True):
        # Log deployment time
        deployment_time = datetime.now().isoformat()
        mlflow.log_param("deployment_time", deployment_time)
        
        # In a real scenario, you would deploy the model to your serving infrastructure
        # For this example, we'll just write the deployment status
        with open(deployment_path, 'w') as f:
            f.write(f"Model deployed at {deployment_time}")
        
        mlflow.log_metric("deployment_status", 1)

# Define the pipeline
@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML pipeline with MLflow tracking'
)
def ml_pipeline(
    data_path: str,
    preprocessing_params: Dict[str, float] = {"threshold": 0.5},
    hyperparameters: Dict[str, float] = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
):
    # Start MLflow run
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ML Pipeline Example")
    
    # Pipeline steps
    preprocess_op = preprocess_data(data_path, preprocessing_params)
    
    train_op = train_model(
        preprocess_op.outputs['processed_data_path'],
        hyperparameters
    )
    
    evaluate_op = evaluate_model(
        train_op.outputs['model_path'],
        preprocess_op.outputs['processed_data_path']
    )
    
    deploy_op = deploy_model(train_op.outputs['model_path'])
    
    # Add success condition
    deploy_op.add_is_exit_handler(lambda: True)

# Example usage
def run_pipeline():
    client = kfp.Client(host='http://localhost:8888')
    
    # Compile the pipeline
    pipeline_func = ml_pipeline
    pipeline_filename = pipeline_func.__name__ + '.yaml'
    kfp.compiler.Compiler().compile(pipeline_func, pipeline_filename)
    
    # Create an experiment
    exp = client.create_experiment(name='ml-pipeline-experiment')
    
    # Submit the pipeline
    run = client.run_pipeline(
        experiment_id=exp.id,
        job_name='ml-pipeline-run',
        pipeline_package_path=pipeline_filename,
        params={
            'data_path': '/path/to/your/data.csv',
            'preprocessing_params': {'threshold': 0.5},
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
    )

if __name__ == '__main__':
    run_pipeline()
