from kfp.components import create_component_from_func

def create_dataset_version(
    data_path: str,
    version: str,
    description: str = None
) -> str:
    """Creates a versioned dataset in MLflow."""
    import mlflow
    
    with mlflow.start_run():
        mlflow.log_artifact(data_path, f"datasets/{version}")
        if description:
            mlflow.log_param("description", description)
        return f"datasets/{version}"

dataset_version_op = create_component_from_func(
    create_dataset_version,
    base_image='python:3.9',
    packages_to_install=['mlflow']
)