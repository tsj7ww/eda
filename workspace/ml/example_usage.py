import mlflow
import kfp
from kfp import dsl

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.93)

# Kubeflow pipeline
@dsl.pipeline(
    name='Example Pipeline',
    description='Example pipeline with MLflow integration'
)
def example_pipeline():
    with dsl.ExitHandler(exit_op=None):
        with mlflow.start_run():
            # Pipeline steps here
            pass

# Run pipeline
client = kfp.Client(host='http://localhost:8888')
client.create_run_from_pipeline_func(example_pipeline, arguments={})