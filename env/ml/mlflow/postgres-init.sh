#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE kubeflow;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mlflow" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS mlflow;
    
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        artifact_location VARCHAR(255),
        lifecycle_stage VARCHAR(32),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX idx_experiment_name ON experiments(name);
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "kubeflow" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS kubeflow;
    
    CREATE TABLE IF NOT EXISTS pipelines (
        pipeline_id UUID PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        parameters JSONB,
        status VARCHAR(50)
    );

    CREATE INDEX idx_pipeline_name ON pipelines(name);
    CREATE INDEX idx_pipeline_status ON pipelines(status);
EOSQL