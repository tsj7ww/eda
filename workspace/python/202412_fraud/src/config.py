import os
import sys
import logging
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import argparse

# Setup logging
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Configuration manager for the fraud detection system.
    
    This class handles loading, saving, and managing configuration from:
    - Default values
    - Configuration files (JSON/YAML)
    - Environment variables
    - Command-line arguments
    
    It provides a unified interface for accessing configuration settings
    across all modules in the system.
    """
    
    def __init__(self, config_file: Optional[str] = None, 
                 env_prefix: str = "FRAUD_", 
                 parse_args: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
            env_prefix: Prefix for environment variables
            parse_args: Whether to parse command-line arguments
        """
        # Default configuration
        self.config = self._get_default_config()
        
        # Environment variable prefix
        self.env_prefix = env_prefix
        
        # Load configuration from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load configuration from environment variables
        self.load_from_env()
        
        # Load configuration from command-line arguments
        if parse_args:
            self.load_from_args()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            "general": {
                "random_state": 42,
                "output_dir": "output",
                "log_level": "INFO",
                "log_file": "fraud_detection.log",
            },
            "data": {
                "time_column": "Time",
                "amount_column": "Amount",
                "target_column": "Class",
                "feature_prefix": "V",
                "entity_columns": [],
                "test_size": 0.2,
                "validation_size": 0.2,
            },
            "preprocessing": {
                "pca_scaling": "standard",
                "amount_scaling": "robust",
                "time_scaling": "standard",
                "handle_missing": "impute",
                "outlier_handling": "clip",
                "outlier_threshold": 3.0,
            },
            "feature_engineering": {
                "create_time_features": True,
                "time_windows": [300, 3600, 86400],
                "create_amount_features": True,
                "amount_bins": 10,
                "log_amount": True,
                "create_pca_aggregates": True,
                "create_pca_interactions": False,
                "max_interactions": 10,
                "create_statistical_features": True,
                "create_entity_features": False,
                "create_network_features": False,
                "perform_feature_selection": False,
                "feature_selection_method": "mutual_info",
                "feature_selection_k": 50,
            },
            "training": {
                "model_type": "gbm",
                "model_implementation": "xgboost",
                "perform_optimization": True,
                "optimization_metric": "pr_auc",
                "optimization_trials": 100,
                "use_cv": True,
                "cv_folds": 5,
                "sampling_method": "none",
                "sampling_ratio": 0.1,
                "optimize_threshold": True,
                "threshold_criterion": "f1",
                "early_stopping_rounds": 10,
                "balance_classes": True,
                "model_dir": "models",
                "model_name": "fraud_model",
                "n_jobs": -1,
            },
            "gbm_params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "dl_params": {
                "batch_size": 256,
                "epochs": 50,
                "patience": 10,
                "hidden_layers": [64, 32],
                "dropout_rate": 0.3,
            },
            "evaluation": {
                "cost_matrix": {
                    "tp_benefit": 200,
                    "fp_cost": 10,
                    "fn_cost": 200,
                    "tn_benefit": 0,
                },
                "generate_plots": True,
                "save_reports": True,
                "report_dir": "reports",
            }
        }
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            return False
        
        try:
            # Determine file type from extension
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    loaded_config = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return False
            
            # Update configuration
            self._deep_update(self.config, loaded_config)
            logger.info(f"Loaded configuration from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            return False
    
    def load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format:
            {env_prefix}{SECTION}__{KEY}
            
        Example:
            FRAUD_TRAINING__MODEL_TYPE=gbm
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Remove prefix
                key = env_var[len(self.env_prefix):]
                
                # Split into section and key
                parts = key.split('__')
                
                if len(parts) == 2:
                    section, key = parts
                    section = section.lower()
                    key = key.lower()
                    
                    if section in self.config and key in self.config[section]:
                        # Convert value to the right type
                        current_value = self.config[section][key]
                        
                        if isinstance(current_value, bool):
                            # Convert string to boolean
                            value = value.lower() in ('true', 'yes', '1', 'y')
                        elif isinstance(current_value, int):
                            # Convert string to integer
                            value = int(value)
                        elif isinstance(current_value, float):
                            # Convert string to float
                            value = float(value)
                        elif isinstance(current_value, list):
                            # Convert comma-separated string to list
                            value = [item.strip() for item in value.split(',')]
                            
                            # Convert list items to the right type
                            if current_value and len(current_value) > 0:
                                element_type = type(current_value[0])
                                if element_type == int:
                                    value = [int(item) for item in value]
                                elif element_type == float:
                                    value = [float(item) for item in value]
                        
                        # Update configuration
                        self.config[section][key] = value
                        logger.debug(f"Updated configuration from environment: {section}.{key} = {value}")
    
    def load_from_args(self) -> None:
        """
        Load configuration from command-line arguments.
        
        Arguments should be in the format:
            --{section}-{key} value
            
        Example:
            --training-model_type gbm
        """
        parser = argparse.ArgumentParser(description='Fraud Detection System')
        
        # Add arguments for each configuration option
        for section, values in self.config.items():
            for key, value in values.items():
                # Format argument name: --{section}-{key}
                arg_name = f"--{section}-{key}"
                
                # Determine argument type
                if isinstance(value, bool):
                    parser.add_argument(arg_name, dest=f"{section}__{key}", 
                                        action='store_true', help=f"{section} {key}")
                    parser.add_argument(f"--no-{section}-{key}", dest=f"{section}__{key}", 
                                        action='store_false', help=f"Disable {section} {key}")
                    parser.set_defaults(**{f"{section}__{key}": value})
                elif isinstance(value, list):
                    parser.add_argument(arg_name, dest=f"{section}__{key}", 
                                        nargs='+', type=type(value[0]) if value else str, 
                                        default=value, help=f"{section} {key}")
                else:
                    parser.add_argument(arg_name, dest=f"{section}__{key}", 
                                        type=type(value), default=value, help=f"{section} {key}")
        
        # Add argument for configuration file
        parser.add_argument('--config', dest='config_file', 
                            type=str, help='Path to configuration file')
        
        # Parse arguments
        args = parser.parse_args()
        
        # Load configuration from file if specified
        if hasattr(args, 'config_file') and args.config_file:
            self.load_from_file(args.config_file)
        
        # Update configuration from arguments
        for arg_name, value in vars(args).items():
            if arg_name == 'config_file':
                continue
                
            if '__' in arg_name:
                section, key = arg_name.split('__')
                if section in self.config and key in self.config[section]:
                    self.config[section][key] = value
                    logger.debug(f"Updated configuration from arguments: {section}.{key} = {value}")
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a path.
        
        Args:
            path: Path to the configuration value (e.g., "training.model_type")
            default: Default value if the path is not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value using a path.
        
        Args:
            path: Path to the configuration value (e.g., "training.model_type")
            value: Value to set
        """
        parts = path.split('.')
        
        # Navigate to the right level
        config = self.config
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Get the full configuration.
        
        Returns:
            Dictionary with the full configuration
        """
        return self.config
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save the configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Determine file type from extension
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return False
            
            logger.info(f"Saved configuration to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            return False
    
    def create_module_config(self, module: str) -> Any:
        """
        Create a configuration object for a specific module.
        
        This method will convert a section of the configuration into
        the appropriate configuration class for a module.
        
        Args:
            module: Name of the module ('preprocessing', 'feature_engineering', 'training', 'evaluation')
            
        Returns:
            Configuration object for the module or None if not found
        """
        if module not in self.config:
            return None
        
        # Return specific module configuration
        if module == 'preprocessing':
            from preprocessing import PreprocessingConfig
            config = PreprocessingConfig()
            
            # Update configuration from settings
            module_config = self.config[module]
            
            # Add data-related settings
            if 'data' in self.config:
                for key in ['time_column', 'amount_column', 'target_column', 'feature_prefix']:
                    if key in self.config['data']:
                        setattr(config, key, self.config['data'][key])
            
            # Add preprocessing-specific settings
            for key, value in module_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Add general settings
            if 'general' in self.config and 'random_state' in self.config['general']:
                config.random_state = self.config['general']['random_state']
            
            return config
            
        elif module == 'feature_engineering':
            from feature_engineering import FeatureEngineeringConfig
            config = FeatureEngineeringConfig()
            
            # Update configuration from settings
            module_config = self.config[module]
            
            # Add data-related settings
            if 'data' in self.config:
                for key in ['time_column', 'amount_column', 'target_column', 'feature_prefix', 'entity_columns']:
                    if key in self.config['data']:
                        setattr(config, key, self.config['data'][key])
            
            # Add feature engineering-specific settings
            for key, value in module_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Add general settings
            if 'general' in self.config and 'random_state' in self.config['general']:
                config.random_state = self.config['general']['random_state']
            
            return config
            
        elif module == 'training':
            from model_training import ModelTrainingConfig
            config = ModelTrainingConfig()
            
            # Update configuration from settings
            module_config = self.config[module]
            
            # Add data-related settings
            if 'data' in self.config and 'target_column' in self.config['data']:
                config.target_column = self.config['data']['target_column']
            
            # Add training-specific settings
            for key, value in module_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Add GBM parameters
            if 'gbm_params' in self.config:
                config.gbm_params = self.config['gbm_params']
            
            # Add deep learning parameters
            if 'dl_params' in self.config:
                config.dl_params = self.config['dl_params']
            
            # Add general settings
            if 'general' in self.config and 'random_state' in self.config['general']:
                config.random_state = self.config['general']['random_state']
            
            return config
        
        # Return the configuration as a dictionary if no specific class is available
        return self.config[module]


# Simple function to create configuration manager
def load_configuration(config_file: Optional[str] = None, 
                      env_prefix: str = "FRAUD_", 
                      parse_args: bool = True) -> ConfigurationManager:
    """
    Load configuration from various sources.
    
    Args:
        config_file: Path to configuration file
        env_prefix: Prefix for environment variables
        parse_args: Whether to parse command-line arguments
        
    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager(config_file, env_prefix, parse_args)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_manager = load_configuration()
    
    # Print configuration
    print("Configuration:")
    for section, values in config_manager.get_full_config().items():
        print(f"\n[{section}]")
        for key, value in values.items():
            print(f"{key} = {value}")
    
    # Get a specific value
    model_type = config_manager.get("training.model_type")
    print(f"\nModel type: {model_type}")
    
    # Save configuration to file
    config_manager.save_to_file("config.json")
    print("\nConfiguration saved to config.json")
    
    # Create module configuration
    training_config = config_manager.create_module_config("training")
    if training_config:
        print("\nTraining configuration created successfully")
        print(f"Model type: {training_config.model_type}")
        print(f"Model implementation: {training_config.model_implementation}")
        print(f"Optimization metric: {training_config.optimization_metric}")
    else:
        print("\nFailed to create training configuration")