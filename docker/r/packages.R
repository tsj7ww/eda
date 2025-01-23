packages <- c(
  # Core Data Science & Tidyverse
  "tidyverse",      # Collection of data science packages
  "data.table",     # Fast data manipulation
  "dtplyr",         # Data.table backend for dplyr
  "lubridate",      # Date/time handling
  "jsonlite",       # JSON handling
  "xml2",           # XML handling
  "readxl",         # Excel files
  "haven",          # SPSS, SAS, Stata files
  
  # Data Manipulation & Preprocessing
  "tidyr",          # Data tidying
  "dplyr",          # Data manipulation
  "stringr",        # String manipulation
  "forcats",        # Factor handling
  "janitor",        # Data cleaning
  "scales",         # Scale functions for visualization
  
  # Visualization
  "ggplot2",        # Grammar of graphics plotting
  "plotly",         # Interactive plots
  "highcharter",    # Interactive charts
  "viridis",        # Color palettes
  "GGally",         # Extension to ggplot2
  "corrplot",       # Correlation plots
  "ggridges",       # Ridgeline plots
  "gganimate",      # Animated plots
  
  # Statistical Analysis
  "stats",          # Basic statistics
  "car",            # Companion to Applied Regression
  "lme4",           # Linear mixed-effects models
  "survival",       # Survival analysis
  "forecast",       # Time series forecasting
  "tseries",        # Time series analysis
  "zoo",            # Time series objects
  "sandwich",       # Robust covariance matrix estimators
  
  # Machine Learning
  "caret",          # ML model training and evaluation
  "tidymodels",     # Collection of ML packages
  "ranger",         # Fast Random Forests
  "xgboost",        # Gradient Boosting
  "glmnet",         # Regularized regression
  "kernlab",        # Kernel-based ML
  "e1071",          # SVM and Naive Bayes
  "nnet",           # Neural Networks
  "rpart",          # Decision Trees
  "randomForest",   # Random Forests
  
  # Deep Learning
  "keras",          # Deep Learning interface
  "tensorflow",     # TensorFlow interface
  
  # Dimensionality Reduction & Clustering
  "umap",           # Uniform Manifold Approximation
  "Rtsne",          # t-SNE
  "factoextra",     # Clustering algorithms & visualization
  "cluster",        # Clustering algorithms
  "dbscan",         # Density-based clustering
  
  # Model Interpretation & Evaluation
  "ROCR",           # ROC curves
  "PRROC",          # Precision-Recall curves
  "lime",           # Local Interpretable Model-agnostic Explanations
  "shapviz",        # SHAP value visualization
  "vip",            # Variable importance plots
  
  # Reporting & Reproducibility
  "rmarkdown",      # Dynamic report generation
  "knitr",          # Dynamic report generation
  "shiny",          # Interactive web apps
  "flexdashboard",  # Interactive dashboards
  "DT",             # Interactive tables
  
  # Development Tools
  "devtools",       # Package development
  "testthat",       # Unit testing
  "roxygen2",       # Documentation
  "profvis",        # Code profiling
  
  # Performance & Parallel Processing
  "future",         # Parallel processing
  "furrr",          # Parallel processing with purrr
  "parallel",       # Parallel processing
  "doParallel",     # Parallel backend
  
  # Database Connections
  "DBI",            # Database interface
  "RPostgres",      # PostgreSQL interface
  "RSQLite",        # SQLite interface
  "odbc"            # ODBC interface
)