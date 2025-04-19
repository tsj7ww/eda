import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    precision_recall_curve,
    average_precision_score, 
    roc_auc_score, 
    f1_score
)

# Setup logging
logger = logging.getLogger(__name__)

class FraudModelEvaluator:
    """
    Comprehensive evaluation tools for fraud detection models.
    
    Features:
    - Calculation of relevant fraud detection metrics
    - Visualization of model performance
    - Threshold optimization for decision making
    - Cost-benefit analysis and business impact assessment
    - Model comparison
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, 
                         threshold: float = 0.5, cost_matrix: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            threshold: Classification threshold used
            cost_matrix: Cost matrix for business impact calculation
                {
                    'tp_benefit': 200,  # Benefit of catching fraud
                    'fp_cost': 10,      # Cost of investigating non-fraud
                    'fn_cost': 200,     # Cost of missing fraud
                    'tn_benefit': 0     # Benefit of not investigating non-fraud
                }
            
        Returns:
            Dictionary of evaluation metrics
        """
        # If y_prob is provided, recalculate y_pred using threshold
        if y_prob is not None:
            y_pred = (y_prob >= threshold).astype(int)
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'threshold': threshold
        }
        
        # Add AUC metrics if probabilities are provided
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Add detection rate and false alarm rate
        metrics['detection_rate'] = metrics['recall']  # Same as recall/TPR
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR
        
        # Add business metrics
        # Default cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = {
                'tp_benefit': 200,  # Benefit of catching fraud
                'fp_cost': 10,      # Cost of investigating non-fraud
                'fn_cost': 200,     # Cost of missing fraud
                'tn_benefit': 0     # Benefit of correctly not flagging legitimate transactions
            }
        
        # Calculate net benefit using cost matrix
        net_benefit = (
            tp * cost_matrix['tp_benefit'] - 
            fp * cost_matrix['fp_cost'] - 
            fn * cost_matrix['fn_cost'] +
            tn * cost_matrix['tn_benefit']
        )
        metrics['net_benefit'] = net_benefit
        
        # Calculate return on investment (ROI)
        total_cost = fp * cost_matrix['fp_cost'] + fn * cost_matrix['fn_cost']
        total_benefit = tp * cost_matrix['tp_benefit'] + tn * cost_matrix['tn_benefit']
        
        # Avoid division by zero
        if total_cost > 0:
            metrics['roi'] = (total_benefit - total_cost) / total_cost
        else:
            metrics['roi'] = float('inf') if total_benefit > 0 else 0
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                              criterion: str = 'f1', 
                              cost_matrix: Optional[Dict[str, float]] = None) -> float:
        """
        Find the optimal threshold for classification based on a criterion.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            criterion: Criterion to optimize ('f1', 'precision_recall_balance', 'cost')
            cost_matrix: Cost matrix for business impact calculation (for 'cost' criterion)
            
        Returns:
            Optimal threshold value
        """
        # Get precision and recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        if criterion == 'f1':
            # Calculate F1 score for each threshold
            f1_scores = []
            for i in range(len(precision)):
                if precision[i] + recall[i] > 0:  # Avoid division by zero
                    f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                else:
                    f1 = 0
                f1_scores.append(f1)
                
            # Find threshold that maximizes F1 score
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])  # Exclude the last value
                return thresholds[best_idx]
                
        elif criterion == 'precision_recall_balance':
            # Find threshold where precision and recall are closest
            differences = np.abs(precision - recall)
            best_idx = np.argmin(differences[:-1])  # Exclude the last value
            return thresholds[best_idx]
            
        elif criterion == 'cost':
            # Define a cost function using the provided cost matrix
            if cost_matrix is None:
                cost_matrix = {
                    'tp_benefit': 200,
                    'fp_cost': 10,
                    'fn_cost': 200,
                    'tn_benefit': 0
                }
                
            costs = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                # Calculate net cost (negative of net benefit)
                net_cost = -(
                    tp * cost_matrix['tp_benefit'] - 
                    fp * cost_matrix['fp_cost'] - 
                    fn * cost_matrix['fn_cost'] +
                    tn * cost_matrix['tn_benefit']
                )
                costs.append(net_cost)
                
            best_idx = np.argmin(costs)
            return thresholds[best_idx]
            
        # Default
        return 0.5
    
    @staticmethod
    def plot_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, 
                    figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """
        Create comprehensive evaluation plots for fraud detection.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Classification threshold to use
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with plots
        """
        # Create binary predictions using threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        axs[0, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axs[0, 0].plot([0, 1], [0, 1], 'k--')
        axs[0, 0].set_xlim([0.0, 1.0])
        axs[0, 0].set_ylim([0.0, 1.05])
        axs[0, 0].set_xlabel('False Positive Rate')
        axs[0, 0].set_ylabel('True Positive Rate')
        axs[0, 0].set_title('Receiver Operating Characteristic (ROC) Curve')
        axs[0, 0].legend(loc='lower right')
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        axs[0, 1].plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        axs[0, 1].axhline(y=sum(y_true)/len(y_true), color='r', linestyle='--', 
                        label=f'Baseline (fraud rate = {sum(y_true)/len(y_true):.3f})')
        axs[0, 1].set_xlim([0.0, 1.0])
        axs[0, 1].set_ylim([0.0, 1.05])
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].set_title('Precision-Recall Curve')
        axs[0, 1].legend(loc='upper right')
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0])
        axs[1, 0].set_xlabel('Predicted Label')
        axs[1, 0].set_ylabel('True Label')
        axs[1, 0].set_title('Confusion Matrix')
        
        # 4. Normalized Confusion Matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axs[1, 1])
        axs[1, 1].set_xlabel('Predicted Label')
        axs[1, 1].set_ylabel('True Label')
        axs[1, 1].set_title('Normalized Confusion Matrix')
        
        # 5. Threshold Analysis
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        axs[2, 0].plot(thresholds, precisions, label='Precision')
        axs[2, 0].plot(thresholds, recalls, label='Recall')
        axs[2, 0].plot(thresholds, f1_scores, label='F1 Score')
        axs[2, 0].axvline(x=threshold, color='r', linestyle='--', 
                        label=f'Threshold = {threshold:.2f}')
        axs[2, 0].set_xlabel('Threshold')
        axs[2, 0].set_ylabel('Score')
        axs[2, 0].set_title('Metrics vs. Threshold')
        axs[2, 0].legend()
        
        # 6. Probability Distribution
        axs[2, 1].hist(y_prob[y_true==0], bins=50, alpha=0.5, label='Non-Fraud')
        axs[2, 1].hist(y_prob[y_true==1], bins=50, alpha=0.5, label='Fraud')
        axs[2, 1].axvline(x=threshold, color='r', linestyle='--',
                        label=f'Threshold = {threshold:.2f}')
        axs[2, 1].set_xlabel('Predicted Probability')
        axs[2, 1].set_ylabel('Count')
        axs[2, 1].set_title('Probability Distribution')
        axs[2, 1].legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_models(model_results: Dict[str, Dict[str, Any]], figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Compare multiple fraud detection models visually.
        
        Args:
            model_results: Dictionary with model names as keys and their evaluation results as values
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with comparison plots
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Prepare data for bar plots
        models = list(model_results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, results in model_results.items():
            for metric in metrics:
                metric_values[metric].append(results.get(metric, 0))
        
        # 1. Bar plot of precision, recall, F1
        x = np.arange(len(models))
        width = 0.25
        
        axs[0, 0].bar(x - width, metric_values['precision'], width, label='Precision')
        axs[0, 0].bar(x, metric_values['recall'], width, label='Recall')
        axs[0, 0].bar(x + width, metric_values['f1_score'], width, label='F1 Score')
        
        axs[0, 0].set_xlabel('Models')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].set_title('Precision, Recall, and F1 Score Comparison')
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(models)
        axs[0, 0].legend()
        
        # 2. Bar plot of AUC metrics
        axs[0, 1].bar(x - width/2, metric_values['roc_auc'], width, label='ROC AUC')
        axs[0, 1].bar(x + width/2, metric_values['pr_auc'], width, label='PR AUC')
        
        axs[0, 1].set_xlabel('Models')
        axs[0, 1].set_ylabel('AUC')
        axs[0, 1].set_title('AUC Metrics Comparison')
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(models)
        axs[0, 1].legend()
        
        # 3. Confusion matrix counts (TP, FP, TN, FN)
        count_metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
        count_values = {metric: [] for metric in count_metrics}
        
        for model_name, results in model_results.items():
            for metric in count_metrics:
                count_values[metric].append(results.get(metric, 0))
        
        axs[1, 0].bar(x - 3*width/4, count_values['true_positives'], width/2, label='TP')
        axs[1, 0].bar(x - width/4, count_values['false_positives'], width/2, label='FP')
        axs[1, 0].bar(x + width/4, count_values['true_negatives'], width/2, label='TN')
        axs[1, 0].bar(x + 3*width/4, count_values['false_negatives'], width/2, label='FN')
        
        axs[1, 0].set_xlabel('Models')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].set_title('Confusion Matrix Counts')
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(models)
        axs[1, 0].legend()
        
        # 4. Business metric - net benefit
        net_benefits = []
        for model_name, results in model_results.items():
            net_benefits.append(results.get('net_benefit', 0))
        
        axs[1, 1].bar(x, net_benefits, width, label='Net Benefit')
        
        axs[1, 1].set_xlabel('Models')
        axs[1, 1].set_ylabel('Value ($)')
        axs[1, 1].set_title('Business Impact - Net Benefit')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(models)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def feature_importance_plot(feature_names: List[str], importances: np.ndarray, top_n: int = 20, 
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot feature importances.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importance values
            top_n: Number of top features to show
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with feature importance plot
        """
        # Create DataFrame of feature importances
        if len(feature_names) != len(importances):
            raise ValueError("Length of feature_names must match length of importances")
            
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort and take top N
        fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax)
        
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_report(evaluation_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate a text report of evaluation metrics.
        
        Args:
            evaluation_results: Dictionary of evaluation metrics
            output_file: Optional file path to save the report
            
        Returns:
            Report as a string
        """
        report = "=" * 50 + "\n"
        report += "FRAUD DETECTION MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Basic metrics
        report += "BASIC METRICS:\n"
        report += "-" * 50 + "\n"
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        for metric in metrics:
            if metric in evaluation_results:
                report += f"{metric.replace('_', ' ').title()}: {evaluation_results[metric]:.4f}\n"
        
        # AUC metrics
        report += "\nAUC METRICS:\n"
        report += "-" * 50 + "\n"
        auc_metrics = ['roc_auc', 'pr_auc']
        for metric in auc_metrics:
            if metric in evaluation_results:
                report += f"{metric.replace('_', ' ').upper()}: {evaluation_results[metric]:.4f}\n"
        
        # Confusion matrix counts
        report += "\nCONFUSION MATRIX COUNTS:\n"
        report += "-" * 50 + "\n"
        cm_metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
        for metric in cm_metrics:
            if metric in evaluation_results:
                report += f"{metric.replace('_', ' ').title()}: {evaluation_results[metric]}\n"
        
        # Business metrics
        report += "\nBUSINESS IMPACT METRICS:\n"
        report += "-" * 50 + "\n"
        if 'net_benefit' in evaluation_results:
            report += f"Net Benefit: ${evaluation_results['net_benefit']:.2f}\n"
        if 'roi' in evaluation_results:
            report += f"ROI: {evaluation_results['roi']:.2f}\n"
            
        # Additional details
        report += "\nADDITIONAL DETAILS:\n"
        report += "-" * 50 + "\n"
        if 'threshold' in evaluation_results:
            report += f"Classification Threshold: {evaluation_results['threshold']:.4f}\n"
            
        # If 'confusion_matrix' is in evaluation_results, add a formatted version
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            report += "\nCONFUSION MATRIX:\n"
            report += "-" * 50 + "\n"
            report += "                 Predicted\n"
            report += "                 Neg    Pos\n"
            report += f"Actual   Neg    {cm[0][0]:<6} {cm[0][1]:<6}\n"
            report += f"         Pos    {cm[1][0]:<6} {cm[1][1]:<6}\n"
        
        # Save to file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    @staticmethod
    def evaluate_over_time(model_predictors: List[Callable], X_over_time: List[np.ndarray], 
                          y_over_time: List[np.ndarray], time_labels: List[str],
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Evaluate model performance over time to detect concept drift.
        
        Args:
            model_predictors: List of model prediction functions that take X and return probabilities
            X_over_time: List of feature arrays for different time periods
            y_over_time: List of target arrays for different time periods
            time_labels: List of labels for the time periods
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with performance over time
        """
        if not (len(X_over_time) == len(y_over_time) == len(time_labels)):
            raise ValueError("X_over_time, y_over_time, and time_labels must have the same length")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Metrics to track over time
        metrics_over_time = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
        # Calculate metrics for each time period
        for model_idx, model_predictor in enumerate(model_predictors):
            model_metrics = {metric: [] for metric in metrics_over_time.keys()}
            
            for X, y in zip(X_over_time, y_over_time):
                # Get predictions
                y_prob = model_predictor(X)
                y_pred = (y_prob >= 0.5).astype(int)  # Using default threshold of 0.5
                
                # Calculate metrics
                results = FraudModelEvaluator.calculate_metrics(y, y_pred, y_prob)
                
                # Store metrics
                for metric in model_metrics.keys():
                    model_metrics[metric].append(results.get(metric, 0))
            
            # Plot precision, recall, F1 over time
            axs[0, 0].plot(time_labels, model_metrics['precision'], 'o-', label=f'Model {model_idx+1} Precision')
            axs[0, 0].plot(time_labels, model_metrics['recall'], 's-', label=f'Model {model_idx+1} Recall')
            axs[0, 0].plot(time_labels, model_metrics['f1_score'], '^-', label=f'Model {model_idx+1} F1')
        
        axs[0, 0].set_xlabel('Time Period')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].set_title('Precision, Recall, F1 Over Time')
        axs[0, 0].legend()
        
        # Plot AUC metrics over time
        for model_idx, model_predictor in enumerate(model_predictors):
            model_metrics = {metric: [] for metric in ['roc_auc', 'pr_auc']}
            
            for X, y in zip(X_over_time, y_over_time):
                # Get predictions
                y_prob = model_predictor(X)
                
                # Calculate AUC metrics
                model_metrics['roc_auc'].append(roc_auc_score(y, y_prob))
                model_metrics['pr_auc'].append(average_precision_score(y, y_prob))
            
            # Plot AUC metrics
            axs[0, 1].plot(time_labels, model_metrics['roc_auc'], 'o-', label=f'Model {model_idx+1} ROC AUC')
            axs[0, 1].plot(time_labels, model_metrics['pr_auc'], 's-', label=f'Model {model_idx+1} PR AUC')
        
        axs[0, 1].set_xlabel('Time Period')
        axs[0, 1].set_ylabel('AUC')
        axs[0, 1].set_title('AUC Metrics Over Time')
        axs[0, 1].legend()
        
        # Plot class distribution over time
        class_counts = [np.bincount(y.astype(int), minlength=2) for y in y_over_time]
        fraud_rates = [counts[1] / counts.sum() for counts in class_counts]
        
        axs[1, 0].bar(time_labels, fraud_rates)
        axs[1, 0].set_xlabel('Time Period')
        axs[1, 0].set_ylabel('Fraud Rate')
        axs[1, 0].set_title('Fraud Rate Over Time')
        
        # Plot net benefit over time
        for model_idx, model_predictor in enumerate(model_predictors):
            net_benefits = []
            
            for X, y in zip(X_over_time, y_over_time):
                # Get predictions
                y_prob = model_predictor(X)
                y_pred = (y_prob >= 0.5).astype(int)  # Using default threshold
                
                # Calculate net benefit
                results = FraudModelEvaluator.calculate_metrics(y, y_pred, y_prob)
                net_benefits.append(results.get('net_benefit', 0))
            
            # Plot net benefit
            axs[1, 1].plot(time_labels, net_benefits, 'o-', label=f'Model {model_idx+1}')
        
        axs[1, 1].set_xlabel('Time Period')
        axs[1, 1].set_ylabel('Net Benefit ($)')
        axs[1, 1].set_title('Business Impact Over Time')
        axs[1, 1].legend()
        
        plt.tight_layout()
        return fig': 200,     # Cost of missing fraud
                    'tn_benefit': 0     # Benefit of not investigating non-fraud
                }
        """
        Returns:
            Dictionary of evaluation metrics
        """
        # If y_prob is provided, recalculate y_pred using threshold
        if y_prob is not None:
            y_pred = (y_prob >= threshold).astype(int)
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'threshold': threshold
        }
        
        # Add AUC metrics if probabilities are provided
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Add detection rate and false alarm rate
        metrics['detection_rate'] = metrics['recall']  # Same as recall/TPR
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR
        
        # Add business metrics
        # Default cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = {
                'tp_benefit': 200,  # Benefit of catching fraud
                'fp_cost': 10,      # Cost of investigating non-fraud
                'fn_cost': 200,     # Cost of missing fraud
                'tn_benefit': 0     # Benefit of correctly not flagging legitimate transactions
            }
        
        # Calculate net benefit using cost matrix
        net_benefit = (
            tp * cost_matrix['tp_benefit'] - 
            fp * cost_matrix['fp_cost'] - 
            fn * cost_matrix['fn_cost'] +
            tn * cost_matrix['tn_benefit']
        )
        metrics['net_benefit'] = net_benefit
        
        # Calculate return on investment (ROI)
        total_cost = fp * cost_matrix['fp_cost'] + fn * cost_matrix['fn_cost']
        total_benefit = tp * cost_matrix['tp_benefit'] + tn * cost_matrix['tn_benefit']
        
        # Avoid division by zero
        if total_cost > 0:
            metrics['roi'] = (total_benefit - total_cost) / total_cost
        else:
            metrics['roi'] = float('inf') if total_benefit > 0 else 0
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                              criterion: str = 'f1', 
                              cost_matrix: Optional[Dict[str, float]] = None) -> float:
        """
        Find the optimal threshold for classification based on a criterion.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            criterion: Criterion to optimize ('f1', 'precision_recall_balance', 'cost')
            cost_matrix: Cost matrix for business impact calculation (for 'cost' criterion)
            
        Returns:
            Optimal threshold value
        """
        # Get precision and recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        if criterion == 'f1':
            # Calculate F1 score for each threshold
            f1_scores = []
            for i in range(len(precision)):
                if precision[i] + recall[i] > 0:  # Avoid division by zero
                    f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                else:
                    f1 = 0
                f1_scores.append(f1)
                
            # Find threshold that maximizes F1 score
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])  # Exclude the last value
                return thresholds[best_idx]
                
        elif criterion == 'precision_recall_balance':
            # Find threshold where precision and recall are closest
            differences = np.abs(precision - recall)
            best_idx = np.argmin(differences[:-1])  # Exclude the last value
            return thresholds[best_idx]
            
        elif criterion == 'cost':
            # Define a cost function using the provided cost matrix
            if cost_matrix is None:
                cost_matrix = {
                    'tp_benefit': 200,
                    'fp_cost': 10,
                    'fn_cost': 200,
                    'tn_benefit': 0
                }
                
            costs = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                # Calculate net cost (negative of net benefit)
                net_cost = -(
                    tp * cost_matrix['tp_benefit'] - 
                    fp * cost_matrix['fp_cost'] - 
                    fn * cost_matrix['fn_cost'] +
                    # tn * cost_matrix['tn_
                )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample data with PCA features V1-V28, Amount, Time, and Class
    np.random.seed(42)
    n_samples = 1000
    n_features = 28  # V1 to V28
    
    # Create dataset with PCA features
    X = np.random.randn(n_samples, n_features)
    
    # Add Amount feature (transaction amount)
    amount = np.abs(np.random.lognormal(mean=5, sigma=2, size=n_samples))
    
    # Add Time column (seconds elapsed)
    time_col = np.arange(n_samples) * 100  # 100 seconds between transactions
    
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
    print(f"Fraud distribution: {np.bincount(y.astype(int))}")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Class', axis=1).values, 
        df['Class'].values,
        test_size=0.2, 
        random_state=42,
        stratify=df['Class']
    )
    
    # Train a simple model for demonstration
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = FraudModelEvaluator.calculate_metrics(y_test, y_pred, y_prob)
    
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Find optimal threshold
    optimal_threshold = FraudModelEvaluator.find_optimal_threshold(y_test, y_prob)
    print(f"\nOptimal threshold (F1 criterion): {optimal_threshold:.4f}")
    
    # Calculate metrics with optimal threshold
    optimal_y_pred = (y_prob >= optimal_threshold).astype(int)
    optimal_metrics = FraudModelEvaluator.calculate_metrics(y_test, optimal_y_pred, y_prob, optimal_threshold)
    
    print("\nMetrics with optimal threshold:")
    print(f"F1-Score: {optimal_metrics['f1_score']:.4f}")
    print(f"Precision: {optimal_metrics['precision']:.4f}")
    print(f"Recall: {optimal_metrics['recall']:.4f}")
    
    # Generate evaluation report
    report = FraudModelEvaluator.generate_report(optimal_metrics, "evaluation_report.txt")
    print("\nReport saved to evaluation_report.txt")
    
    # Create evaluation plots
    print("\nCreating evaluation plots...")
    fig = FraudModelEvaluator.plot_metrics(y_test, y_prob, optimal_threshold)
    plt.savefig("evaluation_plots.png")
    plt.close(fig)
    print("Plots saved to evaluation_plots.png")
    
    # Feature importance plot
    print("\nCreating feature importance plot...")
    importances = model.feature_importances_
    feature_names = list(df.drop('Class', axis=1).columns)
    fig = FraudModelEvaluator.feature_importance_plot(feature_names, importances, top_n=10)
    plt.savefig("feature_importance.png")
    plt.close(fig)
    print("Feature importance plot saved to feature_importance.png")
    
    # Model comparison
    print("\nComparing models...")
    
    # Train a second model for comparison
    from sklearn.linear_model import LogisticRegression
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_y_pred = lr_model.predict(X_test)
    lr_y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for the second model
    lr_metrics = FraudModelEvaluator.calculate_metrics(y_test, lr_y_pred, lr_y_prob)
    
    # Compare models
    model_results = {
        'Random Forest': optimal_metrics,
        'Logistic Regression': lr_metrics
    }
    
    fig = FraudModelEvaluator.compare_models(model_results)
    plt.savefig("model_comparison.png")
    plt.close(fig)
    print("Model comparison plot saved to model_comparison.png")
    
    # Evaluate over time (simulated data)
    print("\nEvaluating model performance over time...")
    
    # Create simulated data for different time periods
    time_periods = 5
    X_over_time = []
    y_over_time = []
    time_labels = [f"Period {i+1}" for i in range(time_periods)]
    
    # Simulate concept drift by gradually changing the relationship between features and target
    for i in range(time_periods):
        # Create new random data
        X_period = np.random.randn(n_samples, n_features)
        
        # Modify the relationship slightly for each period
        drift_factor = 0.2 * i  # Increasing drift
        
        # Create imbalanced target with changing fraud patterns
        y_period = np.zeros(n_samples)
        
        # Base fraud probability on some features with increasing noise
        fraud_scores = X_period[:, 0] - X_period[:, 1] + drift_factor * np.random.randn(n_samples)
        fraud_indices = np.where(fraud_scores > 2.0)[0]
        
        # Ensure at least 1% fraud
        if len(fraud_indices) < n_samples * 0.01:
            additional_frauds = np.random.choice(
                np.setdiff1d(np.arange(n_samples), fraud_indices),
                int(n_samples * 0.01) - len(fraud_indices),
                replace=False
            )
            fraud_indices = np.concatenate([fraud_indices, additional_frauds])
        
        y_period[fraud_indices] = 1
        
        X_over_time.append(X_period)
        y_over_time.append(y_period)
    
    # Create model predictors
    def rf_predictor(X):
        return model.predict_proba(X)[:, 1]
    
    def lr_predictor(X):
        return lr_model.predict_proba(X)[:, 1]
    
    # Evaluate models over time
    fig = FraudModelEvaluator.evaluate_over_time([rf_predictor, lr_predictor], X_over_time, y_over_time, time_labels)
    plt.savefig("performance_over_time.png")
    plt.close(fig)
    print("Performance over time plot saved to performance_over_time.png")