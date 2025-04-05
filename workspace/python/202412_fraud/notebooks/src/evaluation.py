import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FraudEvaluation')

class FraudEvaluator:
    """
    A utility class for evaluating fraud detection models with consistent metrics
    and visualizations that can be used with any model type.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None, threshold=0.5):
        """
        Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            threshold: Classification threshold used
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure y_pred is binary
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
        # Assuming each TP saves $100, each FP costs $10, and each FN costs $500
        fraud_savings = tp * 200
        investigation_costs = fp * 10
        missed_fraud_costs = fn * 200
        metrics['net_savings'] = fraud_savings - investigation_costs - missed_fraud_costs
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true, y_prob, criterion='f1'):
        """
        Find the optimal threshold for classification based on a criterion.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            criterion: Criterion to optimize ('f1', 'precision_recall_balance', 'cost')
            
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
            # Define a cost function (example: 10*FP + 500*FN)
            costs = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                # Cost function: each FP costs $10, each FN costs $500
                cost = 10 * fp + 500 * fn
                costs.append(cost)
                
            best_idx = np.argmin(costs)
            return thresholds[best_idx]
            
        # Default
        return 0.5
    
    @staticmethod
    def plot_metrics(y_true, y_prob, threshold=0.5, figsize=(20, 16)):
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
    def compare_models(model_results, figsize=(14, 10)):
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
        
        # 4. Business metric - net savings
        net_savings = []
        for model_name, results in model_results.items():
            net_savings.append(results.get('net_savings', 0))
        
        axs[1, 1].bar(x, net_savings, width, label='Net Savings')
        
        axs[1, 1].set_xlabel('Models')
        axs[1, 1].set_ylabel('Value ($)')
        axs[1, 1].set_title('Business Impact - Net Savings')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(models)
        
        plt.tight_layout()
        return fig
        
    @staticmethod
    def feature_importance_plot(feature_names, importances, top_n=20, figsize=(12, 10)):
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
    def generate_report(evaluation_results, output_file=None):
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
        if 'net_savings' in evaluation_results:
            report += f"Net Savings: ${evaluation_results['net_savings']:.2f}\n"
            
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
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report