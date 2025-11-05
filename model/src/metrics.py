"""
Metrics computation: accuracy, AUC, precision, recall, confusion matrix
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def bin_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities [0,1]
        thr: Classification threshold
    
    Returns:
        Dictionary with acc, auc, prec, rec
    """
    y_pred = (y_prob >= thr).astype(int)
    
    # Handle edge cases
    metrics = {}
    
    # Accuracy
    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    
    # AUC (handles all-one or all-zero cases)
    try:
        if len(np.unique(y_true)) == 1:
            # All labels same class - AUC undefined
            metrics["auc"] = float('nan')
        else:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float('nan')
    
    # Precision (handles zero-division)
    try:
        metrics["prec"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["prec"] = 0.0
    
    # Recall (handles zero-division)
    try:
        metrics["rec"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["rec"] = 0.0
    
    return metrics


def confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Compute 2x2 confusion matrix.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities [0,1]
        thr: Classification threshold
    
    Returns:
        2x2 numpy array [[TN, FP], [FN, TP]]
    """
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm
