"""
Evaluation Metrics for Benchmark Comparison

Implements standard ML metrics for comparing trend prediction methods.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    ndcg_score
)
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple


class Evaluator:
    """Computes evaluation metrics for trend prediction"""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
        }
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics for trend scores"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'spearman': spearmanr(y_true, y_pred)[0],
            'pearson': pearsonr(y_true, y_pred)[0]
        }
    
    @staticmethod
    def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict[str, float]:
        """Compute ranking metrics"""
        # Get top-k indices by predicted score
        pred_top_k = np.argsort(y_pred)[-k:]
        true_top_k = np.argsort(y_true)[-k:]
        
        # Precision@K: how many of top-k predictions are in true top-k
        precision_at_k = len(set(pred_top_k) & set(true_top_k)) / k
        
        # NDCG@K
        try:
            ndcg = ndcg_score([y_true], [y_pred], k=k)
        except:
            ndcg = 0.0
        
        return {
            f'precision@{k}': precision_at_k,
            f'ndcg@{k}': ndcg
        }
    
    @staticmethod
    def full_evaluation(
        y_true_class: np.ndarray,
        y_pred_class: np.ndarray,
        y_true_score: np.ndarray,
        y_pred_score: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """Run full evaluation suite"""
        metrics = {}
        
        # Classification
        metrics.update(Evaluator.classification_metrics(y_true_class, y_pred_class))
        
        # Regression
        metrics.update(Evaluator.regression_metrics(y_true_score, y_pred_score))
        
        # Ranking
        metrics.update(Evaluator.ranking_metrics(y_true_score, y_pred_score, k=k))
        
        return metrics
    
    @staticmethod
    def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
        """Format results as a markdown table"""
        if not results:
            return "No results to display"
        
        # Get all metrics from first result
        metrics = list(next(iter(results.values())).keys())
        methods = list(results.keys())
        
        # Header
        header = "| Method | " + " | ".join(metrics) + " |"
        separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
        
        # Rows
        rows = []
        for method in methods:
            values = [f"{results[method].get(m, 0):.4f}" for m in metrics]
            rows.append(f"| {method} | " + " | ".join(values) + " |")
        
        return "\n".join([header, separator] + rows)
