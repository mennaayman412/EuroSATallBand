# ============================================================================
# File: evaluator.py
# Description: Model evaluation and metrics calculation
# ============================================================================

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef
)


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def __init__(self, model):
        self.model = model
        self.prediction_time = 0
        
    def evaluate(self, X_test, Y_test, batch_size=16):
        """
        Evaluate model on test set
        
        Args:
            X_test, Y_test: Test data
            batch_size: Batch size for evaluation
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        test_loss, test_acc = self.model.evaluate(X_test, Y_test, batch_size=batch_size)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc
    
    def predict(self, X_test):
        """
        Make predictions on test set
        
        Args:
            X_test: Test features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.prediction_time = time.time() - start_time
        print(f"Prediction time: {self.prediction_time:.2f} seconds")
        return predictions
    
    def calculate_metrics(self, Y_test, pred_test):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            Y_test: True labels (one-hot)
            pred_test: Predicted probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        y_pred_labels = np.argmax(pred_test, axis=1)
        y_true_labels = np.argmax(Y_test, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_true_labels, y_pred_labels),
            'precision': precision_score(y_true_labels, y_pred_labels, average='weighted'),
            'recall': recall_score(y_true_labels, y_pred_labels, average='weighted'),
            'f1_score': f1_score(y_true_labels, y_pred_labels, average='weighted'),
            'mcc': matthews_corrcoef(y_true_labels, y_pred_labels)
        }
        
        return metrics
    
    def print_metrics(self, metrics, total_params, training_time):
        """Print all evaluation metrics in formatted manner"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-Score:        {metrics['f1_score']:.4f}")
        print(f"MCC:             {metrics['mcc']:.4f}")
        print("-"*50)
        print(f"Model Complexity: {total_params:,} parameters")
        print(f"Training Time:    {training_time:.2f} seconds")
        print(f"Prediction Time:  {self.prediction_time:.2f} seconds")
        print("="*50)
