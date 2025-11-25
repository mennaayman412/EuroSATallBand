# ============================================================================
# File: pipeline.py
# Description: Main pipeline orchestrator
# ============================================================================

from config import Config
from data_loader import DataLoader
from model import SpectrumNetModel
from evaluator import ModelEvaluator


class EuroSATClassifier:
    """Main pipeline class that orchestrates the entire classification workflow"""
    
    def __init__(self, dataset_path):
        self.config = Config()
        self.dataset_path = dataset_path
        self.data_loader = DataLoader(self.config)
        self.model_builder = None
        self.evaluator = None
        
    def load_and_prepare_data(self):
        """Load data and prepare for training"""
        # Load raw data
        X, Y = self.data_loader.load_satellite_data(self.dataset_path)
        
        # Encode labels
        Y_onehot = self.data_loader.prepare_labels(Y)
        
        # Split data
        self.X_train, self.X_val, self.X_test, \
        self.Y_train, self.Y_val, self.Y_test = self.data_loader.split_data(X, Y_onehot)
        
    def build_and_train(self, epochs=40, batch_size=16):
        """Build model and train"""
        # Build model
        self.model_builder = SpectrumNetModel(self.config)
        self.model_builder.build_model(num_classes=self.config.NUM_CLASSES)
        self.model_builder.model.summary()
        
        # Compile
        self.model_builder.compile_model()
        
        # Train
        self.model_builder.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
    def evaluate_model(self):
        """Evaluate trained model"""
        # Load best model
        best_model = self.model_builder.load_best_model()
        
        # Create evaluator
        self.evaluator = ModelEvaluator(best_model)
        
        # Evaluate
        self.evaluator.evaluate(self.X_test, self.Y_test)
        
        # Predict
        predictions = self.evaluator.predict(self.X_test)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(self.Y_test, predictions)
        
        # Print results
        total_params = self.model_builder.get_model_params()
        self.evaluator.print_metrics(
            metrics, 
            total_params, 
            self.model_builder.training_time
        )
        
        return metrics
    
    def run_full_pipeline(self, epochs=40, batch_size=16):
        """Execute complete pipeline from data loading to evaluation"""
        print("Starting EuroSAT Classification Pipeline...")
        
        self.load_and_prepare_data()
        self.build_and_train(epochs=epochs, batch_size=batch_size)
        metrics = self.evaluate_model()
        
        print("\nPipeline completed successfully!")
        return metrics