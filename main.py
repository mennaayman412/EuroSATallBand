# ============================================================================
# File: main.py
# Description: Entry point for running the classification pipeline
# ============================================================================
from pipeline import EuroSATClassifier


def main():
    """Main function to run EuroSAT classification"""
    
    # Set dataset path
    DATASET_PATH = "/kaggle/input/eurosat-dataset/EuroSATallBands"
    
    # Create classifier instance
    classifier = EuroSATClassifier(DATASET_PATH)
    
    # Run full pipeline
    metrics = classifier.run_full_pipeline(epochs=40, batch_size=16)
    
    return metrics


if __name__ == "__main__":
    main()