# ============================================================================
# File: config.py
# Description: Configuration settings for EuroSAT classification
# ============================================================================

class Config:
    """Configuration class for EuroSAT dataset parameters"""
    
    IMG_SIZE = (64, 64)
    NUM_CLASSES = 10
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    SPECTRAL_BANDS = [
        'Coastal Aerosol', 'Blue', 'Green', 'Red', 'Red Edge 1',
        'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4',
        'Water Vapor', 'SWIR Cirrus', 'SWIR 1', 'SWIR 2'
    ]
    NUM_BANDS = 13
    RANDOM_STATE = 42
    
    def __init__(self):
        print("Config Setup completed...")