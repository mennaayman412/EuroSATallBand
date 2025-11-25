# ============================================================================
# File: model.py
# Description: SpectrumNet model architecture and training
# ============================================================================

import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import Config


class SpectrumNetModel:
    """SpectrumNet architecture for satellite image classification"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.training_time = 0
        
    def _residual_block(self, X, filters):
        """
        Create a residual block with concatenated convolutions
        
        Args:
            X: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        shortcut = X
        
        X1 = Conv2D(filters, (1, 1), padding='same')(X)
        X2 = Conv2D(filters, (3, 3), padding='same')(X)
        X = Concatenate()([X1, X2])
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        
        if shortcut.shape[-1] == X.shape[-1]:
            X = Add()([shortcut, X])
            
        return X
    
    def build_model(self, num_classes):
        """
        Build SpectrumNet Lite architecture
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Keras Model
        """
        input_shape = (
            self.config.IMG_SIZE[0], 
            self.config.IMG_SIZE[1], 
            self.config.NUM_BANDS
        )
        
        X_input = Input(input_shape)
        X = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(X_input)
        
        X = self._residual_block(X, 32)
        X = MaxPooling2D((2, 2))(X)
        
        X = self._residual_block(X, 64)
        X = MaxPooling2D((2, 2))(X)
        
        X = GlobalAveragePooling2D()(X)
        X = Dense(128, activation='relu')(X)
        X = Dropout(0.4)(X)
        X = Dense(num_classes, activation='softmax')(X)
        
        self.model = Model(inputs=X_input, outputs=X)
        return self.model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy'):
        """Compile the model with specified parameters"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
    
    def train(self, X_train, Y_train, X_val, Y_val, 
              epochs=40, batch_size=16, checkpoint_path='best_model.h5'):
        """
        Train the model with callbacks
        
        Args:
            X_train, Y_train: Training data
            X_val, Y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            
        Returns:
            History object
        """
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
        
        earlystop = EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, earlystop],
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def load_best_model(self, model_path='best_model.h5'):
        """Load the best saved model"""
        self.model = load_model(model_path)
        return self.model
    
    def get_model_params(self):
        """Get total number of model parameters"""
        return self.model.count_params()
