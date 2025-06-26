
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class OCRModel:
    """CNN-based OCR model for character recognition"""
    
    def __init__(self, num_classes=62, input_shape=(28, 28, 1)):
        """
        Initialize OCR model
        Args:
            num_classes (int): Number of character classes (A-Z, a-z, 0-9 = 62)
            input_shape (tuple): Input image shape
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        print(f"OCR Model initialized")
        print(f"Classes: {num_classes}, Input shape: {input_shape}")
    
    def create_simple_cnn(self):
        """Create a simple CNN model for baseline testing"""
        print("🏗️ Building Simple CNN Model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            #First convolutional block (emphases important features of the image and ingores other ones so it can be more accurate i think)
            layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Second convolutional block (said before)
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Flatten and dense layers (flattens refers to turning 2d mapping into a more simple 1D to get data ready for classifcation in other hidden then output layers)
            layers.Flatten(name='flatten'),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Simple CNN model created!")
        return model
    
    def create_advanced_cnn(self):
        #reate CNN model
        print("Building Advanced CNN Model for Level 5...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block (said before)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Second convolutional block (said before)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Third convolutional block (said before)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Dense layers (said before)
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout4'),
            layers.Dense(256, activation='relu', name='dense2'),
            layers.Dropout(0.5, name='dropout5'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        #compile with advanced optimizer (?)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        print("Advanced CNN model created!")
        return model
    
    def show_model_summary(self):
        """Display model architecture"""
        if self.model is None:
            print("No model created yet!")
            return
        
        print("\n Model Architecture:")
        print("=" * 50)
        self.model.summary()
        
        #calc total parameters (output layer)
        total_params = self.model.count_params()
        print(f"\n Total Parameters: {total_params:,}")
    
    def prepare_data(self, images, labels, apply_augmentation=False):
        
        #Prepare data for training
        #Args:
            #images: List of preprocessed images
            #labels: List of character labels
        #Returns:
            #X_train, X_test, y_train, y_test: Split and prepared data
        
        print("Preparing training data...")
        
        #convert to numpy arrays
        X = np.array(images)
        
        #change shape ot work for CNN (add channel dimension)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        #encode labels (?)
        y_encoded = self.label_encoder.fit_transform(labels)
        y_categorical = keras.utils.to_categorical(y_encoded, self.num_classes)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y_categorical.shape}")
        print(f"Unique classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_[:10]}...")  # Show first 10
        
        #split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )

        # Add this after your train_test_split line:
        if apply_augmentation:
            print("Applying Level 4 rotation augmentation...")
            X_train = self.apply_rotation_augmentation(X_train)
            y_train = np.concatenate([y_train, y_train])
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Train the CNN model
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            print("No model created! Call create_simple_cnn() or create_advanced_cnn() first")
            return
        
        print(f"Training model for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        #train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        #show final accuracy as perecentage output
        final_accuracy = max(self.history.history['val_accuracy'])
        print(f"Best Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        #accuracy plot graph
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        #loss plot graph
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_character(self, image, top_k=3):
        
        #Predict character from preprocessed image
        #Args:
            #image: Preprocessed image (28, 28)
            #top_k: Return top K predictions
        #Returns:
            #predictions: List of (character, confidence) tuples
        
        if self.model is None:
            print("No trained model available!")
            return []
        
        #get images ready
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
        
        #finalise prediction based on models (output layer kinda)
        predictions = self.model.predict(image, verbose=0)
        
        #get most accurate K predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            character = self.label_encoder.inverse_transform([idx])[0]
            confidence = predictions[0][idx]
            results.append((character, confidence))
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        #create place for models (in case i change models file to have multiple different models saved)
        os.makedirs("../models", exist_ok=True)
        
        #save the model to said file
        model_path = f"../models/{filepath}.keras"
        self.model.save(model_path)
        
        # Save label encoder
        encoder_path = f"../models/{filepath}_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f" Model saved to {model_path}")
        print(f" Label encoder saved to {encoder_path}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_path = f"../models/{filepath}.keras"
        encoder_path = f"../models/{filepath}_encoder.pkl"
        
        try:
            #Load model
            self.model = keras.models.load_model(model_path)
            
            #Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print(f" Model loaded from {model_path}")
            print(f" Label encoder loaded from {encoder_path}")
            
        except Exception as e:
            print(f" Error loading model: {e}")

    def apply_rotation_augmentation(self, X_train):
        """Apply ±20° rotation augmentation for Level 4 requirements"""
        from scipy.ndimage import rotate
        import random
        
        augmented_images = []
        
        for img in X_train:
            # Original image
            augmented_images.append(img)
            
            # Create 1 rotated version per original
            angle = random.uniform(-20, 20)  # Random angle between -20 and +20 degrees
            
            # Handle both (28,28) and (28,28,1) shapes
            if len(img.shape) == 3:
                rotated_img = rotate(img[:,:,0], angle, reshape=False, cval=1.0)
                rotated_img = rotated_img.reshape(img.shape)
            else:
                rotated_img = rotate(img, angle, reshape=False, cval=1.0)
            
            # Ensure values stay in [0,1] range
            rotated_img = np.clip(rotated_img, 0, 1)
            
            augmented_images.append(rotated_img)
        
        print(f"Augmentation complete: {len(X_train)} → {len(augmented_images)} images")
        return np.array(augmented_images)

# Test the model class
if __name__ == "__main__":
    print(" Testing OCR Model Class")
    print("=" * 40)
    
    # Create model instance
    ocr_model = OCRModel()
    
    # Create simple model
    ocr_model.create_simple_cnn()
    
    # Show model summary
    ocr_model.show_model_summary()
    
    print("\n OCR Model class ready!")
    print(" Next: Load your character images and train the model!")