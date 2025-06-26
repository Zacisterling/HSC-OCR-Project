
import os
import numpy as np
from preprocessing import ImagePreprocessor
from model import OCRModel
import re

def extract_character_from_filename(filename):
    
    #Extract character label from your filename format
    
    # Split at underscores and get the last part before .png
    parts = filename.replace('.png', '').split('_')
    character = parts[-1]
    
    # Do special symbol 
    symbol_mapping = {
        'exclmark': '!',
        'at': '@', 
        'hash': '#',
        'dollar': '$',
        'percent': '%',
        'ampersand': '&',
        'asterisk': '*',
        'plus': '+',
        'minus': '-',
        'quesmark': '?',
        'lessthan': '<',
        'greaterthan': '>'
    }
    
    return symbol_mapping.get(character, character)

def load_training_data():
    """Load and prepare all character images for training"""
    print(" Loading character images...")
    
    # start preprocessor
    preprocessor = ImagePreprocessor(target_size=(28, 28))
    
    # Get all the image files
    image_dir = "../comprehensive_character_images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    print(f" Found {len(image_files)} character images")
    
    # Process all the images
    images = []
    labels = []
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:  # indicator of how much Progress
            print(f"Processing {i}/{len(image_files)} images...")
        
        # process image
        image_path = os.path.join(image_dir, filename)
        processed_image = preprocessor.process_image(image_path, show_steps=False)
        
        if processed_image is not None:
            #get character info from filename
            character = extract_character_from_filename(filename)
            
            images.append(processed_image)
            labels.append(character)
    
    print(f" Successfully loaded {len(images)} images")
    print(f" Unique characters: {len(set(labels))}")
    print(f" Character samples: {sorted(set(labels))[:10]}...")
    
    return images, labels

def train_ocr_model():
    """Main training function"""
    print(" Starting OCR Model Training")
    print("=" * 50)
    
    #Load training data
    images, labels = load_training_data()
    
    #create the model
    num_classes = len(set(labels))
    print(f"\n Creating model for {num_classes} character classes")
    
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1))
    
    # Create CNN
    ocr_model.create_simple_cnn()
    ocr_model.show_model_summary()
    
    #prep data for training
    X_train, X_test, y_train, y_test = ocr_model.prepare_data(images, labels)
    
    #train
    print("\n Starting training...")
    ocr_model.train_model(
        X_train, y_train, 
        X_test, y_test,
        epochs=30,  # Start with 30 epochs
        batch_size=32
    )
    
    #plot training results
    print("\n Showing training history...")
    ocr_model.plot_training_history()
    
    #save the trained model
    model_name = "ocr_simple_cnn_v1"
    ocr_model.save_model(model_name)
    
    #test a some predictions to see how accurate it currently is (expand later)
    print("\n Testing some predictions...")
    test_predictions(ocr_model, X_test, y_test, labels)
    
    return ocr_model

def test_predictions(model, X_test, y_test, original_labels):
    """Test model predictions on a few samples"""
    
    #test  first 5 
    for i in range(min(5, len(X_test))):
        image = X_test[i]
        
        # get prediction
        predictions = model.predict_character(image, top_k=3)
        
        #get actual label
        actual_idx = np.argmax(y_test[i])
        actual_char = model.label_encoder.inverse_transform([actual_idx])[0]
        
        print(f"\n Test {i+1}:")
        print(f"   Actual: '{actual_char}'")
        print(f"   Predictions:")
        for j, (char, confidence) in enumerate(predictions):
            print(f"     {j+1}. '{char}' - {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    #Start training
    trained_model = train_ocr_model()
    
    print("\n Training completed!")
    print(" Model saved and ready for testing!")