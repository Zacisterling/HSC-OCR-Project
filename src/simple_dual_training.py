
import os
import numpy as np
from preprocessing import ImagePreprocessor
from model import OCRModel
from datetime import datetime

def load_font_dataset():
    """Load only font images (exclude handwritten)"""
    print("📝 Loading Font Dataset...")
    
    preprocessor = ImagePreprocessor(target_size=(28, 28))
    image_dir = "../comprehensive_character_images"
    
    if not os.path.exists(image_dir):
        print(f"❌ Font dataset not found: {image_dir}")
        return None, None
    
    # Get only font images (exclude HANDWRITTEN)
    image_files = [f for f in os.listdir(image_dir) 
                  if f.endswith('.png') and 'HANDWRITTEN' not in f]
    
    print(f"📊 Found {len(image_files)} font images")
    
    images = []
    labels = []
    
    for i, filename in enumerate(image_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(image_files)} font images...")
        
        image_path = os.path.join(image_dir, filename)
        processed_image = preprocessor.process_image(image_path, show_steps=False)
        
        if processed_image is not None:
            character = extract_character_from_filename(filename)
            images.append(processed_image)
            labels.append(character)
    
    unique_chars = sorted(set(labels))
    print(f"✅ Font dataset loaded: {len(images)} images, {len(unique_chars)} classes")
    
    return images, labels

def load_handwritten_dataset():
    """Load only handwritten images from EMNIST"""
    print("✍️ Loading Handwritten Dataset...")
    
    preprocessor = ImagePreprocessor(target_size=(28, 28))
    image_dir = "../emnist_character_images"
    
    if not os.path.exists(image_dir):
        print(f"❌ Handwritten dataset not found: {image_dir}")
        print("💡 Run emnist_integration.py first!")
        return None, None
    
    images = []
    labels = []
    
    # Get all handwritten directories
    class_dirs = [d for d in os.listdir(image_dir) 
                 if os.path.isdir(os.path.join(image_dir, d)) and 'HANDWRITTEN' in d]
    
    processed_count = 0
    for class_dir in class_dirs:
        class_path = os.path.join(image_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        
        # Extract character from directory name
        character = extract_handwritten_character(class_dir)
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            processed_image = preprocessor.process_image(image_path, show_steps=False)
            
            if processed_image is not None:
                images.append(processed_image)
                labels.append(character)
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"  Processing {processed_count} handwritten images...")
    
    unique_chars = sorted(set(labels))
    print(f"✅ Handwritten dataset loaded: {len(images)} images, {len(unique_chars)} classes")
    
    return images, labels

def extract_character_from_filename(filename):
    """Extract character from font filename"""
    parts = filename.replace('.png', '').split('_')
    character = parts[-1]
    
    symbol_mapping = {
        'exclmark': '!', 'at': '@', 'hash': '#', 'dollar': '$',
        'percent': '%', 'ampersand': '&', 'asterisk': '*', 'plus': '+',
        'minus': '-', 'quesmark': '?', 'lessthan': '<', 'greaterthan': '>'
    }
    
    return symbol_mapping.get(character, character)

def extract_handwritten_character(class_dir):
    """Extract character from handwritten directory name"""
    # Format: HANDWRITTEN_UPPER_A, HANDWRITTEN_LOWER_a, HANDWRITTEN_5
    parts = class_dir.split('_')
    return parts[-1]  # Return the character

def train_font_model():
    """Train model on font dataset using existing model.py"""
    print("\n🎨 Training Font Model")
    print("=" * 40)
    
    # Load font dataset
    images, labels = load_font_dataset()
    if images is None:
        return False
    
    # Create model using existing OCRModel class
    num_classes = len(set(labels))
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1))
    ocr_model.create_simple_cnn()
    ocr_model.show_model_summary()
    
    # Prepare data with augmentation
    X_train, X_test, y_train, y_test = ocr_model.prepare_data(
        images, labels, apply_augmentation=True
    )
    
    # Train using existing training method
    ocr_model.train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32)
    
    # Save with descriptive name
    model_name = "font_model_"
    ocr_model.save_model(model_name)
    
    print(f"✅ Font model saved as: {model_name}")
    return True

def train_handwritten_model():
    """Train model on handwritten dataset using existing model.py"""
    print("\n Training Handwritten Model")
    print("=" * 40)
    
    # Load handwritten dataset
    images, labels = load_handwritten_dataset()
    if images is None:
        return False
    
    # Create model using existing OCRModel class
    num_classes = len(set(labels))
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1))
    ocr_model.create_advanced_cnn()
    ocr_model.show_model_summary()
    
    # Prepare data with augmentation
    X_train, X_test, y_train, y_test = ocr_model.prepare_data(
        images, labels, apply_augmentation=False
    )
    
    # Train using existing training method
    ocr_model.train_model(X_train, y_train, X_test, y_test, epochs=15, batch_size=32)
    
    # Save with descriptive name
    model_name = "handwritten_model"
    ocr_model.save_model(model_name)
    
    print(f"✅ Handwritten model saved as: {model_name}")
    return True

def test_model(model_name, test_image_path):
    """Test a saved model on an image"""
    print(f"\n🧪 Testing model: {model_name}")
    
    # Load model
    ocr_model = OCRModel()
    ocr_model.load_model(model_name)
    
    # Preprocess test image
    preprocessor = ImagePreprocessor(target_size=(28, 28))
    processed_image = preprocessor.process_image(test_image_path)
    
    if processed_image is None:
        print("❌ Failed to process test image")
        return
    
    # Make prediction
    predictions = ocr_model.predict_character(processed_image, top_k=3)
    
    print(f"🎯 Predictions for {os.path.basename(test_image_path)}:")
    for i, (char, confidence) in enumerate(predictions):
        print(f"  {i+1}. '{char}' - {confidence:.4f} ({confidence*100:.2f}%)")

def list_available_models():
    """List all available trained models"""
    models_dir = "../models"
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f.replace('.keras', '') for f in os.listdir(models_dir) 
                   if f.endswith('.keras')]
    
    return model_files

if __name__ == "__main__":
    print("🎓 HSC OCR Project - Simple Dual Training")
    print("Train separate models for fonts vs handwritten")
    print("=" * 50)
    
    while True:
        print("\n📋 Main Menu:")
        print("1. Train Font Model")
        print("2. Train Handwritten Model")
        print("3. Test Model")
        print("4. List Available Models")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            train_font_model()
        
        elif choice == "2":
            train_handwritten_model()
        
        elif choice == "3":
            models = list_available_models()
            if not models:
                print("❌ No trained models found!")
                continue
            
            print("\n📋 Available Models:")
            for i, model in enumerate(models):
                print(f"  {i+1}. {model}")
            
            try:
                model_idx = int(input("Select model number: ")) - 1
                model_name = models[model_idx]
                
                test_path = input("Enter test image path: ").strip()
                if os.path.exists(test_path):
                    test_model(model_name, test_path)
                else:
                    print("❌ Test image not found!")
            except (ValueError, IndexError):
                print("❌ Invalid model selection!")
        
        elif choice == "4":
            models = list_available_models()
            if models:
                print("\n📋 Available Models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("❌ No trained models found!")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-5.")