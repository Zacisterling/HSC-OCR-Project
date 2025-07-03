#!/usr/bin/env python3
"""
HSC OCR Project - EMNIST Dataset Integration
Loads and processes EMNIST handwritten characters for Level 5 complexity
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import string

def load_emnist_dataset(save_dir="../emnist_character_images"):
    """
    Load EMNIST dataset and convert to our image format
    
    EMNIST contains handwritten characters that may need preprocessing:
    - Images might be rotated/reflected
    - Need conversion to our 28x28 format
    - Need proper character mapping
    """
    print("🔤 Loading EMNIST Dataset for Level 5 Handwritten Characters")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load EMNIST ByClass dataset (contains A-Z, a-z, 0-9)
        print("📊 Loading EMNIST ByClass dataset...")
        ds, info = tfds.load('emnist/byclass', 
                            split='train', 
                            shuffle_files=True,
                            as_supervised=True,
                            with_info=True)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Total samples: {info.splits['train'].num_examples}")
        print(f"🔤 Number of classes: {info.features['label'].num_classes}")
        
        # EMNIST character mapping (based on EMNIST paper)
        # Classes 0-9: digits 0-9
        # Classes 10-35: uppercase A-Z  
        # Classes 36-61: lowercase a-z
        character_mapping = create_emnist_character_mapping()
        
        # Process and save images
        process_emnist_images(ds, character_mapping, save_dir, samples_per_class=1000)
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading EMNIST: {e}")
        print("💡 Make sure tensorflow-datasets is installed: pip install tensorflow-datasets")
        return False

def create_emnist_character_mapping():
    """
    Create mapping from EMNIST class indices to characters
    Based on EMNIST ByClass specification
    """
    mapping = {}
    
    # Digits 0-9 (classes 0-9)
    for i in range(10):
        mapping[i] = str(i)
    
    # Uppercase A-Z (classes 10-35)
    for i in range(26):
        mapping[i + 10] = chr(ord('A') + i)
    
    # Lowercase a-z (classes 36-61)
    for i in range(26):
        mapping[i + 36] = chr(ord('a') + i)
    
    print(f"📋 Character mapping created: {len(mapping)} classes")
    print(f"   Sample mappings: {dict(list(mapping.items())[:10])}")
    
    return mapping

def process_emnist_images(dataset, character_mapping, save_dir, samples_per_class=1000):
    """
    Process EMNIST images and save in our format
    """
    print(f"🔄 Processing EMNIST images ({samples_per_class} per class)...")
    
    # Count samples per class
    class_counts = {}
    processed_count = 0
    
    # Create class directories
    for class_idx, char in character_mapping.items():
        if char.isupper():
            char_dir = f"HANDWRITTEN_UPPER_{char}"
        elif char.islower():
            char_dir = f"HANDWRITTEN_LOWER_{char}"
        else:
            char_dir = f"HANDWRITTEN_{char}"
        
        class_dir = os.path.join(save_dir, char_dir)
        os.makedirs(class_dir, exist_ok=True)
        class_counts[class_idx] = 0
    
    # Process dataset
    for image, label in dataset:
        label_int = label.numpy()
        
        # Skip if we have enough samples for this class
        if class_counts.get(label_int, 0) >= samples_per_class:
            continue
        
        if label_int in character_mapping:
            character = character_mapping[label_int]
            
            # Process image
            processed_image = preprocess_emnist_image(image.numpy())
            
            if processed_image is not None:
                # Save image
                save_emnist_image(processed_image, character, class_counts[label_int], save_dir)
                class_counts[label_int] += 1
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"  📊 Processed {processed_count} images...")
        
        # Check if we have enough samples for all classes
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    
    # Summary
    total_saved = sum(class_counts.values())
    print(f"\n✅ EMNIST processing complete!")
    print(f"📊 Total images saved: {total_saved}")
    print(f"🔤 Classes processed: {len([c for c in class_counts.values() if c > 0])}")
    
    return total_saved

def preprocess_emnist_image(image):
    """
    Preprocess EMNIST image to match our format
    
    EMNIST images may need:
    - Transposition (rotate/flip)
    - Normalization
    - Format conversion
    """
    try:
        # EMNIST images are 28x28 but may be rotated
        # According to EMNIST paper, images need to be transposed and flipped
        image = np.transpose(image, (1, 0, 2)) if len(image.shape) == 3 else np.transpose(image)
        image = np.flipud(image)  # Flip vertically
        
        # Ensure 28x28 grayscale
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Take first channel if RGB
        
        # Normalize to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Ensure proper size
        if image.shape != (28, 28):
            # Resize if needed
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
            image = np.array(pil_image)
        
        return image
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def save_emnist_image(image, character, count, save_dir):
    """
    Save processed EMNIST image in our filename format
    """
    try:
        # Determine directory name
        if character.isupper():
            dir_name = f"HANDWRITTEN_UPPER_{character}"
        elif character.islower():
            dir_name = f"HANDWRITTEN_LOWER_{character}"
        else:
            dir_name = f"HANDWRITTEN_{character}"
        
        # Create filename
        filename = f"HANDWRITTEN_emnist_28_{character}_{count:04d}.png"
        filepath = os.path.join(save_dir, dir_name, filename)
        
        # Save image
        pil_image = Image.fromarray(image, mode='L')
        pil_image.save(filepath)
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving image {character}_{count}: {e}")
        return False

def verify_emnist_integration():
    """
    Verify EMNIST dataset was processed correctly
    """
    print("\n🔍 Verifying EMNIST Integration")
    print("=" * 30)
    
    save_dir = "../emnist_character_images"
    
    if not os.path.exists(save_dir):
        print(f"❌ EMNIST directory not found: {save_dir}")
        return False
    
    # Count classes and images
    class_dirs = [d for d in os.listdir(save_dir) 
                  if os.path.isdir(os.path.join(save_dir, d)) and 'HANDWRITTEN' in d]
    
    total_images = 0
    class_breakdown = {}
    
    for class_dir in class_dirs:
        class_path = os.path.join(save_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.endswith('.png')]
        total_images += len(images)
        
        # Extract character type
        if 'UPPER' in class_dir:
            char_type = 'Uppercase'
        elif 'LOWER' in class_dir:
            char_type = 'Lowercase'
        else:
            char_type = 'Digits/Symbols'
        
        if char_type not in class_breakdown:
            class_breakdown[char_type] = 0
        class_breakdown[char_type] += 1
    
    print(f"📊 EMNIST Integration Summary:")
    print(f"   Total handwritten classes: {len(class_dirs)}")
    print(f"   Total handwritten images: {total_images}")
    print(f"   Class breakdown: {class_breakdown}")
    
    # Check if we have the expected classes (62 total: 10 digits + 26 upper + 26 lower)
    expected_classes = 62
    if len(class_dirs) >= expected_classes:
        print(f"✅ All expected character classes present!")
        print(f"🎯 Level 5 handwritten requirement: COMPLETE")
    else:
        print(f"⚠️  Expected {expected_classes} classes, found {len(class_dirs)}")
    
    return len(class_dirs) >= expected_classes

if __name__ == "__main__":
    print("🎓 HSC OCR Project - EMNIST Integration for Level 5")
    print("Adding handwritten characters for maximum complexity")
    print("=" * 70)
    
    # Load and process EMNIST
    success = load_emnist_dataset()
    
    if success:
        # Verify integration
        verification_success = verify_emnist_integration()
        
        if verification_success:
            print("\n🎉 EMNIST integration successful!")
            print("🎯 Ready for Level 5 (120% weighting) training!")
            print("\n📋 Next steps:")
            print("1. Update training script to include handwritten data")
            print("2. Train comprehensive model with all character types")
            print("3. Achieve 85%+ accuracy target")
        else:
            print("\n⚠️  EMNIST integration needs attention")
    else:
        print("\n❌ EMNIST integration failed")
        print("💡 Check tensorflow-datasets installation and internet connection")