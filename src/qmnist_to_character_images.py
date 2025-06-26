"""
Convert QMNIST to Character Images Format
Integrates QMNIST digits with your existing character image system
"""

import numpy as np
import os
from PIL import Image
from pathlib import Path

class QMNISTConverter:
    """Convert QMNIST data to character_images format"""
    
    def __init__(self, qmnist_path="../qmnist_dataset/qmnist_processed", 
                 output_path="enhanced_character_images"):
        self.qmnist_path = qmnist_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
    def load_qmnist_data(self):
        """Load processed QMNIST data"""
        try:
            print("📂 Loading QMNIST data...")
            
            images_path = os.path.join(self.qmnist_path, "images.npy")
            labels_path = os.path.join(self.qmnist_path, "labels.npy")
            
            images = np.load(images_path)
            labels = np.load(labels_path)
            
            if labels.ndim > 1:
                labels = labels.flatten()
            
            print(f"✅ Loaded {len(images):,} QMNIST images")
            return images, labels
            
        except Exception as e:
            print(f"❌ Error loading QMNIST: {e}")
            return None, None
    
    def copy_existing_characters(self, source_path="../character_images"):
        """Copy your existing character images to the new enhanced folder"""
        source = Path(source_path)
        
        if not source.exists():
            print(f"⚠️ Source path {source} not found")
            return 0
        
        print(f"📁 Copying existing character images from {source}")
        
        # Copy all PNG files
        png_files = list(source.glob("*.png"))
        copied = 0
        
        for png_file in png_files:
            dest_file = self.output_path / png_file.name
            
            # Copy the file
            img = Image.open(png_file)
            img.save(dest_file)
            copied += 1
        
        print(f"✅ Copied {copied} existing character images")
        return copied
    
    def convert_qmnist_to_images(self, samples_per_digit=1000):
        """Convert QMNIST digits to character image format"""
        images, labels = self.load_qmnist_data()
        
        if images is None:
            return 0
        
        print(f"🔄 Converting QMNIST digits (max {samples_per_digit} per digit)")
        
        # Group by digit
        converted = 0
        
        for digit in range(10):
            # Find all images of this digit
            digit_indices = np.where(labels == digit)[0]
            
            # Take up to samples_per_digit samples
            selected_indices = digit_indices[:samples_per_digit]
            
            print(f"📝 Converting digit {digit}: {len(selected_indices)} images")
            
            for i, idx in enumerate(selected_indices):
                # Get the image
                img_array = images[idx]
                
                # Convert to PIL Image
                # QMNIST is already 0-1, convert to 0-255 for saving
                img_array_uint8 = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array_uint8, mode='L')
                
                # Create filename in your format: FontName_style_size_character.png
                filename = f"QMNIST_digit_28_{digit}_{i:04d}.png"
                filepath = self.output_path / filename
                
                # Save the image
                img.save(filepath)
                converted += 1
        
        print(f"✅ Converted {converted} QMNIST digit images")
        return converted
    
    def analyze_enhanced_dataset(self):
        """Analyze the combined dataset"""
        png_files = list(self.output_path.glob("*.png"))
        
        print(f"\n📊 Enhanced Dataset Analysis:")
        print(f"   Total images: {len(png_files)}")
        
        # Count by character type
        digits = len([f for f in png_files if any(f"_{d}_" in f.name for d in "0123456789")])
        letters = len([f for f in png_files if any(f"_{c}_" in f.name for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")])
        symbols = len([f for f in png_files if any(symbol in f.name for symbol in ["exclmark", "at", "hash", "dollar"])])
        
        print(f"   Digits (0-9): ~{digits} images")
        print(f"   Letters (A-Z, a-z): ~{letters} images") 
        print(f"   Symbols: ~{symbols} images")
        
        # Extract unique characters
        characters = set()
        for filename in png_files:
            parts = filename.name.replace('.png', '').split('_')
            if len(parts) >= 3:
                char_part = parts[-2] if parts[-1].isdigit() else parts[-1]
                
                # Map symbol names back
                symbol_mapping = {
                    'exclmark': '!', 'at': '@', 'hash': '#', 'dollar': '$',
                    'percent': '%', 'ampersand': '&', 'asterisk': '*',
                    'plus': '+', 'minus': '-', 'quesmark': '?',
                    'lessthan': '<', 'greaterthan': '>'
                }
                
                actual_char = symbol_mapping.get(char_part, char_part)
                characters.add(actual_char)
        
        print(f"   Unique characters: {len(characters)}")
        print(f"   Character classes: {sorted(characters)}")
        
        return len(png_files), len(characters)
    
    def create_enhanced_dataset(self, qmnist_samples_per_digit=1000):
        """Create the complete enhanced dataset"""
        print("🚀 Creating Enhanced Character Dataset")
        print("=" * 50)
        
        # Step 1: Copy existing character images
        existing_count = self.copy_existing_characters()
        
        # Step 2: Convert QMNIST digits
        qmnist_count = self.convert_qmnist_to_images(qmnist_samples_per_digit)
        
        # Step 3: Analyze the result
        total_images, total_classes = self.analyze_enhanced_dataset()
        
        print(f"\n🎉 Enhanced Dataset Created!")
        print(f"📊 Total: {total_images:,} images across {total_classes} character classes")
        print(f"📁 Saved to: {self.output_path}")
        
        return total_images, total_classes

if __name__ == "__main__":
    print("🔄 QMNIST to Character Images Converter")
    print("=" * 60)
    
    converter = QMNISTConverter()
    
    # Create the enhanced dataset
    # Start with 1000 samples per digit (10k total from QMNIST)
    total_images, total_classes = converter.create_enhanced_dataset(qmnist_samples_per_digit=1000)
    
    if total_images > 0:
        print(f"\n✅ Success! {total_images:,} images ready for unified training")
        print(f"🎯 {total_classes} character classes available")
        print("\n📝 Next steps:")
        print("1. Use enhanced_character_images folder for training")
        print("2. Run your existing training.py with this expanded dataset")
        print("3. Your model will automatically detect all character classes")
    else:
        print("❌ Dataset creation failed!")