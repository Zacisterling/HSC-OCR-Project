"""
Rename QMNIST Files to Consistent Format
Changes: QMNIST_digit_28_0_0134 -> handwritten_regular_28_0_0134
"""

import os
from pathlib import Path

class QMNISTRenamer:
    """Rename QMNIST files to consistent format"""
    
    def __init__(self, dataset_path="../enhanced_character_images"):
        self.dataset_path = Path(dataset_path)
        
    def rename_qmnist_files(self):
        """Rename all QMNIST files to new format"""
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return 0
        
        # Find all QMNIST files
        qmnist_files = list(self.dataset_path.glob("QMNIST_digit_*.png"))
        
        print(f"🔍 Found {len(qmnist_files)} QMNIST files to rename")
        
        renamed_count = 0
        
        for old_file in qmnist_files:
            try:
                # Parse the old filename
                # Format: QMNIST_digit_28_0_0134.png
                name_parts = old_file.stem.split('_')
                
                if len(name_parts) >= 5:
                    # Extract components
                    digit = name_parts[3]  # The digit (0-9)
                    sequence = name_parts[4]  # The sequence number (0134)
                    
                    # Create new filename
                    # Format: handwritten_regular_28_0_0134.png
                    new_name = f"handwritten_regular_28_{digit}_{sequence}.png"
                    new_path = self.dataset_path / new_name
                    
                    # Rename the file
                    old_file.rename(new_path)
                    
                    renamed_count += 1
                    
                    if renamed_count % 1000 == 0:
                        print(f"   Renamed {renamed_count}/{len(qmnist_files)} files...")
                
            except Exception as e:
                print(f"❌ Error renaming {old_file.name}: {e}")
        
        print(f"✅ Successfully renamed {renamed_count} files")
        return renamed_count
    
    def verify_renaming(self):
        """Verify the renaming was successful"""
        
        # Count old format files (should be 0)
        old_format = list(self.dataset_path.glob("QMNIST_digit_*.png"))
        
        # Count new format files
        new_format = list(self.dataset_path.glob("handwritten_regular_*.png"))
        
        print(f"\n📊 Renaming Verification:")
        print(f"   Old format (QMNIST_digit_*): {len(old_format)} files")
        print(f"   New format (handwritten_regular_*): {len(new_format)} files")
        
        if len(old_format) == 0 and len(new_format) > 0:
            print("✅ Renaming successful!")
        else:
            print("⚠️ Some files may not have been renamed correctly")
        
        # Show sample new filenames
        if new_format:
            print(f"\n📝 Sample new filenames:")
            for i, file in enumerate(new_format[:5]):
                print(f"   {file.name}")
            if len(new_format) > 5:
                print(f"   ... and {len(new_format) - 5} more")
        
        return len(new_format)
    
    def analyze_final_dataset(self):
        """Analyze the final dataset with new naming"""
        
        all_files = list(self.dataset_path.glob("*.png"))
        
        print(f"\n📊 Final Dataset Analysis:")
        print(f"   Total images: {len(all_files)}")
        
        # Count by naming pattern
        handwritten = len([f for f in all_files if f.name.startswith("handwritten_")])
        font_based = len([f for f in all_files if not f.name.startswith("handwritten_")])
        
        print(f"   Handwritten images: {handwritten}")
        print(f"   Font-based images: {font_based}")
        
        # Extract unique characters
        characters = set()
        for filename in all_files:
            parts = filename.name.replace('.png', '').split('_')
            if len(parts) >= 4:
                # Get character part (should be second to last for most files)
                char_part = parts[-2] if len(parts) > 4 else parts[-1]
                
                # Handle symbol mappings
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

if __name__ == "__main__":
    print("🔄 QMNIST File Renamer")
    print("=" * 40)
    
    renamer = QMNISTRenamer()
    
    # Rename the files
    renamed_count = renamer.rename_qmnist_files()
    
    if renamed_count > 0:
        # Verify the renaming
        renamer.verify_renaming()
        
        # Analyze final dataset
        renamer.analyze_final_dataset()
        
        print(f"\n🎉 Renaming complete!")
        print(f"✅ {renamed_count} files renamed to handwritten_regular_28_X_XXXX format")
    else:
        print("❌ No files were renamed")