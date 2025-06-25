from model import OCRModel
from preprocessing import ImagePreprocessor
from training import extract_character_from_filename  # Import your function!
import os
import random

# Load model
ocr_model = OCRModel(num_classes=48)
ocr_model.load_model("ocr_simple_cnn_v1")
preprocessor = ImagePreprocessor()

# Test the same problematic files
test_files = [
    "RobotoMono_regular_32_at.png",
    "timesnewroman_regular_32_exclmark.png", 
    "Tinos_italic_32_ampersand.png"
]

print("Testing with Correct Character Extraction:")
print("=" * 50)

for filename in test_files:
    image_path = f"../character_images/{filename}"
    processed = preprocessor.process_image(image_path, show_steps=False)
    
    if processed is not None:
        # Use YOUR function to extract the expected character
        expected_char = extract_character_from_filename(filename)
        
        predictions = ocr_model.predict_character(processed, top_k=3)
        top_pred, top_conf = predictions[0]
        
        is_correct = "CORRECT" if top_pred == expected_char else "INCORRECT"
        
        print(f"\n{is_correct} File: {filename}")
        print(f"   Expected: '{expected_char}' | Predicted: '{top_pred}' ({top_conf*100:.1f}%)")

exit()