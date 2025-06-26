"""
Simple Multi-Font Character Generator
Uses system fonts that are guaranteed to work
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Settings
size = 28
img_size = (size, size)
font_size = 32

# System fonts that work on Mac
fonts = {
    'Arial': '/System/Library/Fonts/Arial.ttc',
    'Helvetica': '/System/Library/Fonts/Helvetica.ttc', 
    'Times': '/System/Library/Fonts/Times.ttc',
    'Courier': '/System/Library/Fonts/Courier.ttc'
}

# Character sets
uppercase = [chr(i) for i in range(65, 91)]  # A-Z
lowercase = [chr(i) for i in range(97, 123)]  # a-z
numbers = [chr(i) for i in range(48, 58)]     # 0-9
symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']

# Symbol mappings
symbol_names = {
    '!': 'exclmark', '@': 'at', '#': 'hash', '$': 'dollar',
    '%': 'percent', '&': 'ampersand', '*': 'asterisk',
    '+': 'plus', '-': 'minus', '?': 'quesmark',
    '<': 'lessthan', '>': 'greaterthan'
}

# All characters
all_characters = uppercase + lowercase + numbers + symbols

print(f"🎯 Generating {len(all_characters)} characters × {len(fonts)} fonts = {len(all_characters) * len(fonts)} images")

for font_name, font_path in fonts.items():
    if os.path.exists(font_path):
        print(f"\n📝 Processing {font_name}...")
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            for character in all_characters:
                # Create image
                img = Image.new("L", img_size, color="white")
                draw = ImageDraw.Draw(img)
                
                # Center text
                bbox = draw.textbbox((0, 0), character, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                position = (
                    (size - text_width) // 2 - bbox[0],
                    (size - text_height) // 2 - bbox[1]
                )
                
                # Draw and save
                draw.text(position, character, fill="black", font=font)
                
                char_name = symbol_names.get(character, character)
                filename = f"{font_name}_regular_{font_size}_{char_name}.png"
                img.save(filename)
            
            print(f"   ✅ Generated {len(all_characters)} images for {font_name}")
            
        except Exception as e:
            print(f"   ❌ Error with {font_name}: {e}")

print(f"\n🎉 Generation complete!")