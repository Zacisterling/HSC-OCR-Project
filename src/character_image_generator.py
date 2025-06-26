#!/usr/bin/env python3
"""
HSC OCR Project - Comprehensive Character Image Generator
Based on teacher's template but enhanced for Level 5 requirements

Generates images with specified fonts, styles, and characters
Filename format: FONT_style_FONTSIZE_CHARACTER.png
"""

from PIL import Image, ImageDraw, ImageFont
import os
import string

# Settings
size = 28
img_size = (size, size)
font_size = 32
output_dir = "comprehensive_character_images"

# Font configurations - Updated to match your exact font files
font_configs = {
    'Arial': {
        'regular': '../Fonts/Arial-Regular.ttf',
        'bold': '../Fonts/Arial-Bold.ttf',
        'italic': '../Fonts/Arial-Italic.ttf'
    },
    'ComicRelief': {
        'regular': '../Fonts/ComicRelief-Regular.ttf',
        'bold': '../Fonts/ComicRelief-Bold.ttf',
        'italic': '../Fonts/ComicRelief-Regular.ttf' # Comic Relief doesnt have italic
    },
    'Cookie': {
        'regular': '../Fonts/Cookie-Regular.ttf',
        'bold': '../Fonts/Cookie-Regular.ttf',  # Cookie only has regular
        'italic': '../Fonts/Cookie-Regular.ttf'
    },
    'CormorantGaramond': {
        'regular': '../Fonts/CormorantGaramond-Regular.ttf',
        'bold': '../Fonts/CormorantGaramond-Bold.ttf',
        'italic': '../Fonts/CormorantGaramond-Italic.ttf'
    },
    'CourierNew': {
        'regular': '../Fonts/CourierNew-Regular.ttf',
        'bold': '../Fonts/CourierNew-Bold.ttf',
        'italic': '../Fonts/CourierNew-Italic.ttf'
    },
    'Oxygen': {
        'regular': '../Fonts/Oxygen-Regular.ttf',
        'bold': '../Fonts/Oxygen-Bold.ttf',
        'italic': '../Fonts/Oxygen-Light.ttf' # italic replacement
    },
    'RobotoMono': {
        'regular': '../Fonts/RobotoMono-Regular.ttf',
        'bold': '../Fonts/RobotoMono-Bold.ttf',
        'italic': '../Fonts/RobotoMono-Italic.ttf'
    },
    'Tangerine': {
        'regular': '../Fonts/Tangerine-Regular.ttf',
        'bold': '../Fonts/Tangerine-Bold.ttf',
        'italic': '../Fonts/Tangerine-Regular.ttf'  # Tangerine only has regular and bold
    },
    'TimesNewRoman': {
        'regular': '../Fonts/TimesNewRoman-Regular.ttf',
        'bold': '../Fonts/TimesNewRoman-Bold.ttf',
        'italic': '../Fonts/TimesNewRoman-Italic.ttf'
    },
    'Tinos': {
        'regular': '../Fonts/Tinos-Regular.ttf',
        'bold': '../Fonts/Tinos-Bold.ttf',
        'italic': '../Fonts/Tinos-Italic.ttf'
    }
}

# Character sets
uppercase_letters = [chr(i) for i in range(65, 91)]  # A-Z
lowercase_letters = [chr(i) for i in range(97, 123)]  # a-z
numbers = [chr(i) for i in range(48, 58)]  # 0-9
special_chars = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']

# Special character name mapping for safe filenames
special_char_names = {
    '!': 'exclmark',
    '@': 'at',
    '#': 'hash',
    '$': 'dollar', 
    '%': 'percent',
    '&': 'ampersand',
    '*': 'asterisk',
    '+': 'plus',
    '-': 'minus',
    '?': 'quesmark',
    '<': 'lessthan',
    '>': 'greaterthan'
}

def create_character_image(character, font_path, font_name, style, font_size):
    """
    Create a single character image
    """
    try:
        # Load the font
        font = ImageFont.truetype(font_path, font_size)
        
        # Create a new white image
        img = Image.new("L", img_size, color="white")
        draw = ImageDraw.Draw(img)

        # Get size of text to center it
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((0 - bbox[0] + ((size - text_width) // 2)),
                   (0 - bbox[1] + ((size - text_height) // 2)))
        
        # Draw the character
        draw.text(position, character, fill="black", font=font)

        # Generate filename: FONT_style_FONTSIZE_CHARACTER.png
        if character in special_char_names:
            char_name = special_char_names[character]
        else:
            char_name = character

        filename = f"{font_name}_{style}_{font_size}_{char_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        img.save(filepath)
        return True
        
    except Exception as e:
        print(f"Error creating {character} with {font_name} {style}: {e}")
        return False

def generate_comprehensive_dataset():
    """
    Generate the complete dataset with all fonts, styles, and characters
    """
    print("🎓 HSC OCR Project - Comprehensive Dataset Generator")
    print("=" * 55)
    print(f"Generating images with:")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  📐 Image size: {size}x{size} pixels")
    print(f"  🔤 Font size: {font_size}")
    print(f"  🎨 Fonts: {len(font_configs)} fonts")
    print(f"  📝 Styles: regular, bold, italic")
    print(f"  📊 Characters: {len(uppercase_letters + lowercase_letters + numbers + special_chars)} characters")
    print()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all characters
    all_characters = uppercase_letters + lowercase_letters + numbers + special_chars
    
    total_images = len(font_configs) * 3 * len(all_characters)  # fonts * styles * characters
    print(f"🎯 Target: {total_images} total images")
    print()

    successful_images = 0
    failed_images = 0

    # Generate images for each font, style, and character
    for font_name, font_paths in font_configs.items():
        print(f"📝 Processing font: {font_name}")
        
        for style, font_path in font_paths.items():
            print(f"  🎨 Style: {style}")
            
            # Check if font file exists
            if not os.path.exists(font_path):
                print(f"    ❌ Font file not found: {font_path}")
                failed_images += len(all_characters)
                continue
            
            # Process each character
            for character in all_characters:
                success = create_character_image(character, font_path, font_name, style, font_size)
                if success:
                    successful_images += 1
                else:
                    failed_images += 1
            
            print(f"    ✅ Completed {style} style")
        
        print(f"✅ Completed font: {font_name}")
        print()

    # Final summary
    print("📊 GENERATION SUMMARY")
    print("=" * 25)
    print(f"✅ Successful images: {successful_images}")
    print(f"❌ Failed images: {failed_images}")
    print(f"📁 Images saved to: {output_dir}/")
    print(f"📝 Filename format: FONT_style_FONTSIZE_CHARACTER.png")
    
    if failed_images > 0:
        print(f"\n⚠️  {failed_images} images failed to generate.")
        print("   Common causes:")
        print("   - Missing font files in Fonts/ directory")
        print("   - Font doesn't support certain characters")
        print("   - Font file corruption")

    print(f"\n🎯 Dataset ready for Level 5 OCR training!")
    
    return successful_images, failed_images

def check_font_availability():
    """
    Check which font files are available before generation
    """
    print("🔍 Checking font file availability...")
    print("=" * 40)
    
    available_fonts = []
    missing_fonts = []
    
    for font_name, font_paths in font_configs.items():
        print(f"\n📝 {font_name}:")
        font_available = True
        
        for style, font_path in font_paths.items():
            if os.path.exists(font_path):
                print(f"  ✅ {style}: {font_path}")
            else:
                print(f"  ❌ {style}: {font_path} (MISSING)")
                font_available = False
        
        if font_available:
            available_fonts.append(font_name)
        else:
            missing_fonts.append(font_name)
    
    print(f"\n📊 FONT AVAILABILITY SUMMARY")
    print(f"✅ Available fonts: {len(available_fonts)}")
    print(f"❌ Missing fonts: {len(missing_fonts)}")
    
    if missing_fonts:
        print(f"\n⚠️  Missing fonts: {', '.join(missing_fonts)}")
        print("   Please download these fonts and place them in the Fonts/ directory")
        print("   You can find many of these fonts at:")
        print("   - Google Fonts (fonts.google.com)")
        print("   - System fonts directories")
    
    return available_fonts, missing_fonts

if __name__ == "__main__":
    print("🎓 HSC OCR Project - Enhanced Character Generator")
    print("Based on teacher's template with Level 5 enhancements")
    print("=" * 60)
    
    # Check font availability first
    available_fonts, missing_fonts = check_font_availability()
    
    if not available_fonts:
        print("\n❌ No fonts available! Please add font files to the Fonts/ directory.")
        print("\n📋 Required font files:")
        for font_name, font_paths in font_configs.items():
            print(f"  {font_name}:")
            for style, path in font_paths.items():
                print(f"    {style}: {path}")
    else:
        print(f"\n🚀 Proceeding with {len(available_fonts)} available fonts...")
        
        # Ask user confirmation
        user_input = input("\nProceed with image generation? (y/N): ")
        if user_input.lower() in ['y', 'yes']:
            successful, failed = generate_comprehensive_dataset()
            
            if successful > 0:
                print(f"\n🎉 Success! Generated {successful} character images.")
                print("🎯 Ready for Level 5 OCR training!")
            else:
                print("\n❌ No images were generated successfully.")
        else:
            print("❌ Generation cancelled.")