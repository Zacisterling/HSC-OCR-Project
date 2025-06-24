
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import os

class ImagePreprocessor:
    """Handles all image preprocessing operations for OCR"""
    
    def __init__(self, target_size=(28, 28)):
        """
        Initialize preprocessor
        Args:
            target_size (tuple): Target image size (width, height)
        """
        self.target_size = target_size
        print(f" ImagePreprocessor initialized with target size: {target_size}")
    
    def load_image(self, image_path):
        """
        Load image from file path
        Args:
            image_path (str): Path to image file
        Returns:
            numpy.ndarray: Loaded image or None if failed
        """
        try:
            if not os.path.exists(image_path):
                print(f" Image not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as backup
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
            print(f" Loaded image: {image_path} - Shape: {image.shape}")
            return image
        except Exception as e:
            print(f" Error loading image {image_path}: {e}")
            return None
    
    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f" Converted to grayscale - Shape: {gray.shape}")
            return gray
        print(" Image already grayscale")
        return image
    
    def resize_image(self, image):
        """Resize image to target size"""
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        print(f" Resized to {self.target_size}")
        return resized
    
    def normalize_image(self, image):
        """Normalize pixel values to 0-1 range"""
        normalized = image.astype(np.float32) / 255.0
        print(f" Normalized - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
        return normalized
    
    def remove_noise(self, image):
        """Remove noise using Gaussian blur"""
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        print(" Noise removal applied")
        return denoised
    
    def enhance_contrast(self, image):
        """Enhance image contrast"""
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%
        result = np.array(enhanced)
        print(" Contrast enhanced")
        return result
    
    def rotate_image(self, image, angle):
        """
        Rotate image by specified angle (for data augmentation)
        Args:
            image: Input image
            angle: Rotation angle in degrees
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                borderValue=255, flags=cv2.INTER_LINEAR)
        print(f" Rotated by {angle} degrees")
        return rotated
    
    def process_image(self, image_path, show_steps=False):
        """
        Complete preprocessing pipeline for Level 5 complexity
        Args:
            image_path (str): Path to image
            show_steps (bool): Whether to show each processing step
        Returns:
            numpy.ndarray: Processed image ready for ML model
        """
        print(f"\n Processing: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # Step 1: Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        original = image.copy()
        
        # Step 2: Convert to grayscale
        image = self.convert_to_grayscale(image)
        
        # Step 3: Enhance contrast (helps with handwriting)
        image = self.enhance_contrast(image)
        
        # Step 4: Remove noise
        image = self.remove_noise(image)
        
        # Step 5: Resize to standard size
        image = self.resize_image(image)
        
        # Step 6: Normalize pixel values
        image = self.normalize_image(image)
        
        print(f" Processing complete! Final shape: {image.shape}")
        
        if show_steps:
            self.visualize_preprocessing(original, image, image_path)
        
        return image
    
    def process_batch(self, image_folder):
        """
        Process all images in a folder
        Args:
            image_folder (str): Path to folder containing images
        Returns:
            list: List of processed images and their filenames
        """
        processed_images = []
        filenames = []
        
        if not os.path.exists(image_folder):
            print(f" Folder not found: {image_folder}")
            return processed_images, filenames
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(image_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"\n Processing {len(image_files)} images from {image_folder}")
        print("=" * 50)
        
        for filename in image_files:
            image_path = os.path.join(image_folder, filename)
            processed = self.process_image(image_path)
            
            if processed is not None:
                processed_images.append(processed)
                filenames.append(filename)
        
        print(f"\n Successfully processed {len(processed_images)} images")
        return processed_images, filenames
    
    def visualize_preprocessing(self, original, processed, image_path):
        """Show before/after preprocessing comparison"""
        plt.figure(figsize=(12, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        if len(original.shape) == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title(f"Original\n{os.path.basename(image_path)}")
        plt.axis('off')
        
        # Processed image
        plt.subplot(1, 3, 2)
        plt.imshow(processed, cmap='gray')
        plt.title(f"Processed\n{self.target_size}")
        plt.axis('off')
        
        # Histogram comparison
        plt.subplot(1, 3, 3)
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
        
        plt.hist(original_gray.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        plt.hist((processed * 255).astype(np.uint8).flatten(), bins=50, alpha=0.7, label='Processed', density=True)
        plt.title("Pixel Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_augmented_data(self, image, num_variations=5):
        """
        Create augmented versions for training (Level 5 complexity)
        Args:
            image: Input image
            num_variations: Number of augmented versions to create
        Returns:
            list: List of augmented images
        """
        augmented = [image]  # Include original
        
        for i in range(num_variations):
            variant = image.copy()
            
            # Random rotation (±20 degrees for Level 5)
            angle = np.random.uniform(-20, 20)
            variant = self.rotate_image(variant, angle)
            
            # Random noise
            noise = np.random.normal(0, 0.02, variant.shape)
            variant = np.clip(variant + noise, 0, 1)
            
            augmented.append(variant)
        
        print(f" Created {len(augmented)} augmented variations")
        return augmented

# Test the preprocessor
if __name__ == "__main__":
    print(" Testing Image Preprocessor")
    print("=" * 40)
    
    preprocessor = ImagePreprocessor(target_size=(28, 28))
    
    # Test with a sample image (you'll need to provide a real image path)
    # test_image = "path/to/your/test/image.jpg"
    # processed = preprocessor.process_image(test_image, show_steps=True)
    
    print("\n ImagePreprocessor module ready!")
    print("To test: provide an image path and call process_image()")