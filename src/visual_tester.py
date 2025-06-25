
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from model import OCRModel
from preprocessing import ImagePreprocessor

class VisualOCRTester:
    #Visual window for testing OCR predictions
    
    def __init__(self):
        #start the visual tester
        self.root = tk.Tk()
        self.root.title("🤖 HSC OCR Character Recognition System")
        self.root.geometry("1200x800")
        
        #load model
        self.ocr_model = OCRModel(num_classes=48)
        self.ocr_model.load_model("ocr_simple_cnn_v1")
        self.preprocessor = ImagePreprocessor()
        
        self.setup_ui()
        
    def setup_ui(self):
        #Create the user interface
        # main window frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # title
        title_label = tk.Label(main_frame, text="HSC OCR Character Recognition System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        #Buttons
        tk.Button(control_frame, text="📁 Select Image File", 
                 command=self.select_single_image, 
                 font=("Arial", 12), bg="#4CAF50", fg="white",
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="📂 Test Folder", 
                 command=self.select_folder, 
                 font=("Arial", 12), bg="#2196F3", fg="white",
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="🎲 Random Test", 
                 command=self.random_test, 
                 font=("Arial", 12), bg="#FF9800", fg="white",
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        #Results frame
        results_frame = tk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create matplotlib image of character
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("OCR Analysis Results", fontsize=16, fontweight='bold')
        
        #Embed matplotlib in tkinter (?)
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        #status label
        self.status_label = tk.Label(main_frame, text="🤖 Model loaded and ready! Select an image to test.", 
                                   font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Clear the initial plots
        self.clear_plots()
        
    def clear_plots(self):
        """Clear all plots"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.ax1.text(0.5, 0.5, "Original Image\nwill appear here", 
                     ha='center', va='center', fontsize=12, alpha=0.5)
        self.ax2.text(0.5, 0.5, "Processed Image\nwill appear here", 
                     ha='center', va='center', fontsize=12, alpha=0.5)
        self.ax3.text(0.5, 0.5, "Prediction Results\nwill appear here", 
                     ha='center', va='center', fontsize=12, alpha=0.5)
        self.ax4.text(0.5, 0.5, "Confidence Analysis\nwill appear here", 
                     ha='center', va='center', fontsize=12, alpha=0.5)
        
        self.canvas.draw()
    
    def select_single_image(self):
        """Select and analyze a single image"""
        file_path = filedialog.askopenfilename(
            title="Select Character Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def select_folder(self):
        """Select folder and test multiple images"""
        folder_path = filedialog.askdirectory(title="Select Folder with Character Images")
        
        if folder_path:
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if image_files:
                #Test first image in folder
                first_image = os.path.join(folder_path, image_files[0])
                self.analyze_image(first_image)
                
                messagebox.showinfo("Folder Testing", 
                                  f"Found {len(image_files)} images. Showing first image: {image_files[0]}")
            else:
                messagebox.showwarning("No Images", "No image files found in selected folder.")
    
    def random_test(self):
        """Test a random image from character_images folder"""
        try:
            images_folder = "../character_images"
            if os.path.exists(images_folder):
                image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
                if image_files:
                    import random
                    random_image = random.choice(image_files)
                    image_path = os.path.join(images_folder, random_image)
                    self.analyze_image(image_path)
                else:
                    messagebox.showwarning("No Images", "No images found in character_images folder.")
            else:
                messagebox.showwarning("Folder Not Found", "character_images folder not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Random test failed: {str(e)}")
    
    def analyze_image(self, image_path):
        """Analyze a single image and display results"""
        try:
            self.status_label.config(text=f"Processing: {os.path.basename(image_path)}")
            self.root.update()
            
            #Load and process image
            original_image = cv2.imread(image_path)
            if original_image is None:
                messagebox.showerror("Error", "Could not load image file.")
                return
            
            processed_image = self.preprocessor.process_image(image_path, show_steps=False)
            if processed_image is None:
                messagebox.showerror("Error", "Could not process image.")
                return
            
            #Get predictions
            predictions = self.ocr_model.predict_character(processed_image, top_k=5)
            
            # Display results
            self.display_results(original_image, processed_image, predictions, image_path)
            
            # Update status
            top_pred, top_conf = predictions[0]
            self.status_label.config(text=f"Prediction: '{top_pred}' ({top_conf*100:.1f}% confidence)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_label.config(text="Analysis failed")
    
    def display_results(self, original, processed, predictions, image_path):
        """Display analysis results in the plots"""
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 1. Original Image
        self.ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        self.ax1.set_title(f"Original Image\n{os.path.basename(image_path)}", fontweight='bold')
        
        # 2. Processed Image
        self.ax2.imshow(processed, cmap='gray')
        self.ax2.set_title("Processed Image\n(28x28, normalized)", fontweight='bold')
        
        # 3. Top Predictions (simplified - no emoji)
        self.ax3.axis('off')
        pred_text = "TOP PREDICTIONS:\n\n"
        for i, (char, conf) in enumerate(predictions[:5]):
            confidence_bar = "=" * int(conf * 15)  # Simple text bar
            pred_text += f"{i+1}. '{char}' - {conf*100:.1f}%\n   [{confidence_bar}]\n\n"
        
        self.ax3.text(0.05, 0.95, pred_text, transform=self.ax3.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax3.set_title("Prediction Results", fontweight='bold')
        
        # 4. Confidence Chart
        chars = [pred[0] for pred in predictions[:5]]
        confs = [pred[1] * 100 for pred in predictions[:5]]
        colors = ['green', 'blue', 'orange', 'purple', 'red']
        
        bars = self.ax4.bar(chars, confs, color=colors)
        self.ax4.set_title("Confidence Scores (%)", fontweight='bold')
        self.ax4.set_ylabel("Confidence (%)")
        self.ax4.set_ylim(0, 100)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confs):
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Force refresh the display
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
        self.canvas.flush_events()
        self.root.update()
    
    def run(self):
        #Start the visual tester
        print("Starting Visual OCR Tester...")
        self.root.mainloop()

if __name__ == "__main__":
    app = VisualOCRTester()
    app.run()