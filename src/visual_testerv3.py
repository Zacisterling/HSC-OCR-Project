"""
Simple OCR Tester - Bulletproof version for macOS
Uses the proven pattern from digit recognition tutorials
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import random

from model import OCRModel
from preprocessing import ImagePreprocessor

class SimpleOCRTester:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HSC OCR System")
        self.root.geometry("600x500")
        
        # Load model
        print("Loading model...")
        self.ocr_model = OCRModel(num_classes=48)
        self.ocr_model.load_model("ocr_simple_cnn_v1")
        self.preprocessor = ImagePreprocessor()
        
        self.current_image_path = None
        self.current_predictions = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="HSC OCR Character Recognition", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Simple buttons
        tk.Button(btn_frame, text="Select Image", command=self.select_image,
                 width=15, height=2, bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Random Test", command=self.random_test,
                 width=15, height=2, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Analyze", command=self.analyze,
                 width=15, height=2, bg="orange").pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(self.image_frame, text="No image selected")
        self.image_label.pack(expand=True)
        
        # Results frame
        results_frame = tk.Frame(self.root)
        results_frame.pack(pady=10, fill=tk.X)
        
        # File info
        self.file_info = tk.Label(results_frame, text="File: None")
        self.file_info.pack()
        
        # Main prediction
        self.main_pred = tk.Label(results_frame, text="Prediction: -", 
                                 font=("Arial", 20, "bold"))
        self.main_pred.pack(pady=5)
        
        # Confidence
        self.confidence = tk.Label(results_frame, text="Confidence: -", 
                                  font=("Arial", 14))
        self.confidence.pack()
        
        # Show plots button
        self.plot_btn = tk.Button(results_frame, text="Show Detailed Analysis", 
                                 command=self.show_plots, state=tk.DISABLED,
                                 bg="purple", fg="white", height=2)
        self.plot_btn.pack(pady=10)
        
        # Status
        self.status = tk.Label(self.root, text="Ready!")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.load_image(file_path)
    
    def random_test(self):
        try:
            images_folder = "../character_images"
            if os.path.exists(images_folder):
                files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
                if files:
                    random_file = random.choice(files)
                    image_path = os.path.join(images_folder, random_file)
                    self.load_image(image_path)
                else:
                    messagebox.showwarning("Error", "No images found!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_image(self, image_path):
        try:
            self.current_image_path = image_path
            
            # Load and display image
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Update file info
            filename = os.path.basename(image_path)
            self.file_info.config(text=f"File: {filename}")
            
            # Reset predictions
            self.main_pred.config(text="Prediction: Click 'Analyze'")
            self.confidence.config(text="Confidence: -")
            self.plot_btn.config(state=tk.DISABLED)
            
            self.status.config(text=f"Loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def analyze(self):
        if not self.current_image_path:
            messagebox.showwarning("Error", "Please select an image first!")
            return
        
        try:
            self.status.config(text="Analyzing...")
            self.root.update()
            
            # Process image
            processed = self.preprocessor.process_image(self.current_image_path, show_steps=False)
            if processed is None:
                messagebox.showerror("Error", "Could not process image")
                return
            
            # Get predictions
            predictions = self.ocr_model.predict_character(processed, top_k=5)
            self.current_predictions = predictions
            
            # Display results
            top_char, top_conf = predictions[0]
            self.main_pred.config(text=f"Prediction: '{top_char}'")
            self.confidence.config(text=f"Confidence: {top_conf*100:.1f}%")
            
            # Enable plot button
            self.plot_btn.config(state=tk.NORMAL)
            
            self.status.config(text=f"Complete! Predicted '{top_char}' with {top_conf*100:.1f}% confidence")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.status.config(text="Analysis failed")
    
    def show_plots(self):
        if not self.current_predictions or not self.current_image_path:
            messagebox.showwarning("Error", "No analysis results to show!")
            return
        
        try:
            # Load images for plotting
            original = cv2.imread(self.current_image_path)
            processed = self.preprocessor.process_image(self.current_image_path, show_steps=False)
            
            # Create matplotlib figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"OCR Analysis: {os.path.basename(self.current_image_path)}", fontsize=16)
            
            # Original image
            ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Processed image
            ax2.imshow(processed, cmap='gray')
            ax2.set_title("Processed (28x28)")
            ax2.axis('off')
            
            # Predictions text
            ax3.axis('off')
            pred_text = "Top 5 Predictions:\n\n"
            for i, (char, conf) in enumerate(self.current_predictions):
                pred_text += f"{i+1}. '{char}' - {conf*100:.1f}%\n"
            ax3.text(0.1, 0.9, pred_text, transform=ax3.transAxes, 
                    fontsize=12, verticalalignment='top')
            
            # Confidence chart
            chars = [pred[0] for pred in self.current_predictions]
            confs = [pred[1] * 100 for pred in self.current_predictions]
            
            ax4.bar(chars, confs, color=['green', 'blue', 'orange', 'red', 'purple'])
            ax4.set_title("Confidence Scores")
            ax4.set_ylabel("Confidence (%)")
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.show()  # This opens in separate window like your polynomial code
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not create plots: {e}")
    
    def run(self):
        print("Starting Simple OCR Tester...")
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleOCRTester()
    app.run()