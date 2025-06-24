
import os
import sys

def main():
    """Main program entry point"""
    print("HSC OCR Project - Character Recognition System")
    print("=" * 50)
    
    # Display menu
    while True:
        print("\nChoose an option:")
        print("1. Training Mode - Train the model")
        print("2. Testing Mode - Test the model")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            training_mode()
        elif choice == "2":
            testing_mode()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def training_mode():
    """Handle training mode"""
    print("\n Training Mode Selected")
    print("This will train your OCR model with the dataset...")
    print("(Training functionality will be implemented next)")

def testing_mode():
    """Handle testing mode"""
    print("\n Testing Mode Selected")
    print("(Testing functionality will be implemented next)")

if __name__ == "__main__":
    main()