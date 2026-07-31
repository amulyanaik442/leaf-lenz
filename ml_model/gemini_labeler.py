import os
import sys
import shutil
import time
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please create a '.env' file in the project root directory and add:")
    print("GEMINI_API_KEY=your_gemini_api_key_here")
    sys.exit(1)

genai.configure(api_key=api_key)

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNLABELED_DIR = os.path.join(BASE_DIR, "unlabeled_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "data")

# Setup folder structure
os.makedirs(UNLABELED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# System prompt for Gemini
PROMPT = """
Analyze the plant leaf in this photo.
1. Identify the plant species (e.g. Tomato, Potato, Apple, Grape, Mango, etc.). Use Title Case (replace spaces with underscores).
2. Determine if it is healthy or has a disease. If diseased, identify the disease.
3. Output the result strictly in this format: [PlantName]___[DiseaseOrConditionName]
   - If the plant is healthy, use the word 'healthy' for condition.
   - Replace any spaces in plant or disease names with underscores.
   - Make sure it matches our standard format. E.g., 'Tomato___Late_blight', 'Potato___healthy', 'Grape___black_rot', etc.

Provide ONLY the formatted string (e.g., PlantName___ConditionName) as your final response. Do not include markdown, bold text, bullet points, or additional text.
"""

def main():
    print(f"Checking for unlabeled images in: {UNLABELED_DIR}")
    
    # List supported image files
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    unlabeled_files = [f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith(supported_extensions)]
    
    if not unlabeled_files:
        print(f"No unlabeled images found in: {UNLABELED_DIR}")
        print("Please place some plant/leaf images (.jpg, .png, etc.) inside that folder and run this script again.")
        return

    print(f"Found {len(unlabeled_files)} image(s) to process.")
    
    # Initialize the Gemini model
    # Gemini 1.5 Flash is selected for fast inference and generous free tier
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Failed to initialize Gemini model: {e}")
        return

    success_count = 0
    fail_count = 0

    for idx, filename in enumerate(unlabeled_files, 1):
        file_path = os.path.join(UNLABELED_DIR, filename)
        print(f"[{idx}/{len(unlabeled_files)}] Processing {filename}...")
        
        try:
            # Load the image
            image = Image.open(file_path)
            
            # Request classification from Gemini
            response = model.generate_content([PROMPT, image])
            
            # Clean and sanitize response
            raw_label = response.text.strip()
            # Remove any markdown formatting Gemini might have accidentally included
            clean_label = raw_label.replace("`", "").replace("*", "").replace("\n", "").strip()
            
            # Basic validation of the format: Plant___Condition
            if "___" not in clean_label:
                print(f"  ⚠️ Warning: API returned invalid format: '{clean_label}'. Skipping.")
                fail_count += 1
                continue
                
            print(f"  🏷️ Labeled as: {clean_label}")
            
            # Define destination path
            target_class_dir = os.path.join(OUTPUT_DIR, clean_label)
            os.makedirs(target_class_dir, exist_ok=True)
            
            # Move the labeled file to its directory
            dest_file_path = os.path.join(target_class_dir, filename)
            shutil.move(file_path, dest_file_path)
            print(f"  ✅ Moved to: {os.path.relpath(dest_file_path, BASE_DIR)}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            fail_count += 1
            
        # Add a small delay between requests to stay within free-tier rate limits (15 RPM)
        if idx < len(unlabeled_files):
            time.sleep(4)  # 4 seconds delay = max 15 requests per minute

    print("\n--- Processing Complete ---")
    print(f"Successfully labeled: {success_count}")
    print(f"Failed to label: {fail_count}")
    print(f"All labeled data is organized under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
