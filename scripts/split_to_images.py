import argparse
import os
from app.storage import RAW_SCANS_DIR
import fitz  # This is PyMuPDF

def split_pdf_to_images(source_path, output_folder, image_format="jpg"):
    # 1. Validation and Setup
    if not os.path.exists(source_path):
        print(f"Error: The file '{source_path}' was not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")

    # 2. Open the PDF
    try:
        doc = fitz.open(source_path)
        print(f"Processing '{source_path}' ({len(doc)} pages)...")

        # 3. Iterate through pages
        for i, page in enumerate(doc):
            # --- QUALITY CONTROL ---
            # PDFs are vector-based, Images are pixel-based. 
            # We zoom in 2x (matrix=2,2) to get higher resolution (clearness).
            # If the text is still too blurry for your code, change 2 to 3.
            matrix = fitz.Matrix(2, 2) 
            pix = page.get_pixmap(matrix=matrix)
            
            # Construct filename (e.g., scan_1.jpg)
            output_filename = f"scan_{i+1}.{image_format}"
            full_output_path = os.path.join(output_folder, output_filename)
            
            # Save the image
            pix.save(full_output_path)
        
        print(f"Success! {len(doc)} images saved to '{output_folder}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split PDF directly into images.")
    
    parser.add_argument("--input", required=True, help="Path to the big PDF")
    
    # Default output is /data/raw_scans relative to where you run this
    parser.add_argument("--output", default=str(RAW_SCANS_DIR), help="Output folder")
    
    # Option to switch between jpg and png
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Image format")

    args = parser.parse_args()

    split_pdf_to_images(args.input, args.output, args.format)
