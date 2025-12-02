import argparse
import os
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(source_path, output_folder):
    # 1. Check if the input file actually exists
    if not os.path.exists(source_path):
        print(f"Error: The file '{source_path}' was not found.")
        return

    # 2. Create output folder if it doesn't exist (exist_ok prevents errors if it does)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created folder: {output_folder}")

    # 3. Read the PDF
    try:
        reader = PdfReader(source_path)
        print(f"Processing '{source_path}' containing {len(reader.pages)} pages...")

        # 4. Loop through pages and save them
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            
            # Start naming at scan_1 instead of scan_0 for readability
            output_filename = f"scan_{i+1}.pdf" 
            full_output_path = os.path.join(output_folder, output_filename)
            
            with open(full_output_path, "wb") as output_pdf:
                writer.write(output_pdf)
        
        print(f"Success! {len(reader.pages)} files saved to '{output_folder}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# This block handles the command line input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a PDF into single pages.")
    
    # This sets up the "--input" flag
    parser.add_argument("--input", required=True, help="The path to the big PDF file")
    
    # This sets up the "--output" flag
    parser.add_argument("--output", required=True, help="The folder to save the scans in")

    args = parser.parse_args()

    # Pass the arguments to the function
    split_pdf(args.input, args.output)