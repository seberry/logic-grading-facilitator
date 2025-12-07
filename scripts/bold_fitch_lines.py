import cv2
import numpy as np
import os
import argparse

def highlight_proof_scopes(image_path, output_suffix="_highlighted"):
    """
    Reads an image, detects vertical assumption lines, and draws 
    transparent colored regions extending to the right of each line.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load '{image_path}'. Check filename/path.")
        return None
    
    # 2. Convert to Grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Adaptive Thresholding (Standard shadow/noise removal)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 91, 15
    )
    
    # 4. Isolate Vertical Lines
    # Kernel: 1px wide, 40px tall. 
    # Increased height to 40 to ensure we don't pick up tall letters like 'l' or 'f' as easily.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # 5. Find Contours (distinct vertical lines)
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right (helps with consistent coloring order)
    # x coordinate is b[0]
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    # 6. Define Highlight Colors (BGR format)
    # Using CMY (Cyan, Magenta, Yellow) because they blend into nice secondaries (Blue, Red, Green)
    colors = [
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    # Working on a copy of the image
    output_img = img.copy()
    
    # 7. Iterate and Draw Highlights
    img_h, img_w = img.shape[:2]
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter: Ignore tiny noise (lines shorter than 60px)
        if h < 60:
            continue
            
        # Create a temporary overlay for this single scope
        overlay = output_img.copy()
        
        # Cycle through the colors
        color = colors[i % len(colors)]
        
        # Draw a filled rectangle from the line (x) to the right edge (img_w)
        # We add a small padding to y to avoid cutting off the very top/bottom of letters
        cv2.rectangle(overlay, (x, y), (img_w, y + h), color, -1)
        
        # Blend this layer onto the main image
        # Alpha: 0.2 means 20% color, 80% original text. 
        # As layers stack (nest), the color will become richer/mixed.
        alpha = 0.2
        output_img = cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0)
        
        # Optional: Draw a solid dark line over the actual bar to ensure it remains visible
        cv2.line(output_img, (x, y), (x, y + h), (50, 50, 50), 2)

    # 8. Save Output
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}{output_suffix}{ext}"
    cv2.imwrite(output_path, output_img)
    
    print(f"Processed: Found {len(contours)} scope lines.")
    print(f"Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight logic proof scopes.")
    parser.add_argument("files", nargs="*", help="List of image files to process")
    args = parser.parse_args()
    
    if not args.files:
        # Fallback for quick testing
        print("No files provided. Usage: python script.py image1.png image2.png")
    else:
        for file in args.files:
            if os.path.exists(file):
                highlight_proof_scopes(file)
            else:
                print(f"Skipping '{file}': File not found.")