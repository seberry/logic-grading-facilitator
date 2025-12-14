import cv2
import numpy as np
import os
import argparse

def merge_vertical_segments(contours, x_threshold=10, y_gap_threshold=50):
    """
    Merges contours that are aligned vertically (same column) and close together (gap).
    Returns a list of bounding boxes (x, y, w, h).
    """
    # 1. Convert contours to bounding boxes [x, y, w, h]
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # 2. Sort primarily by X (position), secondarily by Y (vertical start)
    # This groups lines that are in the same vertical column
    boxes.sort(key=lambda b: (b[0], b[1]))

    merged_boxes = []
    if not boxes:
        return []
        
    # Start with the first box
    curr_x, curr_y, curr_w, curr_h = boxes[0]

    for next_box in boxes[1:]:
        nx, ny, nw, nh = next_box

        # Check alignment (X) and proximity (Y)
        # 1. Are they roughly in the same column? (Center alignment might be better, but left-edge is usually fine for lines)
        same_column = abs(curr_x - nx) < x_threshold
        
        # 2. Is the top of the next box close to the bottom of the current box?
        # current_bottom = curr_y + curr_h
        gap = ny - (curr_y + curr_h)
        close_vertically = gap < y_gap_threshold

        if same_column and close_vertically:
            # MERGE: Create a new bounding box covering both
            new_x = min(curr_x, nx)
            new_y = min(curr_y, ny)
            # The width extends to the max right edge
            new_w = max(curr_x + curr_w, nx + nw) - new_x
            # The height extends to the max bottom edge
            new_h = max(curr_y + curr_h, ny + nh) - new_y
            
            # Update current "active" box
            curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
        else:
            # No merge: push the current accumulated box and start a new one
            merged_boxes.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = nx, ny, nw, nh
            
    # Append the final box
    merged_boxes.append((curr_x, curr_y, curr_w, curr_h))
    return merged_boxes

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
    # Kernel: 1px wide, 80px tall. 
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- NEW STEP: Merge broken lines ---
    # We replace the simple contour sort with our merge logic
    final_boxes = merge_vertical_segments(contours, x_threshold=15, y_gap_threshold=60)
    
    # 6. Define Highlight Colors (CMY)
    colors = [
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    # Working on a copy of the image
    output_img = img.copy()
    
    # 7. Iterate and Draw Highlights (Iterating our merged boxes now)
    img_h, img_w = img.shape[:2]
    
    for i, (x, y, w, h) in enumerate(final_boxes):
        # Filter: Ignore tiny noise (lines shorter than 80px)
        if h < 80:
            continue
            
        # Create a temporary overlay for this single scope
        overlay = output_img.copy()
        
        # Cycle through the colors
        color = colors[i % len(colors)]
        
        # Draw a filled rectangle from the line (x) to the right edge (img_w)
        cv2.rectangle(overlay, (x, y), (img_w, y + h), color, -1)
        
        # Blend this layer onto the main image
        # Alpha: 0.2 means 20% color, 80% original text. 
        alpha = 0.2
        output_img = cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0)
        
        # Draw a solid dark line over the actual bar to ensure it remains visible
        cv2.line(output_img, (x, y), (x, y + h), (50, 50, 50), 2)

    # 8. Save Output
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}{output_suffix}{ext}"
    cv2.imwrite(output_path, output_img)
    
    print(f"Processed: Found {len(final_boxes)} scope lines (merged).")
    print(f"Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight logic proof scopes.")
    parser.add_argument("files", nargs="*", help="List of image files to process")
    args = parser.parse_args()
    
    if not args.files:
        print("No files provided. Usage: python bold_fitch_lines.py image1.png image2.png")
    else:
        for file in args.files:
            if os.path.exists(file):
                highlight_proof_scopes(file)
            else:
                print(f"Skipping '{file}': File not found.")