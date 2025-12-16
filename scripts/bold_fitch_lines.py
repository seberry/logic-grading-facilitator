import cv2
import numpy as np
import os
import argparse

def merge_vertical_segments(contours, x_threshold=10, y_gap_threshold=50):
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: (b[0], b[1]))

    merged_boxes = []
    if not boxes:
        return []
        
    curr_x, curr_y, curr_w, curr_h = boxes[0]

    for next_box in boxes[1:]:
        nx, ny, nw, nh = next_box
        # Check alignment (X) and proximity (Y)
        same_column = abs(curr_x - nx) < x_threshold
        gap = ny - (curr_y + curr_h)
        close_vertically = gap < y_gap_threshold

        if same_column and close_vertically:
            # MERGE
            new_x = min(curr_x, nx)
            new_y = min(curr_y, ny)
            new_w = max(curr_x + curr_w, nx + nw) - new_x
            new_h = max(curr_y + curr_h, ny + nh) - new_y
            curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
        else:
            # PUSH AND RESET
            merged_boxes.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = nx, ny, nw, nh
            
    merged_boxes.append((curr_x, curr_y, curr_w, curr_h))
    return merged_boxes

def highlight_proof_scopes(image_path, output_suffix="_highlighted", debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load '{image_path}'")
        return None
    
    filename, ext = os.path.splitext(image_path)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 91, 15
    )
    if debug: 
        cv2.imwrite(f"{filename}_debug_1_binary{ext}", binary)
        print(f"Saved: {filename}_debug_1_binary{ext}")
    
    # 3. DILATION (The Fix)
    # Kernel: (1, 6) means 1px tall, 6px wide. 
    # This smears the line horizontally to catch the 'kink' without adding height.
    dilate_kernel = np.ones((1, 6), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    
    if debug: 
        cv2.imwrite(f"{filename}_debug_1b_dilated{ext}", dilated)
        print(f"Saved: {filename}_debug_1b_dilated{ext}")

    # 4. Vertical Filter (Applied to the FAT lines)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)
    
    if debug: 
        cv2.imwrite(f"{filename}_debug_2_vertical_mask{ext}", vertical_mask)
        print(f"Saved: {filename}_debug_2_vertical_mask{ext}")
    
    # 5. Find Contours
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        # Draw what the computer "sees" (The Red Boxes)
        debug_contours = img.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite(f"{filename}_debug_3_raw_contours{ext}", debug_contours)
        print(f"Saved: {filename}_debug_3_raw_contours{ext}")

    # 6. Merge Segments
    # x_threshold=20 allows for more slant/wobble
    final_boxes = merge_vertical_segments(contours, x_threshold=20, y_gap_threshold=100)
    
    # 7. Draw Final Highlights
    output_img = img.copy()
    img_h, img_w = img.shape[:2]
    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255)] 
    
    for i, (x, y, w, h) in enumerate(final_boxes):
        if h < 80: continue 
        
        # CHEAT: Extend top upwards by 25px just to be safe
        y_adjusted = max(0, y - 25) 
        h_adjusted = h + 25
            
        overlay = output_img.copy()
        color = colors[i % len(colors)]
        
        cv2.rectangle(overlay, (x, y_adjusted), (img_w, y_adjusted + h_adjusted), color, -1)
        
        alpha = 0.2
        output_img = cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0)
        cv2.line(output_img, (x, y_adjusted), (x, y_adjusted + h_adjusted), (50, 50, 50), 2)

    output_path = f"{filename}{output_suffix}{ext}"
    cv2.imwrite(output_path, output_img)
    print(f"Final output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="List of image files")
    # This enables the --debug flag
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    args = parser.parse_args()
    
    if args.files:
        for file in args.files:
            if os.path.exists(file):
                highlight_proof_scopes(file, debug=args.debug)
            else:
                print(f"Skipping '{file}': File not found.")
    else:
        print("Usage: python bold_fitch_lines.py image.jpg --debug")