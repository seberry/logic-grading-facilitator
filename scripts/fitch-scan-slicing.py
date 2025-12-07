import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def process_with_intensity_filter(image_path, target_lines):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- NEW: LIGHTING NORMALIZATION & GREY REMOVAL ---
    
    # A. Estimate the background (the paper)
    # We use a large kernel to "close" over the text, leaving just the paper color + shadows
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)
    
    # B. Normalize: Divide raw image by background
    # This makes the background pure white (255) everywhere, eliminating shadows.
    # We add +1 to avoid division by zero.
    norm_img = (gray.astype(np.float32) / (background.astype(np.float32) + 1)) * 255
    norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
    
    # C. "Turn up the contrast until lines disappear"
    # Now that lighting is flat:
    # - Paper is ~255
    # - Ruled Lines are ~200-230
    # - Pen Ink is ~0-100
    # We set a HARD threshold at 160. This keeps Ink, kills Lines.
    _, binary_no_lines = cv2.threshold(norm_img, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up pepper noise (tiny dots)
    clean_kernel = np.ones((2,2), np.uint8)
    binary_clean = cv2.morphologyEx(binary_no_lines, cv2.MORPH_OPEN, clean_kernel)
    
    # --------------------------------------------------

    # 2. Horizontal Projection (Right-Side Only Trick)
    # We still ignore the left 15% to avoid the thick vertical logic bar
    roi_start = int(binary_clean.shape[1] * 0.15) 
    row_sums = np.sum(binary_clean[:, roi_start:], axis=1)

    # 3. Smoothing
    SMOOTHING_WINDOW = 35 # Slightly tighter window since lines are gone
    kernel_smooth = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    row_sums_smoothed = np.convolve(row_sums, kernel_smooth, mode='same')
    
    # Normalize
    max_val = np.max(row_sums_smoothed)
    if max_val == 0: max_val = 1 # Avoid div/0
    row_sums_normalized = row_sums_smoothed / max_val

    # 4. Auto-Tuning Loop (Same as before)
    best_threshold = 0.10
    best_count = 0
    found_match = False
    
    print(f"Scanning for {target_lines} lines...")
    
    # We sweep looking for the target count
    for t in np.arange(0.05, 0.50, 0.02):
        # Count "humps" in the graph
        is_ink = row_sums_normalized > t
        transitions = np.diff(is_ink.astype(int))
        # transitions == 1 means we entered a text line
        current_count = np.sum(transitions == 1)
        
        print(f"   Threshold {t:.2f} -> Found {current_count} lines")
        
        if current_count == target_lines:
            best_threshold = t
            found_match = True
            break # Found it!
            
    if not found_match:
        print("Warning: Exact match not found, using last best guess.")

    # 5. Slicing for Display
    sliced_img = img.copy()
    height, width = binary_clean.shape
    
    in_ink = False
    ink_start = 0
    
    # Visualization Colors
    box_color = (0, 0, 255) # Red
    
    # Draw boxes around lines
    for y in range(height):
        is_ink_row = row_sums_normalized[y] > best_threshold
        
        if is_ink_row and not in_ink:
            in_ink = True
            ink_start = y
        elif not is_ink_row and in_ink:
            ink_end = y
            # Draw a rectangle around this "Line Zone"
            cv2.rectangle(sliced_img, (0, ink_start), (width, ink_end), box_color, 2)
            in_ink = False

    # --- PLOTTING ---
    plt.figure(figsize=(15, 6))

    # Show the "Grey Lines Removed" version
    plt.subplot(1, 3, 1)
    plt.imshow(binary_clean, cmap='gray')
    plt.title("Step 1: Intensity Filter\n(Keeps Ink, Drops Grey Lines)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.plot(row_sums_normalized, range(height))
    plt.gca().invert_yaxis()
    plt.title(f"Step 2: Projection\n(Thresh: {best_threshold:.2f})")
    plt.axvline(x=best_threshold, color='r', linestyle='--')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sliced_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Step 3: Detected {target_lines} Lines")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True)
    parser.add_argument("-n", "--lines", type=int, default=7)
    args = parser.parse_args()

    process_with_intensity_filter(args.image, args.lines)