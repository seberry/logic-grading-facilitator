"""
Icon Polishing Script
Makes backgrounds transparent and zooms in on content for better icon appearance
"""

from PIL import Image
import os

def make_white_transparent(img):
    """Convert pure white pixels to transparent"""
    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get pixel data
    data = img.getdata()
    
    # Create new data with white pixels made transparent
    new_data = []
    for item in data:
        # If pixel is pure white (RGB 255, 255, 255), make it transparent
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))  # White with alpha = 0 (transparent)
        else:
            new_data.append(item)
    
    # Update the image with new data
    img.putdata(new_data)
    return img

def get_content_bounding_box(img):
    """Find the bounding box of non-transparent pixels"""
    # Convert to RGBA if needed
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get dimensions
    width, height = img.size
    
    # Initialize bounding box
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    # Iterate through pixels to find content bounds
    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            # Check if pixel is not fully transparent
            if pixel[3] > 0:  # Alpha channel > 0
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    
    return (min_x, min_y, max_x, max_y)

def crop_to_content(img, padding=20):
    """Crop image to content with padding"""
    bbox = get_content_bounding_box(img)
    min_x, min_y, max_x, max_y = bbox
    
    # Add padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(img.width, max_x + padding)
    max_y = min(img.height, max_y + padding)
    
    # Crop to the bounding box
    return img.crop((min_x, min_y, max_x, max_y))

def polish_icon(input_path, output_path, padding=20):
    """Polish a single icon: make white transparent and crop to content"""
    print(f"\nProcessing: {input_path}")
    
    try:
        # Load the image
        img = Image.open(input_path)
        original_size = img.size
        print(f"  Original size: {original_size}")
        
        # Step 1: Make white transparent
        print("  Making background transparent...")
        img = make_white_transparent(img)
        
        # Step 2: Crop to content
        print("  Cropping to content...")
        img = crop_to_content(img, padding)
        new_size = img.size
        print(f"  New size: {new_size}")
        
        # Save the polished image
        img.save(output_path, 'PNG')
        print(f"  ✓ Saved: {output_path}")
        
        return True, new_size
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, None

def convert_to_ico(input_png, output_ico, size=256):
    """Convert PNG to ICO format"""
    try:
        print(f"\nConverting to ICO: {input_png} → {output_ico}")
        img = Image.open(input_png)
        # Resize to square for better icon quality
        img_square = img.resize((size, size), Image.Resampling.LANCZOS)
        img_square.save(output_ico, format='ICO', sizes=[(size, size)])
        print(f"  ✓ Saved: {output_ico}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def polish_all_icons():
    """Polish all three icons and regenerate the classwork ICO"""
    print("=" * 60)
    print("Icon Polishing Script")
    print("=" * 60)
    
    # Icons to process
    icons = [
        ("icon_classwork.png", "icon_classwork.png"),
        ("icon_examgrader.png", "icon_examgrader.png"),
        ("icon_absence_basket.png", "icon_absence_basket.png"),
    ]
    
    # Process each PNG icon
    results = {}
    for input_name, output_name in icons:
        success, new_size = polish_icon(input_name, output_name, padding=15)
        results[input_name] = (success, new_size)
    
    # Regenerate classwork ICO from polished version
    if results["icon_classwork.png"][0]:
        convert_to_ico("icon_classwork.png", "icon_classwork.ico", size=256)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for icon_name, (success, size) in results.items():
        status = "✓" if success else "✗"
        size_info = f" → {size}" if size else ""
        print(f"{status} {icon_name}{size_info}")
    
    print("\n✅ All icons polished and ready to use!")

if __name__ == "__main__":
    polish_all_icons()