"""
Classwork Icon Processing Script
- Crops the classwork icon from source
- Makes background transparent
- Crops tightly to content (removes magenta strip)
- Pads to square with equal transparent margins
- Generates multi-resolution .ico file
"""

from PIL import Image, ImageOps
import os

def make_magenta_transparent(img, tolerance=50):
    """Convert magenta background pixels to transparent"""
    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get pixel data
    data = img.getdata()
    
    # Create new data with magenta pixels made transparent
    # Magenta is roughly (255, 0, 255)
    new_data = []
    for item in data:
        # Check if pixel matches magenta (with tolerance)
        if (abs(item[0] - 255) <= tolerance and
            abs(item[1] - 0) <= tolerance and
            abs(item[2] - 255) <= tolerance):
            new_data.append((255, 255, 255, 0))  # Make transparent
        else:
            new_data.append(item)
    
    # Update the image with new data
    img.putdata(new_data)
    return img

def trim_bottom_transparency(img, transparent_threshold=0.90):
    """Remove rows from bottom that are mostly transparent"""
    width, height = img.size
    
    # Scan from bottom up
    rows_to_remove = 0
    for y in range(height - 1, -1, -1):
        # Count transparent pixels in this row
        transparent_count = 0
        for x in range(width):
            pixel = img.getpixel((x, y))
            if pixel[3] == 0:  # Alpha channel is 0 (fully transparent)
                transparent_count += 1
        
        # If this row is mostly transparent, remove it
        transparent_ratio = transparent_count / width
        if transparent_ratio >= transparent_threshold:
            rows_to_remove += 1
            print(f"  Row {y}: {transparent_ratio:.2%} transparent - marking for removal")
        else:
            # Found content row, stop scanning
            print(f"  Row {y}: {transparent_ratio:.2%} transparent - found content")
            break
    
    if rows_to_remove > 0:
        print(f"  Removing {rows_to_remove} rows from bottom")
        new_height = height - rows_to_remove
        img = img.crop((0, 0, width, new_height))
        print(f"  New size after trim: {img.size}")
    
    return img

def crop_to_content(img):
    """Crop image to content using Pillow's getbbox()"""
    # getbbox() returns (left, top, right, bottom) of non-transparent pixels
    bbox = img.getbbox()
    
    if bbox is None:
        print("  Warning: No content found (image is all transparent)")
        return img
    
    print(f"  Content bbox: {bbox}")
    return img.crop(bbox)

def pad_to_square(img):
    """Pad image to square with equal transparent margins (no squashing)"""
    width, height = img.size
    
    print(f"  Current size: {width}x{height}")
    
    # Determine square size (max dimension)
    square_size = max(width, height)
    
    # Calculate how much padding needed on each side
    pad_left = (square_size - width) // 2
    pad_right = square_size - width - pad_left
    pad_top = (square_size - height) // 2
    pad_bottom = square_size - height - pad_top
    
    print(f"  Padding to square: {square_size}x{square_size}")
    print(f"  Left padding: {pad_left}, Right padding: {pad_right}")
    print(f"  Top padding: {pad_top}, Bottom padding: {pad_bottom}")
    
    # Create new square transparent canvas
    square = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
    
    # Paste image centered
    square.paste(img, (pad_left, pad_top))
    
    return square

def create_ico_from_png(png_path, ico_path, sizes=[256, 128, 64, 48, 32, 16]):
    """Create ICO file with multiple sizes from PNG"""
    print(f"\nCreating ICO: {png_path} → {ico_path}")
    
    img = Image.open(png_path)
    
    # Create list of resized images for different sizes
    icon_images = []
    for size in sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        icon_images.append(resized)
        print(f"  Added {size}x{size}")
    
    # Save as ICO - Pillow handles the sizes automatically when we pass the list
    img.save(ico_path, format='ICO')
    print(f"  ✓ Saved: {ico_path}")
    
    return True

def process_classwork_icon(source_file):
    """Process the classwork icon from the source image"""
    print("=" * 70)
    print("Classwork Icon Processing - Tight Crop → Square Padding → ICO")
    print("=" * 70)
    
    # Load the source image
    print(f"\nLoading source file: {source_file}")
    img = Image.open(source_file)
    print(f"Source dimensions: {img.size}")
    
    # Step 1: Crop the classwork icon (first 1/3 of the image)
    print("\n" + "=" * 70)
    print("Step 1: Cropping classwork icon (left 1/3 of image)")
    print("=" * 70)
    
    width, height = img.size
    icon_width = width // 3
    left = 0
    right = icon_width
    box = (left, 0, right, height)
    
    icon = img.crop(box)
    print(f"Initial crop: {icon.size}")
    
    # Step 2: Make magenta background transparent (with higher tolerance)
    print("\n" + "=" * 70)
    print("Step 2: Making magenta background transparent (higher tolerance)")
    print("=" * 70)
    icon_transparent = make_magenta_transparent(icon, tolerance=50)
    print(f"  ✓ Magenta pixels made transparent")
    
    # Step 3: Trim transparent rows from bottom (removes magenta strip and blank space)
    print("\n" + "=" * 70)
    print("Step 3: Trimming transparent rows from bottom")
    print("=" * 70)
    icon_trimmed = trim_bottom_transparency(icon_transparent, transparent_threshold=0.90)
    print(f"After bottom trim: {icon_trimmed.size}")
    
    # Step 4: Crop tightly to content (removes any remaining extra space)
    print("\n" + "=" * 70)
    print("Step 4: Cropping tightly to content")
    print("=" * 70)
    icon_content = crop_to_content(icon_trimmed)
    print(f"After tight crop: {icon_content.size}")
    
    # Step 5: Pad to square with equal transparent margins
    print("\n" + "=" * 70)
    print("Step 5: Padding to square (centered, no squashing)")
    print("=" * 70)
    icon_square = pad_to_square(icon_content)
    print(f"After square padding: {icon_square.size}")
    
    # Step 6: Save as PNG
    output_png = "icon_classwork.png"
    icon_square.save(output_png, 'PNG')
    print(f"\n✓ Saved PNG: {output_png}")
    
    # Step 7: Generate ICO file
    print("\n" + "=" * 70)
    print("Step 6: Generating multi-resolution .ico file")
    print("=" * 70)
    output_ico = "icon_classwork.ico"
    create_ico_from_png(output_png, output_ico)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ {output_png} - {icon_square.size} (square, tightly cropped)")
    print(f"✓ {output_ico} - Multi-size (16x16 to 256x256)")
    
    print("\n✅ Classwork icon processed successfully!")
    print("Background transparent → Tightly cropped → Centered square → Ready for use!")

if __name__ == "__main__":
    source_file = "Gemini_Generated_Image_fg6z5yfg6z5yfg6z.png"
    
    if os.path.exists(source_file):
        process_classwork_icon(source_file)
    else:
        print(f"Error: {source_file} not found!")
        print("Please make sure the source file is in the current directory.")