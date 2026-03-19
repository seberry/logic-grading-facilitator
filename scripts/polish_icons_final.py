"""
Final Icon Polishing Script
Crops to content, pads to square (no squashing), and generates clean icons
"""

from PIL import Image, ImageOps
import os

def get_background_color(img):
    """Sample (0,0) pixel to get background color"""
    pixel = img.getpixel((0, 0))
    return pixel

def make_background_transparent(img, bg_color, tolerance=30):
    """Convert background pixels to transparent"""
    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get pixel data
    data = img.getdata()
    
    # Create new data with background pixels made transparent
    new_data = []
    for item in data:
        # Check if pixel matches background color (with tolerance)
        if (abs(item[0] - bg_color[0]) <= tolerance and
            abs(item[1] - bg_color[1]) <= tolerance and
            abs(item[2] - bg_color[2]) <= tolerance):
            new_data.append((255, 255, 255, 0))  # Make transparent
        else:
            new_data.append(item)
    
    # Update the image with new data
    img.putdata(new_data)
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

def pad_to_square(img, padding=10):
    """Pad image to square with equal transparent margins"""
    width, height = img.size
    
    # Determine square size (max dimension + padding)
    square_size = max(width, height) + (2 * padding)
    
    print(f"  Current size: {width}x{height}")
    print(f"  Padding to: {square_size}x{square_size}")
    
    # Create new square transparent canvas
    square = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
    
    # Calculate centered position
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    
    # Paste image centered
    square.paste(img, (paste_x, paste_y))
    
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

def process_and_crop_icons(source_file, padding=10):
    """Process magenta-background image and crop three icons properly"""
    print("=" * 70)
    print("Final Icon Polishing Script - Crop → Pad Square → Resize")
    print("=" * 70)
    
    # Load the source image
    print(f"\nLoading source file: {source_file}")
    img = Image.open(source_file)
    print(f"Source dimensions: {img.size}")
    
    # Sample background color from (0,0)
    bg_color = get_background_color(img)
    print(f"Background color at (0,0): {bg_color}")
    
    # Make background transparent
    print("\n" + "=" * 70)
    print("Step 1: Making background transparent...")
    print("=" * 70)
    img_transparent = make_background_transparent(img, bg_color, tolerance=30)
    
    # Get dimensions
    width, height = img_transparent.size
    icon_width = width // 3
    print(f"Icon width: {icon_width} pixels")
    
    # Crop three icons
    icons = []
    icon_names = ["classwork", "examgrader", "absence_basket"]
    
    for i, name in enumerate(icon_names):
        print(f"\n" + "=" * 70)
        print(f"Processing {name} icon...")
        print("=" * 70)
        
        # Calculate crop box
        left = i * icon_width
        right = (i + 1) * icon_width
        box = (left, 0, right, height)
        
        # Crop the icon
        icon = img_transparent.crop(box)
        print(f"Initial crop: {icon.size}")
        
        # Step 1: Crop to content
        print("\nStep 2: Cropping to content...")
        icon_content = crop_to_content(icon)
        print(f"After content crop: {icon_content.size}")
        
        # Step 2: Pad to square
        print("\nStep 3: Padding to square...")
        icon_square = pad_to_square(icon_content, padding=padding)
        print(f"After square padding: {icon_square.size}")
        
        # Save as PNG
        output_file = f"icon_{name}.png"
        icon_square.save(output_file, 'PNG')
        print(f"✓ Saved: {output_file}")
        
        icons.append((name, icon_square))
    
    # Generate multi-size ICO for classwork
    print("\n" + "=" * 70)
    print("Generating classwork .ico with multiple sizes...")
    print("=" * 70)
    classwork_icon = icons[0][1]
    create_ico_from_png("icon_classwork.png", "icon_classwork.ico")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, icon in icons:
        print(f"✓ icon_{name}.png - {icon.size} (square, no squashing)")
    print(f"✓ icon_classwork.ico - Multi-size (16x16 to 256x256)")
    
    print("\n✅ All icons processed successfully!")
    print("Background transparent → Cropped to content → Padded to square → Ready for use!")

if __name__ == "__main__":
    # Try magenta file
    magenta_file = "Gemini_Generated_Image_fg6z5yfg6z5yfg6z.png"
    
    if os.path.exists(magenta_file):
        process_and_crop_icons(magenta_file, padding=15)
    else:
        print(f"Error: {magenta_file} not found!")
        print("Please make sure the magenta-background file is in the current directory.")