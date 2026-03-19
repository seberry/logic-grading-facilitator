"""
Improved Icon Polishing Script
Uses magenta background to accurately detect icon boundaries
"""

from PIL import Image
import os

def get_background_color(img):
    """Sample the (0,0) pixel to get background color"""
    pixel = img.getpixel((0, 0))
    return pixel

def make_background_transparent(img, bg_color, tolerance=0):
    """Convert background pixels to transparent"""
    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get pixel data
    data = img.getdata()
    
    # Create new data with background pixels made transparent
    new_data = []
    for item in data:
        # Check if pixel matches background color (with optional tolerance)
        if (abs(item[0] - bg_color[0]) <= tolerance and
            abs(item[1] - bg_color[1]) <= tolerance and
            abs(item[2] - bg_color[2]) <= tolerance):
            new_data.append((255, 255, 255, 0))  # Make transparent
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

def process_and_crop_icons(source_file, padding=15):
    """Process magenta-background image and crop three icons"""
    print("=" * 60)
    print("Improved Icon Polishing Script")
    print("=" * 60)
    
    # Load the source image
    print(f"\nLoading source file: {source_file}")
    img = Image.open(source_file)
    print(f"Image dimensions: {img.size}")
    
    # Sample background color from (0,0)
    bg_color = get_background_color(img)
    print(f"Background color at (0,0): {bg_color}")
    
    # Make background transparent
    print("Making background transparent...")
    img_transparent = make_background_transparent(img, bg_color, tolerance=30)
    
    # Get dimensions
    width, height = img_transparent.size
    icon_width = width // 3
    
    print(f"Icon width: {icon_width} pixels")
    
    # Crop three icons
    icons = []
    icon_names = ["classwork", "examgrader", "absence_basket"]
    
    for i, name in enumerate(icon_names):
        print(f"\nCropping {name} icon...")
        
        # Calculate crop box
        left = i * icon_width
        right = (i + 1) * icon_width
        box = (left, 0, right, height)
        
        # Crop the icon
        icon = img_transparent.crop(box)
        print(f"  Crop box: {box}")
        
        # Find content bounding box
        bbox = get_content_bounding_box(icon)
        min_x, min_y, max_x, max_y = bbox
        print(f"  Content bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        # Crop to content with padding
        padded_min_x = max(0, min_x - padding)
        padded_min_y = max(0, min_y - padding)
        padded_max_x = min(icon.width, max_x + padding)
        padded_max_y = min(icon.height, max_y + padding)
        
        icon_cropped = icon.crop((padded_min_x, padded_min_y, padded_max_x, padded_max_y))
        print(f"  Final size: {icon_cropped.size}")
        
        # Save the icon
        output_file = f"icon_{name}.png"
        icon_cropped.save(output_file, 'PNG')
        print(f"  ✓ Saved: {output_file}")
        
        icons.append((name, icon_cropped))
    
    # Generate classwork ICO
    print("\n" + "=" * 60)
    print("Generating classwork .ico file...")
    classwork_icon = icons[0][1]  # First icon is classwork
    
    # Resize to square for better icon quality
    size = 256
    classwork_square = classwork_icon.resize((size, size), Image.Resampling.LANCZOS)
    classwork_square.save("icon_classwork.ico", format='ICO', sizes=[(size, size)])
    print(f"✓ Saved: icon_classwork.ico ({size}x{size})")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, icon in icons:
        print(f"✓ icon_{name}.png - {icon.size}")
    print(f"✓ icon_classwork.ico - {size}x{size}")
    
    print("\n✅ All icons processed successfully!")
    print("Background removed, icons cropped to content, and .ico regenerated.")

if __name__ == "__main__":
    # Try the new magenta file first
    magenta_file = "Gemini_Generated_Image_fg6z5yfg6z5yfg6z.png"
    
    if os.path.exists(magenta_file):
        process_and_crop_icons(magenta_file, padding=15)
    else:
        print(f"Error: {magenta_file} not found!")
        print("Please make sure the magenta-background file is in the current directory.")