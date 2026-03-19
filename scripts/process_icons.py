"""
Icon Processing Script
Crops gemini_generated_icons.png into three separate icons and converts one to .ico format
"""

from PIL import Image
import os

def process_icons():
    # Paths
    source_file = "gemini_generated_icons.png"
    output_dir = "."  # Save in root directory
    
    # Output files
    classwork_png = os.path.join(output_dir, "icon_classwork.png")
    examgrader_png = os.path.join(output_dir, "icon_examgrader.png")
    absence_basket_png = os.path.join(output_dir, "icon_absence_basket.png")
    classwork_ico = os.path.join(output_dir, "icon_classwork.ico")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found in the current directory.")
        return False
    
    try:
        # Open the source image
        print(f"Opening {source_file}...")
        img = Image.open(source_file)
        width, height = img.size
        
        print(f"Image dimensions: {width}x{height}")
        
        # Calculate icon width (divide by 3)
        icon_width = width // 3
        print(f"Each icon will be approximately {icon_width} pixels wide")
        
        # Crop the three icons (left, middle, right)
        # Left icon: (0, 0, icon_width, height)
        print("Cropping left icon (classwork)...")
        left_box = (0, 0, icon_width, height)
        left_icon = img.crop(left_box)
        left_icon.save(classwork_png)
        print(f"Saved: {classwork_png}")
        
        # Middle icon: (icon_width, 0, 2*icon_width, height)
        print("Cropping middle icon (exam grader)...")
        middle_box = (icon_width, 0, 2 * icon_width, height)
        middle_icon = img.crop(middle_box)
        middle_icon.save(examgrader_png)
        print(f"Saved: {examgrader_png}")
        
        # Right icon: (2*icon_width, 0, width, height)
        print("Cropping right icon (absence basket)...")
        right_box = (2 * icon_width, 0, width, height)
        right_icon = img.crop(right_box)
        right_icon.save(absence_basket_png)
        print(f"Saved: {absence_basket_png}")
        
        # Convert classwork icon to .ico format
        print("Converting classwork icon to .ico format...")
        # ICO files work best with square icons
        # We'll use a 256x256 size for good quality
        left_icon_square = left_icon.resize((256, 256), Image.Resampling.LANCZOS)
        left_icon_square.save(classwork_ico, format='ICO', sizes=[(256, 256)])
        print(f"Saved: {classwork_ico}")
        
        print("\n✅ All icons processed successfully!")
        print(f"\nGenerated files:")
        print(f"  • {classwork_png}")
        print(f"  • {examgrader_png}")
        print(f"  • {absence_basket_png}")
        print(f"  • {classwork_ico}")
        
        return True
        
    except Exception as e:
        print(f"Error processing icons: {e}")
        return False

if __name__ == "__main__":
    process_icons()