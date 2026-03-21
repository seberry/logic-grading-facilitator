from PIL import Image

# Verify PNG
print("=" * 70)
print("Verifying icon_classwork.png")
print("=" * 70)
img = Image.open('icon_classwork.png')
print(f'Dimensions: {img.size}')
print(f'Mode: {img.mode}')
print(f'Has transparent pixels: {img.mode == "RGBA"}')

# Verify ICO
print("\n" + "=" * 70)
print("Verifying icon_classwork.ico")
print("=" * 70)
ico = Image.open('icon_classwork.ico')
print(f'Mode: {ico.mode}')
print(f'Current size: {ico.size}')
print(f'Formats available: {ico.info.get("formats", "N/A")}')

print("\n✅ Both files created successfully!")