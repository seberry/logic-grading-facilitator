import cv2
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. PASTE YOUR LLM OUTPUT HERE
# ==========================================
llm_json_response = """
{
"regions": [
{
"id": 1,
"content_type": "main_proof",
"box_2d": [100, 50, 400, 900]
},
{
"id": 2,
"content_type": "sub_proof_indented",
"box_2d": [210, 150, 750, 900]
}
]
}
"""
# (Note: I put a dummy box roughly where your text is. 
# Replace the content inside """ """ with what your LLM gives you.)

# ==========================================
# 2. CONFIGURATION
# ==========================================
IMAGE_PATH = "fitch_sample2.jpg"  # Ensure this matches your filename

def verify_boxes():
    # Load Image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load {IMAGE_PATH}")
        return
    
    # Get Dimensions for scaling 0-1000 -> Pixels
    h, w, _ = img.shape
    
    try:
        data = json.loads(llm_json_response)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        return

    # Prepare Visualization
    annotated_img = img.copy()
    crops = []

    print(f"Found {len(data['regions'])} regions.")

    for region in data['regions']:
        # Extract normalized coords
        # OLD WAY (Matches your original prompt, but LLM ignored it)
        # norm_ymin, norm_xmin, norm_ymax, norm_xmax = region['box_2d']

        # NEW WAY (Matches what the LLM actually output: xmin, ymin, xmax, ymax)
        norm_xmin, norm_ymin, norm_xmax, norm_ymax = region['box_2d']
        
        # Convert to Pixels
        ymin = int((norm_ymin / 1000) * h)
        ymax = int((norm_ymax / 1000) * h)
        xmin = int((norm_xmin / 1000) * w)
        xmax = int((norm_xmax / 1000) * w)

        # 1. Draw Rectangle on Main Image (Red)
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
        
        # 2. Add Label
        label = f"{region['id']}: {region.get('content_type', 'region')}"
        cv2.putText(annotated_img, label, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 3. Store Crop for display
        # specific check to ensure valid crop
        if ymax > ymin and xmax > xmin:
            crop = img[ymin:ymax, xmin:xmax]
            crops.append((label, crop))

    # --- DISPLAY RESULTS ---
    
    # Show full annotated page
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("LLM Bounding Box Proposals")
    plt.axis('off')
    plt.show()

    # Show individual crops (Grid view)
    if crops:
        rows = (len(crops) // 2) + 1
        plt.figure(figsize=(12, 4 * rows))
        for i, (label, crop_img) in enumerate(crops):
            plt.subplot(rows, 2, i + 1)
            plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            plt.title(label)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_boxes()