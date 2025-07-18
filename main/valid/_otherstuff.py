from PIL import Image
import os

def get_image_dimensions(image_path):
    """
    Opens an image file and prints its width and height.

    Args:
        image_path (str): The path to the image file.
    """
    # Check if the file exists before trying to open it
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return

    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Get width and height
            width, height = img.size
            print(f"✅ Image: {os.path.basename(image_path)}")
            print(f"  - Width:  {width} pixels")
            print(f"  - Height: {height} pixels")
    except Exception as e:
        print(f"❌ Error: Could not read the image file. Reason: {e}")

# --- Configuration ---
# ⚠️ IMPORTANT: Replace this with the path to one of your image files.
IMAGE_FILE_PATH = r'./IMG_1707_JPEG.rf.192b409b40f23c3d6914b30958e34547.jpg' # Or .png, .jpeg, etc.

get_image_dimensions(IMAGE_FILE_PATH)