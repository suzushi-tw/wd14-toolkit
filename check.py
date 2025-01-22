from PIL import Image
import os

def resave_image(path):
    try:
        img = Image.open(path)
        new_path = os.path.splitext(path)[0] + "_resaved.png"
        img.save(new_path)
        print(f"{path}: Image re-saved as {new_path}")
    except Exception as e:
        print(f"{path}: Error - {str(e)}")

# List of problematic images
problematic_images = [
    r"C:\Users\huang\Downloads\#Shinano (Azur Lane) Pictures, Images on pixiv, Japan\101629176_p0.png",
    r"C:\Users\huang\Downloads\#Shinano (Azur Lane) Pictures, Images on pixiv, Japan\116646570_p0.png",
    r"C:\Users\huang\Downloads\#Shinano (Azur Lane) Pictures, Images on pixiv, Japan\116646570_p1.png"
]

for img_path in problematic_images:
    resave_image(img_path)