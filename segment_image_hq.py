import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO
import zipfile
from segment_anything import sam_model_registry, SamPredictor

def seg_everything(image_path):
    # Load the model
    sam_checkpoint = "../sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    hq_token_only = False 

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # We need to set the point and box inputs.
    # In this case, we're assuming they're None (i.e., no specific points or boxes)
    input_point, input_label, input_box = None, None, None

    # Generate masks
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False, hq_token_only=False)
    sorted_anns = sorted(zip(masks, scores), key=(lambda x: x[1]), reverse=True)  # Sort by scores instead of area

    zip_buffer = BytesIO()
    for idx, (mask, score) in enumerate(sorted_anns, 0):
        # Convert mask to the same format as the original code
        mask = (mask * 255).astype(np.uint8)

        # Find the bounding box
        y_indices, x_indices = np.where(mask)
        left, right = x_indices.min(), x_indices.max()
        top, bottom = y_indices.min(), y_indices.max()

        mask_cropped = Image.fromarray(mask[top:bottom+1, left:right+1])

        # Apply the mask to the original image
        original_image = Image.open(image_path).convert('RGBA')  # Ensure original image is in RGBA mode
        ### region = original_image.crop((left, top, right, bottom))

        # 获取遮罩的大小
        mask_width, mask_height = mask_cropped.size

        # 根据遮罩的大小裁剪原始图像
        region = original_image.crop((left, top, left + mask_width, top + mask_height)) 

        # Create an alpha mask
        alpha = mask_cropped.convert('L')  # Convert mask to mode 'L' (grayscale)

        # Combine the original image with the alpha mask
        result = Image.fromarray(np.array(region), mode='RGBA')  # Ensure result image is in RGBA mode
        result.putalpha(alpha)

        # Save the result to the zip file
        result_bytes = BytesIO()
        result.save(result_bytes, format="PNG")
        result_bytes.seek(0)
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

        # If it's the first slice, save it separately
        if idx == 0:
            seg_0_filename = f"{image_path.rsplit('.', 1)[0]}_seg_0.png"
            result.save(seg_0_filename, format="PNG")
    
    # Save the zip file
    with open(f"{image_path.rsplit('.', 1)[0]}.zip", 'wb') as f:
        f.write(zip_buffer.getvalue())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment an image and save the result as a zip file.')
    parser.add_argument('image_path', type=str, help='The path of the image to be segmented.')
    args = parser.parse_args()
    seg_everything(args.image_path)