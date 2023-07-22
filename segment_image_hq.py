import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO
import zipfile
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def seg_everything(image_path):
    # Load the model
    sam_checkpoint = "./pretrained_checkpoint/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    hq_token_only = False 

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)


    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    # 调用生成器
    mask_dicts = mask_generator.generate(image)

    # We need to set the point and box inputs.
    # In this case, we're assuming they're None (i.e., no specific points or boxes)
    #input_point, input_label, input_box = None, None, None

    # Generate masks
    #masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False, hq_token_only=False)
    #sorted_anns = sorted(zip(masks, scores), key=(lambda x: x[1]), reverse=True)  # Sort by scores instead of area
    sorted_anns = sorted(mask_dicts, key=(lambda x: x['area']), reverse=True)

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1

    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(img)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            # Apply the mask to the original image
            original_image = Image.open(image_path).convert('RGBA')  # Ensure original image is in RGBA mode
            region = original_image.crop((left, top, right, bottom))

            # Create an alpha mask
            alpha = mask_cropped.convert('L')  # Convert mask to mode 'L' (grayscale)

            # Combine the original image with the alpha mask
            result = Image.fromarray(np.array(region), mode='RGBA')  # Ensure result image is in RGBA mode
            result.putalpha(alpha)

            # Save the result to the zip file
            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
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