from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLO11n model
model = YOLO("yolo11x-seg.pt")

def segment_image(folder, file):
    image_path = f"{folder}/{file}"
    # Run inference
    results = model(image_path)

    # Get first result
    r = results[0]

    # Get original image
    img = r.orig_img
    def write_image(image):
        path = f"{folder}-segmented/{file}"
                    # Create output directory if it doesn't exist
        output_dir = f"{folder}-segmented"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite(path, image)
        print(f"Image saved to {path}")


    if r.masks is None:
        print("No masks found in the image.")
        write_image(img)
        return

   
    # Check classes and masks
    classes = r.boxes.cls.cpu().numpy()  # class indices
    names = model.names  # class name mapping
    masks = r.masks.data.cpu().numpy()  # masks (N, H, W)

    found_person = False

    for i, class_id in enumerate(classes):
        label = names[int(class_id)]
        if label == "person":
            # Get person mask
            mask = masks[i]
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Make sure the mask matches the image size
            if mask_uint8.shape != img.shape[:2]:
                mask_uint8 = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]))

            # Convert to 3-channel for masking color image
            mask_3ch = cv2.merge([mask_uint8] * 3)

            # Apply mask to original image
            person_segment = cv2.bitwise_and(img, mask_3ch)

            # Optional: Make background transparent (Alpha)
            b, g, r = cv2.split(person_segment)
            alpha = mask_uint8
            rgba = cv2.merge([b, g, r, alpha])  # BGRA image

            # Save the segmented image
            write_image(rgba)

            found_person = True
            break

    if not found_person:           
        write_image(img)
        print("No person found in the image.")


# Input image path
# input_path = "input.jpg"
# segment_image(input_path)

# Read the files in folder
import os
folder_paths = ["data/query"]

for folder_path in folder_paths:
    input_files = os.listdir(folder_path)
    for file in input_files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            # input_file_path = os.path.join(folder_path, file)
            segment_image(folder_path, file)