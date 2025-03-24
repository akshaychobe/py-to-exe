
import cv2
import numpy as np
import os
from ultralytics import YOLO

output_dir = r"\\DAQ-001085\Rüstposition\Requirements\Master_Templates"

# Perform processing on warped images after the initial workflow
print("Processing warped images for big and small inserts...")

# Path to the model for big and small inserts
insert_model_path = r"\\DAQ-001085\Rüstposition\Requirements\Models\best_insert_segmentation_model.pt"

# Directory where warped images are saved
new_image = r"\\DAQ-001085\Rüstposition\Requirements\warped_dir\P04168 000001_1_Versuch_2024-11-06 10_49_02.jpg"  # Assuming warped images are saved here during the initial workflow

# Initialize YOLO model for big and small inserts
insert_model = YOLO(insert_model_path)

# Parameters for rectangle and slot division
rectangle_top_left = (170, 228)
rectangle_bottom_right = (2091, 430)
slot_width = (rectangle_bottom_right[0] - rectangle_top_left[0]) // 4  # Divide rectangle into 4 equal parts

# Tolerance for centroid deviation
tolerance = 15

# Insert dimensions
insert_dimensions = {
    "big": (110, 190),  # Width x Height
    "small": (60, 190),  # Width x Height
}

# Perform inference on warped images
warped_results = insert_model.predict(
    source=new_image,
    save=False,  # Do not save YOLO's default annotations; process them manually
    device="cpu",
    conf=0.75,
    show_labels=False,
    show_boxes=True,
    retina_masks=True
)

for result in warped_results:
    image_path = result.path
    warped_image = cv2.imread(image_path)

    if warped_image is None:
        print(f"Could not load image: {image_path}")
        continue

    # Initialize `expected_inserts` and `predefined_centroids` for the current image
    expected_inserts = {slot: {"big": 0, "small": 0} for slot in range(1, 5)}
    predefined_centroids = {slot: {"big": [], "small": []} for slot in range(1, 5)}

    # Divide slots
    slots = {}
    for i in range(4):
        slot_start_x = rectangle_top_left[0] + i * slot_width
        slot_end_x = slot_start_x + slot_width
        slots[i + 1] = ((slot_start_x, rectangle_top_left[1]), (slot_end_x, rectangle_bottom_right[1]))
        cv2.rectangle(warped_image, slots[i + 1][0], slots[i + 1][1], (255, 0, 0), 2)  # Blue slot rectangles

    # Process detected inserts
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        cls_id = int(cls.item())  # Class ID: 0 for big, 1 for small
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        insert_type = "big" if cls_id == 0 else "small"

        for slot, ((slot_x1, slot_y1), (slot_x2, slot_y2)) in slots.items():
            if slot_x1 <= centroid_x <= slot_x2 and slot_y1 <= centroid_y <= slot_y2:
                # Update `expected_inserts` and `predefined_centroids`
                expected_inserts[slot][insert_type] += 1
                predefined_centroids[slot][insert_type].append((centroid_x, centroid_y))

                # Draw bounding boxes
                cv2.rectangle(warped_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green bounding box
                cv2.circle(warped_image, (centroid_x, centroid_y), radius=5, color=(0, 0, 255), thickness=-1)  # Red centroid

    # Save the updated warped image
    output_image_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, output_image_name)
    cv2.imwrite(output_image_path, warped_image)
    
    # Save the results to a text file named after the image
    text_file_name = os.path.basename(image_path).replace(".png", ".txt").replace(".tif", ".txt").replace(".jpg", ".txt")
    text_file_path = os.path.join(output_dir, text_file_name)
    with open(text_file_path, "w") as f:
        f.write("expected_inserts = {\n")
        for slot, counts in expected_inserts.items():
            f.write(f"    {slot}: {counts},\n")
        f.write("}\n\n")
    
        f.write("predefined_centroids = {\n")
        for slot, centroids in predefined_centroids.items():
            f.write(f"    {slot}: {centroids},\n")
        f.write("}\n\n")
    
        f.write(f"Tolerance = {tolerance}\n")
    
    print(f"Results saved for {image_path}:")
    print(f" - Image: {output_image_path}")
    print(f" - Text: {text_file_path}")

