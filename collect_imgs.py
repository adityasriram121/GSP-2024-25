import cv2
import os
import time
import argparse
# CLI args for camera index
parser = argparse.ArgumentParser(description='Capture images for a class label')
parser.add_argument('--cam', type=int, default=None, help='Camera index (overrides CAM_INDEX env)')
args, unknown = parser.parse_known_args()
cam_index = args.cam if args.cam is not None else int(os.environ.get('CAM_INDEX', '0'))

# Initialize the camera
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f'Failed to open camera index {cam_index}. Try a different index.')
    raise SystemExit(1)

# Create data directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Get class name from user
class_name = input("Enter the class name for the images: ")
class_dir = os.path.join(data_dir, class_name)

# Create class directory if it doesn't exist
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

# Initialize counter for images
counter = 0

print("Press 'c' to capture an image, 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('Capture', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, save the image
    if key == ord('c'):
        counter += 1
        img_name = f"{class_name}_{counter}.jpg"
        img_path = os.path.join(class_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Image saved: {img_name}")
        
        # Optional: Add a small delay to avoid accidental multiple captures
        time.sleep(0.5)

    # If 'q' is pressed, quit the program
    elif key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Total images captured for {class_name}: {counter}")





