import cv2
import os
import time

# Initialize the camera
cap = cv2.VideoCapture(1)  # Use 0 for default camera, change if necessary

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