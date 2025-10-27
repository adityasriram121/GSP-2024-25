import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'

# Initialize lists for data and labels
data = []
labels = []

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print("Data directory does not exist.")
else:
    print("Data directory found.")

# Process each image in the data directory
print("Processing images...")
for dir_ in os.listdir(DATA_DIR):
    # Check if directory is empty
    if not os.listdir(os.path.join(DATA_DIR, dir_)):
        print(f"No images found in directory: {dir_}")
        continue
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Check if image was loaded successfully
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)
        else:
            print(f"No hands detected in {img_path}")

# Report data collection status
print("Data collected:", len(data), "items.")
print("Labels collected:", len(labels), "items.")

# Attempt to save data to 'data.pickle'
print("Saving data to data.pickle...")
try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data successfully saved to data.pickle.")
except Exception as e:
    print("An error occurred while saving:", e)

try:
    hands.close()
except Exception:
    pass

