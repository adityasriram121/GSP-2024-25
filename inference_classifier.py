import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Load the trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe and OpenCV components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Label dictionary for predictions

def extract_full_landmark_data(hand_landmarks):
    """
    Extracts x and y coordinates for all 21 landmarks.
    """
    data_aux = []
    for landmark in hand_landmarks.landmark:
        data_aux.extend([landmark.x, landmark.y])  # Collect x, y coordinates for each landmark
    return data_aux

def predict_sign(data):
    """
    Uses the model to predict the hand sign based on landmark data.
    """
    try:
        prediction = model.predict([data])[0]
        return str(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "?"

def draw_predictions(frame, label, bounding_box):
    """
    Draws the bounding box and prediction label on the frame.
    """
    x1, y1, x2, y2 = bounding_box
    # Draw bounding box around the hand
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Display predicted label
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(int(os.environ.get('CAM_INDEX', '0')))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame for a mirror effect
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract all landmark coordinates
                data_aux = extract_full_landmark_data(hand_landmarks)
                predicted_character = predict_sign(data_aux)

                # Calculate bounding box for display
                x_values = [lm.x for lm in hand_landmarks.landmark]
                y_values = [lm.y for lm in hand_landmarks.landmark]
                x1, y1 = int(min(x_values) * W) - 10, int(min(y_values) * H) - 10
                x2, y2 = int(max(x_values) * W) + 10, int(max(y_values) * H) + 10

                # Draw prediction on the frame
                draw_predictions(frame, predicted_character, (x1, y1, x2, y2))
        else:
            # Display a message if no hand is detected
            cv2.putText(frame, "Place hand in view for detection", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()


