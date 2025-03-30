import cv2
import numpy as np
import torch
import pyautogui
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from gesture_cnn import GestureCNN

# Load Model
model_path = "/home/jinwoo/Desktop/hand-sign-control-ML/gesture_model.pth"
num_classes = 663     # Make sure this matches your trained model's output classes
model = GestureCNN(num_classes)

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("[INFO] Model Loaded Successfully!")
except Exception as e:
    print(f"[ERROR] Failed to Load Model: {e}")
    exit()

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Open Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    exit()

print("[INFO] Webcam Opened Successfully. Press 'q' to exit.")

# Gesture to Action Mapping
gesture_map = {
    0: "Volume Up",
    1: "Volume Down",
    2: "Next Tab",
    3: "Previous Tab",
    4: "Pause/Play"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image!")
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Hand Region
            x_min, y_min, x_max, y_max = 10000, 10000, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            # Predict Gesture
            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (64, 64))
                img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
                img = transform(img).unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    output = model(img)
                    predicted_class = torch.argmax(output).item()

                gesture_name = gesture_map.get(predicted_class, "Unknown")
                print(f"[INFO] Detected Gesture: {gesture_name}")

                cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Perform Action
                if gesture_name == "Volume Up":
                    print("[ACTION] Increasing Volume")
                    pyautogui.press("volumeup")
                elif gesture_name == "Volume Down":
                    print("[ACTION] Decreasing Volume")
                    pyautogui.press("volumedown")
                elif gesture_name == "Next Tab":
                    print("[ACTION] Switching to Next Tab")
                    pyautogui.hotkey("ctrl", "tab")
                elif gesture_name == "Previous Tab":
                    print("[ACTION] Switching to Previous Tab")
                    pyautogui.hotkey("ctrl", "shift", "tab")
                elif gesture_name == "Pause/Play":
                    print("[ACTION] Pausing/Playing Video")
                    pyautogui.press("space")

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting Program...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program Ended Successfully!")
