import cv2
import torch
import numpy as np
import pyautogui
import mediapipe as mp
import time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
class GestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 5)  # Only 5 gestures

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load model
model = GestureModel().to(device)
checkpoint = torch.load('/home/jinwoo/Desktop/hand-sign-control-ML/gesture_model.pth', map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture mappings
GESTURES = {
    0: ('Thumbs Up', 'volumeup', "Volume Increased"),
    1: ('Thumbs Down', 'volumedown', "Volume Decreased"),
    2: ('Left Swipe', 'prevtrack', "Rewinded 10 seconds"),
    3: ('Right Swipe', 'nexttrack', "Forwarded 10 seconds"),
    4: ('Stop', 'playpause', "Playback Paused")
}

# Cooldown settings
last_action_time = 0
cooldown_time = 1  # 1 second

# Video capture setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    action_text = "No gesture detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get bounding box
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Expand bounding box
        expand = 20
        x_min, y_min = max(0, x_min - expand), max(0, y_min - expand)
        x_max, y_max = min(w, x_max + expand), min(h, y_max + expand)

        # Crop and preprocess hand image
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size > 0:
            input_tensor = transform(cv2.resize(hand_img, (128, 128))).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, gesture_id = torch.max(probabilities, dim=1)

            confidence, gesture_id = confidence.item(), gesture_id.item()

            if time.time() - last_action_time > cooldown_time and confidence > 0.9:
                if gesture_id in GESTURES:
                    gesture_name, key, action_text = GESTURES[gesture_id]
                    pyautogui.press(key)
                    last_action_time = time.time()
                    print(f"Action: {action_text} | Gesture: {gesture_name} | Confidence: {confidence:.2f}")

    # Display on screen
    cv2.putText(frame, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
