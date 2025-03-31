import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Hand landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0

# Track previous X position of index finger for swipes
prev_index_x = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            middle_tip = landmarks[MIDDLE_TIP]
            ring_tip = landmarks[RING_TIP]
            pinky_tip = landmarks[PINKY_TIP]
            wrist = landmarks[WRIST]

            # Gesture Recognition
            # 1. Thumbs Up (Increase Volume)
            if thumb_tip.y < index_tip.y and thumb_tip.y < pinky_tip.y:
                pyautogui.press("volumeup")
                cv2.putText(frame, "Volume Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 2. Thumbs Down (Decrease Volume)
            elif thumb_tip.y > index_tip.y and thumb_tip.y > pinky_tip.y:
                pyautogui.press("volumedown")
                cv2.putText(frame, "Volume Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 3. Left Swipe (Rewind 10s)
            if prev_index_x is not None and (index_tip.x - prev_index_x) < -0.05:
                pyautogui.press("left")
                cv2.putText(frame, "Rewind 10s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 4. Right Swipe (Jump forward 10s)
            elif prev_index_x is not None and (index_tip.x - prev_index_x) > 0.05:
                pyautogui.press("right")
                cv2.putText(frame, "Forward 10s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 5. Stop (All Fingers Extended → Pause Video)
            fingers_up = sum(1 for f in [index_tip, middle_tip, ring_tip, pinky_tip] if f.y < wrist.y)
            if fingers_up == 4:
                pyautogui.press("space")
                cv2.putText(frame, "Paused", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Update previous index finger position
            prev_index_x = index_tip.x

    # Show output
    cv2.imshow("Gesture Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
