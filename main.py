import cv2
import numpy as np
import pyautogui

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range for hand detection
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to filter skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply Gaussian Blur to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) > 5000:  # Ignore small detections
            # Get the bounding box of the hand
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Find the center of the hand
            hand_x = x + w // 2
            hand_y = y + h // 2
            cv2.circle(frame, (hand_x, hand_y), 5, (0, 0, 255), -1)

            # Map hand position to screen coordinates
            screen_x = np.interp(hand_x, (100, 540), (0, screen_width))
            screen_y = np.interp(hand_y, (50, 380), (0, screen_height))

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

    # Display the frame
    cv2.imshow("Hand Tracking Mouse Control", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
