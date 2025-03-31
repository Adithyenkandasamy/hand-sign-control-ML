import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1  # Focus on one hand for better performance
)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Hand landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0

# Define additional palm landmark indices
THUMB_MCP = 2  # Metacarpophalangeal joint of the thumb
INDEX_MCP = 5  # Metacarpophalangeal joint of the index finger
MIDDLE_MCP = 9  # Metacarpophalangeal joint of the middle finger
RING_MCP = 13  # Metacarpophalangeal joint of the ring finger
PINKY_MCP = 17  # Metacarpophalangeal joint of the pinky

# Variables to track gesture states
pinch_active = False
thumbs_up_active = False
thumbs_down_active = False
two_fingers_active = False
three_fingers_active = False
four_fingers_active = False
palm_active = False  # Add palm detection state

# YouTube control mode - set to True by default to always enable video controls
youtube_mode = True

# Cooldown mechanism
last_action_time = 0
cooldown = 0.5  # seconds between actions

# Disable mouse scroll events for palm gesture
pyautogui.FAILSAFE = True

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return ((landmark1.x - landmark2.x) ** 2 + 
            (landmark1.y - landmark2.y) ** 2) ** 0.5

def is_finger_extended(finger_tip, finger_mcp, wrist, threshold=0.1):
    """Check if a finger is extended based on tip position relative to MCP"""
    # Check if the tip is significantly higher (lower y) than the MCP
    return finger_tip.y < finger_mcp.y - threshold

def is_palm_showing(landmarks):
    """Detect if the palm is showing to the camera"""
    # Get relevant landmarks
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    wrist = landmarks[WRIST]
    
    # Check if all fingers are extended and spread out
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    
    # Check if all fingers are extended
    fingers_extended = (
        is_finger_extended(thumb_tip, thumb_mcp, wrist) and
        is_finger_extended(index_tip, index_mcp, wrist) and
        is_finger_extended(middle_tip, middle_mcp, wrist) and
        is_finger_extended(ring_tip, ring_mcp, wrist) and
        is_finger_extended(pinky_tip, pinky_mcp, wrist)
    )
    
    # Check if fingers are spread apart (not pinched together)
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    fingers_spread = (
        thumb_index_dist > 0.1 and
        index_middle_dist > 0.05 and
        middle_ring_dist > 0.05 and
        ring_pinky_dist > 0.05
    )
    
    return fingers_extended and fingers_spread

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break
        
    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    # Add a black background for better text visibility
    info_panel = frame.copy()
    cv2.rectangle(info_panel, (30, 70), (300, 320), (0, 0, 0), -1)
    cv2.addWeighted(info_panel, 0.3, frame, 0.7, 0, frame)
    
    current_time = time.time()
    action_ready = current_time - last_action_time > cooldown
    
    # Display frame dimensions and status
    h, w, c = frame.shape
    cv2.putText(frame, f"Frame: {w}x{h}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display YouTube mode status
    cv2.putText(frame, "YouTube Control Mode", (30, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
    
    detection_status = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            detection_status = "Hand Detected"
            
            # Get landmark positions
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            middle_tip = landmarks[MIDDLE_TIP]
            ring_tip = landmarks[RING_TIP]
            pinky_tip = landmarks[PINKY_TIP]
            wrist = landmarks[WRIST]
            
            # Get MCPs for better gesture detection
            thumb_mcp = landmarks[THUMB_MCP]
            index_mcp = landmarks[INDEX_MCP]
            middle_mcp = landmarks[MIDDLE_MCP]
            ring_mcp = landmarks[RING_MCP]
            pinky_mcp = landmarks[PINKY_MCP]
            
            # Calculate distances
            thumb_index_dist = calculate_distance(thumb_tip, index_tip)
            
            # Check if fingers are extended relative to their MCPs
            index_extended = is_finger_extended(index_tip, index_mcp, wrist)
            middle_extended = is_finger_extended(middle_tip, middle_mcp, wrist)
            ring_extended = is_finger_extended(ring_tip, ring_mcp, wrist)
            pinky_extended = is_finger_extended(pinky_tip, pinky_mcp, wrist)
            
            # Track current gesture status
            current_gesture = "None"
            
            # Check for palm gesture first (to prevent other gestures from triggering)
            if is_palm_showing(landmarks):
                current_gesture = "Palm"
                cv2.putText(frame, "Palm Detected (No Action)", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                palm_active = True
                continue
            else:
                palm_active = False
            
            # 1. Thumbs Up (Volume Up)
            if (thumb_tip.y < thumb_mcp.y and 
                not index_extended and 
                not middle_extended and 
                not ring_extended and 
                not pinky_extended):
                
                current_gesture = "Thumbs Up"
                cv2.putText(frame, "Volume Up", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if action_ready and not thumbs_up_active:
                    pyautogui.press('up')
                    last_action_time = current_time
                    thumbs_up_active = True
            else:
                thumbs_up_active = False  
                
            # 2. Thumbs Down (Volume Down)
            if (thumb_tip.y > thumb_mcp.y + 0.05 and
                not index_extended and 
                not middle_extended and 
                not ring_extended and 
                not pinky_extended):
                
                current_gesture = "Thumbs Down"
                cv2.putText(frame, "Volume Down", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if action_ready and not thumbs_down_active:
                    pyautogui.press('down')
                    last_action_time = current_time
                    thumbs_down_active = True
            else:
                thumbs_down_active = False
                
            # 3. Two Fingers (Index & Middle Extended) → Rewind 10s
            if (index_extended and 
                middle_extended and 
                not ring_extended and 
                not pinky_extended):
                
                current_gesture = "Two Fingers"
                cv2.putText(frame, "Rewind 10s", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                if action_ready and not two_fingers_active:
                    pyautogui.press("left")
                    last_action_time = current_time
                    two_fingers_active = True
            else:
                two_fingers_active = False
                
            # 4. Three Fingers (Index, Middle & Ring Extended) → Forward 10s
            if (index_extended and 
                middle_extended and 
                ring_extended and 
                not pinky_extended):
                
                current_gesture = "Three Fingers"
                cv2.putText(frame, "Forward 10s", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                if action_ready and not three_fingers_active:
                    pyautogui.press("right")
                    last_action_time = current_time
                    three_fingers_active = True
            else:
                three_fingers_active = False
                
            # 5. Four Fingers (but not palm) - Toggle Full Screen
            if (index_extended and 
                middle_extended and 
                ring_extended and 
                pinky_extended and
                not is_palm_showing(landmarks)):
                
                current_gesture = "Four Fingers"
                cv2.putText(frame, "Toggle Full Screen", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if action_ready and not four_fingers_active:
                    pyautogui.press('f')
                    last_action_time = current_time
                    four_fingers_active = True
            else:
                four_fingers_active = False
                
            # 6. Pinch Gesture (Play/Pause)
            if thumb_index_dist < 0.05:  # Fingers close together
                current_gesture = "Pinch"
                cv2.putText(frame, "Play/Pause", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if action_ready and not pinch_active:
                    pyautogui.press("space")
                    last_action_time = current_time
                    pinch_active = True
            else:
                pinch_active = False
            
            # Display current gesture
            cv2.putText(frame, f"Gesture: {current_gesture}", (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display hand detection status
    cv2.putText(frame, detection_status, (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display controls information
    cv2.putText(frame, "YouTube Controls:", (30, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
    cv2.putText(frame, "- Pinch: Play/Pause", (40, 345), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- Thumbs Up/Down: Volume", (40, 365), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 2 Fingers: Rewind", (40, 385), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 3 Fingers: Forward", (40, 405), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 4 Fingers: Full Screen", (40, 425), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    # Show output
    cv2.imshow("YouTube Gesture Control", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()