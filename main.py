import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser

# Initialize MediaPipe Hands with optimal performance settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video for better performance
    max_num_hands=2,
    min_detection_confidence=0.5,  # Reduced slightly for better performance
    min_tracking_confidence=0.5    # Reduced slightly for better performance
)

# Set up simplified landmark drawing for better performance
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

# Open webcam with lower resolution for better performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced from 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Reduced from 480
cap.set(cv2.CAP_PROP_FPS, 30)           # Limit FPS

# Hand landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0
THUMB_MCP = 2
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# YouTube control mode
youtube_mode = True

# Cooldown mechanism - shorter for more responsive control
last_action_time = 0
cooldown = 0.5  # seconds between actions

# Gesture recognition tracking
last_gesture = None
gesture_frames = 0
required_consecutive_frames = 2  # Reduced for faster response

# Disable mouse scroll events for palm gesture
pyautogui.FAILSAFE = True

# Process frames at reduced rate for performance
process_every_n_frames = 2
frame_count = 0

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks - simplified"""
    return ((landmark1.x - landmark2.x) ** 2 + 
            (landmark1.y - landmark2.y) ** 2) ** 0.5

def is_finger_extended(finger_tip, finger_mcp, wrist, threshold=0.1):
    """Check if a finger is extended based on tip position relative to MCP"""
    return finger_tip.y < finger_mcp.y - threshold

def is_palm_showing(landmarks):
    """Detect if the palm is showing - simplified implementation"""
    # Get relevant landmarks
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    # Get MCPs for better detection
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    
    # Check if fingers are extended
    fingers_extended = (
        is_finger_extended(index_tip, index_mcp, None) and
        is_finger_extended(middle_tip, middle_mcp, None) and
        is_finger_extended(ring_tip, ring_mcp, None) and
        is_finger_extended(pinky_tip, pinky_mcp, None)
    )
    
    return fingers_extended

def are_hands_crossed(hand_landmarks1, hand_landmarks2):
    """Detect if two hands are crossed - simplified"""
    if hand_landmarks1 and hand_landmarks2:
        # Get wrist positions for both hands
        wrist1 = hand_landmarks1.landmark[WRIST]
        wrist2 = hand_landmarks2.landmark[WRIST]
        
        # Get index finger tips for both hands
        index1 = hand_landmarks1.landmark[INDEX_TIP]
        index2 = hand_landmarks2.landmark[INDEX_TIP]
        
        # Check if hands are crossed
        crossed_condition1 = (wrist1.x < wrist2.x) and (index1.x > index2.x)
        crossed_condition2 = (wrist1.x > wrist2.x) and (index1.x < index2.x)
        
        return crossed_condition1 or crossed_condition2
    return False

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break
        
    # Flip and process only every n frames for better performance
    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # Just show the frame without processing
        cv2.imshow("YouTube Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Only convert to RGB for processing to save resources
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Create a simpler UI for better performance
    # Create semi-transparent box for info
    cv2.rectangle(frame, (10, 10), (310, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 100), (310, 220), (0, 0, 0), -1)
    
    current_time = time.time()
    action_ready = current_time - last_action_time > cooldown
    
    # Display status
    cv2.putText(frame, "YouTube Control Mode", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1)
    
    detection_status = "No Hand Detected"
    current_gesture = "None"
    
    # Check for hands
    if results.multi_hand_landmarks:
        # Two hands detection for crossed hands gesture
        if len(results.multi_hand_landmarks) == 2:
            detection_status = "Two Hands"
            
            # Draw hand landmarks with simplified settings
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
            
            # Check for crossed hands
            if are_hands_crossed(results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]):
                current_gesture = "Hands Crossed"
                
                # Track consecutive frames
                if last_gesture == "Hands Crossed":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Hands Crossed"
                
                # Execute with required frames
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.hotkey('ctrl', 'w')
                    last_action_time = current_time
            else:
                if last_gesture == "Hands Crossed":
                    last_gesture = None
                    gesture_frames = 0
        
        # Process single hand gestures
        elif len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks with simplified settings
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
            detection_status = "Hand Detected"
            
            # Get landmark positions
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            middle_tip = landmarks[MIDDLE_TIP]
            ring_tip = landmarks[RING_TIP]
            pinky_tip = landmarks[PINKY_TIP]
            
            # Get MCPs for better gesture detection
            thumb_mcp = landmarks[THUMB_MCP]
            index_mcp = landmarks[INDEX_MCP]
            middle_mcp = landmarks[MIDDLE_MCP]
            ring_mcp = landmarks[RING_MCP]
            pinky_mcp = landmarks[PINKY_MCP]
            
            # Calculate thumb-index distance
            thumb_index_dist = calculate_distance(thumb_tip, index_tip)
            
            # Check if fingers are extended (simplified calculations)
            index_extended = index_tip.y < index_mcp.y - 0.05
            middle_extended = middle_tip.y < middle_mcp.y - 0.05
            ring_extended = ring_tip.y < ring_mcp.y - 0.05
            pinky_extended = pinky_tip.y < pinky_mcp.y - 0.05
            thumb_extended = thumb_tip.y < thumb_mcp.y - 0.05
            
            # Check palm gesture (simplified test)
            if is_palm_showing(landmarks):
                current_gesture = "Palm"
                if last_gesture != "Palm":
                    last_gesture = "Palm"
                    gesture_frames = 1
                else:
                    gesture_frames += 1
            
            # Thumbs Up (Volume Up)
            elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                current_gesture = "Thumbs Up"
                if last_gesture == "Thumbs Up":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Thumbs Up"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    # Single press with modifier keys for efficiency
                    pyautogui.press('up', presses=5)
                    last_action_time = current_time
            
            # Thumbs Down (Volume Down)
            elif not thumb_extended and thumb_tip.y > thumb_mcp.y + 0.05 and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                current_gesture = "Thumbs Down"
                if last_gesture == "Thumbs Down":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Thumbs Down"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press('down', presses=5)
                    last_action_time = current_time
            
            # Two Fingers (Rewind)
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                current_gesture = "Two Fingers"
                if last_gesture == "Two Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Two Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press("left")
                    last_action_time = current_time
            
            # Three Fingers (Forward)
            elif index_extended and middle_extended and ring_extended and not pinky_extended:
                current_gesture = "Three Fingers"
                if last_gesture == "Three Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Three Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press("right")
                    last_action_time = current_time
            
            # Four Fingers (Full Screen)
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                current_gesture = "Four Fingers"
                if last_gesture == "Four Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Four Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press('f')
                    last_action_time = current_time
            
            # Pinch Gesture (Play/Pause)
            elif thumb_index_dist < 0.05:
                current_gesture = "Pinch"
                if last_gesture == "Pinch":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Pinch"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press("space")
                    last_action_time = current_time
            
            # No recognized gesture
            else:
                last_gesture = None
                gesture_frames = 0
    else:
        # Reset if no hands detected
        last_gesture = None
        gesture_frames = 0
    
    # Display status information - simplified
    cv2.putText(frame, f"Status: {detection_status}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Gesture: {current_gesture}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Simplified controls display
    cv2.putText(frame, "Controls:", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
    cv2.putText(frame, "Pinch: Play/Pause | Thumbs: Volume", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    cv2.putText(frame, "2 Fingers: Rewind | 3 Fingers: Forward", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    cv2.putText(frame, "4 Fingers: Fullscreen | Cross: Close", (20, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    
    # Show output
    cv2.imshow("YouTube Gesture Control", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()