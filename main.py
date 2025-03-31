import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2  # Changed to 2 to detect two hands for crossed gesture
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
palm_active = False
hands_crossed_active = False

# YouTube control mode - set to True by default to always enable video controls
youtube_mode = True

# Cooldown mechanism
last_action_time = 0
cooldown = 0.5  # seconds between actions

# Gesture recognition timing variables
thumbs_up_start_time = 0
thumbs_down_start_time = 0
two_fingers_start_time = 0
three_fingers_start_time = 0
four_fingers_start_time = 0
pinch_start_time = 0
hands_crossed_start_time = 0
continuous_action_interval = 3.0  # seconds between repeating actions

# Minimum gesture hold time (NEW)
min_gesture_time = 1.0  # Only activate after holding for 1 second

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

def are_hands_crossed(hand_landmarks1, hand_landmarks2):
    """Detect if two hands are crossed"""
    if hand_landmarks1 and hand_landmarks2:
        # Get wrist positions for both hands
        wrist1 = hand_landmarks1.landmark[WRIST]
        wrist2 = hand_landmarks2.landmark[WRIST]
        
        # Get index finger tips for both hands
        index1 = hand_landmarks1.landmark[INDEX_TIP]
        index2 = hand_landmarks2.landmark[INDEX_TIP]
        
        # Check if hands are crossed
        # Left hand's wrist is on the left but its index finger is on the right
        # Right hand's wrist is on the right but its index finger is on the left
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
        
    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    # Add a black background for better text visibility
    info_panel = frame.copy()
    cv2.rectangle(info_panel, (30, 70), (300, 320), (0, 0, 0), -1)
    cv2.addWeighted(info_panel, 0.3, frame, 0.7, 0, frame)
    
    # Add a separate box for palm detection visualization
    palm_box = frame.copy()
    cv2.rectangle(palm_box, (450, 70), (610, 150), (0, 0, 0), -1)
    cv2.addWeighted(palm_box, 0.3, frame, 0.7, 0, frame)
    
    current_time = time.time()
    action_ready = current_time - last_action_time > cooldown
    
    # Display frame dimensions and status
    h, w, c = frame.shape
    cv2.putText(frame, f"Frame: {w}x{h}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display YouTube mode status
    cv2.putText(frame, "YouTube Control Mode", (30, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
    
    # Initialize palm detection box with default text
    cv2.putText(frame, "Palm Status", (470, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Not Detected", (480, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    detection_status = "No Hand Detected"
    
    # Check for crossed hands first - need two hands for this gesture
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        detection_status = "Two Hands Detected"
        
        # Draw both hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Check if hands are crossed
        if are_hands_crossed(results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]):
            cv2.putText(frame, "Hands Crossed Detected", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Start timing if not already tracking
            if not hands_crossed_active:
                hands_crossed_start_time = current_time
                hands_crossed_active = True
            
            # Check if gesture held long enough
            if hands_crossed_active and (current_time - hands_crossed_start_time >= min_gesture_time):
                hold_time = current_time - hands_crossed_start_time
                cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 275), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Close YouTube website if held for required time and not recently acted
                if action_ready and hold_time >= min_gesture_time:
                    cv2.putText(frame, "ACTION: Closing YouTube", (50, 300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Use JavaScript to close the browser tab (works for YouTube)
                    pyautogui.hotkey('ctrl', 'w')
                    last_action_time = current_time
        else:
            hands_crossed_active = False
    
    # Process single hand gestures
    elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        
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
            
            # Update the palm status box
            cv2.rectangle(frame, (450, 70), (610, 150), (0, 100, 0), 2)  # Green outline
            cv2.putText(frame, "Palm Status", (470, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "DETECTED", (480, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            palm_active = True
            continue
        else:
            palm_active = False
            # Reset palm box to default
            cv2.rectangle(frame, (450, 70), (610, 150), (100, 0, 0), 2)  # Red outline
        
        # 1. Thumbs Up (Volume Up by 10)
        if (thumb_tip.y < thumb_mcp.y and 
            not index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended):
            
            current_gesture = "Thumbs Up"
            cv2.putText(frame, "Volume Up (10%)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Start timing if not already tracking
            if not thumbs_up_active:
                thumbs_up_start_time = current_time
                thumbs_up_active = True
                
            # Show hold duration
            hold_time = current_time - thumbs_up_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Only trigger if held for minimum time
            if thumbs_up_active and hold_time >= min_gesture_time:
                # Initial press
                if action_ready:
                    # Press up arrow 10 times for 10% volume increase
                    for _ in range(10):
                        pyautogui.press('up')
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Volume +10%", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            thumbs_up_active = False  
            
        # 2. Thumbs Down (Volume Down by 10)
        if (thumb_tip.y > thumb_mcp.y + 0.05 and
            not index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended):
            
            current_gesture = "Thumbs Down"
            cv2.putText(frame, "Volume Down (10%)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Start timing if not already tracking
            if not thumbs_down_active:
                thumbs_down_start_time = current_time
                thumbs_down_active = True
            
            # Show hold duration
            hold_time = current_time - thumbs_down_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Only trigger if held for minimum time
            if thumbs_down_active and hold_time >= min_gesture_time:
                # Initial press
                if action_ready:
                    # Press down arrow 10 times for 10% volume decrease
                    for _ in range(10):
                        pyautogui.press('down')
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Volume -10%", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
            
            # Start timing if not already tracking
            if not two_fingers_active:
                two_fingers_start_time = current_time
                two_fingers_active = True
            
            # Show hold duration
            hold_time = current_time - two_fingers_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Only trigger if held for minimum time
            if two_fingers_active and hold_time >= min_gesture_time:
                # Initial press
                if action_ready:
                    pyautogui.press("left")
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Rewind", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
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
            
            # Start timing if not already tracking
            if not three_fingers_active:
                three_fingers_start_time = current_time
                three_fingers_active = True
            
            # Show hold duration
            hold_time = current_time - three_fingers_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Only trigger if held for minimum time
            if three_fingers_active and hold_time >= min_gesture_time:
                # Initial press
                if action_ready:
                    pyautogui.press("right")
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Forward", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
            
            # Start timing if not already tracking
            if not four_fingers_active:
                four_fingers_start_time = current_time
                four_fingers_active = True
            
            # Show hold duration
            hold_time = current_time - four_fingers_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 225), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Only trigger if held for minimum time
            if four_fingers_active and hold_time >= min_gesture_time:
                if action_ready:
                    pyautogui.press('f')
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Fullscreen", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            four_fingers_active = False
            
        # 6. Pinch Gesture (Play/Pause)
        if thumb_index_dist < 0.05:  # Fingers close together
            current_gesture = "Pinch"
            cv2.putText(frame, "Play/Pause", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Start timing if not already tracking
            if not pinch_active:
                pinch_start_time = current_time
                pinch_active = True
            
            # Show hold duration
            hold_time = current_time - pinch_start_time
            cv2.putText(frame, f"Holding: {hold_time:.1f}s", (50, 275), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Only trigger if held for minimum time
            if pinch_active and hold_time >= min_gesture_time:
                if action_ready:
                    pyautogui.press("space")
                    last_action_time = current_time
                    
                # Show that action was performed
                cv2.putText(frame, "ACTION: Play/Pause", (50, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            pinch_active = False
        
        # Display current gesture
        cv2.putText(frame, f"Gesture: {current_gesture}", (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Reset all gesture states if no hands detected
        thumbs_up_active = False
        thumbs_down_active = False
        two_fingers_active = False
        three_fingers_active = False
        four_fingers_active = False
        pinch_active = False
        hands_crossed_active = False
    
    # Display hand detection status
    cv2.putText(frame, detection_status, (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display controls information
    cv2.putText(frame, "YouTube Controls:", (30, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
    cv2.putText(frame, "- Pinch: Play/Pause (hold 1s)", (40, 345), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- Thumbs Up/Down: Volume +/-10% (hold 1s)", (40, 365), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 2 Fingers: Rewind (hold 1s)", (40, 385), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 3 Fingers: Forward (hold 1s)", (40, 405), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- 4 Fingers: Full Screen (hold 1s)", (40, 425), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "- Cross Hands: Close YouTube (hold 1s)", (40, 445), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    # Show output
    cv2.imshow("YouTube Gesture Control", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()