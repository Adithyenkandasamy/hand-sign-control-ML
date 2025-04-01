import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands with optimal performance settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Set up simplified landmark drawing for better performance
drawing_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

# Open webcam with lower resolution for better performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

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

# Cooldown mechanism - increased for better control
last_action_time = 0
cooldown = 1.0  # seconds between actions

# Gesture recognition tracking
last_gesture = None
gesture_frames = 0
required_consecutive_frames = 5  # Increased for better accuracy

# Disable mouse scroll events for palm gesture
pyautogui.FAILSAFE = True

# Process frames at reduced rate for performance
process_every_n_frames = 1  # Process every frame for better responsiveness
frame_count = 0

# Store previous gestures to prevent unintended actions
previous_gestures = []
gesture_history_length = 3

# Track hand positions for swing detection
previous_hand_positions = []
position_history_length = 10

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return ((landmark1.x - landmark2.x) ** 2 + 
            (landmark1.y - landmark2.y) ** 2) ** 0.5

def is_finger_extended(finger_tip, finger_mcp, wrist, threshold=0.1):
    """Check if a finger is extended based on tip position relative to MCP"""
    return finger_tip.y < finger_mcp.y - threshold

def is_finger_closed(finger_tip, finger_mcp, threshold=0.05):
    """Check if a finger is closed (not extended)"""
    return finger_tip.y > finger_mcp.y - threshold

def is_palm_showing(landmarks):
    """Detect if the palm is showing - improved implementation"""
    # Get relevant landmarks
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    # Get MCPs for better detection
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    
    # Check if all fingers are extended and spread apart
    fingers_extended = (
        is_finger_extended(thumb_tip, thumb_mcp, None, threshold=0.1) and
        is_finger_extended(index_tip, index_mcp, None, threshold=0.1) and
        is_finger_extended(middle_tip, middle_mcp, None, threshold=0.1) and
        is_finger_extended(ring_tip, ring_mcp, None, threshold=0.1) and
        is_finger_extended(pinky_tip, pinky_mcp, None, threshold=0.1)
    )
    
    # Check distances between fingertips to ensure they're spread
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    fingers_spread = (
        index_middle_dist > 0.1 and
        middle_ring_dist > 0.1 and
        ring_pinky_dist > 0.1
    )
    
    return fingers_extended and fingers_spread

def detect_swing_gesture(current_position, position_history, threshold=0.15):
    """Detect a hand swing motion based on position history"""
    if len(position_history) < 5:  # Need enough history to detect swing
        return False
    
    # Calculate x-directional movement
    x_movement = abs(current_position.x - position_history[0].x)
    
    # Check if hand moved significantly horizontally with minimal vertical movement
    y_movement = abs(current_position.y - position_history[0].y)
    
    # Return true if horizontal movement exceeds threshold and vertical movement is minimal
    return x_movement > threshold and y_movement < threshold / 2

def is_thumbs_up(landmarks):
    """Improved thumbs up detection"""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    thumb_ip = landmarks[3]  # Thumb IP joint
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    wrist = landmarks[WRIST]
    
    # Check thumb is pointing up
    thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y
    
    # Check other fingers are curled in
    other_fingers_down = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    
    # Ensure thumb is raised significantly higher than other fingers
    thumb_raised = thumb_tip.y < index_tip.y - 0.1
    
    # Check thumb is pointing up relative to the wrist
    thumb_direction = thumb_tip.y < wrist.y
    
    return thumb_up and other_fingers_down and thumb_raised and thumb_direction

def is_thumbs_down(landmarks):
    """Improved thumbs down detection"""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    thumb_ip = landmarks[3]  # Thumb IP joint
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    wrist = landmarks[WRIST]
    
    # Check thumb is pointing down
    thumb_down = thumb_tip.y > thumb_ip.y and thumb_tip.y > thumb_mcp.y
    
    # Check other fingers are curled in
    other_fingers_down = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    
    # Ensure thumb is lower than other fingers
    thumb_lowered = thumb_tip.y > index_tip.y + 0.05
    
    # Check thumb is pointing down relative to the wrist
    thumb_direction = thumb_tip.y > wrist.y
    
    return thumb_down and other_fingers_down and thumb_lowered and thumb_direction

def are_hands_crossed(hand_landmarks1, hand_landmarks2):
    """Detect if two hands are crossed - improved with more robust checks"""
    if hand_landmarks1 and hand_landmarks2:
        # Get wrist positions for both hands
        wrist1 = hand_landmarks1.landmark[WRIST]
        wrist2 = hand_landmarks2.landmark[WRIST]
        
        # Get middle finger tips for both hands (more reliable than index)
        middle1 = hand_landmarks1.landmark[MIDDLE_TIP]
        middle2 = hand_landmarks2.landmark[MIDDLE_TIP]
        
        # Calculate distance between wrists
        wrist_distance = calculate_distance(wrist1, wrist2)
        
        # Check if hands are crossed
        crossed_condition1 = (wrist1.x < wrist2.x) and (middle1.x > middle2.x)
        crossed_condition2 = (wrist1.x > wrist2.x) and (middle1.x < middle2.x)
        
        # Ensure wrists are a reasonable distance apart
        return (crossed_condition1 or crossed_condition2) and wrist_distance > 0.1
    return False

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break
        
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Process frame
    frame_count += 1
    if frame_count % process_every_n_frames == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
    
    current_time = time.time()
    action_ready = current_time - last_action_time > cooldown
    
    current_gesture = "None"
    
    # Check for hands
    if results.multi_hand_landmarks:
        # Two hands detection for crossed hands gesture
        if len(results.multi_hand_landmarks) == 2:
            # Draw hand landmarks
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
                
                # Execute with required frames with extra verification
                if gesture_frames >= required_consecutive_frames and action_ready:
                    # Only close if we've seen crossed hands consistently
                    all_crossed = all(g == "Hands Crossed" for g in previous_gestures[-3:]) if previous_gestures else False
                    if all_crossed:
                        pyautogui.hotkey('ctrl', 'w')
                        last_action_time = current_time
            else:
                if last_gesture == "Hands Crossed":
                    last_gesture = None
                    gesture_frames = 0
        
        # Process single hand gestures
        elif len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
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
            
            # Track wrist position for swing detection
            previous_hand_positions.append(wrist)
            if len(previous_hand_positions) > position_history_length:
                previous_hand_positions.pop(0)
            
            # Check if fingers are extended
            index_extended = index_tip.y < index_mcp.y - 0.1
            middle_extended = middle_tip.y < middle_mcp.y - 0.1
            ring_extended = ring_tip.y < ring_mcp.y - 0.1
            pinky_extended = pinky_tip.y < pinky_mcp.y - 0.1
            thumb_extended = thumb_tip.y < thumb_mcp.y - 0.1
            
            # Check for swing gesture
            if detect_swing_gesture(wrist, previous_hand_positions):
                current_gesture = "Swing"
                if last_gesture == "Swing":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Swing"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    # Close YouTube tab when swing is detected
                    pyautogui.hotkey('ctrl', 'w')
                    last_action_time = current_time
            
            # Check palm gesture - for play/pause
            elif is_palm_showing(landmarks):
                # Add an extra check that all fingers are spread out
                finger_spread = (
                    calculate_distance(index_tip, middle_tip) > 0.1 and
                    calculate_distance(middle_tip, ring_tip) > 0.1 and
                    calculate_distance(ring_tip, pinky_tip) > 0.1
                )
                
                if finger_spread:
                    current_gesture = "Palm"
                    if last_gesture == "Palm":
                        gesture_frames += 1
                    else:
                        gesture_frames = 1
                        last_gesture = "Palm"
                    
                    # Play/pause video on palm gesture
                    if gesture_frames >= required_consecutive_frames and action_ready:
                        pyautogui.press("space")
                        last_action_time = current_time
            
            # Thumbs Up (Volume Up) - completely rewritten detection
            elif is_thumbs_up(landmarks):
                current_gesture = "Thumbs Up"
                if last_gesture == "Thumbs Up":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Thumbs Up"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press('up', presses=3)
                    last_action_time = current_time
            
            # Thumbs Down (Volume Down) - completely rewritten detection
            elif is_thumbs_down(landmarks):
                current_gesture = "Thumbs Down"
                if last_gesture == "Thumbs Down":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Thumbs Down"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press('down', presses=3)
                    last_action_time = current_time
            
            # Two Fingers (Rewind) - improved detection
            elif (index_extended and 
                  middle_extended and 
                  not ring_extended and 
                  not pinky_extended and
                  calculate_distance(index_tip, middle_tip) < 0.1):
                
                current_gesture = "Two Fingers"
                if last_gesture == "Two Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Two Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press("left")
                    last_action_time = current_time
            
            # Three Fingers (Forward) - improved detection
            elif (index_extended and 
                  middle_extended and 
                  ring_extended and 
                  not pinky_extended and
                  calculate_distance(index_tip, middle_tip) < 0.1 and
                  calculate_distance(middle_tip, ring_tip) < 0.1):
                
                current_gesture = "Three Fingers"
                if last_gesture == "Three Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Three Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press("right")
                    last_action_time = current_time
            
            # Four Fingers (Full Screen) - improved and distinct from palm
            elif (index_extended and 
                  middle_extended and 
                  ring_extended and 
                  pinky_extended and
                  not thumb_extended):  # Thumb not extended differentiates from palm
                
                current_gesture = "Four Fingers"
                if last_gesture == "Four Fingers":
                    gesture_frames += 1
                else:
                    gesture_frames = 1
                    last_gesture = "Four Fingers"
                
                if gesture_frames >= required_consecutive_frames and action_ready:
                    pyautogui.press('f')
                    last_action_time = current_time
            
            # No recognized gesture
            else:
                last_gesture = None
                gesture_frames = 0
    else:
        # Reset if no hands detected
        last_gesture = None
        gesture_frames = 0
    
    # Display current gesture on screen
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Update gesture history
    if current_gesture != "None":
        previous_gestures.append(current_gesture)
        if len(previous_gestures) > gesture_history_length:
            previous_gestures.pop(0)
    
    # Show output with gesture overlay
    cv2.imshow("YouTube Gesture Control", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()