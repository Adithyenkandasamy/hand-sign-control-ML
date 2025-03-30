import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame!")
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)  

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
