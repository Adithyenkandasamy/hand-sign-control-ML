import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = GestureCNN(num_classes)
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output).item()

    # Display gesture prediction
    cv2.putText(frame, f"Gesture: {train_dataset.classes[predicted_class]}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
