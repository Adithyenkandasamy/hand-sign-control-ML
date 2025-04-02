import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
from sklearn.model_selection import train_test_split
import pyautogui
import time
import glob

# Optimized constants for memory efficiency
IMAGE_HEIGHT = 64     # Reduced from 120
IMAGE_WIDTH = 64      # Reduced from 120
NUM_CLASSES = 5

def load_image(image_path):
    """Load and preprocess a single image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to load image: {image_path}")
            
        # Resize image
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return blank image if loading fails
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

def find_image_files(base_dir, extensions=['jpg', 'jpeg', 'png']):
    """Find all image files in the directory and subdirectories."""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(base_dir, '**', f'*.{ext}')
        image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(image_files)} image files in {base_dir}")
    return image_files

def create_dataset_from_images(base_dir):
    """Create a dataset from image files based on directory names or filename patterns."""
    image_files = find_image_files(base_dir)
    
    if not image_files:
        print("No image files found!")
        return None
    
    # Define gesture keywords to look for in filenames
    gesture_keywords = {
        'thumbs_up': 'Thumbs_Up',
        'thumbsup': 'Thumbs_Up',
        'thumbs-up': 'Thumbs_Up',
        'thumbs_down': 'Thumbs_Down',
        'thumbsdown': 'Thumbs_Down',
        'thumbs-down': 'Thumbs_Down',
        'left_swipe': 'Left_Swipe',
        'leftswipe': 'Left_Swipe',
        'left-swipe': 'Left_Swipe',
        'right_swipe': 'Right_Swipe',
        'rightswipe': 'Right_Swipe',
        'right-swipe': 'Right_Swipe',
        'stop': 'Stop',
        'halt': 'Stop',
        'hand': 'Stop'
    }
    
    # Create dataset
    data = []
    
    for img_path in image_files:
        img_path_lower = img_path.lower()
        filename = os.path.basename(img_path_lower)
        
        # Try to determine gesture from filename or directory
        gesture = None
        for keyword, label in gesture_keywords.items():
            if keyword in filename or keyword in img_path_lower:
                gesture = label
                break
        
        # If we couldn't determine the gesture, skip this image
        if gesture is None:
            continue
            
        data.append({
            'Image Path': img_path,
            'Gesture': gesture
        })
    
    if not data:
        print("No labeled images found!")
        return None
        
    return pd.DataFrame(data)

def create_efficient_model():
    """Create a memory-efficient CNN model for gesture recognition."""
    model = Sequential()
    
    # Efficient CNN architecture
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Global average pooling to reduce parameters
    model.add(GlobalAveragePooling2D())
    
    # Dense layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    return model

def train_gesture_model(train_csv_path, images_base_dir, model_save_path='youtube_gesture_model.h5'):
    # First check if train_csv_path exists, if not try to create dataset from images
    if not os.path.exists(train_csv_path):
        print(f"CSV file {train_csv_path} not found.")
        print("Attempting to create dataset from images...")
        train_data = create_dataset_from_images(images_base_dir)
        
        if train_data is None:
            print("Could not create dataset from images. Exiting.")
            return None, None
    else:
        # Load the CSV data
        print("Loading training data from CSV...")
        try:
            try:
                train_data = pd.read_csv(train_csv_path, sep=';')
            except:
                train_data = pd.read_csv(train_csv_path)
                
            print("\nCSV columns found:", train_data.columns.tolist())
            print("\nFirst few rows of data:")
            print(train_data.head())
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print("Attempting to create dataset from images...")
            train_data = create_dataset_from_images(images_base_dir)
            
            if train_data is None:
                print("Could not create dataset from images. Exiting.")
                return None, None

    # Map gesture names to numerical values
    gesture_map = {
        'Thumbs_Up': 0, 
        'Thumbs_Down': 1,
        'Left_Swipe': 2,
        'Right_Swipe': 3,
        'Stop': 4
    }

    # Process the data into standard format
    processed_data = pd.DataFrame()
    
    # Determine column names based on what's available
    image_col = None
    gesture_col = None
    
    for col in train_data.columns:
        if 'image' in col.lower() or 'path' in col.lower() or 'file' in col.lower():
            image_col = col
            break
    
    for col in train_data.columns:
        if 'gesture' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            gesture_col = col
            break
    
    if not image_col:
        image_col = train_data.columns[0]  # Default to first column
    
    if not gesture_col:
        if len(train_data.columns) > 1:
            gesture_col = train_data.columns[1]  # Default to second column
        else:
            # Try to extract gesture from the image path/name
            print("No gesture column found. Extracting gestures from image names...")
            train_data['Extracted_Gesture'] = train_data[image_col].apply(
                lambda x: next((g for g, _ in gesture_map.items() if g.lower() in str(x).lower()), None)
            )
            gesture_col = 'Extracted_Gesture'
    
    # Create standardized columns
    processed_data['Image Path'] = train_data[image_col].astype(str)
    processed_data['Gesture'] = train_data[gesture_col]
    
    # Clean gestures to match expected format
    def clean_gesture(g):
        if not isinstance(g, str):
            return None
        for known_g in gesture_map.keys():
            if known_g.lower() in g.lower():
                return known_g
        return None
    
    processed_data['Gesture'] = processed_data['Gesture'].apply(clean_gesture)
    processed_data = processed_data.dropna(subset=['Gesture'])
    
    # Map gestures to numerical values
    processed_data['Classification'] = processed_data['Gesture'].map(gesture_map)
    processed_data = processed_data.dropna(subset=['Classification'])
    
    print(f"\nProcessed {len(processed_data)} valid samples")
    if len(processed_data) < 10:
        print("Warning: Very few valid samples. Training may not be effective.")
        return None, None

    # Split data
    train_df, val_df = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    # Create image data generator to augment data
    from keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Prepare X and y data
    X_train = np.array([load_image(path) for path in train_df['Image Path']])
    y_train = train_df['Classification'].values
    
    X_val = np.array([load_image(path) for path in val_df['Image Path']])
    y_val = val_df['Classification'].values
    
    # Create model
    print("Creating efficient model...")
    model = create_efficient_model()
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train model with data augmentation
    print(f"Training model with {len(train_df)} samples...")
    
    # Use a small batch size for memory efficiency
    batch_size = 8
    
    # Use data augmentation during training
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=5,  # Set a higher number and rely on early stopping
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Save the model
    print(f"Training complete. Saving model to {model_save_path}...")
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    return model, history

def predict_youtube_gestures(model_path='youtube_gesture_model.h5'):
    """Predict gestures in real-time using webcam and control YouTube."""
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Initialize variables
    frame_buffer = []
    buffer_size = 5  # Use a small buffer for smoothing predictions
    last_predictions = []
    
    # Gesture labels with YouTube actions
    gesture_labels = {
        0: "Thumbs Up - Volume Up",
        1: "Thumbs Down - Volume Down",
        2: "Left Swipe - Rewind 10s",
        3: "Right Swipe - Forward 10s",
        4: "Stop - Pause/Play"
    }
    
    # For cooldown between actions
    last_action_time = 0
    cooldown = 1.0  # 1 second cooldown
    
    print("Starting YouTube gesture control. Press 'q' to quit.")
    print("Make sure YouTube is open and in focus!")
    print("Waiting 3 seconds to give you time to switch to YouTube...")
    time.sleep(3)
    print("Gesture control active!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam!")
            break
        
        # Mirror the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
            
        # Preprocess frame
        frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        
        # Add to buffer for prediction smoothing
        frame_buffer.append(frame_normalized)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
        
        # Make prediction
        input_data = np.expand_dims(frame_normalized, axis=0)
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Add to predictions list for smoothing
        last_predictions.append(predicted_class)
        if len(last_predictions) > buffer_size:
            last_predictions.pop(0)
        
        # Use most common prediction in buffer
        from collections import Counter
        most_common = Counter(last_predictions).most_common(1)
        smoothed_class = most_common[0][0]
        
        # Only perform action if confidence is high enough and cooldown has passed
        current_time = time.time()
        if confidence > 0.7 and current_time - last_action_time > cooldown:
            # Execute action
            if smoothed_class == 0:  # Thumbs Up - Volume Up
                pyautogui.press('up')
                print("Action: Volume Up")
            elif smoothed_class == 1:  # Thumbs Down - Volume Down
                pyautogui.press('down')
                print("Action: Volume Down")
            elif smoothed_class == 2:  # Left Swipe - Rewind 10s
                pyautogui.press('j')
                print("Action: Rewind 10s")
            elif smoothed_class == 3:  # Right Swipe - Forward 10s
                pyautogui.press('l')
                print("Action: Forward 10s")
            elif smoothed_class == 4:  # Stop - Pause/Play
                pyautogui.press('k')
                print("Action: Play/Pause")
                
            last_action_time = current_time
            
        # Display result on frame
        label = f"{gesture_labels[smoothed_class]} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show a visual confidence bar
        bar_length = int(confidence * 200)
        cv2.rectangle(frame, (10, 40), (10 + bar_length, 50), (0, 255, 0), -1)
        
        # Display frame
        cv2.imshow('YouTube Gesture Control', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("YouTube gesture control stopped.")

def main():
    # Ask user what they want to do
    print("YouTube Gesture Control - Improved Version")
    print("1. Train model")
    print("2. Run YouTube control")
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        print("Training model...")
        train_csv_path = input("Enter path to training CSV file (press Enter to skip and use image folders): ")
        if not train_csv_path:
            train_csv_path = 'nonexistent_file.csv'  # Will trigger image-based dataset creation
            
        images_base_dir = input("Enter path to images directory: ")
        if not images_base_dir:
            images_base_dir = '.'
            
        model_save_path = input("Enter path to save model (default: youtube_gesture_model.h5): ")
        if not model_save_path:
            model_save_path = 'youtube_gesture_model.h5'
            
        print(f"Images directory: {images_base_dir}")
        print(f"Model will be saved to: {model_save_path}")
            
        model, history = train_gesture_model(train_csv_path, images_base_dir, model_save_path)
        if model is None:
            print("Training failed. Please check error messages above.")
        
    elif choice == '2':
        print("Launching YouTube gesture control...")
        model_path = input("Enter path to model file (default: youtube_gesture_model.h5): ")
        if not model_path:
            model_path = 'youtube_gesture_model.h5'
            
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            print("Would you like to train a model first? (y/n): ")
            train_first = input().lower()
            if train_first == 'y':
                images_base_dir = input("Enter path to images directory: ")
                if not images_base_dir:
                    images_base_dir = '.'
                model, _ = train_gesture_model('nonexistent_file.csv', images_base_dir, model_path)
                if model is None:
                    return
            else:
                return
            
        predict_youtube_gestures(model_path)
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()