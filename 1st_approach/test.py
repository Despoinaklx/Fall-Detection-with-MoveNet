import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MoveNet model
interpreter = tf.lite.Interpreter(
    model_path=r"C:\Users\d2907\Downloads\cyberphysical\1st_approach\3.tflite")
interpreter.allocate_tensors()

# Settings
fall_videos_path = r"add path"      
no_fall_videos_path = r"add path"
confidence_threshold = 0.3
orientation_threshold = 20
deviation_threshold = 5

# --- Keypoint Utility Functions ---
def calculate_orientation(keypoints):
    try:
        shoulder_mid = np.mean([keypoints[5], keypoints[6]], axis=0)
        hip_mid = np.mean([keypoints[11], keypoints[12]], axis=0)
        delta_y = hip_mid[1] - shoulder_mid[1]
        delta_x = hip_mid[0] - shoulder_mid[0]
        angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        return angle
    except (IndexError, ValueError):
        return None

def calculate_velocities(previous_positions, current_positions):
    try:
        velocities = [abs(curr[1] - prev[1]) for curr, prev in zip(current_positions, previous_positions)]
        return velocities
    except (IndexError, ValueError):
        return []


# --- Fall Detection Functions ---
def detect_fall_with_speed(previous_positions, current_positions, deviation_threshold=5):
    if not previous_positions or not current_positions:
        return False
    try:
        velocities = [abs(curr[1] - prev[1]) for curr, prev in zip(current_positions, previous_positions)]
        max_velocity = max(velocities)
        print(f"Max Speed: {max_velocity:.2f}")
        return max_velocity > deviation_threshold
    except IndexError:
        return False


def detect_fall_with_orientation(prev_orientation, current_orientation, orientation_threshold=20):
    if prev_orientation is None or current_orientation is None:
        return False
    angle_change = abs(current_orientation - prev_orientation)
    print(f"Angle Change: {angle_change:.2f}")
    return angle_change > orientation_threshold

# --- Video Processing Function ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    previous_positions = None
    previous_orientation = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()

        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0, 0]
        valid_keypoints = [(kp[1], kp[0]) for kp in keypoints_with_scores if kp[2] > confidence_threshold]
        
        if len(valid_keypoints) < 2:
           previous_positions = None
           previous_orientation = None
           continue
        
        current_orientation = calculate_orientation(valid_keypoints)
        current_positions = valid_keypoints
        
        velocities = calculate_velocities(previous_positions, current_positions) if previous_positions else []

        fall_detected = detect_fall_with_speed(previous_positions, current_positions, deviation_threshold) or \
            detect_fall_with_orientation(previous_orientation, current_orientation, orientation_threshold)
        
        previous_positions = valid_keypoints
        previous_orientation = current_orientation

        if fall_detected:
            cap.release()
            return 1  # Fall detected

    cap.release()
    return 0  # No fall detected


# --- Evaluation Functions ---
def calculate_confusion_matrix(y_true, y_pred):
    """Calculates the confusion matrix metrics (TP, FP, TN, FN)."""
    cm = confusion_matrix(y_true, y_pred)
    return cm


def evaluate_model():
    """Evaluates the fall detection model using the provided dataset."""
    y_true = []
    y_pred = []
    fp_videos = []  # Store names of False Positive videos
    fn_videos = []  # Store names of False Negative videos

    # Process fall videos
    for video_file in os.listdir(fall_videos_path):
        video_path = os.path.join(fall_videos_path, video_file)
        if video_file.endswith(".mp4"):
            print(f"Processing fall video: {video_file}")
            y_true.append(1)
            prediction = process_video(video_path)
            y_pred.append(prediction)
            if prediction == 0:  # False Negative
                fn_videos.append(video_file)

    # Process no-fall videos
    for video_file in os.listdir(no_fall_videos_path):
        video_path = os.path.join(no_fall_videos_path, video_file)
        if video_file.endswith(".mp4"):
            print(f"Processing no-fall video: {video_file}")
            y_true.append(0)
            prediction = process_video(video_path)
            y_pred.append(prediction)
            if prediction == 1:  # False Positive
                fp_videos.append(video_file)

    # Calculate confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Print the names of incorrectly classified videos
    print("\nFalse Positive Videos:")
    for video_name in fp_videos:
        print(f"  - {video_name}")
    
    print("\nFalse Negative Videos:")
    for video_name in fn_videos:
        print(f"  - {video_name}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["No Fall", "Fall"]))

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
