# main.py
import cv2
import numpy as np
import tensorflow as tf

from pose_estimation import draw_keypoints, draw_connections, load_model, EDGES
from fall_detection import calculate_orientation, calculate_velocities, detect_fall_with_speed, detect_fall_with_orientation
from utils import log_fall

# --- 1. Load the Model and Setup ---
# Load the MoveNet model from the specified .tflite file.
# This model is used for pose estimation, i.e., detecting keypoints of a human body.
model_path = r"C:\Users\d2907\Downloads\cyberphysical\1st_approach\3.tflite"
# Allocate tensors for the interpreter, preparing it to run inference.
interpreter = load_model(model_path)

# --- 2. Fall Detection and main loop ---
def fall_detection_live():
    # Open video file.
    cap = cv2.VideoCapture(r"C:\Users\d2907\Downloads\cyberphysical\1st_approach\07.mp4")
    # Initialize previous keypoints positions and orientation.
    previous_positions = None
    previous_orientation = None

    # Process video frames until there are no more frames or user quits.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Copy the frame for processing.
        img = frame.copy()
        # Resize the frame to match the input size of the model.
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        # Cast the image data type to float32.
        input_image = tf.cast(img, dtype=tf.float32)
        # Get the input and output details of the model.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Set the input tensor with the processed image.
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        # Run the inference to get the keypoints.
        interpreter.invoke()
        # Extract keypoints and confidence scores.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0, 0]
         # Filter keypoints with confidence above 0.3.
        valid_keypoints = [(kp[1], kp[0]) for kp in keypoints_with_scores if kp[2] > 0.3]

        # If not enough keypoints are detected, continue to the next frame.
        if len(valid_keypoints) < 2:
            previous_positions = None
            previous_orientation = None
            continue

        # Calculate the current orientation of the person.
        current_orientation = calculate_orientation(valid_keypoints)

         # Calculate velocities of keypoints if there are previous positions.
        velocities = calculate_velocities(previous_positions, valid_keypoints) if previous_positions else []

         # Detect a fall based on either speed or orientation change.
        fall_detected = detect_fall_with_speed(velocities, speed_threshold=15) or \
                        detect_fall_with_orientation(previous_orientation, current_orientation, orientation_threshold=20)

        # Store the current keypoint positions for the next frame.
        previous_positions = valid_keypoints
        # Store the current orientation for the next frame.
        previous_orientation = current_orientation

        # If fall is detected print message, log it and add text to the frame.
        if fall_detected:
            print("Fall detected!")
            log_fall()
            cv2.putText(frame, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw lines between the keypoints.
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        # Draw keypoints on the frame.
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        # Show the processed frame with detected keypoints and fall indicators.
        cv2.imshow('Fall Detection', frame)
        # Break the loop if the user presses the "q" key.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all cv2 windows.
    cap.release()
    cv2.destroyAllWindows()

# --- 5. Entry Point ---
if __name__ == "__main__":
     # Start the fall detection process.
    fall_detection_live()