import cv2
import numpy as np
import tensorflow as tf

# --- 1. Setup and Model Loading ---

# Load the MoveNet model from the specified .tflite file.
# This model is used for pose estimation, i.e., detecting keypoints of a human body.
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\d2907\Downloads\cyberphysical\1st_approach\3.tflite")
# Allocate tensors for the interpreter, preparing it to run inference.
interpreter.allocate_tensors()


# --- 2. Helper Functions for Visualization ---

# Function to draw circles (keypoints) on the frame.
def draw_keypoints(frame, keypoints, confidence_threshold):
    # Get the frame dimensions (height,weidth,channels).
    y, x, c = frame.shape
    # Scale the keypoint coordinates to match the frame size and remove the extra dimension.
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # Iterate through the keypoints.
    for i, kp in enumerate(shaped):
        # Get the y-coordinate, x-coordinate, and confidence score for each keypoint.
        ky, kx, kp_conf = kp
        # If the confidence is above the threshold, draw the keypoint.
        if kp_conf > confidence_threshold:
            # Draw a circle at the keypoint position (x, y), with color green, and filled.
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            # Put the keypoint index (0-16) next to the circle in green color.
            cv2.putText(frame, str(i), (int(kx), int(ky) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# A dictionary that maps pairs of keypoint indices to a color label ('m', 'c', or 'y').
# Used to connect keypoints with lines for visualization.
EDGES = {(0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
         (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y', (11, 13): 'm',
         (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'}

# Function to draw lines (connections) between keypoints.
def draw_connections(frame, keypoints, edges, confidence_threshold):
     # Get the frame dimensions (height, width, channels).
    y, x, c = frame.shape
     # Scale the keypoint coordinates to match the frame size and remove the extra dimension.
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # Define a color map for different edges.
    color_map = {'m': (255, 0, 255), 'c': (255, 255, 0), 'y': (0, 255, 255)}
    # Loop through all the defined edges.
    for edge, color_key in edges.items():
        # Get the two keypoints forming the edge.
        p1, p2 = edge
        # Get the y, x coordinates and confidence of the keypoints.
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        # If the confidence of both keypoints is above the threshold, draw the line.
        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            # Get the color for the line based on the key.
            color = color_map[color_key]
             # Adjust the line thickness based on the vertical distance between keypoints.
            thickness = 2 if abs(y2 - y1) > 20 else 1
            # Draw a line between the two keypoints.
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)