import numpy as np

# Function to calculate the orientation of the person based on keypoints (shoulder and hip positions).
def calculate_orientation(keypoints):
    try:
        # Calculate the midpoint between the left and right shoulders.
        shoulder_mid = np.mean([keypoints[5], keypoints[6]], axis=0)
        # Calculate the midpoint between the left and right hips.
        hip_mid = np.mean([keypoints[11], keypoints[12]], axis=0)
         # Calculate the vertical difference between hip and shoulder.
        delta_y = hip_mid[1] - shoulder_mid[1]
        # Calculate the horizontal difference between hip and shoulder.
        delta_x = hip_mid[0] - shoulder_mid[0]
          # Calculate the angle of orientation in radians and convert to degrees.
        angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        print(f"Orientation Angle: {angle:.2f}")
        return angle
    except (IndexError, ValueError):
        # Handle cases where the required keypoints are not available.
        return None

# Function to calculate the velocities of keypoints between two frames.
def calculate_velocities(previous_positions, current_positions):
    try:
        # Calculate the distance between corresponding keypoints in the previous and current frames.
        velocities = [np.linalg.norm(np.array(curr) - np.array(prev)) for curr, prev in zip(current_positions, previous_positions)]
        return velocities
    except (IndexError, ValueError):
        # Handle cases where the required keypoints are not available.
        return []

# Function to detect a fall based on the speed of keypoints.
def detect_fall_with_speed(velocities, speed_threshold=15):
    if not velocities:
        return False
    # Get the maximum speed among all keypoints.
    max_speed = max(velocities)
    print(f"Max Speed: {max_speed:.2f}")
    # Detect a fall if the maximum speed is greater than the threshold.
    return max_speed > speed_threshold

# Function to detect a fall based on a significant change in orientation angle between two frames.
def detect_fall_with_orientation(prev_orientation, current_orientation, orientation_threshold=20):
    if prev_orientation is None or current_orientation is None:
        return False
    # Calculate the absolute difference in orientation angles.
    angle_change = abs(current_orientation - prev_orientation)
    print(f"Angle Change: {angle_change:.2f}")
    # Detect a fall if the angle change is greater than the threshold.
    return angle_change > orientation_threshold
