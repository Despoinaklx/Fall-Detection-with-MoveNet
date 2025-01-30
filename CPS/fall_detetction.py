import cv2
import numpy as np
import tensorflow as tf

# Φόρτωση του MoveNet μοντέλου
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\d2907\Downloads\cyberphysical\CPS\3.tflite")
interpreter.allocate_tensors()

# Σχεδίαση σημείων-κλειδιών
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Σχεδίαση συνδέσεων μεταξύ αρθρώσεων
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    color_map = {'m': (255, 0, 255), 'c': (255, 255, 0), 'y': (0, 255, 255)}

    for edge, color_key in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            color = color_map[color_key]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# Υπολογισμός orientation
def calculate_orientation(keypoints):
    try:
        shoulder_mid = np.mean([keypoints[5], keypoints[6]], axis=0)
        hip_mid = np.mean([keypoints[11], keypoints[12]], axis=0)
        delta_y = hip_mid[1] - shoulder_mid[1]
        delta_x = hip_mid[0] - shoulder_mid[0]
        angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        return angle
    except IndexError:
        return None

# Ανίχνευση πτώσης με βάση την ταχύτητα
def detect_fall(previous_positions, current_positions, threshold=20):
    if not previous_positions or not current_positions:
        return False

    try:
        velocities = [abs(current[1] - prev[1]) for current, prev in zip(current_positions, previous_positions)]
        max_velocity = max(velocities)
        return max_velocity > threshold
    except IndexError:
        return False

# Ανίχνευση πτώσης με βάση την αλλαγή προσανατολισμού
def detect_fall_with_orientation(prev_orientation, current_orientation, orientation_threshold=30):
    if prev_orientation is None or current_orientation is None:
        return False
    angle_change = abs(current_orientation - prev_orientation)
    return angle_change > orientation_threshold

# Λειτουργία για ανίχνευση πτώσης από κάμερα σε πραγματικό χρόνο
def fall_detection_live():
    #cap = cv2.VideoCapture(0)  # Χρήση της κάμερας
    cap = cv2.VideoCapture(r"C:\Users\d2907\Downloads\cyberphysical\CPS\07.mp4")
    previous_positions = None
    previous_orientation = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Προετοιμασία εικόνας για MoveNet
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Επεξεργασία με το MoveNet
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()

        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0, 0]
        valid_keypoints = [(kp[1], kp[0]) for kp in keypoints_with_scores if kp[2] > 0.3]

        if len(valid_keypoints) < 2:
            previous_positions = None
            previous_orientation = None
            continue

        # Υπολογισμός orientation
        current_orientation = calculate_orientation(valid_keypoints)

        # Ανίχνευση πτώσης
        fall_detected = detect_fall(previous_positions, valid_keypoints, threshold=20) or \
                        detect_fall_with_orientation(previous_orientation, current_orientation, orientation_threshold=30)

        previous_positions = valid_keypoints
        previous_orientation = current_orientation

        if fall_detected:
            print("Fall detected!")
            cv2.putText(frame, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Σχεδίαση αρθρώσεων και συνδέσεων
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        cv2.imshow('Fall Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Εκκίνηση ανίχνευσης πτώσης
if __name__ == "__main__":
    fall_detection_live()
