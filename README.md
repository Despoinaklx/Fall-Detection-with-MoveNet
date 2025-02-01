# Fall Detection System using MoveNet

This project implements a fall detection system using the MoveNet pose estimation model and computer vision techniques. It processes video input, identifies key body points, calculates orientation and speed, and detects falls based on predefined thresholds.

## Project Structure

The project is organized into the following modules:

-   **`main.py`**: The main execution script which orchestrates the fall detection process by importing and using functions from other modules.
-   **`pose_estimation.py`**: Contains the logic for loading the MoveNet model and drawing keypoints and connections on the video frame.
-   **`fall_detection.py`**: Contains the fall detection logic based on orientation changes and speed of keypoint movement.
-   **`utils.py`**: Contains utility functions, in this case only a logging function.
-   **`falls.log`**: File where fall detections are logged.

There is also a **test.py** file for classification performance. 

I used these datasets 

- **`fall`**                          https://drive.google.com/drive/folders/1sXZ3oKdmdfrku2IpATMXC5SW3qEgaUE7?usp=drive_link
- **`no_fall`**                       https://drive.google.com/drive/folders/1LXS3OBtgv3RFs3o7Br5TLXJOw4PAtv5y?usp=drive_link


## Requirements

-   **Python 3.6 or higher**
-   **TensorFlow**
-   **OpenCV**
-   **NumPy**

You also need the MoveNet TensorFlow Lite model file. 

**Example Videos**

This project includes two example videos to help you understand how it works:

fall.mp4: A video demonstrating a fall scenario.

no_fall.mp4: A video demonstrating a scenario without a fall.

To use these videos, or your own videos, replace the raw path in the cv2.VideoCapture() function in the main.py file with the path of the video you want to analyse.


## How it Works

Pose Estimation:
The pose_estimation.py module loads the MoveNet TensorFlow Lite model, which is used for detecting keypoints of the human body from a video frame.
The module contains function for drawing the detected keypoints and their connections on the video frame.

Fall Detection:
The fall_detection.py module calculates the person's orientation based on the position of shoulders and hips using the keypoints provided by the pose estimation.
It calculates the velocity of the keypoint movement between frames.
The fall is detected by comparing these calculations against the predefined thresholds.

Main Loop (main.py):
The main script handles video capture using OpenCV.
It passes the video frames to the pose_estimation module to obtain keypoints.
It passes the keypoints to fall_detection to check for fall conditions.
If a fall is detected, it prints a message on the terminal, logs the event into the falls.log and draws text in the current frame.


## Strengths and Limitations

**Strengths**


Implements fall detection based on orientation and speed changes.

Utilizes the MoveNet pose estimation model for accurate body keypoint detection.

Modular code structure for better organization and reusability.

Visual feedback with keypoint drawings on video frames.

Easy to run.

Logs of fall detections are saved in a log file.

**Limitations**

Detection accuracy may be affected by poor video quality.

The system may trigger false positives (incorrectly identify falls) due to significant changes in body orientation that are not actual falls. For example, actions such as bending, kneeling, or sudden shifts can cause the orientation threshold to be exceeded, leading to a false detection.

The system may fail to detect slow-paced falls since they do not exceed the speed threshold. It's primarily designed to identify rapid falls, but may miss gradual movements that can also lead to a fall.

Detection thresholds are fixed and may not be optimal for all individuals and situations.

Designed for single-person analysis.

Possible false positives may occur.

The model relies on the accuracy of the MoveNet algorithm.

Requires sufficient computing resources to run efficiently.

The project has not been tested for different scenarios and may fail to detect falls under certain circumstances.

This is not a final and perfect product.
