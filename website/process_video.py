import cv2
import argparse
import mediapipe as mp
import numpy as np
import math
from object_detection import ObjectDetection  
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b,c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)

                    radians = np.arctan2(c[1] - b[1] , c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians*180.0/np.pi)

                    if angle > 180.0:
                        angle = 360 - angle
                    

                    return angle

# Initialize object detection
od = ObjectDetection()

# Initialize video capture
cap = cv2.VideoCapture("c:/Users/User/Documents/source_code/Overhand serve Fast Speed.mp4")

# Initialize pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        (class_ids, scores, boxes) = od.detect(frame)

        # Perform pose detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw object detection bounding boxes
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract pose landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            print("Angle:", angle)

            # Visualize angle on frame
            cv2.putText(frame, str(angle),
                        tuple(np.multiply(elbow, [500, 700]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add "right shoulder angle" text
            cv2.putText(frame, "Right Shoulder Angle: {:.2f}".format(angle),
                        (10, 30),  # Position of the text (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Render pose detection
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Combined Feed', frame)

        key = cv2.waitKey(10)
        if key == ord('q') or key == 27:  # Break the loop if 'q' or ESC key is pressed
            break

cap.release()
cv2.destroyAllWindows()