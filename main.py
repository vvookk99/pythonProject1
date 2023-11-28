import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise

(명령행 옵션, 실행방법)
ap = argparse.ArgumentParser()
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Type of activity to do',
                required=False)
args = vars(ap.parse_args())

args = vars(ap.parse_args())

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if args["video_source"] is not None:
    cap = cv2.VideoCapture("Exercise Videos/" + args["video_source"])
else:
    cap = cv2.VideoCapture(0)  # webcam

cap.set(3, 800)  # width
cap.set(4, 480)  # height

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    count = 0  # movement of exercise
    status = True  # state of move
    accuracy = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # result_screen = np.zeros((250, 400, 3), np.uint8)

        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        # recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        # make detection
        results = pose.process(frame)
        # recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            count, status, accuracy = TypeOfExercise(landmarks).calculate_exercise(
                args["exercise_type"], count, status, accuracy)
        except:
            pass

        frame = score_table(args["exercise_type"], frame, count, status, accuracy)

        # render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=2),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=2),
        )

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from utils import *


class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_neck(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_mouth = detection_body_part(self.landmarks, "MOUTH_RIGHT")
        l_mouth = detection_body_part(self.landmarks, "MOUTH_LEFT")
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")

        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]
        mouth_avg = [(r_mouth[0] + l_mouth[0]) / 2,
                     (r_mouth[1] + l_mouth[1]) / 2]
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        return abs(180 - calculate_angle(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        # calculate angle of the avg shoulder
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]

        # calculate angle of the avg hip
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        # calculate angle of the avg knee
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        knee_avg = [(r_knee[0] + l_knee[0]) / 2, (r_knee[1] + l_knee[1]) / 2]

        return calculate_angle(shoulder_avg, hip_avg, knee_avg)


import numpy as np
from body_part_angle import BodyPartAngle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def squat(self, count, status, accuracy):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

        if status:
            if 98 > avg_leg_angle >= 90:
                count += 1
                status = False
                accuracy = 100

        else:
            if avg_leg_angle >= 170:
                status = True
                accuracy = 0

            if 170 > avg_leg_angle >= 166:
                status = True
                accuracy = 5

            if 166 > avg_leg_angle >= 162:
                status = True
                accuracy = 10

            if 162 > avg_leg_angle >= 158:
                status = True
                accuracy = 15

            if 158 > avg_leg_angle >= 154:
                status = True
                accuracy = 20

            if 154 > avg_leg_angle >= 150:
                status = True
                accuracy = 25

            if 150 > avg_leg_angle >= 146:
                status = True
                accuracy = 30

            if 146 > avg_leg_angle >= 142:
                status = True
                accuracy = 35

            if 142 > avg_leg_angle >= 138:
                status = True
                accuracy = 40

            if 138 > avg_leg_angle >= 134:
                status = True
                accuracy = 45

            if 134 > avg_leg_angle >= 130:
                status = True
                accuracy = 50

            if 130 > avg_leg_angle >= 126:
                status = True
                accuracy = 55

            if 126 > avg_leg_angle >= 122:
                status = True
                accuracy = 60

            if 122 > avg_leg_angle >= 118:
                status = True
                accuracy = 65

            if 118 > avg_leg_angle >= 114:
                status = True
                accuracy = 70

            if 114 > avg_leg_angle >= 110:
                status = True
                accuracy = 75

            if 110 > avg_leg_angle >= 106:
                status = True
                accuracy = 80

            if 106 > avg_leg_angle >= 102:
                status = True
                accuracy = 85

            if 102 > avg_leg_angle >= 98:
                status = True
                accuracy = 90

            if 90 > avg_leg_angle >= 82:
                status = True
                accuracy = 90

            if avg_leg_angle < 82:
                count -= 1
                status = True
                accuracy = 0

        return [count, status, accuracy]

    def calculate_exercise(self, exercise_type, count, status, accuracy):
        if exercise_type == "squat":
            count, status, accuracy = TypeOfExercise(self.landmarks).squat(
                count, status, accuracy)

        return [count, status, accuracy]

import mediapipe as mp
import pandas as pd
import numpy as np
import cv2

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]


def detection_body_parts(landmarks):
    body_parts = pd.DataFrame(columns=["body_part", "x", "y"])

    for i, lndmrk in enumerate(mp_pose.PoseLandmark):
        lndmrk = str(lndmrk).split(".")[1]
        cord = detection_body_part(landmarks, lndmrk)
        body_parts.loc[i] = lndmrk, cord[0], cord[1]

    return body_parts

def score_table(exercise, frame, count, status, accuracy):
    cv2.putText(frame, "Excercise : " + exercise.replace("-", " "),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "Count : " + str(count), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status : " + str(status), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Accuracy : " + str(accuracy), (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame