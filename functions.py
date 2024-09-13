import cv2
import mediapipe as mp
import time
import numpy as np
from numpy import greater
import utils
import math
import pandas as pd
import pyttsx3

from ultralytics import YOLO
from ultralytics import YOLOWorld
import cvzone
from datetime import datetime, timedelta

# variables for direction alert
change_dir_counter = 0
dir_warning_counter = 0
vis_warning_counter = 0
warning_count = 0
visibility_counter = 0

# variables 
frame_counter =0
TOTAL_BLINKS =0
frame_counter =0

# constants 
CLOSED_EYES_FRAME =1
FONTS =cv2.FONT_HERSHEY_COMPLEX

map_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=1,circle_radius=1)


start_time = time.time()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]
    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def direction_estimator_1(extreme_right_circle_right_eye, extreme_left_circle_right_eye, gaze_center, l_eye_threshold, r_eye_threshold):
    # input :- takes 3 tuples
    # output :- returns the direction
    dist_gaze_and_rightOfRight = extreme_right_circle_right_eye[0] - gaze_center[0]
    dist_gaze_and_leftOfRight = gaze_center[0] - extreme_left_circle_right_eye[0]
    eye_width = extreme_right_circle_right_eye[0] - extreme_left_circle_right_eye[0]
    if dist_gaze_and_rightOfRight < (eye_width * r_eye_threshold):
        direction = "Right"
    elif dist_gaze_and_leftOfRight < (eye_width * l_eye_threshold):
        direction = "Left"
    else:
        direction = "Center"
    return direction

def direction_estimator_2(r_eye_pts, gaze_center):
    distance = {}
    for i in range(0, 16):
        dist = abs(gaze_center[0] - r_eye_pts[i][0])
        distance[i] = dist
    top_5_smallest = sorted(distance.items(), key=lambda x: x[1])[:5]
    print(top_5_smallest)
    keys = [item[0] for item in top_5_smallest]
    required_keys_for_left = {2, 3, 13, 14}
    required_keys_for_right = {6, 7, 10, 11}
    if required_keys_for_left.issubset(keys):
        direction = "Left"
    elif required_keys_for_right.issubset(keys):
        direction = "Right"
    else:
        direction = "Center"
    return direction


def points_on_circle(center, radius, num_points):
    points = []
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append((x, y))
    return points


def draw_sharingan(frame, center, radius):
    sharingan_clr = (19, 19, 175)
    black_clr = (9, 9, 9)
    cv2.circle(frame, center, radius, black_clr, int(radius * 0.125))

    # Extra cirlces just for fun
    cv2.circle(frame, center, int(radius * 0.875), sharingan_clr, int(radius * 0.15)) 
    cv2.circle(frame, center, int(radius * 0.725), sharingan_clr, int(radius * 0.15)) # sharingan points
    cv2.circle(frame, center, int(radius * 0.575), sharingan_clr, int(radius * 0.15))
    cv2.circle(frame, center, int(radius * 0.425), black_clr, int(radius * 0.1)) 
    cv2.circle(frame, center, int(radius * 0.325), sharingan_clr, -1)   

    # Get points on the border of the circle (sharingan points)
    border_points = points_on_circle(center, int(radius * 0.5), 3)
    # Draw the sharingan points
    for point in border_points:
        cv2.circle(frame, point, int(radius * 0.075) + 1, black_clr, -1)


def draw_mesh(frame, r_eye_pts, gaze_center):
    distance = {}
    for i in range(0, 16):
        dist = abs(gaze_center[0] - r_eye_pts[i][0])
        cv2.line(frame, gaze_center, r_eye_pts[i], (19, 19, 175), 1)
        distance[i] = dist


def eye_track(ret, frame, rgb_frame, results):
    global frame_counter, CEF_COUNTER, TOTAL_BLINKS, frame_counter, eye_points

    # Left eyes indices 
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

    # right eyes indices
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

    frame_counter +=1 # frame counter

    mesh_coords = landmarksDetection(frame, results, False)

    l_eye_pts = []
    for i in range(0, 16):
        pt = mesh_coords[RIGHT_EYE[i]]
        l_eye_pts.append(pt)   
    # Draw a line connecting all points
    pts_array = np.array(l_eye_pts, np.int32)
    pts_array = pts_array.reshape((-1, 1, 2))

    r_eye_pts = []
    for i in range(0, 16):
        pt = mesh_coords[LEFT_EYE[i]]
        r_eye_pts.append(pt)
    # Draw a line connecting all points
    pts_array = np.array(r_eye_pts, np.int32)
    pts_array = pts_array.reshape((-1, 1, 2))

    ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

#     if ratio > 5.5:
#         counter_threshold +=1
#     else:
#         if counter_threshold > CLOSED_EYES_FRAME:
#             TOTAL_BLINKS +=1
#             counter_threshold = 0

    frame_h, frame_w, _ = frame.shape
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    
    if landmark_points:
        landmarks = landmark_points[0].landmark   
        
        # Get coordinates of the four points around the eye
        eye_points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in range(474, 478)]   
    
    # Draw circle approximating the eye
    if len(eye_points) == 4:
        center, radius = cv2.minEnclosingCircle(np.array(eye_points))
        center = (int(center[0]), int(center[1]))
        radius = int(radius * 0.75)
        # draw_sharingan(frame, center, radius)            

    direction = direction_estimator_1(r_eye_pts[8], r_eye_pts[0], center, 0.4, 0.3)
    
    return direction


def head_pose(frame, results):
    img_h, img_w, Img_c = frame.shape
    face_3d = []
    face_2d = []

    for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])

                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)

        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        if y < -10:
            text = "Left"
        elif y > 10:
            text = "Right"
        elif x < -18:
            text = "Down"
        elif x > 15:
            text = "Up"
        else:
            text = "Center"

    return text
        

def calculate_distance(distance_pixel, distance_cm, success, image):
    # get correlation coefficients
    coff = np.polyfit(distance_pixel, distance_cm, 2)

    # perform face detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)
    
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(image)
    bbox_list, eyes_list = [], []
    if results.detections:
        for detection in results.detections:

            # get bbox data
            bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
            bbox_list.append(bbox)

            # get the eyes landmark
            left_eye = detection.location_data.relative_keypoints[0]
            right_eye = detection.location_data.relative_keypoints[1]
            eyes_list.append([(int(left_eye.x * iw), int(left_eye.y * ih)),
                              (int(right_eye.x * iw), int(right_eye.y * ih))])

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for bbox, eye in zip(bbox_list, eyes_list):

        # calculate distance between left and right eye
        dist_between_eyes = np.sqrt(
            (eye[0][1] - eye[1][1]) ** 2 + (eye[0][0] - eye[1][0]) ** 2)

        # calculate distance in cm
        a, b, c = coff
        distance_cm = a * dist_between_eyes ** 2 + b * dist_between_eyes + c
        distance_cm -= 0

        return distance_cm

alerts  = {"visibility": ["Attention: Your face is not visible to the camera."],
      "direction": ["Alert: It seems you are not facing the camera."],
       "object": ["Warning: An important object has been detected."] }

################################################################################################################################################

model = YOLO('yolov8s.pt')

# Define the class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
# Define the desired features
desired_features = ["person", "book", "cell phone"]

# Initialize the timer
alert_timer = 0
alert_triggered = False

# Object detection function with simplified calculations
def obj_detect(ret, image):
    global alert_timer, alert_triggered, start_time

    results = model.predict(image, device='cpu')
    count = [0] * len(desired_features)  # Initialize count for desired features

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])  # Get the class ID
            class_name = classNames[cls_id]

            if class_name in desired_features:
                count[desired_features.index(class_name)] += 1  # Increment count for detected class

    # Calculate FPS based on time elapsed
    end = time.time()
    totalTime = end - start_time
    fps = 1 / totalTime if totalTime > 0 else 0
    start_time = end

    # Trigger alert if thresholds are exceeded
    if count[0] > 1 or count[1] > 0 or count[2] > 0:
        alert_timer += 1
        if alert_timer > 15:
            alert_triggered = True
            alert_timer = 0
            return False  # Alert triggered
    return True  # No alert


################################################################################################################################################

def run(camera):
    global change_dir_counter, start_time, dir_warning_counter, visibility_counter, vis_warning_counter, warning_count, alerts
    
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)  # Resize for uniform input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert frame to RGB

    results = face_mesh.process(rgb_frame)  # Process the frame for face landmarks
    direction, head_direction = '', ''
    obj_d = True
    fps = 0

    # If face landmarks are detected
    if results.multi_face_landmarks:
        head_direction = head_pose(rgb_frame, results)  # Get head direction

        if head_direction in ["Center", "Up"]:
            eye_direction = eye_track(ret, frame, rgb_frame, results)  # Track eye direction
            direction = eye_direction
        else:
            direction = head_direction

        # Calculate FPS
        end = time.time()
        totalTime = end - start_time
        fps = 1 / totalTime if totalTime > 0 else 0
        start_time = end

        # Monitor direction changes
        if direction in ["Right", "Left", "Up"]:
            change_dir_counter += 1
            if change_dir_counter > 20:
                change_dir_counter = 0
                dir_warning_counter += 1
                warning_count += 1
                return False, direction, head_direction, fps, obj_d, alerts["direction"][0]
            return True, direction, head_direction, fps, obj_d, None
        else:
            obj_d = obj_detect(ret, frame)
            if not obj_d:
                return False, direction, head_direction, fps, obj_d, alerts["object"][0] 
            return True, direction, head_direction, fps, obj_d, None
    else:
        # If no face detected, increment visibility counter
        end = time.time()
        totalTime = end - start_time
        fps = 1 / totalTime if totalTime > 0 else 0
        start_time = end

        visibility_counter += 1
        if visibility_counter > 20:
            visibility_counter = 0
            change_dir_counter = 0
            vis_warning_counter += 1
            warning_count += 1
            return False, direction, head_direction, fps, obj_d, alerts["visibility"][0] 
        return True, direction, head_direction, fps, obj_d, None



