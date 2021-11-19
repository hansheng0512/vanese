import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# CSV Header
fieldnames = ['left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'result']
csv_file_object = open('dataset.csv', 'w', encoding='UTF8', newline='')
writer = csv.DictWriter(csv_file_object, fieldnames=fieldnames)
writer.writeheader()

# os.listdir
# For static images:
IMAGE_FILES = ['images/1.jpg', 'images/1.jpg']
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue

    # NOSE = 0
    # LEFT_EYE_INNER = 1
    # LEFT_EYE = 2
    # LEFT_EYE_OUTER = 3
    # RIGHT_EYE_INNER = 4
    # RIGHT_EYE = 5
    # RIGHT_EYE_OUTER = 6
    # LEFT_EAR = 7
    # RIGHT_EAR = 8
    # MOUTH_LEFT = 9
    # MOUTH_RIGHT = 10
    # LEFT_SHOULDER = 11
    # RIGHT_SHOULDER = 12
    # LEFT_ELBOW = 13
    # RIGHT_ELBOW = 14
    # LEFT_WRIST = 15
    # RIGHT_WRIST = 16
    # LEFT_PINKY = 17
    # RIGHT_PINKY = 18
    # LEFT_INDEX = 19
    # RIGHT_INDEX = 20
    # LEFT_THUMB = 21
    # RIGHT_THUMB = 22
    # LEFT_HIP = 23
    # RIGHT_HIP = 24
    # LEFT_KNEE = 25
    # RIGHT_KNEE = 26
    # LEFT_ANKLE = 27
    # RIGHT_ANKLE = 28
    # LEFT_HEEL = 29
    # RIGHT_HEEL = 30
    # LEFT_FOOT_INDEX = 31
    # RIGHT_FOOT_INDEX = 32

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)

    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # CSV Writer
    left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    right_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    left_elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    left_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    right_elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    right_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

    writer.writerows([{
        'left_shoulder_x': left_shoulder_x,
        'left_shoulder_y': left_shoulder_y,
        'right_shoulder_x': right_shoulder_x,
        'right_shoulder_y': right_shoulder_y,
        'left_elbow_x': left_elbow_x,
        'left_elbow_y': left_elbow_y,
        'right_elbow_x': right_elbow_x,
        'right_elbow_y': right_elbow_y,
        'result': 1
    }])

    # print(
    #     f'LEFT_SHOULDER coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height})'
    # )
    # print(
    #     f'RIGHT_SHOULDER coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height})'
    # )
    # print(
    #     f'LEFT_ELBOW coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height})'
    # )
    # print(
    #     f'RIGHT_ELBOW coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height})'
    # )

    # # Display Annotated Image
    # cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))
    # cv2.waitKey(0)
    #
    # # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)