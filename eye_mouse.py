import cv2
import mediapipe as mp
import pyautogui
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize mediapipe face mesh and hand detectors
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hand_detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Get screen size
screen_w, screen_h = pyautogui.size()

# Define constants
offset = 20
imgSize = 300
labels = ["Hello", "1", "2", "I love you"]

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgOutput = frame.copy()

    # Process frame for face landmarks
    face_output = face_mesh.process(rgb_frame)
    landmark_points = face_output.multi_face_landmarks

    # Process frame for hand landmarks and gesture classification
    hands, img = hand_detector.findHands(frame, flipType=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop is not None and imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            gesture_label = labels[index]

            if gesture_label == "1" and distance > 100:
                pyautogui.press("volumeup")
            elif gesture_label == "I love you" and distance > 100:
                pyautogui.press("volumedown")

            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Keep these lines to show the processed and cropped hand images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Handle eye-based clicking
    if landmark_points:
        landmarks = landmark_points[0].landmark
        left_eye_landmarks = [landmarks[145], landmarks[159]]
        for landmark in left_eye_landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if abs(left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.01:
            pyautogui.click()
            pyautogui.sleep(1)

    # Handle hand-based mouse movement
    if hands:
        for hand in hands:
            lmList = hand['lmList']
            index_x = index_y = thumb_x = thumb_y = 0
            for id, lm in enumerate(lmList):
                x, y, _ = lm
                if id == 8:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    index_x = screen_w / frame_w * x
                    index_y = screen_h / frame_h * y
                if id == 4:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    thumb_x = screen_w / frame_w * x
                    thumb_y = screen_h / frame_h * y

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            if distance < 100:
                pyautogui.moveTo(index_x, index_y)
            else:
                if gesture_label == "1":
                    pyautogui.press("volumeup")
                elif gesture_label == "I love you":
                    pyautogui.press("volumedown")

    cv2.imshow('Eye and Hand Controlled Mouse', frame)
    # Comment out or remove this line to prevent the 'Hand Gesture' window from showing
    # cv2.imshow('Hand Gesture', imgOutput)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
