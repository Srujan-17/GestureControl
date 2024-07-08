import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe face mesh and hands detectors
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam capture
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame for face landmarks
    face_output = face_mesh.process(rgb_frame)
    landmark_points = face_output.multi_face_landmarks
    
    # Process frame for hand landmarks
    hand_output = hand_detector.process(rgb_frame)
    hands = hand_output.multi_hand_landmarks

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
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            index_x = index_y = thumb_x = thumb_y = 0
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                if id == 8:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    index_x = screen_w / frame_w * x
                    index_y = screen_h / frame_h * y
                if id == 4:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    thumb_x = screen_w / frame_w * x
                    thumb_y = screen_h / frame_h * y

            if abs(index_y - thumb_y) < 100:
                pyautogui.moveTo(index_x, index_y)

    cv2.imshow('Eye and Hand Controlled Mouse', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cam.release()
cv2.destroyAllWindows()
