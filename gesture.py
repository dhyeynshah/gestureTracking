# All comments, debugging and numpy calculation is done with the help of ChatGPT (50-50)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen size for scaling
screen_w, screen_h = pyautogui.size()

# To prevent multiple clicks for one gesture
click_down = False

# Function to calculate distance between two points
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get landmarks as (x, y) in pixel coords
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        # Index fingertip is landmark 8
        index_finger_tip = landmarks[8]

        # Thumb tip is landmark 4
        thumb_tip = landmarks[4]

        # Move mouse cursor based on index finger tip position (scaled to screen)
        screen_x = np.interp(index_finger_tip[0], [0, w], [0, screen_w])
        screen_y = np.interp(index_finger_tip[1], [0, h], [0, screen_h])
        pyautogui.moveTo(screen_x, screen_y)

        # Check if thumb and index finger tips are close for click
        dist = distance(index_finger_tip, thumb_tip)

        # If fingers close and not already clicked, do left click
        if dist < 40:
            if not click_down:
                click_down = True
                pyautogui.click()
                print("Left Click")
                cv2.putText(frame, "Left Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        else:
            click_down = False

    else:
        cv2.putText(frame, "Show your hand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
