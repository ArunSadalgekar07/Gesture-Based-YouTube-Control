import cv2 as cv
import mediapipe as mp
import pyautogui
from pynput.keyboard import Key, Controller
import time

# Initialize Keyboard Controller
keyboard = Controller()

last_action_time = 0  # Store last action timestamp
action_delay = 1  # Minimum time gap (in seconds) between actions

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Finger tip IDs for detection
fingerTipIds = [4, 8, 12, 16, 20]

# Capture Video
video = cv.VideoCapture(0)

while True:
    success, image = video.read()
    image = cv.flip(image, 1)  # Mirror the image for better UX
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process Image for Hand Detection
    results = hands.process(image_rgb)
    landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for index, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([index, cx, cy])

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Gesture Recognition
    fingers_open = []

    if landmarks_list:
        for tipId in fingerTipIds:
            if tipId == 4:  # Thumb
                if landmarks_list[tipId][1] > landmarks_list[tipId - 1][1]:  # Thumb right
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)
            else:
                if landmarks_list[tipId][2] < landmarks_list[tipId - 2][2]:  # Other fingers
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)

        count_fingers_open = fingers_open.count(1)

        # Gesture Control Logic
        if fingers_open == [1, 1, 1, 1, 1]:
            current_time = time.time()
            if current_time - last_action_time > action_delay:  # Execute if enough time passed
                cv.putText(image, "Play", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                keyboard.press("k")  # System play/pause toggle
                keyboard.release("k")
                last_action_time = current_time  # Update last action time
            
        
        elif fingers_open == [1, 0, 0, 0, 0]:  # Thumbs Up
            cv.putText(image, "Volume Up", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            pyautogui.press("volumeup")

        elif fingers_open == [0, 0, 0, 0, 1]:  # Pinky Up
            cv.putText(image, "Volume Down", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pyautogui.press("volumedown")



    # Show Video
    cv.imshow("Gesture-Based Media Control", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
