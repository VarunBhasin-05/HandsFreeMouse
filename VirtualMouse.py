import cv2
import numpy as np
import mediapipe as mp
import pyautogui

#Configuration
max_hands = 1
detection_conf = 0.7
tracking_conf = 0.7
smoothness = 3        # lower = faster cursor, higher = smoother
click_distance = 40   # max distance (in screen pixels) to register a click
show_debug = False    # set True to print coordinates each frame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(
    max_num_hands=max_hands,
    min_detection_confidence=detection_conf,
    min_tracking_confidence=tracking_conf
)

# Initialize webcam
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# We’ll keep a running “previous” cursor position for smoothing
prev_mouse_x, prev_mouse_y = 0, 0
mouse_x, mouse_y = 0, 0

try:
    while True:
        ok, frame = cam.read()
        if not ok:
            print("⚠️  Camera read failed; exiting.")
            break
        # Mirror the camera so movement feels natural
        frame = cv2.flip(frame, 1)
        # Run MediaPipe on RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_tracker.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # Grab coordinates for index and middle fingertips
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ix, iy = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            mx, my = int(middle_tip.x * screen_w), int(middle_tip.y * screen_h)
            # Determine which fingers are “up”
            fingers = [0] * 5
            fingers[0] = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y <
                             hand.landmark[mp_hands.HandLandmark.THUMB_IP].y)
            fingers[1] = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
                             hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)
            fingers[2] = int(hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
                             hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)
            # Cursor move: only index finger up
            if fingers[1] == 1 and fingers[2] == 0:
                mouse_x = prev_mouse_x + (ix - prev_mouse_x) // smoothness
                mouse_y = prev_mouse_y + (iy - prev_mouse_y) // smoothness
                pyautogui.moveTo(mouse_x, mouse_y)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                if show_debug:
                    print(f"🖱️  Move to ({mouse_x}, {mouse_y})")
            # Click: index + middle fingers up and touching
            if fingers[1] == 1 and fingers[2] == 1:
                gap = np.hypot(mx - ix, my - iy)
                if gap < click_distance:
                    pyautogui.click()
                    if show_debug:
                        print("🖱️  Click")
                    cv2.waitKey(300)  # Debounce so we don’t double‑click
            # Draw landmarks just so we can see what’s happening
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        # Show the video feed
        cv2.imshow("Hand Mouse (ESC to quit)", frame)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cam.release()
    cv2.destroyAllWindows()