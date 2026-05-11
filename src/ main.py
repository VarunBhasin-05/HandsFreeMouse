import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller
import time
import tkinter

W_CAM, H_CAM = 640, 480
RECT_PADDING = 100
SMOOTHING = 5
CLICK_DISTANCE = 35

mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

root = tkinter.Tk()
SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()
root.destroy()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    exit()

cap.set(3, W_CAM)
cap.set(4, H_CAM)

p_loc_x, p_loc_y = 0, 0

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        cv2.rectangle(img, (RECT_PADDING, RECT_PADDING),
                      (W_CAM - RECT_PADDING, H_CAM - RECT_PADDING), (255, 0, 255), 2)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            lms = hand_lms.landmark

            h, w, c = img.shape
            ix, iy = int(lms[8].x * w), int(lms[8].y * h)
            mx, my = int(lms[12].x * w), int(lms[12].y * h)

            index_up = lms[8].y < lms[6].y
            middle_up = lms[12].y < lms[10].y

            if index_up and not middle_up:
                x_mapped = np.interp(ix, (RECT_PADDING, W_CAM - RECT_PADDING), (0, SCREEN_W))
                y_mapped = np.interp(iy, (RECT_PADDING, H_CAM - RECT_PADDING), (0, SCREEN_H))

                curr_x = p_loc_x + (x_mapped - p_loc_x) / SMOOTHING
                curr_y = p_loc_y + (y_mapped - p_loc_y) / SMOOTHING

                mouse.position = (curr_x, curr_y)
                p_loc_x, p_loc_y = curr_x, curr_y

                cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED)

            elif index_up and middle_up:
                dist = np.hypot(ix - mx, iy - my)
                if dist < CLICK_DISTANCE:
                    cv2.circle(img, (ix, iy), 12, (255, 255, 0), cv2.FILLED)
                    mouse.click(Button.left, 1)
                    time.sleep(0.15)

            draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
