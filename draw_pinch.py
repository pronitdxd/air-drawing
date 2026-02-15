import cv2
import mediapipe as mp
import numpy as np
import math

# ------------------------------
# Setup
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

canvas = None
prev_x, prev_y = None, None

# ------------------------------
# Colors and Brush
# ------------------------------
colors = [
    (0, 0, 0),        # black
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (255, 255, 255)   # white
]

color_index = 1
current_color = colors[color_index]

brush_thickness = 8
eraser_thickness = 25

SMOOTHING = 0.7
pinky_was_up = False

# ------------------------------
# Finger Detection Helper
# ------------------------------
def finger_is_up(tip_id, base_id, hand_landmarks):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[base_id].y

# ------------------------------
# Main Loop
# ------------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            # Get landmark positions
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]

            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

            # ------------------------------
            # 1️⃣ Drawing with pinch
            # ------------------------------
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 40:  # pinch detected

                if prev_x is None:
                    prev_x, prev_y = x2, y2

                # Smoothing
                x_smooth = int(SMOOTHING * prev_x + (1 - SMOOTHING) * x2)
                y_smooth = int(SMOOTHING * prev_y + (1 - SMOOTHING) * y2)

                cv2.line(canvas, (prev_x, prev_y),
                         (x_smooth, y_smooth),
                         current_color,
                         brush_thickness)

                prev_x, prev_y = x_smooth, y_smooth
            else:
                prev_x, prev_y = None, None

            # ------------------------------
            # 2️⃣ Color change with pinky
            # ------------------------------
            pinky_is_up = finger_is_up(20, 18, hand)

            if pinky_is_up and not pinky_was_up:
                color_index = (color_index + 1) % len(colors)
                current_color = colors[color_index]

            pinky_was_up = pinky_is_up

            # ------------------------------
            # 3️⃣ Eraser (all fingers up)
            # ------------------------------
            index_up = finger_is_up(8, 6, hand)
            middle_up = finger_is_up(12, 10, hand)
            ring_up = finger_is_up(16, 14, hand)
            pinky_up = finger_is_up(20, 18, hand)

            if index_up and middle_up and ring_up and pinky_up:
                cv2.circle(canvas, (x2, y2), eraser_thickness, (0, 0, 0), -1)

            # Cursor
            cv2.circle(frame, (x2, y2), 8, current_color, -1)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ------------------------------
    # Color Palette Display
    # ------------------------------
    for i, c in enumerate(colors):
        cv2.rectangle(frame, (10 + i*40, 10),
                      (40 + i*40, 40),
                      c, -1)

        if i == color_index:
            cv2.rectangle(frame, (10 + i*40, 10),
                          (40 + i*40, 40),
                          (255, 255, 255), 2)

    # ------------------------------
    # Merge Canvas + Frame Cleanly
    # ------------------------------
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    combined = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Drawing - Press Q to Exit", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):   # clear canvas
        canvas = np.zeros_like(frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
