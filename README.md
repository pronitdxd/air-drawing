# Air Drawing 

Draw in the air using only your hand and a webcam — no stylus, no touchscreen, no mouse. Uses **MediaPipe Hands** to track your index finger and thumb in real time, letting you paint on a live camera feed with pinch gestures, switch colors with your pinky, and erase with an open hand.

---

## How It Works

```
Webcam frame
    │
    ▼
MediaPipe Hands  →  21 landmark coordinates per hand
    │
    ├── Thumb tip (lm[4]) + Index tip (lm[8])
    │       Distance < 40px  →  PINCH DETECTED  →  Draw stroke on canvas
    │       Distance ≥ 40px  →  Pen lifted (no stroke)
    │
    ├── Pinky raised (lm[20] above lm[18])
    │       Rising edge (was down, now up)  →  Cycle to next color
    │
    └── All 4 fingers up (index + middle + ring + pinky)
            →  Eraser mode (circle wipe on canvas)
    │
    ▼
Smoothed coordinates (SMOOTHING = 0.7 exponential blend)
    │
    ▼
cv2.line() drawn on persistent canvas layer
    │
    ▼
Bitmasked composite: canvas merged over webcam frame
```

---

## Features

- **Pinch to draw** — thumb-to-index distance under 40px activates the pen; release the pinch to lift it
- **Exponential smoothing** — coordinates are blended 70/30 with the previous position, eliminating shaky lines
- **8-color palette** — black, red, green, blue, yellow, magenta, cyan, white — displayed as swatches in the top-left corner
- **Pinky gesture to cycle colors** — rising-edge detection means one raise = one color change (no repeat firing)
- **Open-hand eraser** — all four fingers raised activates a circle eraser at the index fingertip
- **Live canvas overlay** — drawing persists on a transparent layer composited over the live camera feed using bitmasking, so strokes look painted onto the real scene
- **Adjustable brush thickness** — brush at 8px, eraser at 25px radius

---

## File Overview

| File | Purpose |
|---|---|
| `draw_pinch.py` | Main application — full drawing loop with pinch, color, and eraser gestures |
| `finger_track.py` | Standalone finger tracking test / utility |
| `hand_test.py` | MediaPipe hand detection sanity check |
| `camera_test.py` | Webcam feed test |
| `howtorun.txt` | Quick start instructions |

---

## Installation

### Prerequisites

- Python 3.9+
- A webcam

### Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### Run

```bash
python draw_pinch.py
```

---

## Controls

| Gesture | Action |
|---|---|
| Pinch (thumb + index close) | Draw / paint stroke |
| Open pinch | Lift pen |
| Raise pinky (flick up) | Cycle to next color |
| All 4 fingers open | Eraser mode |
| Press `C` | Clear canvas |
| Press `Q` | Quit |

---

## Color Palette

| Index | Color |
|---|---|
| 0 | Black |
| 1 | Red *(default)* |
| 2 | Green |
| 3 | Blue |
| 4 | Yellow |
| 5 | Magenta |
| 6 | Cyan |
| 7 | White |

---

## Requirements

```
opencv-python
mediapipe
numpy
```

---

## Author

**Pronit Das**
B.Tech/M.Tech Integrated CSE (Cybersecurity) · National Forensic Sciences University, Dharwad
[github.com/pronitdxd](https://github.com/pronitdxd)
