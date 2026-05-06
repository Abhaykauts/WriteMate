import time
import cv2
import numpy as np
import pickle
import sounddevice as sd
import json
import keyboard
from vosk import Model, KaldiRecognizer


# ================= SVG =================
def save_to_svg(text, filename="output.svg", scale=5, spacing=15):
    letters = {
        "A": "M0,10 L5,0 L10,10 M2,6 L8,6",
        "B": "M0,0 L0,10 L6,10 L8,8 L8,6 L6,5 L0,5 M6,5 L8,4 L8,2 L6,0 L0,0",
        "C": "M10,0 L2,0 L0,5 L2,10 L10,10",
        "D": "M0,0 L0,10 L6,10 L10,5 L6,0 L0,0",
        "E": "M10,0 L0,0 L0,10 L10,10 M0,5 L7,5",
        "F": "M0,0 L0,10 M0,0 L10,0 M0,5 L7,5",
        "G": "M10,2 L8,0 L2,0 L0,5 L2,10 L8,10 L10,8 L10,6 L6,6",
        "H": "M0,0 L0,10 M10,0 L10,10 M0,5 L10,5",
        "I": "M0,0 L10,0 M5,0 L5,10 M0,10 L10,10",
        "J": "M10,0 L10,8 L8,10 L2,10 L0,8",
        "K": "M0,0 L0,10 M10,0 L0,5 L10,10",
        "L": "M0,0 L0,10 L10,10",
        "M": "M0,10 L0,0 L5,5 L10,0 L10,10",
        "N": "M0,10 L0,0 L10,10 L10,0",
        "O": "M2,0 L8,0 L10,5 L8,10 L2,10 L0,5 L2,0",
        "P": "M0,10 L0,0 L6,0 L10,3 L6,6 L0,6",
        "Q": "M2,0 L8,0 L10,5 L8,10 L2,10 L0,5 L2,0 M6,6 L10,10",
        "R": "M0,10 L0,0 L6,0 L10,3 L6,6 L0,6 M6,6 L10,10",
        "S": "M10,0 L2,0 L0,3 L2,5 L8,5 L10,7 L8,10 L0,10",
        "T": "M0,0 L10,0 M5,0 L5,10",
        "U": "M0,0 L0,8 L2,10 L8,10 L10,8 L10,0",
        "V": "M0,0 L5,10 L10,0",
        "W": "M0,0 L2,10 L5,5 L8,10 L10,0",
        "X": "M0,0 L10,10 M10,0 L0,10",
        "Y": "M0,0 L5,5 L10,0 M5,5 L5,10",
        "Z": "M0,0 L10,0 L0,10 L10,10"
    }

    x, y = 10, 10
    paths = []

    for ch in text.upper():
        if ch == " ":
            x += 10 * scale + spacing
            continue

        if ch in letters:
            cmds = letters[ch].split(" ")
            t = []
            for c in cmds:
                if c[0] in ["M", "L"]:
                    px, py = map(int, c[1:].split(","))
                    t.append(f"{c[0]}{x+px*scale},{y+py*scale}")
            paths.append(f'<path d="{" ".join(t)}" stroke="black" fill="none"/>')

        x += 10 * scale + spacing

    with open(filename, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg">{"".join(paths)}</svg>')

    print("✅ SVG saved:", filename)


# ================= VOICE =================
def voice_mode():
    print("\n🎤 Voice Mode (Press Q to stop)\n")

    model = Model("vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, 16000)

    buffer = ""

    def callback(indata, frames, time_, status):
        nonlocal buffer

        if rec.AcceptWaveform(bytes(indata)):
            text = json.loads(rec.Result()).get("text", "").upper()
            if text:
                buffer += " " + text
                print("Buffer:", buffer.strip())

    with sd.RawInputStream(samplerate=16000, blocksize=8000,
                           dtype='int16', channels=1, callback=callback):

        while True:
            if keyboard.is_pressed('q'):
                print("\nSaving...")
                save_to_svg(buffer.strip())
                break
            time.sleep(0.1)


# ================= GESTURE =================
class mpHands:
    import mediapipe as mp

    def __init__(self):
        self.hands = self.mp.solutions.hands.Hands(False, 1)

    def Marks(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        hands = []
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                pts = [(int(l.x * width), int(l.y * height)) for l in h.landmark]
                hands.append(pts)
        return hands


def findDistances(hand):
    palm = ((hand[0][0]-hand[9][0])**2 + (hand[0][1]-hand[9][1])**2)**0.5
    mat = np.zeros((len(hand), len(hand)))
    for i in range(len(hand)):
        for j in range(len(hand)):
            mat[i][j] = (((hand[i][0]-hand[j][0])**2 +
                          (hand[i][1]-hand[j][1])**2)**0.5)/palm
    return mat


def findError(g1, g2, keyPoints):
    err = 0
    for r in keyPoints:
        for c in keyPoints:
            err += abs(g1[r][c] - g2[r][c])
    return err


def gesture_mode():
    global width, height
    width, height = 720, 360

    cam = cv2.VideoCapture(0)
    detector = mpHands()
    keyPoints = [0,4,5,9,13,17,8,12,16,20]

    with open("train.pkl", "rb") as f:
        names = pickle.load(f)
        gestures = pickle.load(f)

    buffer = []
    last = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, (width, height))
        hands = detector.Marks(frame)

        if hands and time.time() - last > 1:
            unknown = findDistances(hands[0])
            errors = [findError(g, unknown, keyPoints) for g in gestures]

            m = min(errors)
            idx = errors.index(m)

            if m < 10:
                buffer.append(names[idx])
                print("Buffer:", "".join(buffer))
                last = time.time()

        # draw landmarks
        for hand in hands:
            for pt in keyPoints:
                cv2.circle(frame, hand[pt], 8, (255, 0, 255), 2)

        cv2.putText(frame, "".join(buffer), (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        cv2.imshow("Cam", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            save_to_svg("".join(buffer))
            break

    cam.release()
    cv2.destroyAllWindows()


# ================= MENU =================
mode = input("""
Select Mode:
1 → Gesture Recognition
2 → Voice Recognition
Choice: 
""")

if mode == "1":
    gesture_mode()
elif mode == "2":
    voice_mode()
else:
    print("Invalid choice")