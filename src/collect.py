# src/collect.py
from pathlib import Path
import cv2, json, time
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
LABELS = [l.strip() for l in (ROOT / "labels.txt").read_text().splitlines()]

OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_landmarks(res):
    points = []
    # Pose (33) -> x,y,z,visibility
    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            points += [lm.x, lm.y, lm.z, lm.visibility]
    else:
        points += [0.]* (33*4)

    # Left hand (21), Right hand (21) -> x,y,z
    for hand in ['left', 'right']:
        lms = getattr(res, f'{hand}_hand_landmarks')
        if lms:
            for lm in lms.landmark:
                points += [lm.x, lm.y, lm.z]
        else:
            points += [0.]* (21*3)
    return points  # length = 33*4 + 21*3*2 = 132 + 126 = 258

def record(label, reps=25, seconds=2.0, fps=20):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(model_complexity=1) as holistic:
        for r in range(reps):
            frames = []
            start = time.time()
            while time.time() - start < seconds:
                ok, frame = cap.read()
                if not ok: break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(image)
                feats = extract_landmarks(res)
                frames.append(feats)
                cv2.putText(frame, f"{label} rec {r+1}/{reps}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Collect", frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == 27: break
            seq = {"label": label, "frames": frames, "fps": fps}
            p = OUT_DIR / f"{label}_{int(time.time()*1000)}.json"
            p.write_text(json.dumps(seq))
            print("Saved", p)
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    for lab in LABELS:
        input(f"\nGet ready to perform: '{lab}'. Press Enter to start…")
        record(lab)