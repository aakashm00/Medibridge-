# src/infer.py
import cv2, torch, mediapipe as mp, numpy as np
from pathlib import Path
from src.preprocess import pad_or_sample
from src.collect import extract_landmarks
from src.llm_normalize import normalize as norm_text
from src.train import Model

ROOT = Path(__file__).resolve().parent.parent
LABELS = [l.strip() for l in (ROOT / "labels.txt").read_text().splitlines()]

T, D = 40, 258
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = Model()
model.load_state_dict(torch.load(ROOT / "sign_med_transformer.pt", map_location="cpu"))
model.eval().to(device)

def _predict_seq(seq_frames):
    x = pad_or_sample(seq_frames)[None, ...]  # [1,T,D]
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(xt)
        p = torch.softmax(logits, -1)[0].cpu().numpy()
    top = int(p.argmax())
    return LABELS[top], float(p[top])

def run_live(threshold=0.60, seconds=2.0, fps=20, frame_callback=None, running=lambda: True):
    """
    Generator yielding (pred, conf).
    - frame_callback(rgb_frame) is called each loop for Streamlit display.
    - running() should return True while we should continue (Streamlit stop control).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check macOS Privacy > Camera permissions for your terminal/IDE.")

    with mp.solutions.holistic.Holistic(model_complexity=1) as mp_h:
        buff = []
        try:
            while running():
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # display to Streamlit if requested
                if frame_callback is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_callback(frame_rgb)

                # landmark extraction
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = mp_h.process(rgb)
                feats = extract_landmarks(res)
                buff.append(feats)
                if len(buff) > int(seconds * fps):
                    buff = buff[-int(seconds * fps):]

                if len(buff) >= 10:
                    pred, conf = _predict_seq(buff)
                    if conf >= threshold:
                        yield norm_text(pred), conf
                        buff.clear()
        finally:
            cap.release()

if __name__ == "__main__":
    for pred, conf in run_live():
        print("PRED:", pred, conf, "->", norm_text(pred))