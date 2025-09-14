# src/preprocess.py
from pathlib import Path
import numpy as np, json, random

# Project root and paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [l.strip() for l in (ROOT / "labels.txt").read_text().splitlines()]
lab2id = {l: i for i, l in enumerate(LABELS)}

T, D = 40, 258

def normalize(seq):
    arr = np.array(seq)  # [t, 258]
    # Center by hip midpoint if available (pose idxs 23,24)
    # Pose landmarks are first 132 vals: each lm is 4 numbers
    def get_xy(idx):  # pose idx to x,y
        b = idx*4
        return arr[:, b], arr[:, b+1]
    try:
        lx, ly = get_xy(23); rx, ry = get_xy(24)
        cx = (lx+rx)/2.0; cy = (ly+ry)/2.0
        arr[:, 0::4] -= cx[:, None]  # pose x
        arr[:, 1::4] -= cy[:, None]  # pose y
        # Hands x,y are after 132; shift them too (x at 132+0::3)
        arr[:, 132+0::3] -= cx[:, None]
        arr[:, 132+1::3] -= cy[:, None]
    except Exception:
        pass
    # Scale by shoulder width if possible (pose 11,12)
    def dist(p,q):
        px,py = get_xy(p); qx,qy = get_xy(q)
        return np.maximum(np.sqrt((px-qx)**2+(py-qy)**2), 1e-3)
    try:
        s = dist(11,12)[:,None]
        arr[:, 0::4] /= s; arr[:, 1::4] /= s
        arr[:, 132+0::3] /= s; arr[:, 132+1::3] /= s
    except Exception:
        pass
    return arr

def pad_or_sample(frames):
    x = normalize(frames)
    if len(x) >= T:
        idx = np.linspace(0, len(x)-1, T).astype(int)
        x = x[idx]
    else:
        pad = np.repeat(x[-1:], T-len(x), axis=0)
        x = np.concatenate([x, pad], 0)
    return x.astype(np.float32)
def build_dataset():
    LABELS = [l.strip() for l in (Path(__file__).resolve().parent.parent / "labels.txt").read_text().splitlines()]
    lab2id = {l:i for i,l in enumerate(LABELS)}

    X, y = [], []
    for p in RAW.glob("*.json"):
        obj = json.loads(p.read_text())
        X.append(pad_or_sample(obj["frames"]))
        y.append(lab2id[obj["label"]])

    if not X:
        print("No JSONs found in data/raw/. Collect some first!")
        return

    X = np.stack(X)           # [N, T, D]
    y = np.array(y, int)

    N = len(y)
    idx = list(range(N))
    random.shuffle(idx)
    tr = int(N*0.85)

    np.savez_compressed(OUT/"dataset.npz",
                        X=X[idx], y=y[idx], split=np.array([tr], int))
    print("Saved", OUT/"dataset.npz", "N=", N)

# only run dataset build if called directly, NOT when imported
if __name__ == "__main__":
    build_dataset()