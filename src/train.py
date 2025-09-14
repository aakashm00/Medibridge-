# src/train.py
import torch, math, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA = np.load(ROOT / "data" / "processed" / "dataset.npz", allow_pickle=False)

X, y = DATA["X"], DATA["y"]
tr = int(DATA["split"][0])
num_classes = int(y.max() + 1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
T, D = X.shape[1], X.shape[2]

class DS(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

train_ds, val_ds = DS(X[:tr], y[:tr]), DS(X[tr:], y[tr:])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_dl   = DataLoader(val_ds, batch_size=128)

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x:[B,T,d]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class Model(nn.Module):
    def __init__(self, d_in=D, d_model=256, nhead=8, depth=4, mlp=512, n_cls=num_classes):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pe  = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, mlp, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_cls)
        )
    def forward(self, x):
        h = self.inp(x)
        h = self.pe(h)
        h = self.enc(h)          # [B,T,d]
        h = h.mean(dim=1)        # temporal average pooling
        return self.cls(h)

model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

best = 0.0
for epoch in range(40):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward(); opt.step()
    # val
    model.eval(); correct=total=0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmin(dim=-1) if False else model(xb).argmax(dim=-1)
            correct += (pred==yb).sum().item(); total += yb.numel()
    acc = correct/total if total else 0.0
    print(f"epoch {epoch+1} val_acc={acc:.3f}")
    if acc>best:
        best=acc
        torch.save(model.state_dict(), "sign_med_transformer.pt")
        print("↑ saved best")
print("Best val_acc", best)
