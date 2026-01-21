# train.py
import torch
from torch.optim import Adam
from midi_data import load_midis_recursive
from model import TinyTransformer
from tqdm import trange
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, "1")

seqs = load_midis_recursive(PATH)

# vocab = MIDI pitches 0–127
model = TinyTransformer(128)
opt = Adam(model.parameters(), 1e-3)
print(seqs)
for epoch in range(10):
    for seq in seqs[:10]:
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        print(x.shape, x.dtype, x.min().item(), x.max().item())
        y = torch.tensor(seq[1:]).unsqueeze(0)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()
        print("epoch", epoch, "loss", loss.item())
# After training loop
MODEL_PATH = os.path.join(BASE_DIR, "tiny_transformer.pth")  # where to save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
