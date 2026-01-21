# train.py
import torch
from torch.optim import Adam
from midi_data import load_midis
from model import TinyTransformer
from tqdm import trange
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, "1")

seqs = load_midis(PATH)

# vocab = MIDI pitches 0–127
model = TinyTransformer(128)
opt = Adam(model.parameters(), 1e-3)

for epoch in range(10):
    for seq in seqs:
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        y = torch.tensor(seq[1:]).unsqueeze(0)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()
    print("epoch", epoch, "loss", loss.item())
