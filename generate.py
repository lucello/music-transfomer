# generate.py
import torch
import pretty_midi
from model import TinyTransformer
import os

# ------------------------------
# Paths relative to script
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tiny_transformer.pth")
OUTPUT_PATH = os.path.join(BASE_DIR, "generated.mid")

# ------------------------------
# Load the model
# ------------------------------
model = TinyTransformer(128)  # must match your training initialization
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ------------------------------
# Generate sequence
# ------------------------------
seq = torch.randint(0, 128, (1, 1), dtype=torch.long)  # start token

for _ in range(300):
    logits = model(seq)  # forward pass
    probs = torch.softmax(logits[0, -1], dim=0)
    next_note = torch.multinomial(probs, 1)
    seq = torch.cat([seq, next_note.view(1, 1)], dim=1)

# ------------------------------
# Convert to MIDI
# ------------------------------
midi = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)  # acoustic piano
t = 0.0
for p in seq[0]:
    note = pretty_midi.Note(velocity=100, pitch=int(p), start=t, end=t + 0.3)
    inst.notes.append(note)
    t += 0.3

midi.instruments.append(inst)
midi.write(OUTPUT_PATH)
print(f"Saved generated MIDI to {OUTPUT_PATH}")
