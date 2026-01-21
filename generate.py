# generate.py
import torch
import pretty_midi
from model import TinyTransformer

model = TinyTransformer(128)
model.load_state_dict(torch.load("model.pt"))
model.eval()

seq = torch.randint(0, 128, (1, 1))
for _ in range(300):
    logits = model(seq)
    next = torch.multinomial(torch.softmax(logits[0, -1], 0), 1)
    seq = torch.cat([seq, next.view(1, 1)], 1)

midi = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(0)
t = 0
for p in seq[0]:
    inst.notes.append(pretty_midi.Note(100, int(p), t, t + 0.3))
    t += 0.3
midi.instruments.append(inst)
midi.write("generated.mid")
