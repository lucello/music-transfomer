# midi_data.py
import os
import pretty_midi
import torch


def load_midis(path, max_files=200):
    sequences = []
    for fname in os.listdir(path)[:max_files]:
        if not fname.endswith(".mid"):
            continue
        midi = pretty_midi.PrettyMIDI(os.path.join(path, fname))
        notes = []
        for inst in midi.instruments:
            for n in inst.notes:
                notes.append((n.start, n.pitch))
        notes.sort()
        seq = [pitch for _, pitch in notes]
        if len(seq) > 50:
            sequences.append(seq)
    return sequences
