# midi_data.py
import os
import pretty_midi


def load_midi_file(fname):
    """Load a single MIDI file and return a sequence of pitches."""
    midi = pretty_midi.PrettyMIDI(fname)
    notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    notes.sort()
    # print(notes)
    # drop notes which are too high?

    seq = [pitch for (t, pitch) in notes]
    if len(seq) > 50:
        return seq

    else:
        return None


def load_midis(path, max_files=200):
    """Load all MIDI files in a single folder (non-recursive)."""
    sequences = []
    for fname in os.listdir(path)[:max_files]:
        if not fname.lower().endswith(".mid"):
            continue
        seq = load_midi_file(os.path.join(path, fname))
        if seq is not None:
            sequences.append(seq)
    return sequences


def load_midis_recursive(base_path, max_files=None):
    """Walk through all subfolders and load MIDI files."""
    midi_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))

    if max_files is not None:
        midi_files = midi_files[:max_files]

    sequences = []
    for fname in midi_files:
        seq = load_midi_file(fname)
        if seq is not None:
            sequences.append(seq)

    return sequences
