# model.py
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d=128, nhead=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(2**15, d)
        enc = nn.TransformerEncoderLayer(d, nhead)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)
        h = self.tr(h.transpose(0, 1))
        return self.out(h.transpose(0, 1))
