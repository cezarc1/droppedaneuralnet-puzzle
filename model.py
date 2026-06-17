from typing import override

import torch
from torch import nn

from puzzle_types import HIDDEN_DIM, INPUT_DIM, LayerPermutation, Piece, PieceMap


class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _load_linear(linear: nn.Linear, piece: Piece) -> None:
    with torch.no_grad():
        linear.weight.copy_(piece.weight)
        linear.bias.copy_(piece.bias)


def build_model(pieces: PieceMap, perm: LayerPermutation) -> nn.Sequential:
    """Assemble the recovered model from a solved permutation.

    perm is (inp0, out0, inp1, out1, ..., last): 48 (inp, out) block pairs
    followed by the single-output head.
    """
    *block_pieces, last_idx = perm
    blocks = []
    for inp_idx, out_idx in zip(block_pieces[::2], block_pieces[1::2], strict=True):
        block = Block(INPUT_DIM, HIDDEN_DIM)
        _load_linear(block.inp, pieces[inp_idx])
        _load_linear(block.out, pieces[out_idx])
        blocks.append(block)

    last = LastLayer(INPUT_DIM, 1)
    _load_linear(last.layer, pieces[last_idx])
    return nn.Sequential(*blocks, last).eval()
