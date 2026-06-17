from dataclasses import dataclass
from typing import NamedTuple, NewType

import torch

PieceIdx = NewType("PieceIdx", int)  # 0-96: index into the 97 pieces
BlockIdx = NewType("BlockIdx", int)  # 0-47: position in the 48-block sequence

HIDDEN_DIM = 96
INPUT_DIM = 48


class Piece(NamedTuple):
    weight: torch.Tensor
    bias: torch.Tensor


class BlockPair(NamedTuple):
    inp: PieceIdx
    out: PieceIdx


class BlockWeights(NamedTuple):
    w_inp: torch.Tensor  # (96, 48)
    b_inp: torch.Tensor  # (96,)
    w_out: torch.Tensor  # (48, 96)
    b_out: torch.Tensor  # (48,)


type PieceMap = dict[PieceIdx, Piece]
type PieceIds = tuple[PieceIdx, ...]
type BlockPairs = tuple[BlockPair, ...]
type BlockWeightsByIndex = tuple[BlockWeights, ...]
type BlockOrder = tuple[BlockIdx, ...]
type LayerPermutation = tuple[PieceIdx, ...]


@dataclass(frozen=True, slots=True)
class PuzzleData:
    pieces: PieceMap
    inp_pieces: PieceIds
    out_pieces: PieceIds
    last_piece: PieceIdx
    x: torch.Tensor  # (N, 48) input samples
    y_pred: torch.Tensor  # (N,) target predictions
