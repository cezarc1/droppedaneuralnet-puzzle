import math
import time
from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from model import build_model
from puzzle_types import (
    HIDDEN_DIM,
    INPUT_DIM,
    BlockIdx,
    BlockOrder,
    BlockPair,
    BlockPairs,
    BlockWeights,
    BlockWeightsByIndex,
    LayerPermutation,
    Piece,
    PieceIds,
    PieceIdx,
    PieceMap,
    PuzzleData,
)

TOL = 1e-9  # treat error below this as "solved"
EPS = 1e-12  # small number to avoid dividing by zero (not machine epsilon)
SEARCH_SAMPLES = 1000  # rows used to order; full set is only for verification
EXPECTED_BLOCKS = 48
EXPECTED_LAST_LAYERS = 1


def load_pieces(pieces_dir: Path) -> PieceMap:
    # .pth keys are exactly Piece's fields (weight, bias), so ** unpacks cleanly.
    return {
        PieceIdx(i): Piece(
            **torch.load(
                pieces_dir / f"piece_{i}.pth", map_location="cpu", weights_only=True
            )
        )
        for i in range(97)
    }


def identify_piece_types(
    pieces: PieceMap,
) -> tuple[PieceIds, PieceIds, PieceIdx]:
    inp_pieces = tuple(
        sorted(
            i
            for i, piece in pieces.items()
            if piece.weight.shape == (HIDDEN_DIM, INPUT_DIM)
        )
    )
    out_pieces = tuple(
        sorted(
            i
            for i, piece in pieces.items()
            if piece.weight.shape == (INPUT_DIM, HIDDEN_DIM)
        )
    )
    last_pieces = tuple(
        sorted(i for i, piece in pieces.items() if piece.weight.shape == (1, INPUT_DIM))
    )
    if (
        len(inp_pieces) != EXPECTED_BLOCKS
        or len(out_pieces) != EXPECTED_BLOCKS
        or len(last_pieces) != EXPECTED_LAST_LAYERS
    ):
        raise ValueError(
            "expected 48 input pieces, 48 output pieces, and 1 last piece; "
            f"found {len(inp_pieces)} input, {len(out_pieces)} output, "
            f"{len(last_pieces)} last"
        )
    return inp_pieces, out_pieces, last_pieces[0]


def find_block_pairs(
    pieces: PieceMap,
    inp_pieces: PieceIds,
    out_pieces: PieceIds,
) -> BlockPairs:
    """Match each input half to its output half (best one-to-one via Hungarian).

    Score = trace(W_out @ W_in): for a block whose halves belong together this is
    strongly negative; for a mismatch it is ~0. Minimizing the total picks the pairing.
    """
    w_inp = torch.stack([pieces[i].weight for i in inp_pieces])  # (48, 96, 48)
    w_out = torch.stack([pieces[o].weight for o in out_pieces])  # (48, 48, 96)
    pairing_scores = torch.einsum("jcd,idc->ij", w_out, w_inp).numpy()
    inp_rows, out_cols = linear_sum_assignment(pairing_scores)
    return tuple(
        BlockPair(inp_pieces[r], out_pieces[c])
        for r, c in zip(inp_rows, out_cols, strict=True)
    )


def precompute_block_weights(
    pieces: PieceMap,
    pairs: BlockPairs,
) -> BlockWeightsByIndex:
    return tuple(
        BlockWeights(
            w_inp=pieces[pair.inp].weight,
            b_inp=pieces[pair.inp].bias,
            w_out=pieces[pair.out].weight,
            b_out=pieces[pair.out].bias,
        )
        for pair in pairs
    )


def apply_block(x: torch.Tensor, block: BlockWeights) -> torch.Tensor:
    return x + torch.relu(x @ block.w_inp.T + block.b_inp) @ block.w_out.T + block.b_out


def forward_states(
    x: torch.Tensor,
    block_weights: BlockWeightsByIndex,
    order: Sequence[BlockIdx],
) -> tuple[torch.Tensor, ...]:
    """The running signal before each block: (x, block0(x), block1(block0(x)), ...)."""
    return tuple(
        accumulate(
            order,
            lambda curr, idx: apply_block(curr, block_weights[idx]),
            initial=x,
        )
    )


def output_gradients(
    states: tuple[torch.Tensor, ...],
    block_weights: BlockWeightsByIndex,
    order: Sequence[BlockIdx],
    w_head: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """How much the prediction moves if you nudge the running signal at each position.

    One backward pass: start from the head weights at the end and walk back through
    each block, reusing the signals cached by forward_states. One gradient per position.
    """
    n = states[0].shape[0]
    length = len(order)
    g = w_head.unsqueeze(0).expand(n, -1).contiguous()  # gradient at the final position
    grads = [torch.empty_like(g) for _ in range(length + 1)]
    grads[length] = g
    for t in range(length - 1, -1, -1):  # reverse order
        block = block_weights[order[t]]
        gate = (states[t] @ block.w_inp.T + block.b_inp > 0).to(g.dtype)  # ReLU mask
        g = g + ((g @ block.w_out) * gate) @ block.w_inp
        grads[t] = g
    return tuple(grads)


def l1_seed(block_weights: BlockWeightsByIndex) -> BlockOrder:
    """Rough starting order: smallest output weights first (a stand-in for depth)."""
    sizes = torch.tensor([block.w_out.abs().sum() for block in block_weights])
    return tuple(BlockIdx(i) for i in sizes.argsort().tolist())


def eval_order(
    start_state: torch.Tensor,
    block_weights: BlockWeightsByIndex,
    order: Sequence[BlockIdx],
    last_layer: Piece,
    y_pred: torch.Tensor,
) -> float:
    """Mean-squared error of running `order` from `start_state` through the head."""
    curr = start_state
    for idx in order:
        curr = apply_block(curr, block_weights[idx])
    predictions = (curr @ last_layer.weight.T + last_layer.bias).squeeze()
    return ((predictions - y_pred) ** 2).mean().item()


def refine_order_with_adjoint(
    block_weights: BlockWeightsByIndex,
    last_layer: Piece,
    w_head: torch.Tensor,
    x: torch.Tensor,
    y_pred: torch.Tensor,
    initial_order: BlockOrder,
    *,
    conf_thresh: float = 1.0,
    max_rounds: int = 80,
) -> BlockOrder:
    """Refine a seed order by predicting each swap, not re-running the network.

    Once per round we cache the running signal at each position and the output
    gradients (see output_gradients). Swapping the blocks at positions t, t+1 changes
    the signal locally by `delta` (two block evaluations). Everything after is
    unchanged, so the change in prediction is, to first order, `grad . delta`; the
    change in mean-squared error is `mean(2 * error * change + change**2)`. Negative
    means the swap helps. We apply the confident, non-overlapping beneficial swaps
    (those large vs their sampling noise), refresh, and repeat until none help.
    """
    order = list(initial_order)
    length = len(order)

    def errors(states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        predictions = (states[-1] @ last_layer.weight.T + last_layer.bias).squeeze()
        return predictions - y_pred

    states = forward_states(x, block_weights, order)
    grads = output_gradients(states, block_weights, order, w_head)
    e = errors(states)

    for _ in range(max_rounds):
        candidates: list[tuple[float, int]] = []
        for t in range(length - 1):
            block_a, block_b = block_weights[order[t]], block_weights[order[t + 1]]
            swapped = apply_block(apply_block(states[t], block_b), block_a)
            change = (grads[t + 2] * (swapped - states[t + 2])).sum(dim=1)
            effect = 2.0 * e * change + change * change
            predicted = effect.mean().item()
            noise = (effect.std(unbiased=True) / math.sqrt(effect.numel())).item()
            if predicted < -TOL and predicted / (noise + EPS) < -conf_thresh:
                candidates.append((predicted, t))
        if not candidates:
            break
        candidates.sort()  # most beneficial (most negative) first
        used: set[int] = set()
        for _predicted, t in candidates:
            if t in used or t + 1 in used:
                continue
            order[t], order[t + 1] = order[t + 1], order[t]
            used.update((t, t + 1))
        states = forward_states(x, block_weights, order)
        grads = output_gradients(states, block_weights, order, w_head)
        e = errors(states)
        if (e * e).mean().item() < TOL:
            break
    return tuple(order)


def build_permutation(
    pairs: BlockPairs,
    order: BlockOrder,
    last_piece: PieceIdx,
) -> LayerPermutation:
    """Build final permutation: (inp0, out0, inp1, out1, ..., last)."""
    return (*(piece for idx in order for piece in pairs[idx]), last_piece)


def solve(data: PuzzleData) -> LayerPermutation:
    """Match the halves, guess the order, fix it cheaply, return the layer order."""
    pairs = find_block_pairs(data.pieces, data.inp_pieces, data.out_pieces)
    print(f"  Paired {len(pairs)} blocks")

    block_weights = precompute_block_weights(data.pieces, pairs)
    last_layer = data.pieces[data.last_piece]
    w_head = last_layer.weight.squeeze(0)
    x_sub, y_sub = data.x[:SEARCH_SAMPLES], data.y_pred[:SEARCH_SAMPLES]

    t_start = time.perf_counter()
    seed = l1_seed(block_weights)
    seed_mse = eval_order(x_sub, block_weights, seed, last_layer, y_sub)
    print(f"  Seed MSE (||W_out||_1, N={SEARCH_SAMPLES}): {seed_mse:.6f}")

    order = refine_order_with_adjoint(
        block_weights, last_layer, w_head, x_sub, y_sub, seed
    )

    print(f"  Ordering solved in {time.perf_counter() - t_start:.3f}s")
    return build_permutation(pairs, order, data.last_piece)


def verify_recovered_predictions(
    predictions: torch.Tensor, y_pred: torch.Tensor
) -> float:
    recovered_mse = ((predictions - y_pred) ** 2).mean().item()
    if recovered_mse >= TOL:
        raise SystemExit(f"Recovered model MSE is non-zero: {recovered_mse:.10f}")
    return recovered_mse


def main() -> None:
    print("Loading pieces...")
    pieces = load_pieces(Path(__file__).parent / "pieces")
    inp_pieces, out_pieces, last_piece = identify_piece_types(pieces)
    print(
        f"  {len(inp_pieces)} input pieces, {len(out_pieces)} output pieces, "
        f"last={last_piece}"
    )

    print("Loading historical data...")
    historical_data = pd.read_csv(Path(__file__).parent / "historical_data.csv")
    x = torch.tensor(historical_data.iloc[:, :INPUT_DIM].values, dtype=torch.float32)
    y_pred = torch.tensor(historical_data["pred"].values, dtype=torch.float32)
    data = PuzzleData(
        pieces=pieces,
        inp_pieces=inp_pieces,
        out_pieces=out_pieces,
        last_piece=last_piece,
        x=x,
        y_pred=y_pred,
    )

    print("Solving...")
    t0 = time.perf_counter()
    permutation = solve(data)
    print(f"Total solve time: {time.perf_counter() - t0:.3f}s")

    print("\nVerifying recovered nn.Module...")
    model = build_model(pieces, permutation)
    with torch.no_grad():
        predictions = model(x).squeeze()
    recovered_mse = verify_recovered_predictions(predictions, y_pred)
    print(f"  Recovered model MSE: {recovered_mse:.10f}")

    print(f"\nPermutation ({len(permutation)} elements):")
    print(",".join(str(p) for p in permutation))


if __name__ == "__main__":
    main()
