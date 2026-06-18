# I Dropped a Neural Net — Solver

Solves the 2nd NN-based [Jane Street puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet) with a sub-second internal solve phase on my machine.

_As of 06/16/2026 this is the fastest public solution I have seen._

This solution builds heavily on [Hyunwoo Park's solution](https://github.com/hynwprk/droppedaneuralnet).

The novel contribution here is using the [adjoint method](https://en.wikipedia.org/wiki/Adjoint_state_method), used in optimal control and computational physics, to try to _predict_ each swap's effect cheaply instead of re-running the full network on every candidate. I have not seen any other public solution use this strategy, yet. Please raise a PR if that is not the case!

### Disclaimer

While I made some nudges and contributions here and there, I heavily used coding agents (Claude) for this puzzle. Pretty scary times, huh?

## Benchmark

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv run python solve_dropped_net.py
```

Example (timings vary by machine/load)

```bash
uv run python solve_dropped_net.py
Loading pieces...
  48 input pieces, 48 output pieces, last=85
Loading historical data...
Solving...
  Paired 48 blocks
  Seed MSE (||W_out||_1, N=1000): 0.081558
  Ordering solved in 0.795s
Total solve time: 0.796s

Verifying recovered nn.Module...
  Recovered model MSE: 0.0000000000

Permutation (97 elements):
43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85
```

For comparison, I benchmarked the other public solution this builds on, plus the fastest public solutions that I could find:

| solution | ordering approach | median full-script wall | internal solve timer |
|---|---|---:|---:|
| **our solution** | L1 seed + adjoint-predicted swaps | **1.28 s** | **0.19 s** |
| [alyxya](https://github.com/alyxya/janestreet-droppedaneuralnet) | greedy order + adjacent-swap bubble sort | 4.36 s | n/a |
| [Park](https://github.com/hynwprk/droppedaneuralnet) | Frobenius seed + bubble-repair hill-climb | 4.97 s | 4.0 s |
| [EugenHotaj](https://github.com/EugenHotaj/droppedaneuralnet) | cosine pairing + ~10k random-swap hill-climb | 196 s | n/a |

All four solvers timmings include language runtime startup, imports, data loading and verification, which dominate the sub-5 s solvers. Some solvers don't provide the actual solve timing.

All solvers run in a CPU-pinned k8s batch job on a single node, w/ a Intel Core i9-13900F CPU (no GPU acceleration).

## The idea, explained by Claude better than I ever could

**You don't need to re-run the network to know whether a swap helps.** We do a forward pass through the network once and record two things — the signal passing through at each step, and how sensitive the final answer is to a small nudge at each step (one extra backward pass, aka the adjoint). From those, the effect of any neighbour swap is a quick local calculation rather than a full re-run.

## Method

**1. Pair.** For every candidate `(input, output)`, compute `trace(W_out · W_in)`. A correctly-coupled block has inverse-related maps, so the product has a strong negative diagonal (trace ≈ −11) versus ≈ 0 for wrong pairs. SciPy's `linear_sum_assignment` (the Hungarian algorithm) picks the optimal one-to-one matching — all 48 pairs, no data needed. Same as pretty much all solutions out there.

But we are not done: pairing only says _which_ two halves form each of the 48 blocks — not the _order_ the blocks run in. A ResNet feeds each block's output into the next, so the order changes the result. Solving the pairing cuts the search from `(48!)² ≈ 10¹²²` down to `48! ≈ 10⁶¹` orderings! Unfortunately still far too many to brute-force.

**2. Seed.** We then sort blocks by ascending L1 norm ([Park's solution uses Frobenius](https://github.com/hynwprk/droppedaneuralnet/pull/1)) of `W_out`. As in the PR L1 seems to converge quicker. Later blocks perturb the residual stream more, so we use weight magnitude as a depth proxy.

**3. Refine.** Instead of re-running the whole network to _score/eval_ each candidate swap, we try to _predict_ its effect, without training another model ;-p, using the [adjoint method](https://en.wikipedia.org/wiki/Adjoint_state_method). Read: we want to do this cheaply!

Basically, we run the network once and remember two things:

1) the running signal at each step
2) How much the final error responds to a nudge at each step (one backward pass).

Swapping two neighbouring blocks changes the signal locally; that combined with that error, the swap's effect on the error is then a couple of cheap matrix multiplies!

Each round we apply the swaps we're confident about and repeat until none help.

**4. Verify.** As a sanity check, we rebuild an `nn.Module` from the recovered permutation and confirm all values on all 10,000 rows from the provided `historical_data.csv` in the `pred` column are the same. This is not counted in the internal solve timing.
