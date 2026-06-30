# I Dropped a Neural Net — Solver

Solves the 2nd NN-based [Jane Street puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet) with a sub-second solve phase in the CPU-pinned benchmark snapshot below.

*As of 06/16/2026 this is the fastest among the public CPU solutions I surveyed under the benchmark protocol below.*

This solution builds heavily on [Hyunwoo Park's solution](https://github.com/hynwprk/droppedaneuralnet).

The novel contribution here is using the [adjoint method](https://en.wikipedia.org/wiki/Adjoint_state_method), used in optimal control and computational physics, to predict each swap's effect cheaply instead of re-running the full network on every candidate. To the best of my search, I have not seen another solver use this strategy yet. Please raise a PR if that is not the case!

For an academic-style writeup, see [this note](academic_puzzle_note.pdf).

### Disclaimer

While I made some nudges and contributions here and there, I heavily used coding agents for this puzzle. Pretty scary times, huh?

## Benchmark

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv run python solve_dropped_net.py
```

Example timings vary by machine/load:

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

For comparison, I benchmarked the public solution this builds on, plus the fastest public solutions that I could find under the same CPU-pinned protocol:

| solution                                                         | ordering approach                            | median full-script wall | internal solve timer |
| ---------------------------------------------------------------- | -------------------------------------------- | ----------------------: | -------------------: |
| **our solution**                                                 | L1 seed + adjoint-predicted swaps            |              **1.28 s** |           **0.19 s** |
| [alyxya](https://github.com/alyxya/janestreet-droppedaneuralnet) | greedy order + adjacent-swap bubble sort     |                  4.36 s |                  n/a |
| [Park](https://github.com/hynwprk/droppedaneuralnet)             | Frobenius seed + bubble-repair hill-climb    |                  4.97 s |                4.0 s |
| [EugenHotaj](https://github.com/EugenHotaj/droppedaneuralnet)    | cosine pairing + ~10k random-swap hill-climb |                   196 s |                  n/a |

All four solvers' timings include language runtime startup, imports, data loading, and verification, which dominate the sub-5 s solvers. Some solvers do not provide the actual solve timing.

All solvers run in a CPU-pinned k8s batch job on a single node, with an Intel Core i9-13900F CPU and no GPU acceleration.

## The idea

After pairing the 48 input/output halves, just like Park did, the remaining task is to put the 48 recovered blocks in the right order.

Now, in theory, we could try every ordering, but there are:

$$48! \approx 10^{61}$$

possible orderings, so maybe we can do better!

A more reasonable approach is to try swapping neighboring blocks and check whether the final model error improves. A reasonable way to do that is:

1. swap two neighboring blocks;
2. run a forward pass through the full network;
3. measure the error (MSE) against the `pred` column in the historical data;
4. keep the swap if it helped lower the error.

That works, but can we avoid some repeated work? Well ... a neighboring swap only directly changes the numbers around those two blocks. Everything before the swap is unchanged, and most of the question is really:

> If this swap changes the 48 numbers at this point in the network, how much should we expect the final prediction to change? Is there a way we can do that?

The adjoint trick gives us a cheap way to estimate that.

For the current guessed order, we run the full network once and save the 48 numbers after every block:

$$x_0, x_1, x_2, \ldots, x_{48}$$

Here $x_t$ means:

> the 48 numbers after the first `t` blocks have run.

Then we do one backward pass. At every position `t`, we compute a 48-number sensitivity vector:

$$g_t = \frac{\partial \hat{y}}{\partial x_t}$$

This vector is the adjoint.

In plain English, $g_t$ tells us:

> If I slightly changed each of the 48 numbers after block `t`, how much would the final prediction move?

So the adjoint is a local price list for the 48 numbers at that point in the network.

You might be familiar with the Jacobian from backprop. The Jacobian describes how small changes move forward through a network. The adjoint uses the same local derivative information in the backward direction: it tells us how much a small change at some intermediate point should affect the final prediction.

Backprop usually uses these sensitivities to update weights. Here, we use them to score potential swaps. Said another way, for a candidate swap, the adjoint lets us estimate how much the final prediction would change without running all later blocks again. We then combine that predicted output change with the current error to estimate whether the swap should lower MSE.

For a batch of historical rows, each row has its own 48-number adjoint vector at each position. In code, those vectors are stacked together, so we can represent them as a matrix. But the intuition is the same: for every row and every block position, the adjoint tells us how much the final prediction cares about the 48 numbers at that point.

Now let’s make the swap scoring concrete.

Suppose two neighboring blocks are currently ordered as $A$ then $B$.

We already ran the current network once, so we have saved the 48 numbers before block $A$. Call that saved value $x_t$.

With the current order, those two blocks produce:

$$x_{t+2} = B(A(x_t))$$

If we swap them, those same two blocks would instead produce:

$$x'_{t+2} = A(B(x_t))$$

So the swap creates a local change:

$$\delta = x'_{t+2} - x_{t+2}$$

This local change is easy to compute. We only have to run the two swapped blocks.

The expensive question is:

> After this changed 48-number value goes through all the later blocks, how much will the final prediction change?

This is where the adjoint helps.

At position $t+2$, we already cached the adjoint sensitivity vector $g_{t+2}$. This vector tells us how much the final prediction cares about each of the 48 numbers at that point.

So we can estimate the swap’s effect on the final prediction with one dot product:

$$\Delta \hat{y} \approx g_{t+2} \cdot \delta$$

In plain English:

> the swap tells us what local change it creates; the adjoint tells us how much that local change matters.

Now we turn that predicted output change into a predicted error change.

If the current prediction error is:

$$e = \hat{y} - y$$

then after the swap, we estimate the new error as:

$$e + \Delta \hat{y}$$

So the predicted change in squared error is:

$$(e + \Delta \hat{y})^2 - e^2 = 2e\Delta \hat{y} + (\Delta \hat{y})^2$$

We average this predicted change over the sampled historical rows. If the average is confidently negative, the swap probably helps.

The important assumption is that the seed order is already close enough that remaining mistakes mostly look like local neighboring inversions. In that regime, a wrong neighboring pair should usually create a detectable MSE improvement when swapped. The adjoint estimate is only a local approximation, so after applying a round of swaps we refresh the saved values and adjoints before scoring the next round.

That is the whole trick: instead of re-running the full network for every neighboring swap, we run the current network once, do one backward pass, and then score many swaps using cheap local calculations.

## Overview

**1. Pair.** For every candidate `(input, output)`, compute `trace(W_out · W_in)`. A correctly-coupled block has inverse-related maps, so the product has a strong negative diagonal, with trace ≈ −11, versus ≈ 0 for wrong pairs. SciPy's `linear_sum_assignment` — the Hungarian algorithm — picks the optimal one-to-one matching: all 48 pairs, no data needed. This is the same basic pairing approach used by pretty much all solutions out there.

But we are not done. Pairing only says *which* two halves form each of the 48 blocks. It does not tell us the *order* the blocks run in.

The network feeds each block's output into the next, so the order changes the result. Solving the pairing cuts the search from:

$$(48!)^2 \approx 10^{122}$$

down to:

$$48! \approx 10^{61}$$

orderings. Unfortunately, that is still far too many to brute-force.

**2. Seed.** We then sort blocks by ascending L1 norm of `W_out`. [Park's solution uses Frobenius norm](https://github.com/hynwprk/droppedaneuralnet/pull/1); in my tests, L1 seems to converge quicker.

Empirically, the size of `W_out` works as a rough depth proxy. Sorting by ascending L1 norm gives an order that is close enough for the swap-refinement step to finish quickly.

**3. Refine.** The L1 seed gives a good rough order, but some neighboring blocks may still be flipped. We fix those with an adjoint-guided local search.

For the current order, we run the 48 blocks once on a sample of historical rows and save the running 48-dimensional value after every block.

Then we do one backward pass. This gives us, at every saved position, a sensitivity vector: how much the final prediction would change if each of those 48 numbers changed slightly.

That sensitivity vector is the adjoint.

For a candidate adjacent swap, we do not re-run the full network. We only run the two involved blocks in the swapped order and compare the local result against the current local result. This gives us a local change vector.

Then we dot that local change vector with the cached adjoint. This predicts how much the final prediction would change if we made the swap.

Finally, we combine that predicted output change with the current prediction error to estimate the swap's effect on MSE.

Swaps with confidently negative predicted MSE-change are applied. We only apply non-overlapping swaps in one round, then refresh the saved values and adjoints and repeat.

So each refinement round looks like this:

1. run the current order once;
2. save the 48 numbers after every block;
3. compute adjoint sensitivities with one backward pass;
4. cheaply score neighboring swaps;
5. apply the confident non-overlapping improvements;
6. repeat until no confident swap remains.

**4. Verify.** As a sanity check, we rebuild an `nn.Module` from the recovered permutation and confirm that all values match the `pred` column on all 10,000 rows from the provided `historical_data.csv`. This verification is not counted in the internal solve timing.
