# Do independently learned fact updates add?

## Prospective matched-schedule experiment

The sentence-ablation and embedding-transplant experiments show distributed,
co-adapted changes. Here we test a simpler constructive hypothesis directly:
can independently learned fact updates be added to build a model of both?
This is not a dataset-only construction, because the component updates are
still trained. It tests a proposed composition law for those updates.

Use exactly the original corpus's lines 80 and 406, in that order:

```
The capital of France is Paris, a city on the Seine.
The capital of Greece is Athens, an ancient Mediterranean city.
```

Keep the original full-corpus compact vocabulary and step-zero weights.
Run 1,024 batch-one optimizer steps, seed 1337, with the original warmup,
40,000-step cosine schedule, Adam parameters, zero decay, and no clipping.
The two sentences are shuffled without replacement with the same iterator
seed. Each receives 512 scheduled exposures. Four conditions use exactly
the same initialization, batches, global optimizer clock, and loss scaling:

* both facts, plus a duplicate baseline checked bytewise after every update;
* France-only contribution: zero Greece's logit gradient after loss backward;
* Greece-only contribution: zero France's logit gradient after loss backward.

An omitted slot is still executed, and **Adam still steps on zero gradients**.
Its momentum is not reset or paused. Thus these controls preserve the global
clock; they are not independently running the old batch-one, 512-update
single-fact experiments. The original fixed loss normalization is retained.

With common initialization `W0` and the two masked-training endpoints `WA,WB`,
freeze these whole-model constructions before measuring their predictions:

```
SUM  = W0 + (WA-W0) + (WB-W0)
MEAN = W0 + 0.5*((WA-W0) + (WB-W0))
```

Compute each scalar in FP64 and round once to its FP32 master representation.
Apply the same formula to every unique tensor, including embeddings,
positions, biases, and norms. Do not choose per-layer coefficients or search
interpolation factors. The tied embedding alias remains tied.

## Controls and measurements

Check common initialization and compact mappings, shape/inventory integrity,
finite values, and unchanged source checkpoint bytes. Evaluate the joint,
France-contribution, Greece-contribution, SUM, and MEAN models on both entire
suffixes plus EOS from their first five tokens. Also report next-token counts
and parameter error relative to the directly trained joint model.

At step one, only the first scheduled sentence has contributed. The other
masked run must remain exactly at initialization, and SUM must match the
joint checkpoint byte-for-byte. This checks arithmetic/provenance, not the
later hypothesis. The duplicate joint run checks deterministic execution.

A failed formula would reject these particular additive constructions under
matched training conditions, not all possible nonlinear combinations or all
task-vector methods. A successful formula would be a two-fact construction,
not evidence that 1,024 facts superpose independently. Differences between
joint and isolated activations and Adam moments remain part of what is tested.
Generated checkpoints and reports remain local. Budget: under one hour.
