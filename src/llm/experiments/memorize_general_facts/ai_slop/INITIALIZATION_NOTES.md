# Fixed-initialization width scaling

The width search keeps QKV and MLP-input initialization standard deviation
at 0.02. Residual output projections keep standard deviation 0.005, using
the original eight-block residual scale even in shallower models. This makes
shared initial tensors identical across depths at a fixed width, but does
not make initial activation statistics invariant across widths.

## Expected initial attention scale

For a pre-LayerNorm input vector `z` of width `d`, let
`v = sum(z[i]^2) / d`. With independently initialized zero-mean Q/K weights
of variance `sigma^2`, each projected coordinate has conditional variance
`sigma^2 * d * v`. For query/key inputs `a, b` and head width `h`, the score
`q(a) dot k(b) / sqrt(h)` has initialization-ensemble variance
`sigma^4 * d^2 * v_a * v_b`. The division by the square root of head width
cancels the dot product's head-width factor, not its input-width dependence.

At initialization LayerNorm has gamma 1 and beta 0, so `v` is close to one
(slightly smaller because epsilon is 1e-5). Thus Q/K coordinate RMS is roughly
`0.02 * sqrt(d)` and marginal score standard deviation roughly `0.0004 * d`.
The latter gives 0.0064, 0.0096, and 0.0128 for widths 16, 24, and 32,
respectively, versus 0.2048 at the original width 512. These are ensemble
estimates, not exact predictions for a fixed seed and correlated corpus inputs.

Flatter initial attention could affect optimization, but this derivation does
not prove slower learning, explain individual failures, or establish that a
different initialization would improve memorization. A width-aware variant
would need its reference width and affected tensors specified and would be a
separate protocol. Rescaling only QKV differs from rescaling embeddings and
every projection. No such variant has been run or substituted into the active
or queued trials.

## Read-only CPU check

An independently repeated FP64 CPU calculation of the first block supports
nearly uniform initial attention. It also finds selective attention in the
trained narrow models, including those that failed the memorization criterion.
They are not stuck with uniform attention in this diagnostic.

| Width | Final checkpoint step | Initial within-query score spread | Final within-query score spread | Initial mean KL from uniform | Final mean KL from uniform |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 20,000 | 0.004933467 | 5.902750133 | 0.000012169 | 1.299750452 |
| 24 | 20,000 | 0.008593010 | 4.057876732 | 0.000036912 | 1.237652182 |
| 32 | 9,984 | 0.010789021 | 2.475036940 | 0.000058195 | 1.012688218 |

All are one-block, one-head models. Widths 16/24 use their final budget-failure
checkpoints; width 32 uses its successful checkpoint. Their different training
histories prevent a causal comparison of the endpoints.

For each of the 10,002 scored queries, scores include only its visible causal
keys. "Within-query score spread" is the square root of the mean of those
queries' population score variances: each query is centered separately and
weighted equally. KL is the equally weighted mean of
`KL(softmax(scores) || uniform over visible keys)`, in nats. This centering and
KL measurement matter: a global score spread could merely reflect row-constant
offsets, which do not affect softmax. The within-query statistic is distinct
from the marginal initialization-ensemble estimate above.

For an `n`-token sentence, scored query positions are 4 through `n-1`. The last
predicts EOS; earlier queries predict the next original token. No EOS input or
padding is needed to calculate these first-block causal scores. This is only a
CPU diagnostic; training still uses the unchanged 1,024-position context.

The replay promotes FP32 checkpoint weights to FP64 and omits native BF16
rounding boundaries and FP32 reduction order. It is **not a native activation
measurement**, a full-model replay, or a new memorization result. Dense layout
is `[input_dim, output_dim]`, with `output = input @ W + bias`; the packed QKV
columns are Q, then K, then V. This agrees with the CPU reference and CUDA
implementation. The first six checkpoint files supply token embeddings,
positions, LayerNorm gamma/beta, QKV weights, and QKV bias.

## Reproduce

Run from the repository root with NumPy and `tokenizers` installed. This reads
only completed checkpoints and does not launch GPU work or write files. It
checks corpus/tokenizer hashes, tensor sizes/finiteness, and the exact query
and visible-pair counts (10,002 and 97,594 respectively).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
from pathlib import Path
import hashlib
import numpy as np
from tokenizers import Tokenizer

repo = Path.cwd()
root = Path('/home/ubuntu/checkpoints/memorize_general_facts')
corpus = repo / 'testdata/general_facts_dataset.txt'
tokenizer = Path('/home/ubuntu/datasets/tokenizer/gpt2/tokenizer.json')
for path, expected in (
    (corpus, '814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c'),
    (tokenizer, '1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e'),
):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
tok = Tokenizer.from_file(str(tokenizer))
tok.no_padding()
tok.no_truncation()
rows = [tok.encode(s, add_special_tokens=False).ids
        for s in corpus.read_text().splitlines()]
assert len(rows) == 1024 and sum(len(ids) - 4 for ids in rows) == 10002

cases = [(16, 'width_depth_long_1', 20000),
         (24, 'width_depth_refine_24', 20000),
         (32, 'width_depth_long_1', 9984)]
for d, run, final_step in cases:
    for step in (0, final_step):
        p = root / run / f'width_{d}' / 'layers_1' / f'step_{step}'
        a = [np.fromfile(p / f'weight_{i}.bin', dtype='<f4').astype(np.float64)
             for i in range(6)]
        assert [x.size for x in a] == [50272*d, 1024*d, d, d, 3*d*d, 3*d]
        assert all(np.isfinite(x).all() for x in a)
        E, P, W = a[0].reshape(50272, d), a[1].reshape(1024, d), a[4].reshape(d, 3*d)
        variances, divergences, pairs = [], [], 0
        for ids in rows:
            x = E[ids] + P[:len(ids)]
            z = (x - x.mean(1, keepdims=True)) / np.sqrt(
                x.var(1, keepdims=True) + 1e-5)
            z = z * a[2] + a[3]
            q = z @ W[:, :d] + a[5][:d]
            k = z @ W[:, d:2*d] + a[5][d:2*d]
            scores = q @ k.T / np.sqrt(d)
            for j in range(4, len(ids)):
                s = scores[j, :j+1]
                pairs += len(s)
                variances.append(s.var())
                lp = s - s.max()
                lp -= np.log(np.exp(lp).sum())
                divergences.append(np.sum(np.exp(lp) * (lp + np.log(len(s)))))
        assert len(variances) == 10002 and pairs == 97594
        print(d, step, 'within_query_score_SD_RMS=', np.sqrt(np.mean(variances)),
              'mean_query_KL_nats=', np.mean(divergences))
PY
```
