# Reproduce the third-attention puzzle from scratch

This guide starts without a tokenizer or checkpoint. It trains the ordinary
four-block model to memorize the corpus, verifies the saved checkpoint, and
then fits a larger pointwise MLP to the third attention block's outputs.
Everything needed beyond the machine prerequisites is downloaded or built by
[`reproduce_puzzle.sh`](reproduce_puzzle.sh). No research branch or private
experiment artifacts are needed.

## 1. Prepare a Linux GPU machine

The tested machine is an NVIDIA GH200 with compute capability **9.0**, Ubuntu
24.04 on ARM64, GCC 13.3, CUDA `nvcc` 13.3.73, `tileiras` 13.3.36, and driver
580.126.20. The repository currently targets **SM90**, so use a Hopper GPU
(for example GH200 or H100). This is not a CPU-only experiment. Other GPU
architectures require changes to the build configuration and are not covered
by these instructions.

Allow space for the CUDA toolkit and the Bazel build cache as well as the run
directory. A fresh machine should have at least **30 GB free** as a practical
starting allowance; the generated checkpoints are much smaller than the
toolchain and build artifacts. This is a budget recommendation, not a measured
minimum. Check available space before starting:

```bash
df -h "$HOME" /tmp
```

On Ubuntu 24.04, install basic tools:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl ca-certificates python3 python3-venv zip unzip
```

Install the **full CUDA 13.3 toolkit**, including its C++ headers, `nvcc`, and
`tileiras`. Python CUDA packages or a working `nvidia-smi` alone are not enough.
For a fresh Ubuntu 24.04 machine, the following selects NVIDIA's repository
for the host CPU architecture and installs the toolkit:

```bash
case "$(uname -m)" in
  aarch64) cuda_repository_arch=sbsa ;;
  x86_64) cuda_repository_arch=x86_64 ;;
  *) echo "These instructions cover ARM64 SBSA and x86-64 only." >&2; exit 1 ;;
esac
cuda_setup_dir="$(mktemp -d)"
curl --fail --location --retry 3 \
  "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${cuda_repository_arch}/cuda-keyring_1.1-1_all.deb" \
  --output "$cuda_setup_dir/cuda-keyring.deb"
sudo dpkg -i "$cuda_setup_dir/cuda-keyring.deb"
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

These commands do **not** install or replace a GPU driver. If your machine does
not already have a working compatible driver, install it and reboot following
NVIDIA's [driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/).
For non-Ubuntu systems or existing CUDA installations, use NVIDIA's
[CUDA 13.3 installation guide](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-installation-guide-linux/index.html)
and [CUDA 13.3 download archive](https://developer.nvidia.com/cuda-13-3-0-download-archive).
Do not replace a working managed GPU driver merely to match the example version.

The Bazel configuration expects the toolkit at `/usr/local/cuda`. The CUDA
installer normally creates that link; verify it selects the intended toolkit.
Set these in every shell from which you build or run the experiment:

```bash
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
/usr/local/cuda/bin/nvcc --version
/usr/local/cuda/bin/tileiras --version
g++ --version
```

`PATH` selects the compiler; `LD_LIBRARY_PATH` lets the executable find CUDA's
shared runtime libraries. The `CUDA Version` shown by `nvidia-smi` describes
driver compatibility, not the installed compiler version. Check `nvcc` itself.

## 2. Get Pluto and the pinned Bazel version

Clone the repository using an account with access, then switch to `main`:

```bash
git clone https://github.com/sanjoy/pluto.git
cd pluto
git switch main
```

If GitHub requests authentication or returns "repository not found", make sure
your account has repository access and authenticate with GitHub. An authorized
SSH clone URL can also be used; an SSH key is not required for HTTPS cloning.

The repository pins Bazel **9.2.0** in `.bazelversion`. Install the official
standalone binary for your CPU architecture, checking its published SHA-256:

```bash
case "$(uname -m)" in
  aarch64)
    bazel_arch=arm64
    bazel_sha256=049dd21f40ad979db11c3ee68c96a42ce75f1185e69ac61ab20de1501427a410
    ;;
  x86_64)
    bazel_arch=x86_64
    bazel_sha256=7668a95db1250f12c40407251e4e203b4ec8bf39bc495d2f485b2d8c99048694
    ;;
  *) echo "Unsupported CPU architecture." >&2; exit 1 ;;
esac
bazel_setup_dir="$(mktemp -d)"
curl --fail --location --retry 3 \
  "https://releases.bazel.build/9.2.0/release/bazel-9.2.0-linux-${bazel_arch}" \
  --output "$bazel_setup_dir/bazel"
printf '%s  %s\n' "$bazel_sha256" "$bazel_setup_dir/bazel" | sha256sum --check || exit 1
install -d "$HOME/.local/bin"
install -m 0755 "$bazel_setup_dir/bazel" "$HOME/.local/bin/bazel"
bazel --version
```

The expected output is `bazel 9.2.0`. If you already use Bazelisk, it can select
the same pinned version instead. See the official
[Bazel release](https://github.com/bazelbuild/bazel/releases/tag/9.2.0) and
[installation options](https://bazel.build/install/ubuntu). The standalone
release includes its Java runtime; this project does not require a separate JDK.

## 3. Run the complete experiment

From the repository root, choose a **new** artifact directory outside the
checkout. The script refuses to overwrite an existing run:

```bash
mkdir -p "$HOME/pluto-runs"
facts_run="$HOME/pluto-runs/facts-puzzle-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/memorize_general_facts/reproduce_puzzle.sh --run_dir "$facts_run"
```

The script builds the optimized C++ binary, creates a local Python environment
with `tokenizers==0.22.0`, downloads the GPT-2 `tokenizer.json` from a pinned
revision and checks its SHA-256, and copies the checked-in 1,024-fact corpus
into the run directory. No `transformers` package is needed. Python is used
for orchestration and independent tokenization/auditing; the model, its
training, and the puzzle experiment run in C++ on the GPU.

It then performs the following stages, stopping on any failed check:

1. **Train the source model from scratch.** Four transformer blocks, hidden
   width 10, one head, MLP width 20, context 27, and the 4,475-token compact
   vocabulary: **48,680 parameters**. Use batch size 32, seed 1337, learning
   rate 0.0012, 100 warmup steps, and cosine decay over at most 120,000 steps.
   Evaluate every 256 steps and stop early once every scored prediction is
   correct. There is no gradient clipping.
2. **Reload and verify the saved checkpoint.** Require zero errors on all
   **10,002 suffix-plus-EOS targets**, and independently check the predictions
   against the corpus tokenized with the downloaded GPT-2 tokenizer. Also
   require all **1,024 autonomous greedy completions** from five-token prompts
   to reproduce the exact fact text. The interactive generation interface omits
   EOS from its text output; EOS predictions are checked separately by the
   teacher-forced audit above.
3. **Capture third-attention activations and audit separation.** The frozen
   source model must still get every scored target right. Equal complete
   hidden-state vectors must never require different scored target tokens.
4. **Fit the puzzle MLP for 300,000 steps**, reporting every 1,000 steps.
   Train a residual `10 -> 150 -> 10` MLP, its input LayerNorm, and the final
   LayerNorm; freeze the source transformer and the tied embedding projection.
   Use seed 3, batch size 32, and learning rate 0.01 decaying to 0.001.

The replacement has **3,160 affine MLP parameters**, or **3,200** including its
two trainable LayerNorms. That exceeds the original **1,380-parameter** path
after attention 3, which includes MLP3, attention4, MLP4, and their LayerNorms.
The frozen embedding/head is common to both paths and excluded from this
comparison.

The first build and downloads may dominate initial setup time. Training speed
depends on the machine. Run in a persistent terminal session if disconnecting
from a remote server. Do not stop after a periodic checkpoint appears: only a
checkpoint that passes the verification stage is suitable for the puzzle.
Source training and its verification share a four-hour safety deadline by
default; this is separate from the 120,000-step learning-rate schedule.

## 4. Interpret the results

The script prints the verified source checkpoint and writes its absolute path
to `checkpoint.txt`. Useful files below the chosen run directory are:

| Path | Contents |
| --- | --- |
| `setup.log`, `build.log`, `provenance.txt` | Setup/build output, source revision, and toolchain details. |
| `model/inputs/` | Snapshots of the executable, corpus, and tokenizer. |
| `model/summary.json` | Source training and verification outcome. |
| `model/trial_000_L4_W10_FF20/` | Source training, checkpoints, and independent prediction/greedy audits. |
| `checkpoint.txt` | The verified source checkpoint selected for the puzzle. |
| `puzzle.log` | Capture/separation checks and readout training statistics. |
| `puzzle/puzzle.html` | Standalone activation plots. |
| `puzzle/training.tsv`, `puzzle/best_mlp/` | Readout history and best readout weights. |

Open `$facts_run/puzzle/puzzle.html` locally in a web browser; it is self-contained.
It shows 1,024 points per position in dimension pairs (1,2), (3,4), etc., with
padding and supplied-prompt rows distinguished from scored rows.

The puzzle directory also contains `run.txt`, `training.tsv`, and `best_mlp/`.
Statistics distinguish correct next tokens from complete facts. The final
`BEST` line greedily generates each fact's suffix plus EOS from its first
five tokens and reports exact completions. Teacher-forced token accuracy is
not the same as whole-fact greedy accuracy.

A previous checkpoint produced 33.40% readout token accuracy and no complete
facts; **that is not a promised result for a newly trained source model**.
Both source training and readout fitting depend on initialization and numerical
details. The relevant question is whether a fully memorized source and
collision-free A3 states nevertheless resist this pointwise readout recipe.
Different accuracy is an experimental result, not necessarily a script error.

A from-scratch validation of this guide's default recipe on the tested GH200
(2026-09-26) measured:

| Stage | Checkpoint/update | Correct scored tokens | Exact greedy facts |
| --- | ---: | ---: | ---: |
| Original source model | 89,600 | 10,002 / 10,002 | 1,024 / 1,024 |
| Best replacement readout during 300,000 updates | 235,000 | 3,594 / 10,002 (35.93%) | 1 / 1,024 |

Source training took about 259 seconds; the 300,000-update readout fit took
about 285 seconds, excluding setup and the initial build. All 10,002 scored
A3 vectors were distinct, and both the source weights and frozen head remained
unchanged during readout training. These are observed results, not cross-machine
bitwise reproducibility or a proof that another fitting recipe cannot succeed.

If source training reaches its budget without memorizing, the script must not
continue with an inadequate checkpoint. Inspect its logs, keep that run's
artifacts, and make another run with a fresh directory. Increasing the step
budget also changes the cosine learning-rate schedule; it is not just extending
the same training trajectory. See the script's `--help` for supported controls.

All generated inputs, checkpoints, logs, plots, and Python environments belong
in the run directory, not Git. Bazel maintains its normal build cache separately.

For an intentional recipe change, specify controls explicitly and use a fresh
run directory. For example, this requests the smallest parameter-budget-matched
readout width without changing the source training recipe:

```bash
bash scripts/memorize_general_facts/reproduce_puzzle.sh \
  --run_dir "$HOME/pluto-runs/facts-puzzle-width66" \
  --model_steps 120000 --puzzle_steps 300000 --mlp_width 66 \
  --training_timeout 14400
```

The script uses the existing compact-model driver for one fixed architecture,
not a model-size sweep. A training-budget failure, failed verification, or
timeout stops the workflow before puzzle fitting, even if a checkpoint file
was written. Preserve the failed run directory for diagnosis; restarting this
script requires a new directory and trains a new model from scratch.

## 5. Reuse the verified checkpoint without retraining

Keep `facts_run` set to the completed run's directory. The commands below use
its immutable executable and input snapshots, not a potentially rebuilt binary.
Use the CUDA environment exports from step 1 in a new terminal.

To repeat only the puzzle fit with the same source weights:

```bash
source_checkpoint="$(cat "$facts_run/checkpoint.txt")"
snapshot_binary="$facts_run/model/inputs/memorize_general_facts"
puzzle_again="$(mktemp -d /tmp/pluto-puzzle.XXXXXX)/run"
"$snapshot_binary" \
  --mode=puzzle --train_mlp --puzzle_checkpoint="$source_checkpoint" \
  --tokenizer="$facts_run/model/inputs" \
  --corpus="$facts_run/model/inputs/corpus.txt" \
  --layers=4 --model_width=10 --attention_heads=1 --feed_forward_width=20 \
  --context_length=27 --compact_vocabulary=true \
  --mlp_width=150 --steps=300000 --eval_every=1000 \
  --batch_size=32 --seed=3 --learning_rate=0.01 \
  --output_dir="$puzzle_again"
echo "Puzzle results: $puzzle_again"
```

`puzzle_again` must not already exist. Choose `--mlp_width=66` to test the
smallest parameter-budget-matched expansion instead; this still uses the same
source checkpoint and does not train the source model again.

To query the original, fully memorized model:

```bash
source_checkpoint="$(cat "$facts_run/checkpoint.txt")"
"$facts_run/model/inputs/memorize_general_facts" \
  --mode=infer_model --infer_checkpoint="$source_checkpoint" \
  --tokenizer="$facts_run/model/inputs" \
  --layers=4 --model_width=10 --attention_heads=1 --feed_forward_width=20 \
  --context_length=27 --compact_vocabulary=true --generation_tokens=27 \
  --prompt="The capital of France is"
```

Omit the `--prompt` line to type prompts interactively. This loads the original
four-block model, not `best_mlp/`, which stores only the replacement readout's
trainable tensors and is not a complete standalone language-model checkpoint.
Exact completion is verified for the dataset's first-five-token prompts; it
is not a guarantee for arbitrary wording.

## 6. Test the workflow without a GPU

From the repository root:

```bash
bash -n scripts/memorize_general_facts/reproduce_puzzle.sh
python3 -B -m unittest discover -s scripts/memorize_general_facts \
  -p reproduce_puzzle_test.py -v
```

These CPU-only tests cover argument handling and refusing unverified, malformed,
or incomplete source-training results. They require no tokenizer download and
do not replace running the actual GPU training and verification stages.
