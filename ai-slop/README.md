# Research artifacts

This directory contains one-off, AI-generated research artifacts and analysis
tools. They are unreviewed experimental work and may contain errors.

- [`weight_analysis`](weight_analysis) contains the analysis tools and their
  tests, including Python modules and C++ probes.
- [`research`](research) contains experiment protocols, reports, and recorded
  evidence. These are historical snapshots: recorded commands, paths, hashes,
  and source inventories retain their original values, including references
  to the former `scripts/weight_analysis` and `research` locations.

The four generated evidence archives remain local and untracked. Links to
those archives do not resolve in a fresh clone; external datasets, checkpoints,
and binaries referenced by historical records are not bundled here.

Run current Python tools from the repository root:

```sh
PYTHONPATH=ai-slop python -m weight_analysis.<tool>
```

Replace `<tool>` with a module name. See the
[tool guide](weight_analysis/README.md) for tool-specific requirements and
examples. Run the CPU Python tests with:

```sh
PYTHONPATH=ai-slop python -m unittest discover \
  -s ai-slop/weight_analysis -t ai-slop -p '*_test.py' -q
```

C++ analysis targets use the `//ai-slop/weight_analysis:` Bazel prefix, for
example `//ai-slop/weight_analysis:causal_probe`.
