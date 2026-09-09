# Independent restart-delta numerical audit

This note accompanies byte-exact copies of the one-off auditor programs and
their original results. No corpus/tokenizer contents, model calls, or GPU
execution were used. No frozen candidates were changed.

The frozen experiment is `/tmp/pluto-restart-delta.fZs3LQ/run`; its `frozen.json`
SHA256 is `a16d908bf7b496d6fbabf601d32d93e1e2bf6af5b83e2a428eec2825a59211ca`.
The copied scripts deliberately retain original absolute paths and exclusive
output creation. They are preserved forensic artifacts, not general CLI tools;
their source hashes and the JSON's recorded paths have not been rewritten.

## Checked scope

- All 35 activity NPZ files and the complete score archive passed their
  recorded hashes, shape checks, and finiteness checks. All other recorded
  artifact/source hashes were also checked before and after the audit.
- Independently recomputed all 30 transient score vectors using scalar
  `math.log`, and all seven shared vectors using Python sorting/grouped
  midranks and scalar summation, without calling the production scorer helpers.
- Reread eight archives: steps 570, 580, 590, 600, 1,780, 4,280, 7,230, and
  8,160. Archive and embedding hashes matched the frozen provenance; complete
  input hashes were unchanged during each read. The strict archive reader was
  shared, so archive parsing was **not** independently implemented.
- Independently recomputed all five endpoint row-norm vectors using `einsum`
  contractions. Maximum absolute difference was approximately `3.11e-15`.
- Recomputed both activity variants for intervals 570→580, 580→590, and 590→600
  using compensated scalar means and different contractions. All comparisons
  passed the recorded absolute/relative tolerances. The other **32 intervals'
  activities were hash-checked, not recomputed from checkpoint bytes**.
- Checked all 42 lists, totaling 5,376 selected IDs and 5,376 shuffled IDs.
  All frozen rankings agree exactly with independent heap sorting of their
  stored score vectors. Independently recomputed arithmetic produces exactly
  the same 42 unordered top-128 sets and their shuffled sets.

## Original failures and supplemental explanation

The initial result is retained with `passed: false` and nine error messages.
It found three shared ordinary-control lists with different within-top-128
ordering. For each list, this caused a ranking mismatch, a correspondingly
reordered shuffle, and a comparison of component columns at different IDs:
three cascading errors per list, **not nine independent numerical defects**.

The lists are `shared_ordinary_after_raw`,
`shared_ordinary_before_adjusted`, and `shared_ordinary_after_adjusted`.
Eight positions differ across them. Every boundary list agrees exactly.

The cause is FP64 summation order in mean-percentile scores. Maximum shared
score difference is `2.22e-16`; mathematically identical sums of integer
midranks can round to equal or adjacent floating-point numbers. For example,
IDs 13,784 and 45,893 at ranks 113–114 of `shared_ordinary_after_raw` both have
the exact doubled-midrank sum 499,557, but the frozen means differ by two
FP64 ulps. Independent arithmetic ties them and sorts by ascending ID.

The supplement checks component scores and positivity **at the frozen IDs**,
where they agree, and distinguishes these facts:

- Frozen ranking versus independently sorted stored FP64 scores: exact match.
- Independent arithmetic versus frozen full scores: within recorded tolerance.
- Independent arithmetic versus unordered top-128 membership: exact match.
- Independent arithmetic versus every internal rank: **not** an exact match.

An integer/rational midrank-sum ordering also preserves every top-128 set.
Exact mathematical tie ordering would require that approach, rather than
assuming all FP64 averaging implementations choose the same internal order.
The preserved freeze remains unchanged. Unordered top-128 corpus-presence and
frequency aggregates, and per-ID positivity, are unaffected; this is not a
claim about every possible rank-sensitive statistic.

## Preserved artifacts

| Repository copy | Original basename | SHA256 |
| --- | --- | --- |
| [restart_delta_independent_check.py](restart_delta_independent_check.py) | `independent_check.py` | `bc7c32daac4edb2ac153ac1ea1a53e8ea3fdea85151a8c50c9a6cf6d2f863567` |
| [restart_delta_independent_ties.py](restart_delta_independent_ties.py) | `independent_check_ties.py` | `76f6d1abfcb3f150958ff6cba093aea16486da1c28edf90c1b6c5bf725bdeade` |
| [restart_delta_independent_check.json](restart_delta_independent_check.json) | `run/independent_check.json` | `640b7db071a7c289e29ce144cf2e5a95824165cf1a4bff93f6df8a8eebbe146c` |
| [restart_delta_independent_ties.json](restart_delta_independent_ties.json) | `run/independent_check_ties.json` | `9b3e8731e997c4cf36ebe93f84b2a05d95e6aff18e53567f54716d88d630add4` |

All originals live below `/tmp/pluto-restart-delta.fZs3LQ/`. Their copies were
compared byte-for-byte. The supplemental interpretation does not overwrite or
erase the initial failing audit record.
