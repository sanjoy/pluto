# Exploratory four-sequence replay hypothesis

This is a NEW hypothesis prompted by the already-observed corpus-assisted
localization results, not a revision of the frozen earlier experiments. It
must not be called independent validation of historical batch settings.

The shared ordinary-after adjusted bag's top-ten retrieved windows align most
closely with replay draws 82, 94, 90, 93, 92, 95, 86, 100, 89, and 98 within
the previously examined first hundred draws. Its weight score centers the
interval N+20→N+30 after a candidate restart. Four sequences per step would
put that interval at draws 81–120. This motivates testing exactly batch size
four, rather than searching many batch sizes/seeds/phases for a favorable fit.

Before new overlap counts, freeze the following tests using only existing
candidate bags. Keep all real and fixed shuffled-ID variants. Do not rerank
IDs or read new checkpoint weights.

| Bag source | Target zero-based draws | Previous | Next | Reset |
| --- | --- | --- | --- | --- |
| Restart shared ordinary-after, raw and adjusted | [80,120) | [40,80) | [120,160) | [0,40) |
| Later shared follow-through, raw and adjusted | [400,440) | [360,400) | [440,480) | [0,40) |

For each prescribed phase, compare overlap of each frozen 128-ID bag with the
union of the sampled 1,025-token input/target windows. Retain every overlap
for seed 17 and seeds 10000–10999, plus per-ID membership for seed 17. Sampler,
native corpus, current prefix split, and context remain exactly the existing
conditional replay definition. Regenerate 480 starts per seed with the scalar
MT19937-64/libstdc++ mapping, or authenticate an existing export and check it
against that independent mapping.

Keep the earlier batch-size-one and batch-size-ten failures intact. A favorable
batch-four result would be evidence for a previously untested phase alignment,
not authenticated historical flags, proof of raw-gradient inversion, or
ordered text extraction. A negative result also stays in the report; do not
expand the batch/phase search in this experiment after seeing the result.
