# Every intermediate next-token readout

Native checkpoint `step_13030`; literal prompt `to be or not to be,`.
Each cell is an argmax after applying the **same final LayerNorm and tied head**
to that stage's residual vector. These are separate diagnostics, not generated
tokens or feedback into the forward pass. Blocks are zero-based; quotes expose
leading spaces and escaped newlines. Probabilities/ranks are in the full JSON.

| Stage | 0: `to` | 1: ` be` | 2: ` or` | 3: ` not` | 4: ` to` | 5: ` be` | 6: `,` |
|---|---|---|---|---|---|---|---|
| Embedding + position | `'to'` | `' be'` | `' or'` | `' not'` | `' to'` | `' be'` | `','` |
| B0 attention | `'to'` | `' be'` | `' or'` | `' not'` | `' to'` | `' be'` | `','` |
| B0 MLP | `'\n\n'` | `' call'` | `' I'` | `' be'` | `' be'` | `' call'` | `' and'` |
| B1 attention | `'\n\n'` | `' call'` | `' I'` | `' be'` | `' be'` | `' call'` | `' bump'` |
| B1 MLP | `' take'` | `' call'` | `' I'` | `' be'` | `' be'` | `' call'` | `' and'` |
| B2 attention | `' take'` | `' call'` | `' I'` | `' be'` | `' be'` | `' call'` | `'\n'` |
| B2 MLP | `' be'` | `' call'` | `' I'` | `' be'` | `' be'` | `' call'` | `'\n'` |
| B3 attention | `' be'` | `' call'` | `' bump'` | `' be'` | `' be'` | `' call'` | `'\n'` |
| B3 MLP | `' be'` | `' call'` | `' bump'` | `' be'` | `' be'` | `' call'` | `'\n'` |
| B4 attention | `' be'` | `' call'` | `' I'` | `' for'` | `' be'` | `' but'` | `'\n'` |
| B4 MLP | `' be'` | `' call'` | `' to'` | `' bump'` | `' be'` | `' call'` | `'\n'` |
| B5 attention | `' be'` | `' call'` | `' to'` | `','` | `' be'` | `'\n'` | `'\n'` |
| B5 MLP | `' If'` | `' call'` | `' I'` | `'Rex'` | `' be'` | `' Lau'` | `' but'` |
| B6 attention | `'Enter'` | `' call'` | `' I'` | `' bump'` | `' be'` | `' your'` | `' but'` |
| B6 MLP | `' Have'` | `' call'` | `' I'` | `' in'` | `' be'` | `' call'` | `' but'` |
| B7 attention | `'\n\n'` | `' call'` | `' I'` | `' in'` | `' be'` | `'\n'` | `' but'` |
| B7 MLP / actual output | `'\n'` | `' the'` | `' no'` | `' in'` | `' be'` | `' the'` | `'\n'` |

See the [short report](PHRASE_FORWARD_TRACE.md) for causal checks and limitations.
