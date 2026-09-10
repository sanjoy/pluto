"""Score the amended word cases without conflating capitalization variants.

Each word event is the probability of its three recorded native token IDs,
conditioned on the same prefix. It does not sum alternate tokenizations or
require a delimiter after the word. Candidate log ratios are paired by corpus
role WITHIN each spelling variant; lowercase cases never become missing or
extra candidates of a title-case context. The first-token and remaining-spelling
contributions are retained, since a low whole-word probability can arise from
either selecting the word or completing its spelling.
"""

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from . import paired_lowercase_cases as cases
from . import paired_lowercase_training as training
from . import paired_supplemental_cases as native

FORMAT = 'pluto-paired-lowercase-word-score-summary-v1'
ARMS = ('original', 'replacement')


def load_cases(path):
    """Validate frozen bytes, role/variant labels, and complete paired contexts."""
    plan = training.read_json(path)
    if plan.get('format') != cases.FORMAT or plan.get('complete') is not True:
        raise ValueError('expected complete amended word cases')
    records = [training.record(path), plan['packed_batch'],
               *plan['provenance'], *plan['sources']]
    training.verify_records(records)
    packed, _ = native._packed(plan)
    contexts = defaultdict(dict)
    for index, case in enumerate(plan['cases']):
        prefix = native._validate_case(case, index, packed, (3,))
        if case['split'] not in ('training', 'test'):
            raise ValueError('invalid corpus split')
        if case['kind'] == 'control':
            if (case.get('spelling_variant') is not None or case.get('candidate_pair') is not None
                    or case['prefix_domain'] != 'shared'):
                raise ValueError('control must not claim a word spelling pair')
            continue
        if case['kind'] != 'word':
            raise ValueError('unsupported case kind')
        variant = case.get('spelling_variant')
        pair = cases.PAIRS.get(variant)
        domain, role = case['prefix_domain'], case['target_source_domain']
        if (pair is None or case.get('candidate_pair') != pair or domain not in ARMS
                or role not in ARMS or case['target'] != pair[role]):
            raise ValueError('case spelling, candidate role, or domain disagrees')
        raw = bytes.fromhex(case['target_source']['bytes_hex'])
        if raw not in (pair[role].encode(), b' ' + pair[role].encode()):
            raise ValueError('native target bytes disagree with exact spelling')
        key = (case['split'], variant, case['context_id'])
        crossing = (domain, role)
        if crossing in contexts[key]:
            raise ValueError('duplicate candidate crossing')
        contexts[key][crossing] = prefix
    if not contexts:
        raise ValueError('no paired word contexts')
    for entries in contexts.values():
        if set(entries) != {(domain, role) for domain in ARMS for role in ARMS}:
            raise ValueError('missing paired word candidate or prefix domain')
        for domain in ARMS:
            if not np.array_equal(entries[domain, 'original'], entries[domain, 'replacement']):
                raise ValueError('paired candidates must use identical prefixes')
    training.verify_records(records)
    return plan, records


def candidate_ratios(items):
    """Positive values favor the original spelling, for this EXACT context."""
    groups = defaultdict(dict)
    for item in items:
        if item['kind'] != 'word':
            continue
        key = (item['split'], item['spelling_variant'], item['context_id'], item['prefix_domain'])
        role = item['target_source_domain']
        if role in groups[key]:
            raise ValueError('duplicate candidate score')
        groups[key][role] = item
    result = []
    for (split, variant, context, domain), pair in sorted(groups.items()):
        if set(pair) != set(ARMS):
            raise ValueError('missing role-paired score')
        a, b = (pair[role] for role in ARMS)
        pieces = [y-x for x, y in zip(a['token_nll'], b['token_nll'])]
        result.append(dict(split=split, spelling_variant=variant, context_id=context,
            prefix_domain=domain, candidate_pair=cases.PAIRS[variant],
            case_indices={role: pair[role]['case_index'] for role in ARMS},
            log_probability_ratio_original_over_replacement=math.fsum(pieces),
            first_piece_log_ratio=pieces[0], conditional_spelling_log_ratio=math.fsum(pieces[1:]),
            per_piece_log_ratios=pieces))
    return result


def summarize(cases_path, scores_directory, output):
    """Consume native FP32 NLL/argmax dumps; all reductions below use FP64."""
    output, scores_directory = Path(output), Path(scores_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(parent == output.resolve() or parent in output.resolve().parents
           for parent in (Path(cases_path).resolve().parent, scores_directory.resolve())):
        raise ValueError('summary output must be outside input directories')
    plan, records = load_cases(cases_path)
    count, length = plan['case_count'], plan['context_length']
    files = {name: training.record(scores_directory / filename) for name, filename in
             (('metadata', 'metadata.json'), ('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin'))}
    metadata = training.read_json(files['metadata']['path'])
    expected = dict(kind='paired_loss_probe', complete=True, temperature=1,
                    byte_order='little', loss_dtype='<f4', argmax_dtype='<i4',
                    loss_file='losses.f32.bin', argmax_file='argmax.i32.bin',
                    case_count=count, passage_count=count, context_length=length,
                    output_shape=[count, length], batch_bytes=plan['packed_batch']['bytes'])
    if (any(metadata.get(key) != value for key, value in expected.items())
            or metadata.get('complete') is not True
            or Path(metadata['batch_file']).resolve() != Path(plan['packed_batch']['path'])
            or ('batch_sha256' in metadata and metadata['batch_sha256'] != plan['packed_batch']['sha256'])):
        raise ValueError('native score metadata disagrees with frozen cases')
    if any(files[key]['bytes'] != count*length*4 for key in ('losses', 'argmax')):
        raise ValueError('native score file size mismatch')
    losses = np.fromfile(files['losses']['path'], dtype='<f4').reshape(count, length)
    argmax = np.fromfile(files['argmax']['path'], dtype='<i4').reshape(count, length)
    if (not np.isfinite(losses).all() or np.any(losses < 0)
            or np.any(argmax < 0) or np.any(argmax >= plan['vocab_size'])):
        raise ValueError('invalid loss or argmax values')
    items, groups = [], defaultdict(list)
    for case in plan['cases']:
        index, rows = case['case_index'], case['scored_rows']
        token_nll = [float(losses[index, row]) for row in rows]
        nll, suffix = math.fsum(token_nll), math.fsum(token_nll[1:])
        predicted = argmax[index, rows].tolist()
        matches = [a == b for a, b in zip(predicted, case['target_ids'])]
        item = dict(case, token_nll=token_nll, sequence_nll=nll,
            sequence_log_probability=-nll, sequence_probability=math.exp(-nll),
            first_piece_nll=token_nll[0], conditional_spelling_nll=suffix,
            conditional_spelling_probability=math.exp(-suffix),
            token_probabilities=[math.exp(-v) for v in token_nll],
            teacher_forced_argmax_ids=predicted, argmax_matches=matches,
            all_three_argmax_match=all(matches))
        items.append(item)
        groups[(case['kind'], case['split'], case['spelling_variant'] or '',
                case['prefix_domain'], case['target'])].append(item)
    summaries = []
    for (kind, split, variant, domain, target), group in sorted(groups.items()):
        mean = lambda key: math.fsum(item[key] for item in group)/len(group)
        summaries.append(dict(kind=kind, split=split, spelling_variant=variant or None,
            prefix_domain=domain, target=target, case_count=len(group),
            mean_sequence_nll=mean('sequence_nll'), mean_first_piece_nll=mean('first_piece_nll'),
            mean_conditional_spelling_nll=mean('conditional_spelling_nll'),
            geometric_mean_sequence_probability=math.exp(-mean('sequence_nll')),
            arithmetic_mean_sequence_probability=mean('sequence_probability'),
            all_three_argmax_match_fraction=mean('all_three_argmax_match')))
    training.verify_records([*records, *files.values()])
    result = dict(format=FORMAT, complete=True, cases=records[0], scores=files,
        probe_metadata=metadata, case_count=count, context_length=length,
        groups=summaries, per_case=items, candidate_ratios=candidate_ratios(items),
        provenance=[*records, *files.values()],
        probability_definition='Temperature-1 teacher-forced probability of the three native IDs; '
            'not all tokenizations and not a delimiter-terminated event.',
        aggregation='Equal weights within each split/variant/domain/target; no title/lowercase pooling.',
        limitations=plan.get('limitations', []))
    training.publish(output, result)
    return result


def compare(original, replacement):
    """Compare the same frozen contexts across models, with explicit sign units."""
    if (original['cases'] != replacement['cases']
            or original['case_count'] != replacement['case_count']
            or any(len(report['per_case']) != report['case_count'] for report in (original,replacement))):
        raise ValueError('model comparisons require identical frozen cases')
    case_deltas = []
    for a, b in zip(original['per_case'], replacement['per_case']):
        keys = ('case_index', 'context_id', 'split', 'kind', 'target', 'target_ids',
                'prefix', 'prefix_domain', 'spelling_variant', 'candidate_pair')
        if any(a.get(key) != b.get(key) for key in keys):
            raise ValueError('model case identities differ')
        case_deltas.append({**{key: a[key] for key in keys},
            'nll_replacement_model_minus_original_model': b['sequence_nll']-a['sequence_nll'],
            'first_piece_nll_delta': b['first_piece_nll']-a['first_piece_nll'],
            'conditional_spelling_nll_delta': b['conditional_spelling_nll']-a['conditional_spelling_nll']})
    ratios = []
    for a, b in zip(original['candidate_ratios'], replacement['candidate_ratios']):
        keys = ('split', 'spelling_variant', 'context_id', 'prefix_domain', 'candidate_pair', 'case_indices')
        if any(a[key] != b[key] for key in keys):
            raise ValueError('model candidate contexts differ')
        ratios.append({**{key: a[key] for key in keys},
            'original_model_log_ratio': a['log_probability_ratio_original_over_replacement'],
            'replacement_model_log_ratio': b['log_probability_ratio_original_over_replacement'],
            'original_minus_replacement_model_log_ratio':
                a['log_probability_ratio_original_over_replacement']-b['log_probability_ratio_original_over_replacement']})
    if len(case_deltas) != original['case_count'] or len(original['candidate_ratios']) != len(replacement['candidate_ratios']):
        raise ValueError('incomplete model comparison')
    return dict(case_deltas=case_deltas, candidate_ratio_deltas=ratios,
        interpretation='Positive NLL delta means the original-trained model assigns more probability '
            'to this candidate. Positive candidate-ratio delta means it favors the original spelling '
            'more than the replacement-trained model does. These are behavioral associations, not causal localization.')
