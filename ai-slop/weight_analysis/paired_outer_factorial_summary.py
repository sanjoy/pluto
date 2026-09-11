"""CPU-only, alias-aware summaries of completed outer-factorial suites.

This is a reporting tool, not another native experiment. A suite may finish
before its enclosing runner: successful recorded native executions and the
exact saved outer construction are required, not just an output filename.
The saved inner reader's likelihood calculations are authenticated here, not
rerun. No model, checkpoint, source evidence, or running controller is changed.
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path

from . import paired_outer_factorial as outer

training = outer.training
reader = outer.reader
CELLS = outer.CELL_NAMES
FORMAT = 'pluto-paired-outer-factorial-summary-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _base(row):
    prefix = row['case']['prediction_prefixes'][0]
    return prefix['causal_prefix_sha256'], prefix['causal_prefix_length']


def _scores(row, positions):
    return {cell: math.fsum(row['cells'][cell]['token_log_probability'][p]
                            for p in positions) for cell in CELLS}


def _metric(rows, positions, target_length):
    """Deduplicate the causal event, including the supplied word history.

    In particular two different fourth targets cannot double-weight the same
    first token or word triple. Suffixes use the first prefix PLUS the entire
    triple, because the given first token changes the suffix's input history.
    """
    unique = {}
    for row in rows:
        key = (_base(row), tuple(row['case']['target_ids'][:target_length]))
        values = _scores(row, positions)
        _require(key not in unique or unique[key] == values,
                 'conflicting duplicate causal event scores')
        unique[key] = values
    means = {cell: math.fsum(v[cell] for v in unique.values()) / len(unique)
             for cell in CELLS}
    return dict(event_count=len(unique), cells={cell: dict(
        mean_log_probability=value, geometric_probability=math.exp(value))
        for cell, value in means.items()}, effects=outer.outer_effect(means))


def _group(rows, identity, word):
    length = len(rows[0]['case']['target_ids'])
    _require(all(len(row['case']['target_ids']) == length for row in rows),
             'different sequence lengths in one group')
    metrics = dict(first=_metric(rows, [0], 1),
                   suffix=_metric(rows, range(1, 3 if word else length),
                                  3 if word else length),
                   sequence=_metric(rows, range(length), length))
    if word:
        metrics['word_three'] = _metric(rows, range(3), 3)
        if length == 4:
            metrics['next_fourth'] = _metric(rows, [3], 4)
    return dict(identity, raw_case_count=len(rows),
                event_count=metrics['sequence']['event_count'], metrics=metrics)


def _preferences(rows, identity):
    contexts = defaultdict(dict)
    pairs = {tuple(sorted(row['case']['candidate_pair'].items())) for row in rows}
    _require(len(pairs) == 1, 'candidate labels differ within preference stratum')
    pair = dict(next(iter(pairs)))
    _require(set(pair) == {'original', 'replacement'}
             and pair['original'] != pair['replacement'], 'ambiguous candidate pair')
    for row in rows:
        case = row['case']
        roles = [role for role, target in pair.items() if target == case['target']]
        _require(len(roles) == 1, 'word target not in candidate pair')
        role = roles[0]
        value = (tuple(case['target_ids'][:3]), _scores(row, range(3)))
        context = contexts[_base(row)]
        _require(role not in context or context[role] == value,
                 'ambiguous or conflicting candidate at same causal prefix')
        context[role] = value
    matched = [context for context in contexts.values() if len(context) == 2]
    odds = [{cell: context['replacement'][1][cell] - context['original'][1][cell]
             for cell in CELLS} for context in matched]
    counts, flips = {}, {}
    for cell in CELLS:
        counts[cell] = dict(original_preferred=sum(o[cell] < 0 for o in odds),
                            replacement_preferred=sum(o[cell] > 0 for o in odds),
                            ties=sum(o[cell] == 0 for o in odds))
        flips[cell] = dict(
            original_to_replacement=sum(o['A'] < 0 < o[cell] for o in odds),
            replacement_to_original=sum(o['A'] > 0 > o[cell] for o in odds),
            tie_to_decisive=sum(o['A'] == 0 and o[cell] != 0 for o in odds),
            decisive_to_tie=sum(o['A'] != 0 and o[cell] == 0 for o in odds))
    return dict(identity, candidate_pair=pair, context_count=len(matched),
        unpaired_context_count=len(contexts)-len(matched),
        mean_log_odds={cell: math.fsum(o[cell] for o in odds)/len(odds)
                       if odds else None for cell in CELLS},
        counts=counts, flips_from_A=flips)


def summarize_cases(readout, prefix_domain='original'):
    """Aggregate a supplied readout; this pure function makes no provenance claim.

    The default is the original corpus's prompt domain in BOTH swap directions.
    Neither candidate spellings, splits, spacing variants, nor prefix domains
    are pooled. A context can occur in several strata; counts are not a claim
    of globally unique prompts, independent samples, or statistical power.
    """
    _require(prefix_domain in ('original', 'replacement'), 'choose one prefix domain')
    words, controls, preferences = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in readout['per_case']:
        case = row['case']
        kind = case['kind']
        _require(kind in ('word', 'word_next_native', 'control', 'shared_piece'),
                 'unknown case kind')
        word = kind in ('word', 'word_next_native')
        if case['prefix_domain'] != (prefix_domain if word else 'shared'):
            continue
        ids = case['target_ids']
        _require(len(ids) == (4 if kind == 'word_next_native' else 3)
                 and all(type(i) is int and i >= 0 for i in ids), 'invalid target IDs')
        prefixes = case['prediction_prefixes']
        _require(len(prefixes) == len(ids) and all(
            type(p['causal_prefix_length']) is int and p['causal_prefix_length'] > 0
            and p['prediction_row'] == p['causal_prefix_length']-1
            and isinstance(p['causal_prefix_sha256'], str)
            and len(p['causal_prefix_sha256']) == 64 for p in prefixes),
            'invalid causal prefix identity')
        _require(set(row['cells']) == set(CELLS), 'missing or extra outer cells')
        for cell in CELLS:
            values = row['cells'][cell]['token_log_probability']
            _require(len(values) == len(ids) and all(type(v) in (int, float)
                and math.isfinite(v) and v <= 0 for v in values),
                'invalid token log probabilities')
        if word:
            key = (kind, case['split'], case['spelling_variant'],
                   case['word_has_leading_space'])
            words[(*key, case['target'])].append(row)
            preferences[key].append(row)
        else:
            controls[(kind, case['split'], case.get('piece_id'))].append(row)
    names = ('kind', 'split', 'spelling_variant', 'word_has_leading_space')
    return dict(prefix_domain=prefix_domain,
        word_groups=[_group(rows, dict(zip((*names, 'target'), key)), True)
                     for key, rows in sorted(words.items())],
        preference_groups=[_preferences(rows, dict(zip(names, key)))
                           for key, rows in sorted(preferences.items())],
        control_groups=[_group(rows, dict(zip(('kind', 'split', 'piece_id'), key)), False)
                        for key, rows in sorted(controls.items())],
        definitions=dict(
            probability='exp(mean(log P)); geometric probability, not arithmetic success rate',
            suffix='P(pieces 2,3 | actual prefix and candidate first piece)',
            next_fourth='One exact supplied token after the three-piece word, not any boundary',
            deduplication='Within each stratum: first causal prefix SHA256+length and scored target IDs; projected word metrics ignore fourth-target aliases',
            preferences='log P(replacement three-piece word) minus log P(original three-piece word) on matching first causal prefixes; suffix normalizers do not cancel',
            interaction='EC-C-E+A in log probability includes softmax-normalizer curvature; not automatically representational interaction',
            controls='Shared-prefix ordinary controls grouped by split; shared-piece controls also separated by piece ID',
            coverage='Counts are per stratum and metric, not globally unique or independent contexts'))


def _inner_records(path, expected_record, binary):
    """Check the unchanged native reader's execution ledger, without rescoring."""
    _require(training.record(path) == expected_record, 'inner readout hash differs')
    report = reader._json(path)
    _require(report.get('format') == 'pluto-embedding-factorial-readout-v1'
             and report['execution_provenance'].get('verified') is True,
             'inner native readout lacks verified execution')
    records = {r['path']: r for r in outer._unique(report['files'])}
    metadata_record = report['native_metadata']
    _require(records.get(metadata_record['path']) == metadata_record,
             'native metadata is not in reader ledger')
    metadata = reader._json(metadata_record['path'])
    _require(metadata.get('complete') is True
             and metadata.get('format') == 'pluto-embedding-factorial-v1'
             and metadata['binary'] == binary['path'] and metadata['temperature'] == 1
             and all(metadata['checks'].get(key) is True for key in reader.CHECKS),
             'native completion, binary, or checks differ')
    directory = Path(metadata_record['path']).parent
    outputs = [metadata_record]
    for cell in reader.CELLS:
        logit_path = str(directory / metadata['cells'][cell]['logits_file'])
        _require(logit_path in records, 'unbound native logits')
        outputs.append(records[logit_path])
    provenance = reader._execution_record(report['execution_provenance']['record']['path'],
        metadata, directory, report['cases'], report['patch'], records, outputs)
    _require(provenance == report['execution_provenance'], 'saved execution provenance differs')
    execution = reader._json(provenance['record']['path'])
    records[execution['log']['path']] = execution['log']
    records[expected_record['path']] = expected_record
    return report, list(records.values())


def analyze(readout_path, output, prefix_domain='original'):
    """Publish a NEW, per-suite summary; never claim the entire runner is done."""
    readout_path, output = Path(readout_path).absolute(), Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), 'summary output must be new')
    run = readout_path.parent.parent
    _require(run not in output.parents, 'write summaries outside the live runner directory')
    input_record = training.record(readout_path)
    saved = reader._json(readout_path)
    request_path = run / 'request.json'
    request_record = training.record(request_path)
    request = reader._json(request_path)
    suite, direction = saved['suite'], saved['direction']
    _require(request.get('format') == saved.get('format') == outer.FORMAT
             and request.get('fixed_step') == saved.get('step') == outer.STEP
             and suite in outer.SUITES and suite in request['suites']
             and readout_path == run/direction/(suite+'_outer_readout.json')
             and Path(request['amendment_root']) == run.parent,
             'unexpected outer request/suite/step/path')
    matches = [item for item in request['directions']
               if item['donor_arm']+'_to_'+item['recipient_arm'] == direction]
    _require(len(matches) == 1, 'outer direction not bound to request')
    item = matches[0]
    reciprocal = [i for i in request['directions']
                  if i['recipient_arm'] == item['donor_arm']
                  and i['donor_arm'] == item['recipient_arm']]
    _require(len(reciprocal) == 1, 'missing reciprocal donor baseline')
    frozen = {r['path']: r for r in outer._unique(request['frozen_inputs'])}
    binary = request['binaries']['factorial_probe']
    _require(frozen.get(binary['path']) == binary, 'probe binary not frozen')
    records = [input_record, request_record, training.record(__file__),
               training.record(Path(__file__).with_name('paired_outer_factorial_summary_test.py')),
               *frozen.values()]
    reports = {}
    for name, entry in (('baseline', item), ('donor', reciprocal[0])):
        path = Path(request['previous_recovery'])/entry['name']/('factorial_'+suite+'_readout.json')
        _require(str(path) in frozen, 'baseline readout not frozen by runner')
        reports[name], more = _inner_records(path, frozen[str(path)], binary)
        records.extend(more)
    for name, field, prefix in (('conditional', 'conditional_readout', 'outer_'),
                               ('control', 'donor_control_readout', 'donor_control_')):
        path = run/direction/(prefix+suite)/'readout.json'
        _require(saved[field]['path'] == str(path), 'outer native suite path differs')
        reports[name], more = _inner_records(path, saved[field], binary)
        records.extend(more)
    _require(reports['conditional']['patch']['paths']['A'] == str(run/direction/'C/step_331')
             and reports['conditional']['patch']['paths']['J'] == str(run/direction/'EC/step_331')
             and reports['conditional']['patch']['selected_rows'] == item['embedding_rows']
             and reports['control']['patch']['selected_rows'] == []
             and reports['control']['patch']['paths']['A'] == item['donor_checkpoint']['path']
             and reports['control']['patch']['paths']['J'] == str(run/direction/'donor_copy/step_331'),
             'conditional/control checkpoint selection differs')
    # Paths alone do not establish C's meaning: authenticate the stored model
    # manifest and check its complete tensor maps against the frozen endpoints.
    models_path = run/direction/'models.json'
    models_record = training.record(models_path)
    _require(models_record in records, 'model construction manifest not execution-bound')
    models = reader._json(models_path)
    expected_c = dict(item['donor_checkpoint']['sha256'])
    _require(len(expected_c) == len(item['recipient_checkpoint']['sha256']) == 100,
             'expected 100 endpoint tensors')
    expected_c['weight_0.bin'] = item['recipient_checkpoint']['sha256']['weight_0.bin']
    _require(models['C']['output']['weights_sha256'] == expected_c
             and models['donor_copy']['output']['weights_sha256'] == item['donor_checkpoint']['sha256']
             and len(models['EC']['output']['weights_sha256']) == 100
             and all(models['EC']['output']['weights_sha256'][key] == value
                     for key, value in expected_c.items() if key != 'weight_0.bin'),
             'C/EC/donor-copy tensor identities differ')
    for name, report_name, side in (('C', 'conditional', 'A'),
                                     ('EC', 'conditional', 'J'),
                                     ('donor_copy', 'control', 'J')):
        weights = reports[report_name]['patch']['weight_records'][side]
        _require({Path(r['path']).name:r['sha256'] for r in weights}
                 == models[name]['output']['weights_sha256'],
                 'model manifest differs from execution-bound checkpoint weights')
    records = outer._unique(records)
    training.verify_records(records)
    outer.assert_donor_parity(reports['control'], reports['donor'])
    fresh = outer.combine_outer(reports['baseline'], reports['conditional'], reports['donor'])
    fresh.update(format=outer.FORMAT, step=outer.STEP, direction=direction, suite=suite,
                 donor_copy_all_logits_byte_equal=True,
                 conditional_readout=saved['conditional_readout'],
                 donor_control_readout=saved['donor_control_readout'])
    _require(fresh == saved, 'saved outer construction differs from authenticated inner readouts')
    result = summarize_cases(saved, prefix_domain)
    result.update(format=FORMAT, complete=True, created_utc=training.now(),
        direction=direction, suite=suite, step=outer.STEP, input=input_record,
        completion_meaning='CPU summary of this successfully measured suite only; not whole-run completion',
        saved_outer_construction_exactly_regenerated=True,
        donor_copy_all_logits_byte_equal=True, original_reader_rescored_logits=False,
        native_executed_by_this_tool=False, whole_assay_completion_claimed=False,
        goal_completion_claimed=False, source_records=records,
        limitations=[*saved['limitations'],
            'C bundles positions, all transformer weights, and final LayerNorm; it is not a localized circuit.',
            'A/E/C/EC use recipient remaining embedding rows; D-minus-EC changes those rows only.',
            'Per-suite native evidence is authenticated, but new native/likelihood recomputation is not performed here.'])
    training.verify_records(records)
    training.publish(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-domain', choices=('original', 'replacement'), default='original')
    args = parser.parse_args(argv)
    analyze(args.readout, args.output, args.prefix_domain)


if __name__ == '__main__':
    main()
