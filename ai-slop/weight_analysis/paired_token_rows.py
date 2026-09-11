"""Reproduce token-row delta rankings from preserved paired-weight reports.

All original/replacement weights are rescanned on the CPU with the same FP64
chunk reduction used by paired_weight_diff. Their hashes, tensor metrics, and
whole-model metrics must match the supplied reports. Only embedding row
energies are retained. Ranking includes every logical token, excludes padding,
and resolves ties by the lower token ID. Energy denominators include physical
padding and count the tied embedding/output head once, like the source reports.

Example: python -m weight_analysis.paired_token_rows --report DIFF.json
  --token-id 45 --token-id 68 --tokenizer-json tokenizer.json --output NEW.json

This performs no model inference and writes only a new exclusive JSON report.
The API accepts GPT2Config for small tests; the CLI uses the current recipe.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest
from .paired_weight_diff import DEFAULT_CHUNK_ELEMENTS, _compare_tensor, _metrics


_REL_TOL = 1e-10
_ABS_TOL = 1e-12


def _file_record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def _read_json(path):
    """Hash the exact bytes parsed, then check for ordinary concurrent writes."""
    path = Path(path).resolve(strict=True)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size,
                             info.st_mtime_ns, info.st_ctime_ns)
    if identity(before) != identity(after) or len(raw) != after.st_size:
        raise ValueError(f'JSON source changed while reading: {path}')
    record = {'path': str(path), 'bytes': len(raw),
              'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw), record


def _snapshot(directory, specs):
    """Reject links/layout changes and record ordinary write/replacement identity."""
    directory = Path(directory)
    expected = {spec.filename for spec in specs}
    actual = {path.name for path in directory.iterdir()
              if path.name.startswith('weight_') and path.name.endswith('.bin')}
    if actual != expected:
        raise ValueError(f'checkpoint weight file set mismatch: {directory}')
    files = {}
    for spec in specs:
        info = (directory / spec.filename).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec.nbytes:
            raise ValueError(f'checkpoint weight shape/type mismatch: {directory / spec.filename}')
        files[spec.filename] = [info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns]
    info = directory.stat()
    return {'directory_identity': [info.st_dev, info.st_ino], 'files': files}


def _number(value, description):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f'{description} must be finite numeric data')
    return value


def _close(actual, expected, description):
    if expected is None or actual is None:
        if actual is not None or expected is not None:
            raise ValueError(f'inconsistent {description}')
        return
    _number(expected, description)
    if not math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
        raise ValueError(f'inconsistent {description}: report={expected}, reread={actual}')


def _check_metrics(reported, sums, description):
    for key, actual in _metrics(sums, False).items():
        if key not in reported:
            raise ValueError(f'missing {description} {key}')
        if key.endswith('_l2') and reported[key] is not None:
            if _number(reported[key], f'{description} {key}') < 0:
                raise ValueError(f'negative {description} {key}')
        _close(actual, reported[key], f'{description} {key}')


def _validate_report(report, config, specs):
    if (report.get('schema_version') != 1
            or report.get('delta_definition') != 'replacement minus original'):
        raise ValueError('expected a paired_weight_diff schema-version-1 report')
    if (not isinstance(report.get('config'), dict)
            or GPT2Config(**report['config']) != config):
        raise ValueError('report config does not match the explicitly selected config')
    if (report.get('tensor_count') != len(specs)
            or report.get('parameter_count') != sum(spec.nbytes // 4 for spec in specs)):
        raise ValueError('report tensor/parameter counts do not match config')
    tensors = report.get('tensors')
    if not isinstance(tensors, list) or len(tensors) != len(specs):
        raise ValueError('report must describe every unique tensor')
    by_name = {}
    for item in tensors:
        name = item['name']
        if name in by_name:
            raise ValueError('duplicate report tensor')
        by_name[name] = item
    if set(by_name) != {spec.name for spec in specs}:
        raise ValueError('report tensor names do not match config')
    for spec in specs:
        item = by_name[spec.name]
        if (item.get('index') != spec.index or item.get('filename') != spec.filename
                or item.get('shape') != list(spec.shape) or item.get('dtype') != spec.dtype
                or item.get('nbytes') != spec.nbytes):
            raise ValueError(f'report tensor shape/layout mismatch: {spec.name}')
    expected_files = {spec.filename for spec in specs}
    for arm in ('original', 'replacement'):
        hashes = report[arm]['weight_sha256']
        if (not isinstance(hashes, dict) or set(hashes) != expected_files
                or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                       for value in hashes.values())):
            raise ValueError(f'incomplete or invalid {arm} report hashes')
    return by_name


def _token_vocabulary(document, config):
    """Labels only: invert GPT-2's byte alphabet without retokenizing text."""
    vocabulary = document.get('model', {}).get('vocab')
    if not isinstance(vocabulary, dict) or len(vocabulary) != config.vocab_size:
        raise ValueError('tokenizer vocabulary does not cover the logical vocabulary')
    inverse = {}
    for raw, token_id in vocabulary.items():
        if (not isinstance(raw, str) or type(token_id) is not int
                or not 0 <= token_id < config.vocab_size or token_id in inverse):
            raise ValueError('invalid or duplicate tokenizer vocabulary ID')
        inverse[token_id] = raw
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {chr(byte): byte for byte in visible}
    for extra, byte in enumerate(byte for byte in range(256) if byte not in visible):
        alphabet[chr(256 + extra)] = byte
    if any(character not in alphabet for raw in inverse.values() for character in raw):
        raise ValueError('tokenizer vocabulary is not a GPT-2 byte alphabet')
    return inverse, alphabet


def _row(token_id, energies, ranks, tokenizer):
    energy = float(energies[token_id])
    result = {'token_id': int(token_id), 'rank': int(ranks[token_id]),
              'delta_l2': math.sqrt(energy), 'delta_energy': energy}
    if tokenizer is not None:
        vocabulary, alphabet = tokenizer
        raw = vocabulary[int(token_id)]
        token_bytes = bytes(alphabet[character] for character in raw)
        result.update(raw_token=raw, decoded=token_bytes.decode('utf-8', errors='replace'),
                      bytes_hex=token_bytes.hex())
    return result


def _share(numerator, denominator):
    return numerator / denominator if denominator else None


def extract(report_paths, token_ids, *, tokenizer_json=None, config=GPT2Config(),
            chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    """Verify supplied reports against current source bytes and rank token rows.

    Selected IDs must be nonempty, unique logical token IDs. Reports may have
    unequal or unknown steps; those cases are explicitly labeled. Initial-
    relative metrics are retained from the hashed report, not independently
    recomputed. Source snapshots detect ordinary mutation, not adversarial races.
    """
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError('chunk_elements must be a positive integer')
    if not isinstance(config, GPT2Config):
        raise ValueError('config must be an explicit GPT2Config')
    selected = list(token_ids)
    if (not selected or any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, np.integer))
                           or not 0 <= value < config.vocab_size for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('selected IDs must be unique nonempty logical token IDs')
    selected = sorted(int(value) for value in selected)
    if isinstance(report_paths, (str, Path)):
        report_paths = [report_paths]
    paths = [Path(path).resolve(strict=True) for path in report_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('provide one or more distinct source reports')
    documents, input_records = [], []
    for path in paths:
        document, record = _read_json(path)
        documents.append(document)
        input_records.append(record)
    tokenizer_record, tokenizer = None, None
    if tokenizer_json is not None:
        document, tokenizer_record = _read_json(tokenizer_json)
        tokenizer = _token_vocabulary(document, config)
    specs = tensor_manifest(config)
    records, snapshots = [], []
    for report, report_record in zip(documents, input_records):
        by_name = _validate_report(report, config, specs)
        checkpoints, sources, steps = [], {}, []
        for arm in ('original', 'replacement'):
            directory = Path(report[arm]['directory']).resolve(strict=True)
            before = _snapshot(directory, specs)
            model = GPT2Checkpoint(directory, config)
            match = re.fullmatch(r'step_([0-9]+)', directory.name)
            step = int(match[1]) if match else None
            if report[arm].get('step_from_directory_name') != step:
                raise ValueError(f'{arm} reported step disagrees with directory name')
            steps.append(step)
            checkpoints.append(model)
            snapshots.append((directory, before))
            sources[arm] = {'directory': str(directory), 'weight_sha256': {},
                            'stat_before_and_after': before}
        totals = {key: [] for key in ('a2', 'b2', 'delta2', 'ab')}
        embedding_sums, row_energy = None, None
        for spec in specs:
            sums, energies, hashes = _compare_tensor(
                [model[spec.name] for model in checkpoints], spec, chunk_elements)
            for arm, digest in zip(('original', 'replacement'), hashes):
                if digest != report[arm]['weight_sha256'][spec.filename]:
                    raise ValueError(f'{arm} source hash mismatch: {spec.filename}')
                sources[arm]['weight_sha256'][spec.filename] = digest
            _check_metrics(by_name[spec.name], sums, spec.name)
            for key in totals:
                totals[key].append(sums[key])
            if spec.name == 'token_embedding.weight':
                embedding_sums, row_energy = sums, energies
        model_sums = {key: math.fsum(values) for key, values in totals.items()}
        _check_metrics(report['model'], model_sums, 'model')
        if row_energy is None or not np.isfinite(row_energy).all() or np.any(row_energy < 0):
            raise ValueError('invalid embedding row energy')
        _close(math.fsum(float(value) for value in row_energy), embedding_sums['delta2'],
               'embedding row energy sum')
        logical = row_energy[:config.vocab_size]
        order = np.lexsort((np.arange(config.vocab_size), -logical))
        ranks = np.empty(config.vocab_size, dtype=np.int64)
        ranks[order] = np.arange(1, config.vocab_size + 1)
        selected_energy = math.fsum(float(logical[token_id]) for token_id in selected)
        embedding_energy, model_energy = embedding_sums['delta2'], model_sums['delta2']
        matched = steps[0] is not None and steps[0] == steps[1]
        initial = report.get('initial')
        records.append({
            'source_report': report_record, 'sources': sources,
            'initial_checkpoint_directory': initial['directory'] if initial else None,
            'step': steps[0] if matched else None, 'original_step': steps[0],
            'replacement_step': steps[1], 'steps_match': matched,
            'model': report['model'], 'embedding_delta_l2': math.sqrt(embedding_energy),
            'embedding_delta_energy': embedding_energy, 'model_delta_energy': model_energy,
            'logical_embedding_delta_energy': math.fsum(float(value) for value in logical),
            'selected_rows_delta_energy': selected_energy,
            'embedding_share_of_model_delta_energy': _share(embedding_energy, model_energy),
            'selected_rows_share_of_embedding_delta_energy': _share(selected_energy, embedding_energy),
            'selected_rows_share_of_model_delta_energy': _share(selected_energy, model_energy),
            'selected_rows': [_row(token_id, logical, ranks, tokenizer) for token_id in selected],
            'top_twenty_rows': [_row(token_id, logical, ranks, tokenizer) for token_id in order[:20]],
            'validation': {'all_source_weights_finite': True, 'all_source_weight_hashes_match': True,
                           'tensor_and_model_metrics_match': True,
                           'source_snapshots_unchanged': True},
        })
    # Recheck all sources after all reports, not only after each individual scan.
    for directory, before in snapshots:
        GPT2Checkpoint(directory, config)
        if _snapshot(directory, specs) != before:
            raise ValueError(f'checkpoint changed during token-row analysis: {directory}')
    for record in [*input_records, *([] if tokenizer_record is None else [tokenizer_record])]:
        if _file_record(record['path']) != record:
            raise ValueError(f'input report/tokenizer changed during analysis: {record["path"]}')
    return {
        'schema_version': 1, 'kind': 'paired-token-row-deltas', 'config': asdict(config),
        'selected_token_ids': selected, 'tokenizer': tokenizer_record, 'records': records,
        'energy_definition': 'Sum of squared FP64 differences between corresponding FP32 weights.',
        'rank_scope': 'All logical vocabulary rows; physical padding excluded from ranking.',
        'rank_order': 'One-based descending delta L2; lower token ID breaks exact ties.',
        'denominator_scope': 'Unique physical parameters including padding; tied embedding/head counted once.',
        'zero_denominator_policy': 'Undefined energy shares are JSON null.',
        'metric_tolerance': {'relative': _REL_TOL, 'absolute': _ABS_TOL},
        'implementation': {'chunk_elements': chunk_elements,
                           'sources': [_file_record(Path(__file__).with_name(name)) for name in
                                       ('paired_token_rows.py', 'paired_weight_diff.py', 'checkpoint.py')]},
        'limitations': ['Weight-delta ranks do not establish word storage or causal importance.',
                        'Initial-relative metrics are retained from source reports, not independently rescanned.',
                        'Source hashes and stat checks detect ordinary mutation, not adversarial concurrent writes.',
                        'Readable token labels do not retokenize or establish tokenizer/checkpoint provenance.'],
    }


def write_report(output, result):
    """Publish only outside source checkpoints, never replacing an existing file."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve()
    for record in result['records']:
        directories = [source['directory'] for source in record['sources'].values()]
        if record['initial_checkpoint_directory'] is not None:
            directories.append(record['initial_checkpoint_directory'])
        for directory in directories:
            source = Path(directory).resolve()
            if resolved == source or source in resolved.parents:
                raise ValueError('token-row report must be outside source checkpoint directories')
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def analyze(report_paths, token_ids, output, *, tokenizer_json=None, config=GPT2Config(),
            chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    result = extract(report_paths, token_ids, tokenizer_json=tokenizer_json,
                     config=config, chunk_elements=chunk_elements)
    write_report(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--token-id', type=int, action='append', required=True)
    parser.add_argument('--tokenizer-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.report, args.token_id, args.output, tokenizer_json=args.tokenizer_json)


if __name__ == '__main__':
    main()
