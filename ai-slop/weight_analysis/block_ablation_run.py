"""Freeze, run and summarize a read-only 39-arm native block-ablation study.

One-off analysis artifact, not production code. Corpus windows are selected
before scoring. All comparisons use teacher-forced fixed inputs, never claim
held-out generalization, and never alter checkpoint files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bindings(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            yield {key: value[key] for key in ('path', 'bytes', 'sha256')}
        for child in value.values():
            yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)


def verify(records):
    for item in records:
        if record(item['path']) != item:
            raise ValueError('changed input: '+item['path'])


def summarize(directory, fixture, trace_root, graph_path):
    metadata = json.loads((directory/'metadata.json').read_text())
    if not metadata['complete'] or len(metadata['arms']) != 39:
        raise ValueError('incomplete native study')
    graph = json.loads(graph_path.read_text())
    pieces = {node['id']: bytes.fromhex(node['bytes_hex']).decode('utf-8', 'backslashreplace')
              for node in graph['nodes']}
    packed = np.fromfile(fixture/'batch_tokens.bin', dtype='<i4').reshape(2, 8, 1024)
    inputs, targets = packed.reshape(2, -1)
    classes = np.fromfile(fixture/'row_classes.u8', dtype='u1')
    primary = np.fromfile(fixture/'primary_mask.u8', dtype='u1').astype(bool)
    baseline_losses = np.fromfile(directory/'clean_before.losses.f32', dtype='<f4')
    baseline_ids = np.fromfile(directory/'clean_before.argmax.i32', dtype='<i4')
    baseline_trace = np.fromfile(directory/'clean_before.trace_logits.f32', dtype='<f4').reshape(4, 50272)
    for clean in ('clean_repeat', 'clean_after'):
        for suffix in ('losses.f32', 'argmax.i32', 'trace_logits.f32'):
            if (directory/(clean+'.'+suffix)).read_bytes() != (directory/('clean_before.'+suffix)).read_bytes():
                raise ValueError('native clean replay differs')
    for item in range(4):
        previous = (trace_root/f'trace_step_{1340+item}'/'logits.f32').read_bytes()
        if baseline_trace[item].tobytes() != previous:
            raise ValueError('packed clean does not reproduce original exact prefix')
    masks = {'all': np.ones(8192, bool), 'primary': primary,
             'primary_no_edge': primary & (classes == 0),
             'primary_correct_edge': primary & (classes == 1),
             'primary_wrong_edge': primary & (classes == 2)}
    report = {'probability_temperature': 1, 'teacher_forced': True,
              'primary_positions': 'local positions 128..1023 in each of eight windows',
              'baseline_exact_replays': True, 'original_trace_exact_replay': True,
              'arms': []}
    for arm in metadata['arms']:
        if not arm['restored_byte_equal']:
            raise ValueError('uncertified restoration')
        losses = np.fromfile(directory/arm['losses'], dtype='<f4').astype(np.float64)
        ids = np.fromfile(directory/arm['argmax'], dtype='<i4')
        logits = np.fromfile(directory/arm['trace_logits'], dtype='<f4').reshape(4, 50272)
        if losses.shape != (8192,) or ids.shape != (8192,) or not np.isfinite(losses).all():
            raise ValueError('bad measurements')
        result = dict(name=arm['name'], indices=arm['indices'], groups={}, trace=[])
        for name, mask in masks.items():
            nll = float(losses[mask].mean())
            result['groups'][name] = dict(
                count=int(mask.sum()), nll=nll, perplexity=math.exp(nll),
                accuracy=float((ids[mask] == targets[mask]).mean()),
                delta_nll=float((losses[mask]-baseline_losses[mask]).mean()),
                baseline_correct_lost=int(np.sum(mask & (baseline_ids == targets) & (ids != targets))),
                baseline_wrong_fixed=int(np.sum(mask & (baseline_ids != targets) & (ids == targets))))
        result['per_window_primary'] = []
        for item in range(8):
            selected = slice(item*1024+128, (item+1)*1024)
            result['per_window_primary'].append(dict(
                nll=float(losses[selected].mean()),
                accuracy=float((ids[selected] == targets[selected]).mean())))
        for item, target in enumerate([1475, 68, 2797, 1750]):
            values = logits[item, :50257].astype(np.float64)
            scaled = np.exp(values-values.max())
            probability = scaled/scaled.sum()
            order = np.argsort(-values, kind='stable')[:5]
            result['trace'].append(dict(
                step=1340+item, target=target, target_probability=float(probability[target]),
                target_rank=int(np.sum(values > values[target]))+1,
                top=[dict(token=int(token), piece=pieces[int(token)], probability=float(probability[token]))
                     for token in order],
                all_probability=float(probability[477]),
                sever_probability=float(probability[1750])))
        if arm['name'].startswith('keep_through_b'):
            block = int(arm['name'].removeprefix('keep_through_b'))
            for item in range(4):
                previous = (trace_root/f'trace_step_{1340+item}'/f'lens.blocks.{block}.after_mlp.f32').read_bytes()
                if previous != logits[item].tobytes():
                    raise ValueError('truncation does not match saved native logit lens')
            result['exact_saved_lens_parity'] = True
        if arm['name'] == 'drop_b7_mlp':
            for item in range(4):
                previous = (trace_root/f'trace_step_{1340+item}'/'lens.blocks.7.after_attention.f32').read_bytes()
                if previous != logits[item].tobytes():
                    raise ValueError('final MLP drop disagrees with saved native lens')
            result['exact_saved_lens_parity'] = True
        if arm['name'] == 'isolated_b0':
            edges = {edge['source']: edge['target'] for edge in graph['edges']}
            for row, source in enumerate(inputs):
                if int(source) in edges and ids[row] != edges[int(source)]:
                    raise ValueError('isolated ablation contradicts B0 automaton')
            result['automaton_confident_edges_all_reproduced'] = True
        report['arms'].append(result)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture.resolve(strict=True)
    manifest = json.loads((fixture/'probe_manifest.json').read_text())
    input_records = {item['path']: item for item in bindings(manifest)}
    additions = [fixture/'probe_manifest.json', args.binary, args.graph, Path(__file__),
                 Path(__file__).with_name('block_ablation_probe.cc'),
                 Path(__file__).with_name('block_ablation_arms.h'),
                 Path(__file__).with_name('causal_probe.cc'),
                 Path(__file__).with_name('token_trace_probe_lib.cc')]
    for item in range(4):
        additions.append(args.trace/f'prefix_step_{1340+item}.i32')
        additions.extend((args.trace/f'trace_step_{1340+item}').glob('lens.*.f32'))
        additions.append(args.trace/f'trace_step_{1340+item}'/'logits.f32')
    for path in additions:
        identity = record(path)
        if identity['path'] in input_records and input_records[identity['path']] != identity:
            raise ValueError('manifest mismatch')
        input_records[identity['path']] = identity
    frozen = sorted(input_records.values(), key=lambda item: item['path'])
    verify(frozen)
    args.output.mkdir()
    output = args.output.resolve()
    command = [str(args.binary.resolve()), '--checkpoint='+manifest['checkpoint'],
               '--batch_tokens='+str(fixture/'batch_tokens.bin'),
               '--trace_directory='+str(args.trace.resolve()),
               '--output_dir='+str(output/'native')]
    write_json(output/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                        command=command, inputs=frozen, expected_arms=39))
    with (output/'stdout.log').open('x') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    verify(frozen)
    if code:
        raise RuntimeError('native probe failed; outputs preserved, exit='+str(code))
    report = summarize(output/'native', fixture, args.trace, args.graph)
    write_json(output/'summary.json', report)
    write_json(output/'result.json', dict(complete=True, input_hashes_reverified=True,
                                        outputs=[record(path) for path in sorted(output.rglob('*')) if path.is_file()]))
    print('Complete; all inputs unchanged, clean and original-trace parity verified.', flush=True)


if __name__ == '__main__':
    main()
