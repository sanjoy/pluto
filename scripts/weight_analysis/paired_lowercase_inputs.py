"""Prepare a preserved, case-matching amendment to paired experiment inputs.

Only a new exclusive directory is written. The original exports and experiment
manifest remain untouched. This adds exeunt->nuveth to Exeunt->Nuveth, preserving
case, and proves native token/byte alignment anew. The authenticated CPU corpus
tokenizer receives a constructed argument vector; saved manifest commands are
never executed. This utility neither creates training state nor starts a GPU
job. A failed preparation leaves its partial new directory for inspection and
does not publish amendments.json.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from . import checkpoint, paired_training, paired_word_cases


REPLACEMENTS = ((b'Exeunt', b'Nuveth'), (b'exeunt', b'nuveth'))


def replace_words(original):
    for source, target in REPLACEMENTS:
        original = original.replace(source, target)
    return original


def verify_alignment(original, replacement, original_ids, replacement_ids,
                     original_offsets, replacement_offsets):
    """Independently verify both exact-case substitutions and all other bytes.

    Each replaced occurrence must span exactly three native tokens in both
    streams. Internal word boundaries may move; outer boundaries, all IDs
    outside the spans, and every other byte boundary must stay identical.
    A token shared inside a replaced word need not change ID: the changed-ID
    count is measured, not assumed to equal three times the occurrence count.
    """
    if not isinstance(original, bytes) or not isinstance(replacement, bytes):
        raise ValueError('expected byte corpora')
    if replacement != replace_words(original) or len(replacement) != len(original):
        raise ValueError('replacement contains unrelated text changes')
    a, b = np.asarray(original_ids), np.asarray(replacement_ids)
    ao, bo = np.asarray(original_offsets), np.asarray(replacement_offsets)
    if (a.ndim != 1 or b.ndim != 1 or len(a) != len(b)
            or any(ids.dtype.kind not in 'iu' or np.any(ids < 0)
                   or np.any(ids >= paired_word_cases.VOCAB_SIZE) for ids in (a, b))):
        raise ValueError('invalid or unequal native token streams')
    for offsets in (ao, bo):
        if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) != len(a)+1
                or offsets[0] != 0 or offsets[-1] != len(original)
                or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError('invalid native byte offsets')
    occupied = np.zeros(len(a), dtype=bool)
    internal = np.zeros(len(a)+1, dtype=bool)
    occurrences = []
    targets = dict(REPLACEMENTS)
    for match in re.finditer(b'Exeunt|exeunt', original):
        source, target = match.group(), targets[match.group()]
        spans = []
        for offsets, text, word in ((ao, original, source), (bo, replacement, target)):
            first = int(np.searchsorted(offsets, match.start(), side='right')-1)
            end = int(np.searchsorted(offsets, match.end(), side='left'))
            if (end-first != 3 or offsets[end] != match.end()
                    or text[offsets[first]:offsets[end]] not in (word, b' '+word)):
                raise ValueError('a replacement does not occupy exactly three native word tokens')
            spans.append((first, end))
        if spans[0] != spans[1] or ao[spans[0][0]] != bo[spans[0][0]]:
            raise ValueError('replacement token coordinates or outer boundaries shifted')
        first, end = spans[0]
        if occupied[first:end].any():
            raise ValueError('overlapping replacement spans')
        occupied[first:end] = True
        internal[first+1:end] = True
        occurrences.append(dict(byte_start=match.start(), token_start=first,
            original_ids=a[first:end].tolist(), replacement_ids=b[first:end].tolist(),
            source_spelling=source.decode('ascii'), target_spelling=target.decode('ascii')))
    if not np.array_equal(a[~occupied], b[~occupied]):
        raise ValueError('token IDs outside replacement spans changed')
    if not np.array_equal(ao[~internal], bo[~internal]):
        raise ValueError('byte boundaries outside replacement spans changed')
    return dict(token_count=len(a), replacements=len(occurrences),
                changed_token_ids=int(np.count_nonzero(a != b)),
                outside_replacement_tokens_identical=True,
                replacement_case_counts=dict(sorted(Counter(o['source_spelling'] for o in occurrences).items())),
                occurrences=occurrences)


def _copy_record(source_record, destination):
    source = Path(source_record['path'])
    if source.is_symlink() or paired_word_cases._record(source) != source_record:
        raise ValueError('frozen CPU tokenizer source changed or is a symlink')
    with source.open('rb') as source_file, destination.open('xb') as target_file:
        shutil.copyfileobj(source_file, target_file)
    result = paired_word_cases._record(destination)
    if (result['sha256'], result['bytes']) != (source_record['sha256'], source_record['bytes']):
        raise ValueError('frozen CPU tokenizer copy differs from its source')
    return result


def _export(binary, tokenizer_dir, text_path, token_path):
    # Fixed argument shape and authenticated copied program, never item['command'].
    command = [binary['path'], str(tokenizer_dir), str(text_path), str(token_path)]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f'native CPU tokenizer failed ({completed.returncode}): {completed.stderr}')
    export = json.loads(completed.stdout)
    item = dict(text=str(text_path), token_ids=str(token_path),
                offsets=str(token_path)+'.offsets.bin', export=export, command=command)
    for key in ('text', 'token_ids', 'offsets'):
        item[key+'_sha256'] = paired_word_cases._record(item[key])['sha256']
    execution = dict(command=command, started_utc=started,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    return item, execution


def prepare(old_manifest_path, output_root):
    """Create authenticated amended corpus exports, but never launch training."""
    record = paired_word_cases._record
    old_manifest_path = Path(old_manifest_path).resolve(strict=True)
    old_root = old_manifest_path.parent
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.parent.resolve(strict=True)
    sources = [record(module.__file__) for module in (checkpoint, paired_training, paired_word_cases)]
    sources.append(record(__file__))
    old_manifest = record(old_manifest_path)
    manifest = json.loads(old_manifest_path.read_text())
    cases_record = record(old_root/'word_cases/cases.json')
    cases = json.loads((old_root/'word_cases/cases.json').read_text())
    if (manifest.get('format') != 'pluto-paired-corpus-training-v1'
            or manifest.get('replacement') != {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True}
            or cases.get('format') != 'pluto-paired-word-cases-v1'
            or cases.get('manifest') != old_manifest):
        raise ValueError('word cases do not authenticate the original uppercase-only experiment')
    used = {r['path']: r for r in (old_manifest, cases_record)}
    expected = {f'{domain}.{split}' for domain in ('original', 'replacement')
                for split in ('full', 'training', 'test')}
    if set(manifest['inputs']) != expected:
        raise ValueError('expected all six original frozen exports')
    old_inputs = {name: paired_word_cases._load_input(item, used)
                  for name, item in sorted(manifest['inputs'].items())}
    for split in ('full', 'training', 'test'):
        a, b = (old_inputs[f'{domain}.{split}'] for domain in ('original', 'replacement'))
        if paired_training.verify_alignment(a['text'], b['text'], a['tokens'], b['tokens'],
                a['offsets'], b['offsets']) != manifest['alignment'][split]:
            raise ValueError('old native alignment differs from authenticated manifest')
    boundary = manifest['split_byte']
    if type(boundary) is not int or boundary <= 0:
        raise ValueError('invalid split byte')
    for domain in ('original', 'replacement'):
        full, train, test = (old_inputs[f'{domain}.{split}'] for split in ('full', 'training', 'test'))
        if (len(train['text']) != boundary or not test['text']
                or train['text']+test['text'] != full['text']
                or not np.array_equal(np.r_[train['tokens'], test['tokens']], full['tokens'])
                or not np.array_equal(np.r_[train['offsets'], test['offsets'][1:]+boundary], full['offsets'])):
            raise ValueError('old full/split exports disagree')
    original_full = old_inputs['original.full']['text']
    if not original_full.count(b'exeunt') or b'nuveth' in original_full.lower():
        raise ValueError('expected surviving lowercase exeunt and an absent control spelling')
    source_binary = manifest['binaries']['tokenize_corpus']
    if Path(source_binary['path']) != old_root/'bin/tokenize_corpus':
        raise ValueError('unexpected frozen CPU tokenizer path')
    asset_records = manifest['tokenizer_files']
    if 'tokenizer.json' not in asset_records:
        raise ValueError('missing frozen tokenizer.json')
    for name, item in asset_records.items():
        if Path(name).name != name or Path(item['path']) != old_root/'tokenizer'/name:
            raise ValueError('unexpected frozen tokenizer asset path')
    for item in [source_binary, *asset_records.values()]:
        if Path(item['path']).is_symlink() or record(item['path']) != item:
            raise ValueError('frozen tokenizer binary or asset changed')
        used[item['path']] = item

    output_root.mkdir(exist_ok=False)
    for name in ('bin', 'tokenizer', 'inputs'):
        (output_root/name).mkdir()
    binary = _copy_record(source_binary, output_root/'bin/tokenize_corpus')
    Path(binary['path']).chmod(0o555)
    assets = {name: _copy_record(item, output_root/'tokenizer'/name)
              for name, item in sorted(asset_records.items())}
    for item in assets.values():
        Path(item['path']).chmod(0o444)
    for item in [binary, *assets.values()]:
        used[item['path']] = item
    inputs = {f'original.{split}': manifest['inputs'][f'original.{split}']
              for split in ('full', 'training', 'test')}
    alignments, comparisons, executions, amended = {}, {}, [], {}
    for split in ('full', 'training', 'test'):
        original = old_inputs[f'original.{split}']
        text_path = output_root/'inputs'/f'replacement.{split}.txt'
        with text_path.open('xb') as stream:
            stream.write(replace_words(original['text']))
        text_path.chmod(0o444)
        item, execution = _export(binary, output_root/'tokenizer', text_path,
                                 output_root/'inputs'/f'replacement.{split}.tokens.bin')
        updated = paired_word_cases._load_input(item, used)
        alignment = verify_alignment(original['text'], updated['text'], original['tokens'],
            updated['tokens'], original['offsets'], updated['offsets'])
        previous = old_inputs[f'replacement.{split}']
        lower_tokens = np.zeros(len(updated['tokens']), dtype=bool)
        lower_boundaries = np.zeros(len(updated['tokens'])+1, dtype=bool)
        for occurrence in alignment['occurrences']:
            if occurrence['source_spelling'] == 'exeunt':
                start = occurrence['token_start']
                lower_tokens[start:start+3] = True
                lower_boundaries[start+1:start+3] = True
        if (not np.array_equal(previous['tokens'][~lower_tokens], updated['tokens'][~lower_tokens])
                or not np.array_equal(previous['offsets'][~lower_boundaries], updated['offsets'][~lower_boundaries])):
            raise ValueError('an old uppercase replacement or unrelated token/boundary changed')
        inputs[f'replacement.{split}'] = item
        amended[split] = updated
        alignments[split] = alignment
        comparisons[split] = dict(
            lowercase_replacements=alignment['replacement_case_counts'].get('exeunt', 0),
            additional_changed_token_ids=int(np.count_nonzero(previous['tokens'] != updated['tokens'])),
            outside_lowercase_tokens_and_boundaries_identical=True)
        executions.append(execution)
    if (amended['training']['text']+amended['test']['text'] != amended['full']['text']
            or not np.array_equal(np.r_[amended['training']['tokens'], amended['test']['tokens']], amended['full']['tokens'])
            or not np.array_equal(np.r_[amended['training']['offsets'], amended['test']['offsets'][1:]+boundary], amended['full']['offsets'])):
        raise ValueError('amended native full exports disagree with split concatenation')
    for item in [*used.values(), *sources]:
        if record(item['path']) != item:
            raise ValueError('input, copied tokenizer, or analysis source changed during amendment')
    result = dict(
        format='pluto-paired-lowercase-amendment-v1', complete=True,
        old_manifest=old_manifest, old_word_cases=cases_record,
        replacement_rules=[dict(source=a.decode(), target=b.decode()) for a, b in REPLACEMENTS],
        split_byte=boundary, inputs=inputs, alignment=alignments,
        comparison_to_old_replacement=comparisons,
        binaries={'tokenize_corpus': binary}, tokenizer_files=assets,
        native_export_executions=executions, provenance=list(used.values()), sources=sources,
        full_and_split_exports_identical=True, old_inputs_preserved=True,
        gpu_work_performed=False, training_launched=False, goal_completion_claimed=False,
        limitations=[
            'This is an input amendment, not a training run, checkpoint conversion, or updated experiment manifest.',
            'Original exports remain at their authenticated old paths; amended replacement exports are new files.',
            'Three-token alignment and counts are measured using the authenticated native CPU tokenizer.',
            'Existing case suites and analysis observers are not updated by this utility.',
            'Only exact Exeunt and exeunt spellings are replaced; other capitalization variants are not inferred.'])
    paired_word_cases._write_json(output_root/'amendments.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.old_manifest, args.output)
    print(json.dumps(dict(output=str(args.output/'amendments.json'),
                         case_counts={s: a['replacement_case_counts'] for s, a in result['alignment'].items()})))


if __name__ == '__main__':
    main()
