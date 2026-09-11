"""Exact early-LN2/MLP donor transplants with the selected tied E rows fixed.

N transfers only early LN2 gamma/beta; F transfers both MLP projections and
their biases. M is their union, not an MLP without normalization. Every other
parameter stays recipient. These are coordinate-defined weight differences,
not claims of unique storage or independently aligned semantic neurons.

Only independent checkpoint copies are created. No training or native forward
is performed; callers must separately authenticate and score native executions.
"""

from dataclasses import asdict
from pathlib import Path
import stat

from . import checkpoint
from . import historical_source_archive as history
from . import paired_complement_localization as core
from . import paired_weight_patch as patcher

GPT2Config = checkpoint.GPT2Config
GPT2Checkpoint = checkpoint.GPT2Checkpoint
tensor_manifest = checkpoint.tensor_manifest
FORMAT = 'pluto-paired-mlp-affine-v1'
CELLS = ('E', 'N', 'F', 'M')
PRODUCTION_EMBEDDING_ROWS = (45, 68, 303, 400, 409, 1475, 2797, 3109, 14364, 21733, 45177)
_SUBSETS = {'E': (), 'N': ('N',), 'F': ('F',), 'M': ('N', 'F')}
_require = core.native._require
SOURCE_RECORDS = tuple(history.file_record(Path(path).resolve()) for path in
    (__file__, patcher.__file__, checkpoint.__file__, core.__file__, core.native.__file__, core.branch.__file__))


def selections(config=GPT2Config()):
    """Return fresh E/N/F/M tensor lists for the early half of an even layout.

    Production has blocks 0..3: N has eight tensors/4,096 parameters and F
    sixteen tensors/8,398,848 parameters. Small even layouts are CPU fixtures.
    The embedding rows are separate from these whole-tensor selections.
    """
    _require(isinstance(config, GPT2Config), 'expected explicit GPT2Config')
    GPT2Config(**asdict(config))
    _require(config.n_layers >= 2 and config.n_layers % 2 == 0, 'need an even block count of at least two')
    count = config.n_layers // 2
    specs = tensor_manifest(config)
    by_name = {spec.name: spec for spec in specs}
    _require(len(specs) == len(by_name) == 12*config.n_layers+4, 'unexpected checkpoint tensor inventory')
    groups = {'N': [], 'F': []}
    for block in range(count):
        expected = (
            ('N', 'ln2.scale', 8, (config.d_model,)),
            ('N', 'ln2.bias', 9, (config.d_model,)),
            ('F', 'mlp.input.weight', 10, (config.d_model, config.d_ff)),
            ('F', 'mlp.input.bias', 11, (config.d_ff,)),
            ('F', 'mlp.output.weight', 12, (config.d_ff, config.d_model)),
            ('F', 'mlp.output.bias', 13, (config.d_model,)),
        )
        for group, suffix, index, shape in expected:
            name = f'blocks.{block}.{suffix}'
            spec = by_name.get(name)
            _require(spec is not None and spec.index == index+12*block
                     and spec.shape == shape and spec.dtype == '<f4', 'MLP/LN2 tensor layout differs')
            groups[group].append(name)
    _require(len(groups['N']) == 2*count and len(groups['F']) == 4*count
             and not set(groups['N']) & set(groups['F']), 'LN2/MLP selections do not partition the early MLP bundle')
    return {'E': [], 'N': sorted(groups['N']), 'F': sorted(groups['F']),
            'M': sorted(groups['N']+groups['F'])}


def _selection(cell, rows, config):
    available = selections(config)
    _require(type(cell) is str and cell in available, 'unknown affine/MLP cell')
    _require(isinstance(rows, (list, tuple)) and rows
             and all(type(row) is int and 0 <= row < config.vocab_size for row in rows)
             and len(set(rows)) == len(rows), 'invalid selected logical embedding rows')
    rows = sorted(rows)
    if config == GPT2Config():
        _require(tuple(rows) == PRODUCTION_EMBEDDING_ROWS, 'production requires the exact eleven frozen E rows')
    return available[cell], rows


def _path(value, *, exists=True):
    path = Path(value).absolute()
    try:
        _require(not path.is_symlink() and path.resolve(strict=exists) == path,
                 'noncanonical or symlink path: '+str(path))
    except OSError as error:
        raise ValueError('cannot resolve path: '+str(path)) from error
    return path


def _inventory(path, specs):
    _require(path.is_dir(), 'checkpoint must be a directory')
    entries = list(path.iterdir())
    expected = {spec.filename for spec in specs}
    _require({entry.name for entry in entries} in (expected, expected | {'patch.json'})
             and all(stat.S_ISREG(entry.lstat().st_mode) for entry in entries),
             'checkpoint must contain only regular weight files and optional patch.json')
    return patcher._snapshot(path, specs)


def _sources():
    for expected in SOURCE_RECORDS:
        _require(history.file_record(expected['path']) == expected, 'loaded model-helper source changed')
    return {record['path']: dict(record) for record in SOURCE_RECORDS}


def validate_model(patch_path, a, d, cell, rows, config=GPT2Config()):
    """Check source identities, exact selected/unselected bytes and inodes.

    Return the existing native-loader model contract. `subset` is [] for E,
    ['N'] for N, ['F'] for F and ['N','F'] for M. The original patch's generic
    donor-transfer schema and historical identities are never relabeled.
    """
    tensors, rows = _selection(cell, rows, config)
    a, d, patch_path = _path(a), _path(d), _path(patch_path)
    _require(patch_path.name == 'patch.json', 'model marker must be patch.json')
    paths = dict(recipient=a, donor=d, patched=patch_path.parent)
    _require(all(paths['patched'] != source and source not in paths['patched'].parents
                 and paths['patched'] not in source.parents for source in (a, d)),
             'source and output checkpoints overlap')
    specs = tensor_manifest(config)
    before = {role: _inventory(path, specs) for role, path in paths.items()}
    records = _sources()
    patch_record = history.file_record(patch_path)
    records[patch_record['path']] = patch_record
    patch = history._json_record(patch_record)
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config),
             'incomplete patch or wrong configuration')
    GPT2Config(**patch['config'])
    row_bytes = config.d_model*4
    _require(patch['selection'] == dict(tensors=tensors, embedding_rows=rows,
        embedding_row_byte_ranges=[[row*row_bytes, (row+1)*row_bytes] for row in rows],
        embedding_is_tied_to_lm_head=True), 'patch selection differs from declared affine/MLP cell')
    _require(patch['sources']['original']['path'] == str(a)
             and patch['sources']['replacement']['path'] == str(d)
             and patch['output']['path'] == str(paths['patched']), 'patch source/output binding differs')
    _require(all(patch['validation'].get(flag) is True for flag in (
        'all_weights_finite', 'sources_unchanged', 'selected_bytes_equal_replacement',
        'unselected_bytes_equal_original', 'no_hardlinks')), 'incomplete patch validation')
    producer = records[str(Path(patcher.__file__).resolve())]
    loader = records[str(Path(checkpoint.__file__).resolve())]
    _require(patch['implementation'] == dict(path=producer['path'], sha256=producer['sha256'],
        checkpoint_loader_sha256=loader['sha256']), 'patcher or checkpoint-loader identity differs')
    hashes = dict(recipient=patch['sources']['original']['weights_sha256'],
                  donor=patch['sources']['replacement']['weights_sha256'],
                  patched=patch['output']['weights_sha256'])
    weights = {role: core.branch._weights(path, hashes[role], records, config) for role, path in paths.items()}
    checkpoints = {role: GPT2Checkpoint(path, config, check_finite=True) for role, path in paths.items()}
    source_inodes = {tuple(value[:2]) for role in ('recipient', 'donor')
                     for value in before[role]['files'].values()}
    for spec in specs:
        actual = checkpoints['patched'][spec.name]
        if spec.index == 0:
            start = 0
            for row in rows:
                _require(core.native._bits_equal(actual[start:row], checkpoints['recipient'][spec.name][start:row])
                         and core.native._bits_equal(actual[row], checkpoints['donor'][spec.name][row]),
                         'selected/unselected embedding bytes differ')
                start = row+1
            _require(core.native._bits_equal(actual[start:], checkpoints['recipient'][spec.name][start:]),
                     'remaining/padded embedding bytes differ')
        else:
            source = 'donor' if spec.name in tensors else 'recipient'
            _require(core.native._bits_equal(actual, checkpoints[source][spec.name]),
                     'selected/unselected tensor bytes differ')
        info = (paths['patched']/spec.filename).lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and (info.st_dev, info.st_ino) not in source_inodes, 'patch is not an independent regular copy')
    for role, path in paths.items():
        _require(_inventory(_path(path), specs) == before[role], 'checkpoint changed during model validation')
    for expected in records.values():
        _require(history.file_record(expected['path']) == expected, 'file changed during model validation')
    return dict(paths={role: str(path) for role, path in paths.items()}, hashes=hashes, weight_records=weights,
        patch=patch_record, subset=list(_SUBSETS[cell]), selection=tensors,
        records=[records[path] for path in sorted(records)])


def build_model(a, d, output, cell, rows, config=GPT2Config()):
    """Build a NEW independent model and validate it; never execute native code."""
    tensors, rows = _selection(cell, rows, config)
    a, d, output = _path(a), _path(d), _path(output, exists=False)
    _require(output.parent.is_dir(), 'output parent must already exist')
    specs = tensor_manifest(config)
    for source in (a, d):
        _inventory(source, specs)
    _sources()
    patcher.create_patch(a, d, output, tensors=tensors, embedding_rows=rows, config=config)
    return validate_model(output/'patch.json', a, d, cell, rows, config=config)
