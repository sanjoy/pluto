"""Verify historical source identities against a pinned, exact-byte archive.

The returned original records remain historical identities. Physical records
name the files actually checked; they do not claim that current code produced
an old run. This module neither executes archived code nor changes evidence.

Each file verification makes two streaming reads of the same open descriptor.
Matching reads and metadata detect observed edits; they are not an atomic
filesystem snapshot. Synchronized edits restored between observations, or
changes after the final check, still require callers to keep inputs quiescent.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat


FORMAT = 'pluto-pre-sharing-analysis-source-snapshot-v1'
_STAT_FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
_JSON_LIMIT = 64 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value):
    _require(type(value) is str and bool(value), 'path must be a nonempty string')
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and '..' not in path.parts,
             'path must be an absolute normalized identity: ' + value)
    return path


def _record_value(value):
    _require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
             'expected an exact path/bytes/sha256 record')
    _path(value['path'])
    _require(type(value['bytes']) is int and value['bytes'] >= 0
             and type(value['sha256']) is str
             and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None,
             'invalid file size or SHA256')
    return dict(value)


def _canonical_file(path):
    _require(path.resolve(strict=True) == path,
             'file or parent is a symlink: ' + str(path))
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), 'expected a regular nonsymlink file: ' + str(path))
    return info


def _signature(info):
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def _read(path, *, retain=False):
    """Require two matching streamed hashes and a stable descriptor/path identity."""
    path = _path(str(path))
    try:
        before = _signature(_canonical_file(path))
        _require(not retain or before[3] <= _JSON_LIMIT, 'JSON exceeds 64 MiB')
        digest, chunks = hashlib.sha256(), []
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            _require(_signature(os.fstat(stream.fileno())) == before,
                     'file changed before hashing: ' + str(path))
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                if retain:
                    chunks.append(block)
            _require(_signature(os.fstat(stream.fileno())) == before,
                     'file changed while hashing: ' + str(path))
            # Equal-size writes can share mtime/ctime at filesystem clock
            # resolution. Observe the bytes again, through the same descriptor.
            stream.seek(0)
            confirmation = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                confirmation.update(block)
            _require(confirmation.hexdigest() == digest.hexdigest(),
                     'file content changed while hashing: ' + str(path))
            _require(_signature(os.fstat(stream.fileno())) == before,
                     'file changed while hashing: ' + str(path))
        _require(_signature(_canonical_file(path)) == before,
                 'file changed while hashing: ' + str(path))
    except OSError as error:
        raise ValueError('cannot verify file: ' + str(path)) from error
    record = dict(path=str(path), bytes=before[3], sha256=digest.hexdigest())
    return record, b''.join(chunks) if retain else None, before


def file_record(path):
    """Record two matching reads; reject symlinks in the file or any parent."""
    return _read(path)[0]


def _json_record(expected):
    actual, contents, _ = _read(expected['path'], retain=True)
    _require(actual == expected, 'pinned file changed: ' + expected['path'])

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('nonfinite JSON number: ' + value)

    return json.loads(contents, object_pairs_hook=pairs, parse_constant=nonfinite)


class SourceArchive:
    """Resolve old repository records only through the caller-pinned archive.

    Construction checks mapping metadata, canonical archive paths, and owning
    request coverage. Payload hashes are checked when verify() requests them.
    No persistent payload-hash cache is kept between verification calls.
    """

    def __init__(self, manifest_path, *, expected_manifest_record, owning_request_record):
        self._manifest_record = _record_value(expected_manifest_record)
        self._owner_record = _record_value(owning_request_record)
        _require(str(_path(str(manifest_path))) == self._manifest_record['path'],
                 'manifest path differs from pinned identity')
        manifest = _json_record(self._manifest_record)
        _require(type(manifest) is dict and manifest.get('format') == FORMAT,
                 'wrong source archive format')
        for name, value in (('all_frozen_repository_inputs_preserved', True),
                            ('original_paths_retained_in_metadata', True),
                            ('original_sources_moved', False),
                            ('analysis_completion_claimed', False),
                            ('source_bytes_unchanged', True)):
            _require(manifest.get(name) is value, 'invalid archive declaration: ' + name)
        _require(manifest.get('owning_run_request') == self._owner_record,
                 'archive owning request differs from pinned identity')
        self.repository = _path(manifest['repository'])
        archive_root = Path(self._manifest_record['path']).parent / 'repository'
        _require(type(manifest.get('files')) is list
                 and type(manifest.get('package_file_count')) is int
                 and manifest['package_file_count'] == len(manifest['files']) > 0,
                 'archive file count differs')
        self._mapping, physical_paths = {}, set()
        for row in manifest['files']:
            _require(type(row) is dict and set(row) == {'original', 'archived'},
                     'invalid source mapping')
            original, physical = _record_value(row['original']), _record_value(row['archived'])
            original_path = _path(original['path'])
            _require(original_path.is_relative_to(self.repository)
                     and original_path != self.repository, 'source lies outside recorded repository')
            _require(physical['path'] == str(archive_root / original_path.relative_to(self.repository)),
                     'archive path does not preserve repository-relative identity')
            _require(all(original[key] == physical[key] for key in ('bytes', 'sha256')),
                     'original and archived content identities differ')
            _require(original['path'] not in self._mapping and physical['path'] not in physical_paths,
                     'duplicate or conflicting source mapping')
            try:
                _canonical_file(Path(physical['path']))
            except OSError as error:
                raise ValueError('missing archived source: ' + physical['path']) from error
            self._mapping[original['path']] = (original, physical)
            physical_paths.add(physical['path'])
        request = _json_record(self._owner_record)
        _require(type(request) is dict and type(request.get('frozen_inputs')) is list,
                 'owning request lacks frozen inputs')
        seen, repository_count = set(), 0
        for value in request['frozen_inputs']:
            record = _record_value(value)
            _require(record['path'] not in seen, 'duplicate owning-request input')
            seen.add(record['path'])
            if Path(record['path']).is_relative_to(self.repository):
                _require(record['path'] in self._mapping
                         and self._mapping[record['path']][0] == record,
                         'owning-request repository input is not preserved exactly')
                repository_count += 1
        _require(type(manifest.get('frozen_repository_input_count')) is int
                 and manifest['frozen_repository_input_count'] == repository_count,
                 'frozen repository input count differs')
        self._check_pins()

    @property
    def manifest_record(self):
        return dict(self._manifest_record)

    @property
    def owning_request_record(self):
        return dict(self._owner_record)

    def _check_pins(self):
        for expected in (self._manifest_record, self._owner_record):
            _require(file_record(expected['path']) == expected,
                     'pinned file changed: ' + expected['path'])

    def verify(self, records):
        """Return ordered {original, physical} bindings without mutating inputs.

        Repeated identical records are retained; conflicting identities fail.
        Each physical file is verified once per call with two streamed hashes,
        then restatted at the end. No hashes persist between verification calls.
        """
        self._check_pins()
        result, logical, checked = [], {}, {}
        for value in records:
            original = _record_value(value)
            previous = logical.setdefault(original['path'], original)
            _require(previous == original, 'conflicting historical record identity')
            if Path(original['path']).is_relative_to(self.repository):
                _require(original['path'] in self._mapping, 'unknown historical repository path')
                saved, expected = self._mapping[original['path']]
                _require(saved == original, 'historical record differs from archived original')
            else:
                expected = original
            if expected['path'] not in checked:
                actual, _, signature = _read(expected['path'])
                checked[expected['path']] = (actual, signature)
            actual, _ = checked[expected['path']]
            _require(actual == expected, 'physical file differs from recorded identity: ' + expected['path'])
            result.append(dict(original=dict(original), physical=dict(actual)))
        self._check_pins()
        for path, (_, signature) in checked.items():
            try:
                _require(_signature(_canonical_file(Path(path))) == signature,
                         'file changed during verification: ' + path)
            except OSError as error:
                raise ValueError('file disappeared during verification: ' + path) from error
        return result

    def physical_records(self, records):
        """Verify historical records and return their real physical identities."""
        return [binding['physical'] for binding in self.verify(records)]
