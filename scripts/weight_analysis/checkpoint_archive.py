"""Strict, read-only embedding access to directory or gzip-tar checkpoints.

Only the embedding payload is materialized. All other member names, types and
sizes are checked while their bytes are streamed past, never extracted to the
filesystem. A gzip stream must reach its real EOF: stopping at tar's zero-block
terminator alone would leave a corrupt CRC or truncated trailer undetected.

Architecture, little-endian FP32 storage, and the meaning of each file come
from the caller's GPT2Config, NOT authenticated producer metadata. Even an
exact size match does not prove the checkpoint's producer or training data.
"""

from dataclasses import asdict
import gzip
import hashlib
from pathlib import Path
import stat
import tarfile
import zlib

import numpy as np

from .checkpoint import GPT2Config, sha256_file, tensor_manifest


_CHUNK = 1024 * 1024


def _identity(path):
    """Do not follow a final-component symlink while inspecting file identity."""
    info = path.lstat()
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _file_record(path):
    before = _identity(path)
    if not stat.S_ISREG(before[2]):
        raise ValueError(f'input is not a regular non-symlink file: {path}')
    digest = sha256_file(path)
    if _identity(path) != before:
        raise ValueError(f'input changed while hashing: {path}')
    return {'path': str(path), 'bytes': before[3], 'sha256': digest,
            'device': before[0], 'inode': before[1],
            'mtime_ns': before[4], 'ctime_ns': before[5]}


def _content_record(record):
    """Use the experiment runner's exact three-field rehash record schema."""
    return {key: record[key] for key in ('path', 'bytes', 'sha256')}


def _embedding(payload, spec):
    if len(payload) != spec.nbytes:
        raise ValueError(f'embedding expected {spec.nbytes} bytes, got {len(payload)}')
    # A bytes-backed view stays genuinely read-only: callers cannot re-enable
    # writes with setflags(), unlike an owning array whose flag was just unset.
    array = np.frombuffer(payload, dtype='<f4').reshape(spec.shape)
    flat = array.reshape(-1)
    for start in range(0, flat.size, _CHUNK):
        if not np.isfinite(flat[start:start + _CHUNK]).all():
            raise ValueError('embedding contains non-finite values, including padding')
    return array


def _directory_files(path, manifest):
    actual = {entry.name: entry for entry in path.iterdir()}
    expected = {spec.filename for spec in manifest}
    if set(actual) != expected:
        raise ValueError('directory file set mismatch; '
                         f'missing={sorted(expected - set(actual))}, '
                         f'extra={sorted(set(actual) - expected)}')
    for spec in manifest:
        info = actual[spec.filename].lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'weight is not a regular non-symlink file: {spec.filename}')
        if info.st_size != spec.nbytes:
            raise ValueError(f'{spec.filename}: expected {spec.nbytes} bytes, '
                             f'got {info.st_size}')
    return [actual[spec.filename] for spec in manifest]


def _read_directory(path, manifest):
    identity = _identity(path)
    files = _directory_files(path, manifest)
    before = [_file_record(file) for file in files]
    payload = files[0].read_bytes()
    array = _embedding(payload, manifest[0])
    after_files = _directory_files(path, manifest)
    after = [_file_record(file) for file in after_files]
    if identity != _identity(path) or before != after:
        raise ValueError('checkpoint directory changed during read')
    return array, {
        'input_files': [_content_record(record) for record in before],
        'weight_files_after': after,
        'directory_identity_unchanged': True,
        'members': [dict(spec.to_dict(), file=record)
                    for spec, record in zip(manifest, before)],
    }


def _member_record(member):
    """Retain the logical tar member's full public header metadata.

    PAX extension metadata is retained too. Offsets are decompressed-tar byte
    offsets, not offsets into the compressed file. Extension headers themselves
    are interpreted by tarfile and are not mislabeled as weight payloads.
    """
    return {
        'name': member.name, 'type': member.type.decode('latin1'),
        'bytes': member.size, 'mode': member.mode, 'uid': member.uid,
        'gid': member.gid, 'uname': member.uname, 'gname': member.gname,
        'mtime': member.mtime, 'linkname': member.linkname,
        'devmajor': member.devmajor, 'devminor': member.devminor,
        'header_offset': member.offset, 'payload_offset': member.offset_data,
        'pax_headers': dict(member.pax_headers),
    }


def _read_archive(path, step, manifest):
    before = _file_record(path)
    top = f'step_{step}'
    expected = {top: None} | {f'{top}/{spec.filename}': spec for spec in manifest}
    seen = set()
    members = []
    payload = None
    try:
        with gzip.open(path, 'rb') as decompressed:
            # gzip handling is deliberately outside tarfile. The streaming tar
            # reader skips unneeded payloads without creating files or holding
            # a whole decompressed checkpoint in memory.
            with tarfile.open(fileobj=decompressed, mode='r|') as archive:
                for member in archive:
                    name = member.name
                    if name not in expected:
                        raise ValueError(f'unexpected archive member: {name!r}')
                    if name in seen:
                        raise ValueError(f'duplicate archive member: {name!r}')
                    seen.add(name)
                    spec = expected[name]
                    if spec is None:
                        if not member.isdir() or member.size != 0 or member.linkname:
                            raise ValueError('top archive member must be an empty directory')
                    else:
                        # In particular, reject hardlinks, symlinks, devices,
                        # directories masquerading as weights, and sparse files.
                        if (not member.isfile() or member.issparse() or
                                member.linkname or any(key.startswith('GNU.sparse')
                                                       for key in member.pax_headers)):
                            raise ValueError(f'weight member is not a plain regular file: {name}')
                        if member.size != spec.nbytes:
                            raise ValueError(f'{name}: expected {spec.nbytes} bytes, '
                                             f'got {member.size}')
                        if spec.index == 0:
                            source = archive.extractfile(member)
                            if source is None:
                                raise ValueError('embedding member has no readable payload')
                            with source:
                                payload = source.read(spec.nbytes + 1)
                    members.append(_member_record(member))

                if seen != set(expected):
                    raise ValueError(f'missing archive members: {sorted(set(expected) - seen)}')

                # Drain through the tar stream first, including its read-ahead
                # buffer. Beyond tar EOF only zero padding is accepted; hidden
                # appended tar members or other trailing payload are rejected.
                # This also consumes the gzip trailer and verifies its CRC.
                while chunk := archive.fileobj.read(_CHUNK):
                    if chunk.strip(b'\0'):
                        raise ValueError('nonzero bytes after archive terminator')

            # Explicitly reach gzip EOF even if tar's buffering implementation
            # changes. Closing GzipFile alone does not validate an unread CRC.
            while chunk := decompressed.read(_CHUNK):
                if chunk.strip(b'\0'):
                    raise ValueError('nonzero bytes after archive terminator')
    except (OSError, EOFError, tarfile.TarError, zlib.error) as error:
        raise ValueError(f'invalid gzip/tar checkpoint: {error}') from error

    if payload is None:
        raise ValueError('embedding payload missing')
    array = _embedding(payload, manifest[0])
    after = _file_record(path)
    if before != after:
        raise ValueError('checkpoint archive changed during read')
    return array, {
        'input_files': [_content_record(before)],
        'archive_before': before, 'archive_after': after,
        'archive_sha256': before['sha256'], 'archive_bytes': before['bytes'],
        'members': members, 'gzip_eof_and_crc_verified': True,
        'trailing_tar_bytes_zero_only': True,
    }


def read_embedding(path, step, config=GPT2Config()):
    """Return immutable physical FP32 embeddings and complete input provenance.

    `path` must be exactly step_N or step_N.tar.gz with N equal to `step`.
    No extraction is performed. Directories must have exactly the manifest's
    weight files, with no extra logs or symlinks. Archives must have exactly one
    corresponding top directory plus those regular files, with no links,
    duplicate names, sparse payloads, traversal, or extra members.

    All input-file bytes are hashed before and after the read, including every
    directory weight (not just its embedding). The returned `input_files`
    records let a caller repeat those full hashes at the end of an experiment.
    These identity checks detect ordinary concurrent modifications, but are
    not a cryptographically authenticated history or a defense against a
    privileged adversary replacing and restoring files between observations.

    Padding is retained and finite-checked here. Candidate/statistical callers
    must explicitly slice array[:config.vocab_size] to exclude padding IDs.
    """
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    if not isinstance(config, GPT2Config):
        raise ValueError('config must be GPT2Config')
    path = Path(path).absolute()
    if path.name not in (f'step_{step}', f'step_{step}.tar.gz'):
        raise ValueError('checkpoint path name does not match requested step')
    identity = _identity(path)
    manifest = tensor_manifest(config)
    if path.name.endswith('.tar.gz') and stat.S_ISREG(identity[2]):
        kind = 'gzip_tar_archive'
        array, provenance = _read_archive(path, step, manifest)
    elif path.name == f'step_{step}' and stat.S_ISDIR(identity[2]):
        kind = 'directory'
        array, provenance = _read_directory(path, manifest)
    else:
        raise ValueError('checkpoint must be a regular archive or non-symlink directory')
    if _identity(path) != identity:
        raise ValueError('checkpoint identity changed during read')
    provenance.update({
        'path': str(path), 'step': step, 'source_kind': kind,
        'config': asdict(config), 'storage_dtype': '<f4',
        'configuration_scope': 'Caller assumptions, not authenticated checkpoint producer metadata',
        'physical_embedding_shape': list(array.shape),
        'logical_vocab_size': config.vocab_size,
        'embedding_sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest(),
        'unique_weight_files': len(manifest),
        'input_files_unchanged_during_read': True,
        'all_physical_embedding_values_finite': True,
        'embedding_readonly': True,
        'other_weight_finiteness_checked': False,
    })
    return array, provenance
