"""Freeze the load-time shared libraries of an unchanged trusted executable.

Bazel test executables often depend on DSOs through build-tree-relative
RUNPATHs. Relocating just the executable therefore does not relocate the test.
This helper inspects the ORIGINAL executable's working RUNPATH, copies only
direct Bazel _solib dependencies under their loader names, and proves that the
relocated executable resolves the same bytes with one explicit environment
override. It neither rebuilds nor modifies either executable, and never runs
the program's main function: ``ldd`` is used only on caller-trusted binaries.

System libraries, the ELF interpreter, and CUDA runtime libraries remain at
their original resolved paths and are hashed too. This is a LOAD-TIME closure,
not a complete runtime attestation: later dlopen() calls (notably the CUDA
driver) are not enumerated by ldd. The caller must separately validate the GPU,
driver, process identity, and actual test results. Only environment overrides
are returned; unrelated inherited environment values are never serialized.

The destination must not exist. An error after creation deliberately leaves
partial output for inspection; it never overwrites or deletes prior evidence.
"""

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess


FORMAT = 'pluto-native-runtime-bundle-v1'
_NAME = r'[A-Za-z0-9_.+\-]+'
_MAPPED = re.compile(rf'^({_NAME}) => (/\S+) \(0x[0-9a-fA-F]+\)$')
_ABSOLUTE = re.compile(r'^(/\S+) \(0x[0-9a-fA-F]+\)$')
_VIRTUAL = re.compile(r'^linux-(?:vdso|gate)\.so\.\d+ \(0x[0-9a-fA-F]+\)$')


def _record(path):
    """Hash a regular resolved file and reject changes during that read."""
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('dependency is not a regular file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError('file changed while hashing: ' + str(path))
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def _verify(records):
    for record in records:
        if _record(record['path']) != record:
            raise ValueError('runtime input changed: ' + record['path'])


def _parse_ldd(stdout):
    """Require an unambiguous complete mapping, including the ELF loader."""
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or _VIRTUAL.fullmatch(line):
            continue
        match = _MAPPED.fullmatch(line)
        if match:
            name, path = match.groups()
        else:
            match = _ABSOLUTE.fullmatch(line)
            if not match:
                raise ValueError('unresolved or malformed ldd output: ' + line)
            path = match[1]
            name = Path(path).name
            if not re.fullmatch(_NAME, name):
                raise ValueError('invalid loader name: ' + name)
        if name in result:
            raise ValueError('duplicate loader name: ' + name)
        result[name] = str(Path(os.path.normpath(path)))
    if not result:
        raise ValueError('ldd did not report any dynamic libraries')
    return result


def _ldd(executable, environment):
    result = subprocess.run(['ldd', str(executable)], env=environment,
                            check=False, capture_output=True, text=True,
                            timeout=30)
    if result.returncode or result.stderr.strip():
        raise ValueError('ldd failed: ' + result.stderr + result.stdout)
    return dict(command=['ldd', str(executable)], returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr,
                libraries=_parse_ldd(result.stdout))


def freeze_runtime(reference_executable, relocated_executable, destination):
    """Return a verified NEW library bundle for a byte-identical executable.

    ``environment`` contains only ``LD_LIBRARY_PATH=destination``. The caller
    must install that override for the child and retain ``frozen_records`` in
    its evidence ledger, verifying them before and after native execution.
    Nonempty AND empty inherited LD_PRELOAD/LD_AUDIT variables are rejected:
    the caller must explicitly unset them rather than relying on their values.

    ``ldd_before``/``ldd_after`` describe actual loader observations, not native
    execution. Every mapped library is either a copied Bazel DSO or an exact
    retained system/CUDA/loader path. No basename-based substitutions or
    fallback to a mutable Bazel tree are accepted in the relocated result.
    """
    for key in ('LD_PRELOAD', 'LD_AUDIT'):
        if key in os.environ:
            raise ValueError('unset inherited ' + key + ' before bundling')
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise ValueError('bundle parent directory does not exist')
    destination = destination.parent.resolve() / destination.name
    reference = Path(reference_executable).resolve(strict=True)
    relocated_arg = Path(relocated_executable).absolute()
    if relocated_arg.is_symlink():
        raise ValueError('relocated executable must not be a symlink')
    relocated = relocated_arg.resolve(strict=True)
    originals = [_record(reference), _record(relocated)]
    if any(originals[0][key] != originals[1][key] for key in ('bytes', 'sha256')):
        raise ValueError('relocated executable bytes differ from reference')

    # Original Bazel RUNPATH supplies its dependencies; do not let inherited
    # search directories silently select a different reference implementation.
    before_env = dict(os.environ)
    before_env.pop('LD_LIBRARY_PATH', None)
    before = _ldd(reference, before_env)
    rows, source_records = [], {}
    for name, raw_path in sorted(before['libraries'].items()):
        raw = Path(raw_path)
        resolved = raw.resolve(strict=True)
        source = _record(resolved)
        source_records[source['path']] = source
        # CUDA's _solib/...toolchain.../libcudart symlink is NOT a project DSO:
        # keep that resolved system CUDA installation rather than copying it.
        bundle = raw.parent.name.startswith('_solib_')
        if bundle and not resolved.is_relative_to(raw.parent.parent.resolve()):
            raise ValueError('Bazel project dependency escapes build tree: ' + raw_path)
        rows.append(dict(name=name, loader_source_path=raw_path, source=source,
                         bundled=bundle))
    _verify([*originals, *source_records.values()])

    destination.mkdir(exist_ok=False)
    copied = []
    for row in rows:
        if not row['bundled']:
            row['runtime_path'] = row['source']['path']
            continue
        target = destination / row['name']
        # Open exclusively, and copy the resolved source bytes rather than its
        # symlink or unmangled basename (different libraries often share one).
        with Path(row['source']['path']).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
        target.chmod(0o444)
        record = _record(target)
        if any(record[key] != row['source'][key] for key in ('bytes', 'sha256')):
            raise ValueError('dependency changed during copy: ' + row['name'])
        row.update(runtime_path=str(target), copy=record)
        copied.append(record)

    overrides = {'LD_LIBRARY_PATH': str(destination)}
    after_env = dict(os.environ)
    after_env.update(overrides)
    after = _ldd(relocated, after_env)
    if set(after['libraries']) != {row['name'] for row in rows}:
        raise ValueError('relocated dynamic dependency names differ')
    for row in rows:
        actual = Path(after['libraries'][row['name']]).resolve(strict=True)
        if actual != Path(row['runtime_path']):
            raise ValueError('relocated dependency resolved outside its pinned path: ' + row['name'])
        observed = _record(actual)
        if any(observed[key] != row['source'][key] for key in ('bytes', 'sha256')):
            raise ValueError('relocated dependency bytes differ: ' + row['name'])
    records = {r['path']: r for r in [*originals, *source_records.values(), *copied]}
    frozen = [records[path] for path in sorted(records)]
    _verify(frozen)
    if {p.name for p in destination.iterdir()} != {r['name'] for r in rows if r['bundled']}:
        raise ValueError('unexpected files in the new runtime directory')
    return dict(format=FORMAT, complete=True, destination=str(destination),
                environment=overrides, frozen_records=frozen, libraries=rows,
                reference_executable=originals[0], relocated_executable=originals[1],
                executable_sha256=originals[0]['sha256'], ldd_before=before,
                ldd_after=after, runtime_dlopen_covered=False,
                gpu_work_performed=False,
                limitation='Load-time closure only; caller separately authenticates runtime-dlopen CUDA driver, GPU identity, and native test results.')
