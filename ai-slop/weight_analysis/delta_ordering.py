"""Freeze a directional checkpoint-delta ordering HEURISTIC, without text.

Only checkpoint embedding endpoints and already-frozen token IDs are read.
No corpus, tokenizer, model forward, gradient inversion, or GPU is involved.
The association is not a next-token probability: AdamW endpoint differences
mix ten updates and unknown moments, and the embedding is a tied LM head.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .checkpoint import GPT2Config
from .checkpoint_archive import read_embedding


BOUNDARIES = (580, 1780, 4280, 7230, 8160)
ARMS = ('directional', 'transpose', 'static', 'permuted_delta_labels')
PERMUTATION_SEED = 20260909
SEED_COUNT = 128
PATH_LENGTH = 8
SEED_POSITION = 3
SEED_LIST = 'shared_boundary_adjusted'
SEED_CANDIDATES_SHA256 = '2aeee3c0b5054c2bf6a63df38a6dcac20b064351839434bc0c48301d05af18e3'


def file_record(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected a regular non-symlink evidence file')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1 << 20):
            digest.update(block)
    after = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if identity(before) != identity(after) or path.is_symlink():
        raise ValueError('evidence changed while hashing')
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest())


def write_json(path, data):
    serialized = json.dumps(data, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as output:
        output.write(serialized)


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON number: ' + value)
    def finite_float(value):
        number=float(value)
        if not np.isfinite(number):
            reject(value)
        return number
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject,
                      parse_float=finite_float)


def load_seeds(path, vocabulary):
    record = file_record(path)
    if record['sha256'] != SEED_CANDIDATES_SHA256:
        raise ValueError('frozen seed candidate file hash does not match')
    parsed = strict_json(Path(path).read_text())
    if (not isinstance(parsed,dict)
            or parsed.get('stage') != 'frozen_unordered_token_candidates_from_weight_deltas'
            or parsed.get('top_k') != SEED_COUNT):
        raise ValueError('unexpected frozen seed artifact')
    ids = parsed.get('candidate_lists', {}).get(SEED_LIST, {}).get('token_ids')
    if (not isinstance(ids, list) or len(ids) != SEED_COUNT
            or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
            or len(set(ids)) != SEED_COUNT):
        raise ValueError('need exactly 128 distinct logical-vocabulary seed IDs')
    if file_record(path) != record:
        raise ValueError('frozen seed artifact changed while loading')
    return ids, record


def matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError('expected a finite nonempty matrix')
    return value


def normalized_geometry(before, after):
    """Exactly the existing FP64 translation/radial fit, retaining direction.

    Inputs contain only logical vocabulary rows. We use the same operation
    order as restart_delta.activities, then normalize nonzero rows with no
    floor or fitted threshold. Exactly zero rows remain zero. Tiny residuals
    can amplify endpoint-rounding noise; the row norms are retained to expose
    that limitation rather than silently suppressing it.
    """
    before, after = matrix(before), matrix(after)
    if before.shape != after.shape:
        raise ValueError('endpoint shapes differ')
    delta = after - before
    translation = delta.mean(axis=0)
    centered = before - before.mean(axis=0)
    residual = delta - translation
    denominator = float(np.sum(centered * centered))
    coefficient = float(np.sum(centered * residual) / denominator) if denominator else 0.0
    residual -= coefficient * centered
    c_norm = np.linalg.norm(centered, axis=1)
    r_norm = np.linalg.norm(residual, axis=1)
    if (not np.isfinite(c_norm).all() or not np.isfinite(r_norm).all()
            or not np.isfinite(coefficient)):
        raise ValueError('nonfinite geometry arithmetic')
    np.divide(centered, c_norm[:, None], out=centered, where=c_norm[:, None] != 0)
    np.divide(residual, r_norm[:, None], out=residual, where=r_norm[:, None] != 0)
    return centered, residual, c_norm, r_norm, dict(
        shape=list(before.shape), shared_translation=translation.tolist(),
        fitted_radial_coefficient_not_authenticated_decay=coefficient,
        centered_before_squared_norm=denominator,
        centered_zero_rows=int(np.count_nonzero(c_norm == 0)),
        residual_zero_rows=int(np.count_nonzero(r_norm == 0)))


class Associations:
    """Batched directional queries without materializing V-by-V matrices.

    For the permuted control, label t addresses original delta row perm[t].
    This ONE mapping is shared by every interval, never re-drawn per path.
    Returned components are [interval, candidate_ID, fixed-node-index], so the
    actual five BLAS scores used to select an edge are retained verbatim.
    """
    def __init__(self, centered, residual, permutation):
        if not centered or len(centered) != len(residual):
            raise ValueError('need matching nonempty interval matrix lists')
        self.centered = [matrix(x) for x in centered]
        self.residual = [matrix(x) for x in residual]
        self.shape = self.centered[0].shape
        if any(x.shape != self.shape for x in self.centered + self.residual):
            raise ValueError('association matrix shapes differ')
        self.permutation = np.asarray(permutation)
        vocabulary = self.shape[0]
        if (self.permutation.shape != (vocabulary,)
                or self.permutation.dtype.kind not in 'iu'
                or not np.array_equal(np.sort(self.permutation), np.arange(vocabulary))):
            raise ValueError('need a complete delta-row label permutation')

    def scores(self, arm, direction, fixed):
        if arm not in ARMS or direction not in ('predecessor', 'successor'):
            raise ValueError('invalid association arm or extension direction')
        fixed = np.asarray(fixed)
        if (fixed.ndim != 1 or not fixed.size or fixed.dtype.kind not in 'iu'
                or np.any(fixed < 0) or np.any(fixed >= self.shape[0])):
            raise ValueError('invalid fixed-node IDs')
        components = np.empty((len(self.centered), self.shape[0], len(fixed)), dtype=np.float64)
        for index, (c, r) in enumerate(zip(self.centered, self.residual)):
            if arm == 'static':
                values = c @ c[fixed].T
            elif arm == 'directional':
                values = c @ r[fixed].T if direction == 'predecessor' else r @ c[fixed].T
            elif arm == 'transpose':
                values = r @ c[fixed].T if direction == 'predecessor' else c @ r[fixed].T
            elif direction == 'predecessor':
                values = c @ r[self.permutation[fixed]].T
            else:
                values = (r @ c[fixed].T)[self.permutation]
            components[index] = values
        scores = np.mean(components, axis=0)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite association scores')
        return scores, components


def greedy_paths(associations, arm, seeds, predecessors=SEED_POSITION,
                 successors=PATH_LENGTH-SEED_POSITION-1):
    """Grow all seeded paths together; each column has its own exclusions.

    np.argmax returns the first maximum, giving ascending-ID ties on the
    actual computed FP64 scores. No-repeat applies within each path; identical
    or overlapping paths across different seeds are retained, not discarded.
    """
    vocabulary = associations.shape[0]
    if (arm not in ARMS or type(predecessors) is not int or type(successors) is not int
            or predecessors < 0 or successors < 0
            or predecessors + successors + 1 > vocabulary
            or not isinstance(seeds, (tuple, list)) or not seeds
            or any(type(token) is not int or not 0 <= token < vocabulary for token in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('invalid path lengths or distinct seed IDs')
    paths = [[seed] for seed in seeds]
    histories = [[] for seed in seeds]
    directions = ['predecessor']*predecessors + ['successor']*successors
    for selection_step, direction in enumerate(directions):
        fixed = [path[0] if direction == 'predecessor' else path[-1] for path in paths]
        scores, components = associations.scores(arm, direction, fixed)
        for column, path in enumerate(paths):
            scores[path, column] = -np.inf
        choices = np.argmax(scores, axis=0)
        for column, choice in enumerate(choices.tolist()):
            path = paths[column]
            source, target = ((choice, fixed[column]) if direction == 'predecessor'
                              else (fixed[column], choice))
            histories[column].append(dict(
                selection_step=selection_step, direction=direction,
                source_id=source, target_id=target,
                excluded_ids_before_selection=list(path),
                score=float(scores[choice, column]),
                interval_components=components[:, choice, column].tolist()))
            path.insert(0, choice) if direction == 'predecessor' else path.append(choice)
    result = []
    for seed, path, history in zip(seeds, paths, histories):
        by_edge = {(item['source_id'], item['target_id']): item for item in history}
        edges = [dict(by_edge[(source, target)], position=index)
                 for index, (source, target) in enumerate(zip(path, path[1:]))]
        result.append(dict(candidate_id=f'{arm}_seed_{seed}', method=arm,
                           seed_id=seed, seed_position=predecessors, token_ids=path,
                           path_score=float(sum(edge['score'] for edge in edges)),
                           edges=edges, selection_order=history))
    return result


def address_sources(arm, source, target, permutation, width):
    """Physical weight_0 row addresses; adjusted rows also depend on all means.

    The fitted radial subtraction and centering are global operations. These
    addresses identify rows entering the formula, not private text storage.
    """
    if arm == 'transpose':
        left, right, kind = target, source, 'adjusted_delta'
    elif arm == 'static':
        left, right, kind = source, target, 'centered_before_embedding'
    elif arm == 'permuted_delta_labels':
        left, right, kind = source, int(permutation[target]), 'adjusted_delta'
    elif arm == 'directional':
        left, right, kind = source, target, 'adjusted_delta'
    else:
        raise ValueError('invalid association arm')
    def location(token):
        return dict(row_id=token, filename='weight_0.bin',
                    byte_interval=[4*width*token,4*width*(token+1)])
    return dict(left_centered_before=location(left), right_kind=kind,
                right_before=location(right),
                right_after=location(right) if kind == 'adjusted_delta' else None)


def run(checkpoint_root, output_dir, protocol, seed_candidates):
    start = time.monotonic()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '4' or os.environ.get('OMP_NUM_THREADS') != '4':
        raise ValueError('run with OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4')
    root = Path(checkpoint_root).resolve(strict=True)
    requested = Path(output_dir).absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(requested)
    output = requested.resolve()
    if output == root or root in output.parents:
        raise ValueError('outputs must be outside checkpoint root')
    config = GPT2Config()
    seeds, seed_record = load_seeds(seed_candidates, config.vocab_size)
    source_dir = Path(__file__).resolve().parent
    sources = {name:file_record(source_dir/name) for name in
               ('delta_ordering.py','checkpoint_archive.py','checkpoint.py')}
    sources['protocol'] = file_record(protocol)
    sources['frozen_seed_candidates'] = seed_record
    paths = {}
    for boundary in BOUNDARIES:
        for step in (boundary,boundary+10):
            options = [root/f'step_{step}.tar.gz',root/f'step_{step}']
            present = [path for path in options if path.exists() or path.is_symlink()]
            if len(present) != 1:
                raise ValueError(f'missing or ambiguous checkpoint {step}')
            paths[step] = present[0]
    output.mkdir()
    plan = dict(schema_version=1, stage='weight_only_directional_ordering_plan',
                created_utc_before_weight_reads=datetime.now(timezone.utc).isoformat(),
                sources=sources, intervals=[[b,b+10] for b in BOUNDARIES],
                checkpoint_paths={str(step):str(path) for step,path in paths.items()},
                seed_ids=seeds, seed_list=SEED_LIST, seed_count=SEED_COUNT,
                logical_vocabulary_size=config.vocab_size, embedding_width=config.d_model,
                arms=list(ARMS), path_length=PATH_LENGTH, seed_position=SEED_POSITION,
                no_repeat_within_path=True, exact_computed_score_tie='lowest token ID',
                permutation_seed=PERMUTATION_SEED,
                permutation_mapping='right delta row for label t is original R[permutation[t]]',
                purpose='Heuristic ordering, not gradient inversion, LM probability, or model execution',
                no_corpus_or_tokenizer_access=True, no_model_execution=True)
    write_json(output/'plan.json',plan)
    files = {'plan':file_record(output/'plan.json')}
    permutation = np.random.Generator(np.random.PCG64(PERMUTATION_SEED)).permutation(config.vocab_size)
    with (output/'permutation.npy').open('xb') as f:
        np.save(f,permutation,allow_pickle=False)
    files['permutation'] = file_record(output/'permutation.npy')
    centered, residual, row_norms, summaries, provenance = [], [], [], [], {}
    for boundary in BOUNDARIES:
        endpoints=[]
        for step in (boundary,boundary+10):
            values,record=read_embedding(paths[step],step)
            if (values.shape != (config.padded_vocab_size,config.d_model)
                    or values.dtype != np.dtype('<f4')
                    or record.get('input_files_unchanged_during_read') is not True):
                raise ValueError('unverified or incorrectly shaped checkpoint embedding')
            endpoints.append(values[:config.vocab_size])
            provenance[str(step)]=record
        c,r,cn,rn,summary=normalized_geometry(*endpoints)
        centered.append(c);residual.append(r)
        row_norms.append((cn,rn))
        key=f'geometry_{boundary}_{boundary+10}'
        with (output/(key+'.npz')).open('xb') as f:
            np.savez(f,centered_before_row_norms=cn,adjusted_delta_row_norms=rn)
        files[key]=file_record(output/(key+'.npz'))
        summary.update(before_step=boundary,after_step=boundary+10,row_norms_file=key)
        summaries.append(summary)
        print(f'Prepared {boundary}->{boundary+10}; no corpus or model access',flush=True)
    association=Associations(centered,residual,permutation)
    candidates=[]
    for arm in ARMS:
        paths_for_arm=greedy_paths(association,arm,seeds)
        for candidate in paths_for_arm:
            for edge in candidate['edges']:
                edge['physical_weight_row_sources']=address_sources(
                    arm,edge['source_id'],edge['target_id'],permutation,config.d_model)
                addresses=edge['physical_weight_row_sources']
                left=addresses['left_centered_before']['row_id']
                right=addresses['right_before']['row_id']
                edge['original_component_row_norms']=[
                    dict(left_centered_before=float(cn[left]),
                         right=float(cn[right] if arm=='static' else rn[right]),
                         right_kind=addresses['right_kind']) for cn,rn in row_norms]
                edge['intervals']=[[b,b+10] for b in BOUNDARIES]
                edge['checkpoint_embedding_hashes']=[
                    dict(before=provenance[str(b)].get('embedding_sha256'),
                         after=provenance[str(b+10)].get('embedding_sha256')) for b in BOUNDARIES]
            candidate['interpretation']='Weight-only directional heuristic; not asserted to be text or a model continuation'
        candidates.extend(paths_for_arm)
        print(f'Computed {arm}: all {len(paths_for_arm)} candidates retained',flush=True)
    if len(candidates)!=len(ARMS)*SEED_COUNT:
        raise ValueError('candidate denominator changed')
    # Candidates are fixed before any external verifier may inspect text.
    with (output/'candidates.jsonl').open('x') as f:
        for candidate in candidates:
            f.write(json.dumps(candidate,allow_nan=False)+'\n')
    files['candidates']=file_record(output/'candidates.jsonl')
    for record in provenance.values():
        for item in record['input_files']:
            if file_record(item['path'])!=item:
                raise ValueError('checkpoint input changed before freeze')
    for record in [*sources.values(),*files.values()]:
        if file_record(record['path'])!=record:
            raise ValueError('source or output evidence changed before freeze')
    result=dict(schema_version=1,complete=True,stage='frozen_weight_only_directional_ordering_heuristic',
                frozen_utc_before_corpus_verification=datetime.now(timezone.utc).isoformat(),
                sources=sources,files=files,checkpoint_provenance=provenance,
                intervals=summaries,arms=list(ARMS),candidates_per_arm=SEED_COUNT,
                candidate_count=len(candidates),path_length=PATH_LENGTH,
                runtime_environment=dict(numpy=np.__version__,OPENBLAS_NUM_THREADS=os.environ['OPENBLAS_NUM_THREADS'],
                                         OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS']),
                elapsed_seconds=time.monotonic()-start,
                limitations=['Checkpoint gaps are candidate restarts, not authenticated restart records.',
                             'Adjusted endpoint deltas are not raw gradients or authenticated decay correction.',
                             'All nonzero rows are normalized; tiny delta residuals can amplify numerical noise.',
                             'Even pure rigid embedding rotation can produce directed E-dot-delta scores; directionality alone is not token-order evidence.',
                             'Scores are computed FP64 cosines, not next-token probabilities; exact numerical ties use ascending IDs.',
                             'Seed IDs and intervals were informed by earlier token-bag results; this is not a held-out checkpoint test.',
                             'No corpus, tokenizer, model forward/backward, training, or GPU work occurs in this extractor.'])
    write_json(output/'frozen.json',result)
    print(f'Frozen {len(candidates)} eight-token candidates in four arms',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-root','output-dir','protocol','seed-candidates'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    run(args.checkpoint_root,args.output_dir,args.protocol,args.seed_candidates)


if __name__=='__main__': main()
