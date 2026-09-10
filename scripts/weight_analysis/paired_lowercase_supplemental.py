"""Freeze amended exact-following-token and shared-subword causal controls.

The main suite measures three-piece spelling; this one adds the ACTUAL next
native token after each word, not a made-up delimiter. Shared-piece cases ask
whether a word-row intervention harms unrelated uses of the same subwords.
Their entire prefix and continuation avoid both uppercase and lowercase
replacement spans and are identical across the two corpora.

Selection uses only the frozen native corpus IDs and seed 17, never model scores
or learned weights. Missing coverage is reported rather than filled with an
example from a replacement span. Every output is new and existing runs/cases
remain untouched. GPU scoring is a separate, subsequent operation.
"""

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np

from . import paired_lowercase_cases as word_cases
from . import paired_lowercase_scores as scores
from . import paired_lowercase_training as training
from . import paired_supplemental_cases as native

FORMAT = 'pluto-paired-lowercase-supplemental-cases-v1'
SOURCES = [training.record(m.__file__) for m in (word_cases,scores,training,native)] + [training.record(__file__)]
SPLITS = ('training','test')
DOMAINS = ('original','replacement')


def prepare(cases_path, output, *, controls_per_piece=8, prefix_tokens=128):
    """Write mixed 4-target word and 3-target control cases for native probes.

    Original word prefixes and candidate IDs are preserved exactly. Only their
    future padding changes when the fourth target is added, so all first-three
    causal predictions must be bit-identical between main/supplemental suites.
    Tests can request smaller control prefixes; production defaults use 128.
    """
    if (type(controls_per_piece) is not int or controls_per_piece<1
            or type(prefix_tokens) is not int or prefix_tokens<1):
        raise ValueError('positive integer control count and prefix length required')
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    main, records=scores.load_cases(cases_path)
    for parent in (Path(cases_path).resolve().parent,Path(main['packed_batch']['path']).parent):
        if output.resolve()==parent or parent in output.resolve().parents:
            raise ValueError('supplemental output must be outside main inputs')
    length=main['context_length']
    if prefix_tokens+3>length+1:
        raise ValueError('control prefix and targets do not fit model context')
    amendment=training.read_json(main['amendment']['path'])
    manifest=training.read_json(main['old_manifest']['path'])
    used={r['path']:r for r in records}
    _,inputs=word_cases._load_and_validate(amendment,manifest,used)
    piece_ids=sorted({token for split in SPLITS for o in amendment['alignment'][split]['occurrences']
                      for domain in DOMAINS for token in o[domain+'_ids']})
    if not piece_ids:
        raise ValueError('no amended word-piece IDs')
    occupied={}
    for split in SPLITS:
        mask=np.zeros(len(inputs['original.'+split]['tokens']),dtype=bool)
        for occurrence in amendment['alignment'][split]['occurrences']:
            start=occurrence['token_start']; mask[start:start+3]=True
        occupied[split]=mask
    packed,_=native._packed(main)
    cases,xs,ys,excluded=[],[],[],[]

    def append(case,prefix,targets):
        x,y,rows=native._sequence(prefix,targets,length)
        case.update(case_index=len(cases),target_ids=targets,scored_rows=rows)
        cases.append(case); xs.append(x); ys.append(y)

    for index,case in enumerate(main['cases']):
        if case['kind']!='word':
            continue
        prefix=native._validate_case(case,index,packed,(3,))
        split,domain,role=case['split'],case['prefix_domain'],case['target_source_domain']
        begin,start=case['prefix']['token_start'],case['prefix']['token_end']
        source,target=inputs[f'{domain}.{split}'],inputs[f'{role}.{split}']
        occurrence=amendment['alignment'][split]['occurrences'][case['occurrence_index']]
        variant,pair=word_cases._variant(occurrence)
        if (occurrence['token_start']!=start or case['spelling_variant']!=variant
                or case['candidate_pair']!=pair or case['target']!=pair[role]
                or case['target_ids']!=occurrence[role+'_ids']
                or case['prefix']!=native._prefix(source,begin,start)
                or not np.array_equal(prefix,source['tokens'][begin:start])):
            raise ValueError('main word case does not match amended native source')
        if start+3>=len(target['tokens']):
            excluded.append(dict(source_case_index=index,reason='no following native token; EOS not invented'))
            continue
        targets=target['tokens'][start:start+4].tolist()
        append(dict(case,kind='word_next_native',source_case_index=index,
            inherited_source_case_index=case.get('source_case_index'),word_token_count=3,
            target_source=native._source(target,start,4),
            next_native_token=dict(id=targets[3],**native._source(target,start+3,1))),prefix,targets)
    rng=random.Random(17)
    coverage={}
    for split in SPLITS:
        source=inputs['original.'+split]; other=inputs['replacement.'+split]
        tokens,mask=source['tokens'],occupied[split]
        cumulative=np.r_[0,np.cumsum(mask,dtype=np.int64)]
        coverage[split]=[]
        for piece in piece_ids:
            starts=np.flatnonzero((tokens==piece)&~mask)
            fitting=starts[(starts>=prefix_tokens)&(starts+3<=len(tokens))]
            eligible=fitting[cumulative[fitting+3]==cumulative[fitting-prefix_tokens]].tolist()
            selected=sorted(rng.sample(eligible,min(controls_per_piece,len(eligible))))
            coverage[split].append(dict(piece_id=piece,outside_replacement_occurrences=len(starts),
                eligible_window_count=len(eligible),selected_token_starts=selected,selected_count=len(selected),
                missing_coverage=not selected,unfilled_requested_count=controls_per_piece-len(selected)))
            for start in selected:
                begin=start-prefix_tokens
                if (np.any(mask[begin:start+3]) or not np.array_equal(tokens[begin:start+3],other['tokens'][begin:start+3])
                        or native._source(source,begin,start+3-begin)['bytes_hex']!=
                           native._source(other,begin,start+3-begin)['bytes_hex']):
                    raise ValueError('shared-piece context overlaps a replacement or differs between corpora')
                append(dict(kind='shared_piece',split=split,context_id=f'{split}:piece:{piece}:start:{start}',
                    piece_id=piece,prefix_domain='shared',target_source_domain='original',target='control_next_3',
                    spelling_variant=None,candidate_pair=None,source_case_index=None,
                    prefix=native._prefix(source,begin,start),target_source=native._source(source,start,3)),
                    tokens[begin:start],tokens[start:start+3].tolist())
    if not cases:
        raise ValueError('no supplemental cases')
    records=list(used.values())
    training.verify_records([*records,*SOURCES])
    output.mkdir()
    batch_path=output/'packed_cases.bin'
    with batch_path.open('xb') as stream:
        np.asarray(xs,dtype='<i4').tofile(stream); np.asarray(ys,dtype='<i4').tofile(stream)
    result=dict(format=FORMAT,complete=True,case_count=len(cases),context_length=length,
        vocab_size=main['vocab_size'],eos_token_id=main['eos_token_id'],candidate_pairs=word_cases.PAIRS,
        amendment=main['amendment'],old_manifest=main['old_manifest'],
        source_word_cases=training.record(cases_path),source_word_packed_batch=main['packed_batch'],
        packed_batch=training.record(batch_path),provenance=records,sources=SOURCES,cases=cases,
        packing='little-endian int32 [all inputs case_count x context_length][all targets case_count x context_length]',
        selection=dict(seed=17,model_outputs_used=False,union_piece_ids=piece_ids,union_piece_count=len(piece_ids),
            shared_piece_prefix_tokens=prefix_tokens,controls_per_piece_requested=controls_per_piece,
            coverage=coverage,excluded_word_cases=excluded,
            algorithm='preserve all amended word cases; sorted native piece IDs and training/test order; random.Random(17).sample'),
        checks=dict(native_amendment_alignment_recomputed=True,all_shared_contexts_avoid_all_replacements=True,
                    all_shared_context_ids_and_bytes_identical_between_corpora=True,goal_completion_claimed=False),
        limitations=['Fourth word target is one actual next native token, not any possible delimiter.',
            'Shared-piece controls are row-stratified, not corpus-frequency weighted; missing rows are explicit.',
            'Title and lowercase contexts are separate; there are no lowercase test occurrences in this corpus.',
            'Prefix positions restart at zero; these are not necessarily their historic training-window positions.',
            'No model ran; these inputs alone do not demonstrate a probability change or causal mechanism.'])
    training.verify_records([*records,*SOURCES])
    training.publish(output/'cases.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=prepare(args.cases,args.output)
    print(f'Prepared {result["case_count"]} amended supplemental cases; {result["selection"]["union_piece_count"]} word-piece rows.')


if __name__=='__main__':
    main()
