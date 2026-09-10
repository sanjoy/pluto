"""Authenticate the amended trajectory and plan matched-step interventions.

The reused original checkpoints keep their real legacy paths. The replacement
checkpoints keep their amended paths. Both complete time budgets, deterministic
controls, actual terminal logs and checkpoint bytes must validate before this
planner selects anything. Unequal final optimizer steps are never substituted
for a matched-step experiment.

All embedding rows come from the actual amended native replacement spans; no
nine/eleven-row count is built into selection. The two directions, earliest and
latest common positive steps, copy controls, embedding transfers, and every
attention/MLP whole-branch and output-write transfer are fixed independently of
observed effects. This produces no checkpoint copies and runs no GPU code.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import checkpoint, embedding_factorial_readout, paired_case_contract
from . import paired_intervention_plan as legacy
from . import paired_lowercase_analysis as trajectory
from . import paired_lowercase_scores, paired_lowercase_supplemental
from . import paired_lowercase_training as training
from . import paired_supplemental_cases as native

FORMAT='pluto-paired-lowercase-intervention-plan-v1'
MODULES=(checkpoint,embedding_factorial_readout,paired_case_contract,legacy,trajectory,
         paired_lowercase_scores,paired_lowercase_supplemental,training,native)
SOURCES=[training.record(m.__file__) for m in MODULES]+[training.record(__file__)]


def canonical(value):
    """TensorSpec tuples become arrays when persisted as JSON."""
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def validate_exports(exports_path,main_path,supplemental_path,*,config=checkpoint.GPT2Config()):
    """Compare each exported byte/row/label to its actual source case selection."""
    main,records=paired_lowercase_scores.load_cases(main_path)
    supplemental=training.read_json(supplemental_path)
    if (supplemental.get('format')!=paired_lowercase_supplemental.FORMAT
            or supplemental.get('complete') is not True
            or supplemental['amendment']!=main['amendment']
            or supplemental['old_manifest']!=main['old_manifest']
            or supplemental['source_word_cases']!=training.record(main_path)
            or supplemental['source_word_packed_batch']!=main['packed_batch']):
        raise ValueError('supplemental cases must reference these exact amended main cases')
    inputs={'main':main,'supplemental':supplemental}
    source_paths={'main':Path(main_path),'supplemental':Path(supplemental_path)}
    arrays={}
    for name,plan in inputs.items():
        if (plan['vocab_size']!=config.vocab_size or plan['context_length']!=config.context_length):
            raise ValueError('case geometry differs from checkpoint configuration')
        records.extend([training.record(source_paths[name]),*plan['provenance'],*plan['sources']])
        array,batch=native._packed(plan); arrays[name]=array; records.append(batch)
        for index,case in enumerate(plan['cases']):
            native._validate_case(case,index,array,(4,) if case['kind']=='word_next_native' else (3,))
            paired_case_contract.metadata(case,amended=True)
    document=training.read_json(exports_path)
    if (document.get('format')!='pluto-paired-lowercase-causal-exports-v1'
            or document.get('complete') is not True or set(document['exports'])!={'main','word_next_native','shared_piece'}
            or document['main']!=training.record(main_path)
            or document['supplemental']!=training.record(supplemental_path)):
        raise ValueError('not the complete native exports for these frozen cases')
    records.extend([training.record(exports_path),document['exporter']])
    for name,item in document['exports'].items():
        source='main' if name=='main' else 'supplemental'
        plan=inputs[source]
        indices=list(range(plan['case_count'])) if name=='main' else [
            i for i,c in enumerate(plan['cases']) if c['kind']==name]
        count=len(indices); nrows=4 if name=='word_next_native' else 3
        if (not indices or item['source_indices']!=indices or item['case_count']!=count
                or item['rows_per_case']!=nrows or item['source_cases']!=training.record(source_paths[source])
                or item['source_packed_batch']!=plan['packed_batch']):
            raise ValueError('export selection does not cover exactly its declared source kind')
        records.extend(item[key] for key in ('cases_json','packed_batch','selected_rows'))
        exported=training.read_json(item['cases_json']['path'])
        expected_cases=[dict(plan['cases'][i],case_index=j,export_source_case_index=i) for j,i in enumerate(indices)]
        if (exported['cases']!=expected_cases or exported['exporter']!=document['exporter']
                or exported['format']!=plan['format'] or exported['amendment']!=plan['amendment']
                or exported['packed_batch']!=item['packed_batch']):
            raise ValueError('export changed source labels, spelling, or source order')
        expected=arrays[source][:,indices,:].astype('<i4')
        if (Path(item['packed_batch']['path']).stat().st_size!=expected.nbytes
                or not np.array_equal(np.fromfile(item['packed_batch']['path'],dtype='<i4'),expected.reshape(-1))):
            raise ValueError('exported native batch differs from source bytes')
        expected_rows=np.asarray([plan['cases'][i]['scored_rows'] for i in indices],dtype='<i4')
        if (Path(item['selected_rows']['path']).stat().st_size!=expected_rows.nbytes
                or not np.array_equal(np.fromfile(item['selected_rows']['path'],dtype='<i4'),expected_rows.reshape(-1))):
            raise ValueError('exported scored rows differ from source rows')
        ledger={}
        embedding_factorial_readout._validate_cases(item['cases_json']['path'],
            dict(case_count=count,rows_per_case=nrows,batch=item['packed_batch']['path'],rows=item['selected_rows']['path']),
            None,ledger,config)
        records.extend(ledger.values())
    records=list({r['path']:r for r in records}.values())
    training.verify_records(records)
    return main,supplemental,document['exports'],records


def prepare(root,summary_path,exports_path,output,*,config=checkpoint.GPT2Config()):
    """Prepare a new immutable plan; completion means PLANNED, never executed."""
    root,summary_path,exports_path,output=(Path(p).absolute() for p in (root,summary_path,exports_path,output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if summary_path!=root/'analysis_trajectory/summary.json':
        raise ValueError('summary must belong to this amended trajectory')
    request=training.read_json(root/'request.json')
    state=training.read_json(root/'state.json')
    summary=training.read_json(summary_path)
    if (summary.get('format')!=trajectory.FORMAT or summary.get('complete') is not True
            or summary.get('goal_completion_claimed') is not False
            or request['amendment_root']!=str(root)):
        raise ValueError('requires a completed authenticated amended trajectory')
    records=[training.record(summary_path),training.record(root/'request.json'),
             *summary['frozen_inputs'],*summary['terminal_records']]
    if training.record(root/'request.json') not in summary['frozen_inputs']:
        raise ValueError('trajectory did not bind this actual training request')
    training.verify_records([*records,*SOURCES])
    fresh=trajectory.validate_completion(root,request,state)
    if (canonical(fresh['plan'])!=summary['plan']
            or fresh['original_reference']!=summary['original_reference']
            or fresh['determinism_verification']!=summary['determinism_verification']
            or any(r not in summary['terminal_records'] for r in fresh['terminal_records'])):
        raise ValueError('trajectory disagrees with independently revalidated completed training')
    verified={i['path']:i['sha256'] for i in fresh['checked']}
    declared={i['path']:i['weight_sha256'] for i in summary['checkpoints']}
    if verified!=declared or len(declared)!=len(summary['checkpoints']):
        raise ValueError('trajectory checkpoint hashes do not match actual training')
    main_path=root/'word_cases/cases.json'; supplemental_path=root/'supplemental_cases/cases.json'
    main,supplemental,exports,case_records=validate_exports(exports_path,main_path,supplemental_path,config=config)
    if (training.record(main_path) not in summary['frozen_inputs'] or main['packed_batch'] not in summary['frozen_inputs']
            or main['amendment']!=request['amendment'] or main['old_manifest']!=request['legacy_manifest']):
        raise ValueError('trajectory did not score these exact amended main cases')
    records.extend(case_records)
    amendment=training.read_json(main['amendment']['path'])
    piece_ids=sorted({token for split in ('training','test') for occurrence in amendment['alignment'][split]['occurrences']
                      for arm in legacy.ARMS for token in occurrence[arm+'_ids']})
    if (not piece_ids or piece_ids!=supplemental['selection']['union_piece_ids']
            or any(type(i) is not int or not 0<=i<config.vocab_size for i in piece_ids)):
        raise ValueError('selected rows differ from actual amended native word pieces')
    pairs,endpoints=legacy._selected_pairs(summary)
    specs=checkpoint.tensor_manifest(config)
    roots={arm:Path(fresh['plan']['arm_roots'][arm]) for arm in legacy.ARMS}
    if roots!={'original':Path(request['legacy_root']),'replacement':root}:
        raise ValueError('actual original/replacement source roots were relabeled')
    for pair in pairs:
        for arm in legacy.ARMS:
            legacy._verify_checkpoint(pair[arm],verified,roots[arm],arm,specs)
    forbidden=[summary_path.parent,main_path.parent,supplemental_path.parent,exports_path.parent,
               *(roots[arm]/arm/'checkpoints' for arm in legacy.ARMS)]
    if (output==root or output in root.parents
            or any(p==output or p in output.parents for p in forbidden)):
        raise ValueError('plan output overlaps preserved inputs')
    interventions=canonical([item for pair in pairs for item in legacy._interventions(pair,specs,piece_ids)])
    records=list({r['path']:r for r in records}.values())
    training.verify_records([*records,*SOURCES])
    output.mkdir()
    result=dict(format=FORMAT,complete=True,patches_materialized=False,goal_completion_claimed=False,
        completion_meaning='Plan and prevalidated native exports only; no intervention has executed.',
        analysis_summary=training.record(summary_path),frozen_inputs=records,implementation=SOURCES,
        config=asdict(config),arm_roots={k:str(v) for k,v in roots.items()},
        selected_steps=[p['original']['step'] for p in pairs],
        step_selection='Earliest and latest common positive optimizer steps, deduplicated; never unequal endpoints.',
        endpoint_context={arm:endpoints[arm] for arm in legacy.ARMS},
        word_piece_ids=piece_ids,interventions=interventions,exports=exports,
        original_reused_not_rerun=True,
        limitations=['Donor transfer tests functional contribution, not a unique physical storage location.',
            'Whole branches include pre-LayerNorm and biases; output-write transfers include projection bias.',
            'Preserve both words absolute probabilities and unrelated/shared-piece damage, not only preference ratios.',
            'Inputs/checkpoints must be revalidated before each materialization and native execution.'])
    training.publish(output/'plan.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','summary','exports','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    prepare(args.root,args.summary,args.exports,args.output)


if __name__=='__main__':
    main()
