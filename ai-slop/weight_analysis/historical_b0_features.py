"""Inspect distributed B0 feature support in two already-recorded Exeunt traces.

The diagnostic contracts captured GELU values with BF16 decoder rows and a
fixed, context-specific final-LayerNorm readout direction. It does NOT rerun the
transformer, predict an ablation effect, or treat a feature as a unique lexical
memory. Other-position scores use that SAME direction only as a diagnostic;
they are not those positions' actual vocabulary probabilities or logits.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint
from .historical_b0_diagnostic import record
from .phrase_reference import bf16


def support_summary(terms):
    """Retain opposing mass; positive-mass fractions are not causal fractions."""
    terms=np.asarray(terms,dtype=np.float64)
    if terms.ndim != 1 or not terms.size or not np.isfinite(terms).all():
        raise ValueError('need finite signed feature terms')
    positive=np.sort(terms[terms>0])[::-1]
    pos=float(positive.sum()); neg=float(terms[terms<0].sum()); net=float(terms.sum())
    order=np.argsort(-terms,kind='stable')
    return dict(positive_count=int(np.count_nonzero(terms>0)),negative_count=int(np.count_nonzero(terms<0)),
        zero_count=int(np.count_nonzero(terms==0)),positive_mass=pos,negative_mass=neg,net=net,
        top_positive=[dict(neuron=int(i),term=float(terms[i])) for i in order[:8] if terms[i]>0],
        positive_mass_fractions={str(k):float(positive[:k].sum()/pos) if pos else None for k in (1,8,32)},
        positive_neurons_to_reach={str(fraction):int(np.searchsorted(np.cumsum(positive),fraction*pos))+1
                                  if pos else None for fraction in (.5,.9)})


def activation_ranking(activations,ids,selected_row,neuron):
    """Activation ranks within this one saved window; no same-token extrapolation."""
    a=np.asarray(activations,dtype=np.float64);ids=np.asarray(ids)
    if (a.ndim!=2 or len(ids)!=len(a) or ids.ndim!=1 or not np.isfinite(a).all()
            or type(selected_row) is not int or not 0<=selected_row<len(a)
            or type(neuron) is not int or not 0<=neuron<a.shape[1]):
        raise ValueError('invalid captured activation geometry')
    values=a[:,neuron]; value=values[selected_row]
    order=np.argsort(-values,kind='stable')[:6]
    return dict(neuron=neuron,selected_activation=float(value),
        rank_by_strictly_greater=1+int(np.count_nonzero(values>value)),
        same_current_token_other_positions=[int(i) for i in np.flatnonzero(ids==ids[selected_row]) if i!=selected_row],
        largest_activations=[dict(position=int(i),activation=float(values[i]),input_token_id=int(ids[i]),
                                 preceding_ids=ids[max(0,i-10):i+1].tolist()) for i in order])


def analyze(manifest_path,run_directory):
    manifest_path=Path(manifest_path).resolve(strict=True);root=Path(run_directory).resolve(strict=True)
    manifest=json.loads(manifest_path.read_text());known={x['original_path']:x for x in manifest['files']}
    used={str(manifest_path):record(manifest_path)}
    if manifest.get('complete') is not True or len(known)!=len(manifest['files']):
        raise ValueError('invalid historical evidence manifest')

    def checked(path,expected=None):
        path=Path(path)
        if path.is_symlink() or not path.is_file():raise ValueError('need a nonsymlink artifact')
        actual=record(path);expected=known[str(path)] if expected is None else expected
        if (actual['bytes'],actual['sha256'])!=(expected['bytes'],expected['sha256']):
            raise ValueError('historical evidence changed: '+str(path))
        used[actual['path']]=actual
        return path

    plan=json.loads(checked(root/'plan.json').read_text())
    source_records={x['path']:x for x in plan['inputs']}
    model=GPT2Checkpoint(plan['checkpoint_directory'])
    for spec in model.manifest:
        file=model.directory/spec.filename
        checked(file,source_records[str(file)])
    cases=[]
    for step in (1341,1342):
        directory=root/f'trace_step_{step}'
        m=json.loads(checked(directory/'metadata.json').read_text())
        a=json.loads(checked(root/f'analysis_step_{step}/analysis.json').read_text())
        if (m.get('complete') is not True or a.get('complete') is not True
                or m['checkpoint_directory']!=str(model.directory)
                or m['token_ids']!=a['context_token_ids']
                or m['target_id']!=a['target_id'] or m['selected_row']!=a['ledger']['row']):
            raise ValueError('historical native/ledger identity mismatch')
        row=m['selected_row']; ids=m['token_ids']; target=a['target_id']; competitor=a['competitor_id']

        def array(name,width):
            item=m['files'][name]
            if item['dtype']!='bf16' or item['shape']!=[len(ids),width]:
                raise ValueError('invalid native tensor descriptor')
            path=checked(directory/item['file'])
            if path.stat().st_size!=len(ids)*width*2:raise ValueError('invalid native tensor size')
            raw=np.fromfile(path,dtype='<u2').astype(np.uint32)<<16
            result=raw.view(np.float32).reshape(item['shape']).astype(np.float64)
            if not np.isfinite(result).all():raise ValueError('nonfinite native tensor')
            return result

        gates=array('blocks.0.gelu',model.config.d_ff)
        residual=array('blocks.7.after_mlp',model.config.d_model)[row]
        dictionary=bf16(model.token_embedding[[target,competitor]]).astype(np.float64)
        centered=residual-residual.mean();std=math.sqrt(float(np.mean(centered*centered))+1e-5)
        direction=(dictionary[0]-dictionary[1])*model['final_norm.scale'].astype(np.float64)/std
        direction-=direction.mean()
        decoder=bf16(model['blocks.0.mlp.output.weight']).astype(np.float64)
        coefficients=decoder@direction; terms=gates[row]*coefficients
        expected=np.asarray(a['ledger']['neurons']['blocks.0'])
        if not np.allclose(terms,expected,rtol=1e-12,atol=1e-12):
            raise ValueError('raw-gate/decoder/readout reconstruction disagrees with stored ledger')
        signed=support_summary(terms)
        scores=gates@coefficients; top_positions=np.argsort(-scores,kind='stable')[:6]
        cases.append(dict(generation_step=step,target_id=target,target_piece=a['target_piece'],
            competitor_id=competitor,competitor_piece=a['competitor_piece'],selected_row=row,
            final_native_margin=a['ledger']['native_margin'],b0_neuron_terms=signed,
            reconstructed_terms_max_abs_error=float(np.max(np.abs(terms-expected))),
            b0_output_bias_term=a['ledger']['terms']['blocks.0.mlp_bias'],
            individual_activation_rankings=[activation_ranking(gates,ids,row,item['neuron'])
                                           for item in signed['top_positive'][:5]],
            fixed_direction_group_projection=dict(selected_value=float(scores[row]),
                selected_rank_by_strictly_greater=1+int(np.count_nonzero(scores>scores[row])),
                largest=[dict(position=int(i),projection=float(scores[i]),input_token_id=ids[i],
                              preceding_ids=ids[max(0,i-10):i+1]) for i in top_positions])))
    sources=[record(Path(__file__)),record(Path(__file__).with_name('checkpoint.py')),
             record(Path(__file__).with_name('phrase_reference.py')),
             record(Path(__file__).with_name('historical_b0_diagnostic.py'))]
    for item in used.values():
        if record(item['path'])!=item:raise ValueError('input changed during analysis')
    return dict(format='pluto-historical-b0-features-v1',complete=True,
        goal_completion_claimed=False,new_deterministic_experiment_result=False,cases=cases,
        provenance=list(used.values()),sources=sources,
        limitations=['Two selected native events from one historical generated continuation, not independent validation.',
            'Only one occurrence of each current word-piece token is present in its inspected window; same-token controls are missing.',
            'Neuron terms use a fixed final-LayerNorm target-minus-rival direction, retain cancellation, and exclude bias.',
            'Other-position projections freeze that selected direction; they are not actual logits or probabilities there.',
            'Feature selection uses these same events; high ranks are exploratory, not held-out specificity estimates.',
            'Activation/linear-accounting contributions are not intervention effects or evidence of necessity.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-manifest','run-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():raise FileExistsError(args.output)
    if args.run_directory.resolve() in args.output.resolve().parents:raise ValueError('output overlaps historical run')
    result=analyze(args.evidence_manifest,args.run_directory)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
