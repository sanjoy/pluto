"""Case/spelling provenance shared by the causal readouts.

Do not extend a two-candidate set to four words: each context has exactly its
case-matching original/replacement pair. Title and lowercase variants remain
separate strata even when their causal input prefix is byte-identical.
Legacy artifacts keep their title-only interpretation. Amended artifacts must
explicitly bind both literal substitutions and identify each candidate's role.
"""

PAIRS={'title':{'original':'Exeunt','replacement':'Nuveth'},
       'lowercase':{'original':'exeunt','replacement':'nuveth'}}
LEGACY=('pluto-paired-word-cases-v1','pluto-paired-supplemental-cases-v1')
AMENDED=('pluto-paired-lowercase-word-cases-v1','pluto-paired-lowercase-supplemental-cases-v1')


def require(value,message):
    if not value: raise ValueError(message)


def validate_plan(plan,register,read_json):
    """Register actual evidence through the caller's before/after hash ledger.

    This validates the spelling/provenance contract, not packed geometry or the
    entire native corpus alignment. Callers additionally validate target bytes,
    prefix hashes, teacher forcing, and execution-time input identities.
    """
    fmt=plan.get('format')
    require(fmt in LEGACY+AMENDED,'unsupported frozen case format')
    amended=fmt in AMENDED
    if amended:
        require(plan.get('complete') is True and plan.get('candidate_pairs')==PAIRS,
                'amended cases need explicit case-matching candidate pairs')
        old_record=plan['old_manifest']; amendment_record=plan['amendment']
        register(old_record['path'],old_record); register(amendment_record['path'],amendment_record)
        amendment=read_json(amendment_record['path'])
        require(amendment.get('format')=='pluto-paired-lowercase-amendment-v1'
                and amendment.get('complete') is True and amendment.get('old_manifest')==old_record
                and amendment.get('replacement_rules')==[{'source':'Exeunt','target':'Nuveth'},
                                                          {'source':'exeunt','target':'nuveth'}],
                'amended replacement policy/provenance differs')
        for item in [*plan.get('provenance',[]),*plan.get('sources',[])]:
            register(item['path'],item)
    else:
        require(plan.get('candidate_words')==PAIRS['title'],'unexpected title-only candidate pair')
        old_record=plan['manifest']; register(old_record['path'],old_record)
    manifest=read_json(old_record['path'])
    require(manifest.get('format')=='pluto-paired-corpus-training-v1'
            and manifest.get('replacement',{}).get('from')=='Exeunt'
            and manifest['replacement'].get('to')=='Nuveth'
            and manifest['replacement'].get('case_sensitive',True) is True,
            'unexpected original paired-corpus manifest')
    for item in plan.get('source_inputs',[]): register(item['path'],item)
    for name in ('source_word_cases','source_word_packed_batch'):
        if name in plan: register(plan[name]['path'],plan[name])
    return amended


def metadata(case,*,amended=False):
    """Normalize labels without inferring a lowercase case in a legacy plan."""
    if case['kind'] not in ('word','word_next_native'):
        require(case.get('spelling_variant') is None and case.get('candidate_pair') is None,
                'control may not declare a word candidate pair')
        return dict(spelling_variant=None,candidate_pair=None,
                    target_source_domain=case.get('target_source_domain','original'))
    variant=case.get('spelling_variant',None if amended else 'title')
    require(variant in PAIRS and (amended or variant=='title'),'invalid spelling variant')
    pair=PAIRS[variant]
    require(case.get('candidate_pair',None if amended else pair)==pair,
            'wrong or missing case-matching candidate pair')
    require(case['target'] in pair.values(),'word identity: target spelling does not match its variant')
    role=next(role for role,word in pair.items() if case['target']==word)
    require(case.get('target_source_domain',None if amended else role)==role,
            'target spelling disagrees with source corpus role')
    return dict(spelling_variant=variant,candidate_pair=pair,target_source_domain=role)


def experiment_identity(plan):
    return plan['amendment'] if plan['format'] in AMENDED else plan['manifest']
