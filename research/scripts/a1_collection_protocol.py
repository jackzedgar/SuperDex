"""Keep frozen collection initialization independent from policy training provenance.

This contract cannot qualify a policy. Callers must still verify the quality
passport, its checkpoint hash, frozen probe and task membership before writing.
"""
FROZEN_COLLECTION_PROTOCOL={'reset_mode':'uniform_reference','reset_memory_mode':'original','policy_noise_sigma':0.1}

def formal_collection_reset_mode(checkpoint,passport):
    cfg=checkpoint.get('configuration',{})
    training_mode=cfg.get('reset_mode','adaptive_anchor')
    version=passport.get('schema_version',1)
    if version==1:
        assert training_mode=='uniform_reference' and passport.get('reset_mode')=='uniform_reference', 'Legacy passports require matching uniform-reference training; new training needs explicit version2 provenance'
    elif version==2:
        assert passport.get('collection_protocol')==FROZEN_COLLECTION_PROTOCOL, 'Frozen collection protocol cannot be changed by a policy passport'
        assert passport.get('training_reset_mode')==training_mode, 'Training provenance must match the checkpoint'
        assert passport.get('reset_mode')=='uniform_reference', 'Compatibility reset_mode field describes collection'
        assert training_mode in ('uniform_reference','formal_episode_start'), 'Unregistered training mode'
        if training_mode=='formal_episode_start':
            assert cfg.get('scientific_revision')==4 and cfg.get('training_memory')=='original', 'Formal-start policy must retain its authorized fourth-revision provenance'
    else:
        raise ValueError('Unsupported quality-passport schema version')
    return 'uniform_reference'
