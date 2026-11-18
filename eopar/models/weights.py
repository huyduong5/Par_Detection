"""CPU-testable structural pretrained validation."""
def checked_state(module, state, allowed_missing=()):
    expected = module.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    bad = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing - set(allowed_missing) or unexpected or bad:
        raise ValueError(f'Pretrained incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={bad}')
    module.load_state_dict(state, strict=False)


def canonical_vmamba_state(state):
    """Translate the documented legacy VSSM key names without relaxing checks.

    Mirrors VSSM._load_from_state_dict's naming migration. Unknown keys remain
    unknown and checked_state still rejects missing keys and shape mismatches.
    Never overwrite a key when old and new spellings occur together.
    """
    import re
    mapped = {}
    for key, value in state.items():
        target = key
        for old, new in (('patch_embed.proj.', 'patch_embed.0.'),
                         ('patch_embed.norm.', 'patch_embed.2.'),
                         ('norm.', 'classifier.norm.'),
                         ('head.', 'classifier.head.')):
            if target.startswith(old):
                target = new + target[len(old):]
                break
        target = re.sub(r'^(layers\.\d+\.blocks\.\d+)\.ln_1\.', r'\1.norm.', target)
        target = re.sub(r'^(layers\.\d+\.blocks\.\d+)\.self_attention\.', r'\1.op.', target)
        target = re.sub(r'^(layers\.\d+\.downsample)\.norm\.', r'\1.1.', target)
        target = re.sub(r'^(layers\.\d+\.downsample)\.reduction\.', r'\1.2.', target)
        if target in mapped:
            raise ValueError(f'Colliding VMamba checkpoint key: {key} -> {target}')
        mapped[target] = value
    return mapped
