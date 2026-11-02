"""CPU-testable structural pretrained validation."""
def checked_state(module, state, allowed_missing=()):
    expected = module.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    bad = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing - set(allowed_missing) or unexpected or bad:
        raise ValueError(f'Pretrained incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={bad}')
    module.load_state_dict(state, strict=False)
