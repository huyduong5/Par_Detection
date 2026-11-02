"""Compare against official OpenPAR function without importing its training code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from eopar.eval.metrics import par_metrics

URL = 'https://raw.githubusercontent.com/Event-AHU/OpenPAR/main/MambaPAR_Empirical_Study/tools/function.py'


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source = urlopen(URL, timeout=30).read()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_pedestrian_metrics')
    scope = dict(np=np, EasyDict=AttrDict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), URL, 'exec'), scope)
    rng = np.random.default_rng(42)
    cases = []
    for n in (1, 7, 101):
        for kind in ('random', 'zero_predictions', 'threshold_boundary'):
            labels = rng.integers(0, 2, (n, 57))
            probs = rng.random((n, 57)) if kind == 'random' else np.full((n, 57), 0 if kind == 'zero_predictions' else .5)
            ours = par_metrics(probs, labels)
            ref = scope['get_pedestrian_metrics'](labels, probs, threshold=.5)
            pairs = dict(mA='ma', instance_accuracy='instance_acc', instance_precision='instance_prec',
                         instance_recall='instance_recall', instance_f1_mean='instance_f1')
            diffs = {key: abs(ours[key] - ref[value]) for key, value in pairs.items()}
            if max(diffs.values()) > 1e-10:
                raise AssertionError(diffs)
            cases.append(dict(n=n, kind=kind, absolute_differences=diffs))
    Path(args.output).write_text(json.dumps(dict(url=URL, sha256=hashlib.sha256(source).hexdigest(),
         threshold=.5, decision='strict >', cases=cases, status='PASS'), indent=2))


if __name__ == '__main__':
    main()
