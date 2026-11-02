import argparse
import contextlib
import time
import unittest
from pathlib import Path
from eopar.train.runtime import ROOT, environment_identity, source_hashes, write_json,digest


class Result(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)
    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append(dict(id=test.id(), state='PASS', seconds=time.monotonic() - self.started))
    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append(dict(id=test.id(), state='FAIL', error=self._exc_info_to_string(error, test)))
    def addError(self, test, error):
        super().addError(test, error)
        self.records.append(dict(id=test.id(), state='ERROR', error=self._exc_info_to_string(error, test)))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'eopar/tests'))
    with (out / 'cpu_tests.log').open('w') as log, contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, resultclass=Result, verbosity=2).run(suite)
    write_json(out / 'cpu.json', dict(state='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
        tests=result.testsRun, records=result.records, skips=[(str(t), reason) for t, reason in result.skipped],
        source_hashes=source_hashes(), environment=environment_identity()))
    # Tests are recorded separately from runtime compatibility hashes.
    import json
    report=json.loads((out/'cpu.json').read_text())
    report['test_hashes']={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'eopar/tests').glob('*.py')}
    write_json(out/'cpu.json',report)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
