"""Run pure CPU numeric/AST fixtures and bind successful source identities. No NN."""
from pathlib import Path
import hashlib
import importlib.util
import json
import unittest
from factory_c_density_common import ROOT,OUT,NAME,IMPLEMENTATIONS


def main():
    test=ROOT/'tests/test_factory_c_density_preparation.py'
    spec=importlib.util.spec_from_file_location('c_density_preparation_tests',test)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    files=[f'scripts/{name}' for name in IMPLEMENTATIONS]+['tests/test_factory_c_density_preparation.py']
    report={'all_checks_passed':result.wasSuccessful(),'tests_run':result.testsRun,
            'source_pins':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
            'scope':'Pure AST source-operation parity and synthetic numeric identities only; no Torch/OpenVINO import,NN/data decode/optimizer/GPU/accuracy benchmark',
            'nn_started':False,'dataset_pixels_opened':False}
    path=OUT/(NAME+'-preparation-fixtures.json')
    if not result.wasSuccessful():
        raise SystemExit('C-density preparation fixtures failed; no protocol/NN allowed')
    if path.exists() and json.loads(path.read_text())!=report:
        if (OUT/(NAME+'-protocol-before-fit.json')).exists():
            raise ValueError('Never replace source-bound fixtures after protocol freeze')
        old=path.read_bytes()
        archive=OUT/(NAME+'-preparation-fixtures-preserved-'+hashlib.sha256(old).hexdigest()[:16]+'.json')
        if archive.exists() and archive.read_bytes()!=old:raise ValueError('Fixture archive collision')
        if not archive.exists():archive.write_bytes(old)
        path.unlink()
    if not path.exists():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'fixture_receipt':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tests_run':result.testsRun}))


if __name__=='__main__':main()
