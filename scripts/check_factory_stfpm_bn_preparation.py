"""Run hardware-free L4 state/selection plus unchanged numerical/search fixtures."""
import io,json,time,unittest
from factory_stfpm_bn_common import ROOT,OUT,IMPLEMENTATIONS,DEPENDENCIES,sha,write_new


def main():
    path=OUT/'preparation-fixtures.json'
    if path.exists():raise ValueError('Preserve first L4 fixture receipt')
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    for pattern in ['test_factory_stfpm_bn_preparation.py','test_factory_stfpm_preparation.py']:
        suite.addTests(loader.discover(str(ROOT/'tests'),pattern=pattern))
    stream=io.StringIO();started=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pins={f'scripts/{p}':sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS}
    pins.update({p:sha(ROOT/p) for p in DEPENDENCIES})
    for relative in ['scripts/check_factory_stfpm_bn_preparation.py','tests/test_factory_stfpm_bn_preparation.py','tests/test_factory_stfpm_preparation.py']:
        pins[relative]=sha(ROOT/relative)
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun>=17,'tests':result.testsRun,
           'elapsed_seconds':time.perf_counter()-started,'test_log':stream.getvalue(),'source_pins':pins,
           'images_decoded':0,'weights_loaded':0,'NN_executed':False}
    write_new(path,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('Pure L4 preparation failed')


if __name__=='__main__':main()
