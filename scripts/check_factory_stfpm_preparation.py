"""Record only hardware-free NumPy/AST/source-identity L preparation fixtures."""
import io,json,time,unittest
from factory_stfpm_common import ROOT,OUT,IMPLEMENTATIONS,DEPENDENCIES,sha,write_new


def main():
    path=OUT/'preparation-fixtures-v2-score-semantics.json'
    if path.exists():raise ValueError('Preserve first fixture receipt')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_stfpm_preparation.py')
    stream=io.StringIO();started=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pins={f'scripts/{p}':sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS}
    pins.update({p:sha(ROOT/p) for p in DEPENDENCIES})
    pins['tests/test_factory_stfpm_preparation.py']=sha(ROOT/'tests/test_factory_stfpm_preparation.py')
    pins['scripts/check_factory_stfpm_preparation.py']=sha(__file__)
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun>=9,'tests':result.testsRun,
           'elapsed_seconds':time.perf_counter()-started,'test_log':stream.getvalue(),'implementation_pins':pins,
           'images_decoded':0,'weights_loaded':0,'NN_executed':False}
    write_new(path,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('Pure preparation failed')


if __name__=='__main__':main()
