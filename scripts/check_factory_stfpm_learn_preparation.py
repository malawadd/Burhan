"""L7 pure source/outcome contracts only; no Torch or image/model execution."""
import io,json,time,unittest,ast
from factory_stfpm_learn_common import ROOT,OUT,IMPLEMENTATIONS,DEPENDENCIES,sha,write_new

def main():
    target=OUT/'preparation-fixtures.json'
    if target.exists():raise ValueError('Preserve first formal preparation receipt')
    for name in IMPLEMENTATIONS:ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_stfpm_learn_preparation.py')
    stream=io.StringIO();tick=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    paths=[*[f'scripts/{name}' for name in IMPLEMENTATIONS],*DEPENDENCIES,
        'scripts/check_factory_stfpm_learn_preparation.py','tests/test_factory_stfpm_learn_preparation.py']
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun==9,'tests':result.testsRun,
        'elapsed_seconds':time.perf_counter()-tick,'test_log':stream.getvalue(),'source_pins':{p:sha(ROOT/p) for p in paths},
        'NN_executed':False,'optimizer_updates':0,'weights_loaded':False,'image_decodes':0}
    write_new(target,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('L7 preparation failed')

if __name__=='__main__':main()
