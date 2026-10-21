"""L8 pure fake-compiled/source/lifetime checks; no actual GPU/NN."""
import io,json,time,unittest
from probe_factory_stfpm_tapped import ROOT,OUT,PINS,sha,write_new


def main():
    path=OUT/'preparation-fixtures.json'
    if path.exists():raise ValueError('Preserve first architecture fixture receipt')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_tapped_segmentation.py')
    log=io.StringIO();started=time.perf_counter();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    pins={p:sha(ROOT/p) for p in [*PINS,'scripts/check_factory_stfpm_tapped_preparation.py','tests/test_factory_tapped_segmentation.py']}
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun==8,'tests':result.testsRun,'elapsed_seconds':time.perf_counter()-started,
           'source_pins':pins,'test_log':log.getvalue(),'fake_compile_only':True,'NN_executed':False,'source_images_decoded':0,'Torch_weights_loaded':0}
    write_new(path,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('L8 fixture failed')


if __name__=='__main__':main()
