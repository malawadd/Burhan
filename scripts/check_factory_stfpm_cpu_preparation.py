"""L5 pure source, CPU opt-in and exact legacy-default checks; no NN."""
import io,json,time,unittest
from factory_stfpm_cpu_common import ROOT,OUT,IMPLEMENTATIONS,DEPENDENCIES,sha,write_new


def main():
    path=OUT/'preparation-fixtures.json'
    if path.exists():raise ValueError('Preserve first L5 fixture receipt')
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    patterns=['test_factory_cpu_metadata.py','test_factory_stfpm_cpu_preparation.py','test_factory_precision_metadata.py','test_factory_winograd_metadata.py','test_factory_stfpm_preparation.py']
    for pattern in patterns:suite.addTests(loader.discover(str(ROOT/'tests'),pattern=pattern))
    stream=io.StringIO();started=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pins={f'scripts/{p}':sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS};pins.update({p:sha(ROOT/p) for p in DEPENDENCIES})
    for relative in ['scripts/check_factory_stfpm_cpu_preparation.py',*[f'tests/{p}' for p in patterns]]:pins[relative]=sha(ROOT/relative)
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun>=24,'tests':result.testsRun,'elapsed_seconds':time.perf_counter()-started,
           'source_pins':pins,'test_log':stream.getvalue(),'source_images_decoded':0,'weights_loaded':0,'NN_executed':False,'fake_compile_only':True}
    write_new(path,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('L5 pure preparation failed')


if __name__=='__main__':main()
