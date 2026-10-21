"""L9 pure arithmetic/fake compilation/ONNX-byte fixtures, no NN."""
import io,json,time,unittest
from probe_factory_stfpm_partition import ROOT,OUT,PINS,sha,write_new


def main():
    path=OUT/'preparation-fixtures.json'
    if path.exists():raise ValueError('Preserve first formal fixture receipt')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_stfpm_partition*.py')
    log=io.StringIO();started=time.perf_counter();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    files=[*PINS,'scripts/check_factory_stfpm_partition_preparation.py','tests/test_factory_stfpm_partition.py','tests/test_factory_stfpm_partition_independent.py']
    report={'all_checks_passed':result.wasSuccessful() and result.testsRun==15,'tests':result.testsRun,'elapsed_seconds':time.perf_counter()-started,
            'source_pins':{p:sha(ROOT/p) for p in files},'test_log':log.getvalue(),'NN_executed':False,'source_images_decoded':0,'Torch_weights_loaded':0,
            'original_Torch_tensor_arithmetic_and_interpolation_only':True,'network_models_constructed':0,'timing_scope':'Contendedpreparationtests_notFPS'}
    write_new(path,report);print(json.dumps(report,indent=2))
    if not report['all_checks_passed']:raise ValueError('L9 fixture failed')


if __name__=='__main__':main()
