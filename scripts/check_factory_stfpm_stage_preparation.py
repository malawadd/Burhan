"""L6 metadata/NumPy/fake-compiled fixture receipt; no source-image decode or NN."""
import io,json,time,unittest
from diagnose_factory_stfpm_stages import ROOT,OUT,PINS,sha,write_new


def main():
    path=OUT/'preparation-fixtures.json'
    if path.exists():raise ValueError('Preserve initial L6 fixture receipt')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_stfpm_stage_diagnosis.py')
    stream=io.StringIO();tick=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pins={p:sha(ROOT/p) for p in [*PINS,'tests/test_factory_stfpm_stage_diagnosis.py','scripts/check_factory_stfpm_stage_preparation.py']}
    value={'all_checks_passed':result.wasSuccessful() and result.testsRun>=6,'tests':result.testsRun,'elapsed_seconds':time.perf_counter()-tick,
           'source_pins':pins,'test_log':stream.getvalue(),'ONNX_metadata_only':True,'source_images_decoded':0,'Torch_weights_loaded':0,'NN_executed':False}
    write_new(path,value);print(json.dumps(value,indent=2))
    if not value['all_checks_passed']:raise ValueError('L6 preparation fixtures failed')


if __name__=='__main__':main()
