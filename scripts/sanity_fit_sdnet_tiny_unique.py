"""Matched64-update tinyTRAIN control: only forced-anchor assignment changes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,shutil
import sanity_fit_sdnet_tiny as previous
from construction_ssd_background_loss import install as background_install
from construction_ssd_matcher import install as unique_install
import scipy
import scipy.optimize._lsap as assignment_binary
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/evaluation/construction-resumption-fastening-tiny-v1'
OUT=ROOT/'output/evaluation/construction-resumption-fastening-tiny-unique-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    receipt=ROOT/'output/evaluation/sdnet-tiny-unique-control-protocol-before-forward.json'
    if OUT.exists()or receipt.exists():raise ValueError('Preserve matchedcontrol')
    original=json.loads((OLD/'protocol-before-forward.json').read_text())
    if sha(previous.__file__)!=original['script_sha256']:raise ValueError('Original tinytrainer changed')
    if sha(previous.BASE)!=original['pinned_initial_state_sha256']:raise ValueError('Original initialstate changed')
    if sha(previous.DATA)!=original['data_manifest_sha256']:raise ValueError('Correcteddata changed')
    if sha(ROOT/'scripts/construction_ssd_background_loss.py')!=original['background_loss_sha256']:raise ValueError('Backgroundloss changed')
    protocol={'only_changed_training_behavior':'SSDforcedbestanchor overwrite replacedbyrectangularuniqueforcedassignment; priors/graph/inference unchanged',
        'original_protocol_sha256':sha(OLD/'protocol-before-forward.json'),'wrapper_sha256':sha(__file__),'matcher_sha256':sha(ROOT/'scripts/construction_ssd_matcher.py'),
        'scipy_version':scipy.__version__,'scipy_path':scipy.__file__,'assignment_binary_sha256':sha(assignment_binary.__file__),'dependency_policy':'ReusealreadyinstalledprojectSciPy via processlocalappendedpath; noinstallation/envchanges','same8photo_selection':True,'same64updates_batch8_seed_optimizer_lrs_andbackgroundloss':True,'CAL_development_owner_pixels_used':False,'new_model_installed':False}
    receipt.write_text(json.dumps(protocol,indent=2)+'\n');installed=[]
    def install_both(model):
        background_install(model);unique_install(model);installed.append(model);return model
    previous.OUT=OUT;previous.install=install_both;previous.main()
    actual=json.loads((OUT/'protocol-before-forward.json').read_text());schedule=json.loads((OUT/'training-history.json').read_text())['batch_ids'];old_schedule=json.loads((OLD/'training-history.json').read_text())['batch_ids']
    if schedule!=old_schedule or actual['selected_records']!=original['selected_records']:raise ValueError('Controlinputs/schedulediffer')
    if sha(OUT/'initial-state.pt')!=sha(OLD/'initial-state.pt'):raise ValueError('Pinned initialstate differs')
    positive_ids={r['id']for r in actual['selected_records']if r['boxes']};expected=[{'step':step+1,'record_id':identity}for step,batch in enumerate(schedule)for identity in batch if identity in positive_ids]
    diagnostics=installed[0].proposal_matcher.assignment_diagnostics
    if len(expected)!=len(diagnostics):raise ValueError('Unexpected matcher invocations')
    for a,b in zip(expected,diagnostics):a.update(b)
    (OUT/'matcher-coverage-all-updates.json').write_text(json.dumps({'protocol_sha256':sha(receipt),'all_inputs_and_batches_identical_to_control':True,'all_gt_counts_ge1':all(all(n>=1 for n in r['unique_positive_anchor_counts'])for r in expected),'records':expected},indent=2)+'\n')
    for p in [Path(__file__),ROOT/'scripts/construction_ssd_matcher.py',receipt]:shutil.copyfile(p,OUT/p.name)
    print('MATCHED_CONTROL_VERIFIED',len(diagnostics),'positive-imageassignments',flush=True)
if __name__=='__main__':main()
