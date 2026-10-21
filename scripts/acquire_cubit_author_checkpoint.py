"""Bounded public-author checkpoint acquisition and non-executing metadata audit.

No torch.load, model construction, inference, fitting or change to current models.
The dataset declaration is not treated as a separate model-license grant.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib,json,pickletools,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/construction/cubit-author-release-v1'
URL='https://drive.google.com/uc?export=download&id=1Ccup5Xgijl_AOX9HcRZ5zeqQ2kE3oqDs'
LIMIT=32*1024*1024


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'acquisition-protocol-before-download.json'
    if protocol.exists():raise RuntimeError('Preserve existing acquisition attempt; no silent retry')
    write(protocol,{'primary_project':'https://benyunzhao.github.io/CUBIT-Dataset/',
                    'author_shared_nano_folder':'https://drive.google.com/drive/folders/1CSNmnecW-CrG_KnUP6PhUhV7x0-PyBzM',
                    'source_url':URL,'maximum_download_bytes':LIMIT,'maximum_total_zip_uncompressed_bytes':96*1024*1024,
                    'script_sha256':sha(Path(__file__)),'scope':'Acquisition/CRC/SHA and pickletools metadata ONLY; no unpickling, NN or claims about classes/performance/overlap/modelrights.',
                    'license':'CUBIT dataset author declaration CC BY4.0; standalone checkpoint license not established; source Ultralytics lineage AGPL. No commercial clearance inferred.'})
    target=OUT/'author-best.pt';part=OUT/'author-best.pt.partial';count=0
    with urlopen(URL,timeout=60) as response,part.open('xb') as stream:
        length=response.headers.get('Content-Length')
        if length and int(length)>LIMIT:raise RuntimeError('Checkpoint exceeds bound')
        headers={'final_url':response.url,'content_type':response.headers.get('Content-Type'),'content_length':length}
        while block:=response.read(65536):
            count+=len(block)
            if count>LIMIT:raise RuntimeError('Checkpoint exceeds stream bound')
            stream.write(block)
    if not zipfile.is_zipfile(part):raise RuntimeError('Public response is not a PyTorch ZIP checkpoint; preserve response')
    with zipfile.ZipFile(part) as archive:
        infos=archive.infolist()
        if len({info.filename for info in infos})!=len(infos) or sum(info.file_size for info in infos)>96*1024*1024:raise RuntimeError('ZIP metadata exceeds bounded unique-member contract')
        if archive.testzip() is not None:raise RuntimeError('Original checkpoint ZIP CRC failed')
        payloads=[info for info in infos if info.filename.endswith('/data.pkl') or info.filename=='data.pkl']
        if len(payloads)!=1 or payloads[0].file_size>8*1024*1024:raise RuntimeError('Unexpected bounded PyTorch metadata layout')
        operations=list(pickletools.genops(archive.read(payloads[0])))
        globals_used=sorted({str(value) for op,value,_ in operations if op.name=='GLOBAL'})
        windows=[]
        for index,(op,value,pos) in enumerate(operations):
            if value in ['names','nc','task','yaml_file']:
                windows.append({'key':value,'pickle_position':pos,'following_metadata_opcodes':[{'op':o.name,'value':v if isinstance(v,(str,int,float,bool,type(None))) else str(v)[:200]} for o,v,_ in operations[index:index+28]]})
    part.replace(target)
    report={'checkpoint':target.relative_to(ROOT).as_posix(),'sha256':sha(target),'bytes':count,'response':headers,
            'acquisition_protocol_sha256':sha(protocol),'zip_crc_pass':True,'zip_members':len(infos),
            'pickle_global_names_nonexecuting':globals_used,'class_and_task_opcode_windows':windows,
            'safety':'pickletools only inspects opcodes; no pickle was executed or model/weights loaded.',
            'license_status':'Standalone checkpoint license remains unverified; no inference or performance claim.'}
    write(OUT/'source-metadata-audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
