"""Versioned bounded real-wall data preparation, no neural inference or fit.

Freeze before source labels/pixels. Original18 diagnostic capture blocks are
excluded; additional train/cal blocks are disjoint. Similarity flags quarantine
training endpoints conservatively rather than declaring independent scenes.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,re,struct,time,zlib
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import xml.etree.ElementTree as ET
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction/finishes-expansion'
OUT=BASE/'mbdd-wall-five-class-v1'
INV=BASE/'source-records/mbdd2025-publisher-inventory.json'
DIAGNOSTIC=BASE/'mbdd2025-wall-photos/asset-manifest.json'
CLASSES=['crack','leakage','abscission','corrosion','bulge']
SEED='mbdd-five-class-blocks-20261020:'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(value.encode()).hexdigest()
def write_once(path,value):
    blob=(json.dumps(value,indent=2)+'\n').encode()
    if path.exists() and path.read_bytes()!=blob:raise RuntimeError('Preserve frozen dataset artifact: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
def block_id(name):
    match=re.fullmatch(r'Hefei(\d+)\.[^.]+',Path(name).name)
    if not match:raise RuntimeError('Unrecognized source identity; do not guess capture group: '+name)
    return (int(match.group(1))-1)//200


def byte_range(url,start,end):
    for attempt in range(5):
        try:
            request=Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'InspectFlow-research/1.0'})
            with urlopen(request,timeout=90) as response:
                if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise RuntimeError('Range denied; refusing wholearchive')
                blob=response.read(end-start+2)
            if len(blob)!=end-start+1:raise RuntimeError('Truncated source range')
            return blob
        except HTTPError as error:
            if error.code!=429 or attempt==4:raise
            pause=max(20,min(60,int(error.headers.get('Retry-After','30'))))
            print('Publisher429; bounded backoff',pause,flush=True);time.sleep(pause)


def unpack(entry,blob,base):
    offset=entry['header_offset']-base
    header=blob[offset:offset+30]
    if header[:4]!=b'PK\x03\x04':raise RuntimeError('OriginalZIPheader differs')
    nl,el=struct.unpack_from('<HH',header,26)
    payload=blob[offset+30+nl+el:offset+30+nl+el+entry['compressed_bytes']]
    raw=zlib.decompress(payload,-15) if entry['compression']==8 else payload if entry['compression']==0 else None
    if raw is None or len(raw)!=entry['bytes'] or f'{zlib.crc32(raw):08x}'!=entry['crc32']:
        raise RuntimeError('OriginalmemberCRC/size differs')
    return raw


def parsed(entry,raw):
    root=ET.fromstring(raw);size=root.find('size');boxes=[]
    for obj in root.findall('object'):
        label=obj.findtext('name')
        if label not in CLASSES:raise RuntimeError('Unexpected source class')
        b=obj.find('bndbox');p=[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
        boxes.append({'label':label,'class_id':CLASSES.index(label),'source_xyxy':p})
    return {'source_name':Path(entry['name']).stem+'.jpg','source_xml_filename':root.findtext('filename'),
            'annotation_member':entry['name'],'annotation_sha256':hashlib.sha256(raw).hexdigest(),
            'frame_size_wh':[int(size.findtext('width')),int(size.findtext('height'))],
            'capture_block_200':block_id(entry['name']),'boxes':boxes}


def visual_hash(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tiny=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    low=cv2.dct(tiny)[:8,:8].ravel()[1:]
    return ''.join('1' if x>float(np.median(low)) else '0' for x in low)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','acquire'],required=True);args=parser.parse_args()
    cv2.setNumThreads(1)
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    OUT.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(INV.read_text(encoding='utf-8'));entries={r['name']:r for r in inventory['entries']};all_offsets=sorted(set(r['header_offset'] for r in entries.values()))
    xml=[r for r in entries.values() if '/Annotations/' in r['name'] and r['name'].endswith('.xml')]
    first=min(r['header_offset'] for r in xml);last=max(r['header_offset'] for r in xml);end=min(o for o in all_offsets if o>last)-1
    diagnostic=json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    forbidden=sorted({block_id(r['annotation_member']) for r in diagnostic['records']})
    blocks=sorted({block_id(r['name']) for r in xml}-set(forbidden),key=lambda b:digest(SEED+'block:'+str(b)))
    cut=int(len(blocks)*.75);split_blocks={'train':blocks[:cut],'calibration':blocks[cut:]}
    protocol_path=OUT/'protocol-before-source-labels-pixels.json'
    if args.stage=='freeze':
        protocol={'schema_version':1,'source':'https://zenodo.org/records/15622584','license':'CC BY4.0','source_inventory_sha256':sha(INV),
                  'protected_diagnostic_manifest_sha256':sha(DIAGNOSTIC),'seed':SEED,'script_sha256':sha(Path(__file__)),
                  'classes':CLASSES,'annotation_range':{'start':first,'end':end,'bytes':end-first+1,'xml_members':len(xml)},
                  'capture_group_proxy':'NumericHefeiID contiguous200-image blocks; entireblock excluded/disjoint, not proven physical-site/sessionIDs',
                  'diagnostic_blocks_excluded':forbidden,'split_blocks':split_blocks,
                  'selection':'Within eachrole first40TRAIN/10CAL originalimages containing eachpublisherclass by SHA256(seed+role+source_name); uniondeduplicated. No further backfill after similarity quarantine.',
                  'maximum_original_image_download_bytes':100_000_000,'maximum_fit_images_before_quarantine':200,'maximum_cal_images_before_quarantine':50,
                  'quarantine':'OriginalSHAduplicates OR63bitDCTpHash Hamming<=6 acrossTRAIN-CAL/TRAIN-diagnostic quarantineTRAIN endpoint; CAL-diagnostic quarantinesCAL. Samephysicalscene may remain undetected; no manualhash-flag override.',
                  'normal_scope':'Count all emptyXMLs but do not assume healthy/certifiednegative from absence. Perclass absenttarget flags on otherdefects are not truehealthyFPR. No DACLwetspot-labelled photo used as moisture negative.',
                  'material_scope':'Publisherleakage is visiblemoisture/stainappearance, not watermeasurement/hiddenleak. Abscission broadmaterialdetachment, notpaint-specific. Bulge2Dappearance, not measureddepth.',
                  'status':'DATA ONLY, no NN/fit. Parent reviews actualQA +independent sourcecontract before TRAINprobe; rootfitdecision later.',
                  'proposed_model':'Distinct5class YOLO640. Reset ALL5semanticclassifierchannels; retain frozenlocalbackbone/boxreg sourceidentity. CPU1max4TRAINsteps capacityprobe then explicitfixed4epoch decision; no current fit authorized by this dataset.'}
        if end-first+1>8_000_000:raise RuntimeError('Annotationdownload exceeds declared8MB bound')
        write_once(protocol_path,protocol);print('FROZEN',sha(protocol_path),'XMLrangebytes',end-first+1,flush=True);return
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for path,expected in [(INV,protocol['source_inventory_sha256']),(DIAGNOSTIC,protocol['protected_diagnostic_manifest_sha256']),(Path(__file__),protocol['script_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('Frozen wall data source differs')
    archive=inventory['publisher']['files'][0]
    cache=OUT/'original-annotation-range.bin'
    if not cache.exists():cache.write_bytes(byte_range(archive['links']['self'],first,end))
    annotation_blob=cache.read_bytes()
    if len(annotation_blob)!=end-first+1:raise RuntimeError('Annotationrange length differs')
    rows=[]
    for entry in xml:
        raw=unpack(entry,annotation_blob,first);row=parsed(entry,raw)
        target=OUT/'original/Annotations'/Path(entry['name']).name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=raw:raise RuntimeError('Original annotation changed')
        target.write_bytes(raw);row['annotation']=target.relative_to(OUT).as_posix();rows.append(row)
    chosen=[]
    for split,group in protocol['split_blocks'].items():
        candidates=sorted([r for r in rows if r['capture_block_200'] in group],key=lambda r:digest(SEED+split+r['source_name']))
        selected={}
        for label in CLASSES:
            matching=[r for r in candidates if any(b['label']==label for b in r['boxes'])]
            for row in matching[:40 if split=='train' else 10]:selected[row['source_name']]=row
        chosen.extend({'split':split,**r} for r in selected.values())
    for row in chosen:
        name='MBDD2025/JPEGImages/'+row['source_name']
        if name not in entries:raise RuntimeError('Original source photo missing')
        row['image_member']=name
    total=sum(entries[r['image_member']]['compressed_bytes']+30+len(r['image_member'])+65535 for r in chosen)
    if total>protocol['maximum_original_image_download_bytes']:raise RuntimeError('Boundedselectedphoto download exceeds100MB')
    selection={'source_protocol_sha256':sha(protocol_path),'annotation_range_sha256':sha(cache),'parsed_original_xml':len(rows),
               'empty_original_xml':sum(not r['boxes'] for r in rows),'bounded_image_range_bytes_upper':total,'selected':chosen}
    write_once(OUT/'selection-before-photo-acquisition.json',selection)
    def acquire(row):
        entry=entries[row['image_member']];start=entry['header_offset'];finish=min(o for o in all_offsets if o>start)-1
        image=OUT/'original/JPEGImages'/row['source_name'];image.parent.mkdir(parents=True,exist_ok=True)
        if image.exists():raw=image.read_bytes()
        else:raw=unpack(entry,byte_range(archive['links']['self'],start,finish),start)
        if len(raw)!=entry['bytes'] or f'{zlib.crc32(raw):08x}'!=entry['crc32']:raise RuntimeError('OriginalimageCRC/size differs')
        if not image.exists():image.write_bytes(raw)
        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if list(frame.shape[1::-1])!=row['frame_size_wh']:raise RuntimeError('XML/rawframe geometry differs')
        for b in row['boxes']:
            x1,y1,x2,y2=b['source_xyxy'];w,h=row['frame_size_wh']
            if not np.isfinite(b['source_xyxy']).all() or not(0<=x1<x2<=w and 0<=y1<y2<=h):raise RuntimeError('Invalidpublisherbox')
        return {**row,'image':image.relative_to(OUT).as_posix(),'image_sha256':hashlib.sha256(raw).hexdigest(),'phash63':visual_hash(frame),'image_bytes':len(raw)}
    acquired=[]
    # Low-concurrency network work; no neural job, no capacity timings.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(acquire,chosen):
            acquired.append(row)
            if len(acquired)%20==0:print('verified wallphotos',len(acquired),'/',len(chosen),flush=True)
    known=[]
    diagnostic_root=DIAGNOSTIC.parent
    for row in diagnostic['records']:
        image=diagnostic_root/row['image'];frame=cv2.imread(str(image))
        known.append({'split':'diagnostic','source_name':row['filename'],'image_sha256':row['image_sha256'],'phash63':visual_hash(frame)})
    flags=[];quarantine=set()
    for i,a in enumerate(acquired):
        refs=[r for r in acquired[i+1:] if r['split']!=a['split']]+known
        for b in refs:
            distance=sum(x!=y for x,y in zip(a['phash63'],b['phash63']))
            exact=a['image_sha256']==b['image_sha256']
            if exact or distance<=6:
                endpoint=a if a['split']=='train' or b['split']=='diagnostic' else b
                quarantine.add(endpoint['source_name'])
                flags.append({'a':a['source_name'],'a_split':a['split'],'b':b['source_name'],'b_split':b['split'],'exact':exact,'phash_hamming':distance,'quarantined':endpoint['source_name']})
    for row in acquired:row['status']='quarantined_crosssplit_similarity' if row['source_name'] in quarantine else 'eligible_pending_independent_review'
    report={'protocol_sha256':sha(protocol_path),'selection_sha256':sha(OUT/'selection-before-photo-acquisition.json'),
            'source':'https://zenodo.org/records/15622584','license':'CC BY4.0','classes':CLASSES,'empty_original_xml':selection['empty_original_xml'],
            'downloaded_photos':len(acquired),'original_image_bytes':sum(r['image_bytes'] for r in acquired),'annotation_range_bytes':len(annotation_blob),
            'diagnostic_18_preserved_and_excluded':True,'crosssplit_flags':flags,'quarantined':sorted(quarantine),'records':acquired,
            'status':'DATA ONLY pendingindependentlabel/source/visualreview; noTRAINprobe/fit yet',
            'scope':'Numericcaptureblocks +conservativesimilarityquarantine; unflaggeddoesnotprovephysicalindependence. NohealthyFPRclaim.'}
    write_once(OUT/'asset-manifest.json',report)
    print(json.dumps({'manifest':str(OUT/'asset-manifest.json'),'sha256':sha(OUT/'asset-manifest.json'),'downloaded':len(acquired),'quarantined':len(quarantine),
                      'image_bytes':report['original_image_bytes'],'emptyXML':report['empty_original_xml']},indent=2),flush=True)


if __name__=='__main__':main()
