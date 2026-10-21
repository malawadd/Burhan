"""Author MTD original TRAIN-only acquisition/GT QA; never opens SDD2 TEST.

All1344 author images are assigned TRAIN before pixels. There is no MTD accuracy
split/claim: physical tile identities across illumination are not supplied.
"""
import argparse,collections,hashlib,json,re,sys,time,urllib.request,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/factory/round4/magnetic-tile-author'
META=OUT/'repository-record-before-images.json'
META_SHA='e49a72ac7e02537bfeecef024eefb03841dc4a3ee46f562f92e3fba9535c62d6'
COMMIT='cc4eb530b57b9cb7225bf76023873043ffd50655'
URL=f'https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/{COMMIT}'
PROTOCOL=OUT/'protocol-before-original-pixels.json';ARCHIVE=OUT/f'author-{COMMIT}.zip'
CATEGORIES=['MT_Blowhole','MT_Break','MT_Crack','MT_Fray','MT_Free','MT_Uneven']
EXPECTED={'MT_Blowhole':115,'MT_Break':85,'MT_Crack':57,'MT_Fray':32,'MT_Free':952,'MT_Uneven':103}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def selection():
    if sha(META)!=META_SHA:raise ValueError('Originalprepixel repository metadata changed')
    metadata=json.loads(META.read_text());tree=metadata['tree']
    if tree['truncated'] or metadata['commit']['sha']!=COMMIT:raise ValueError('Exactfullcommit inventory required')
    blobs={v['path']:v for v in tree['tree'] if v['type']=='blob' and v['path'].startswith('MT_')}
    pairs=[]
    for path,row in sorted(blobs.items()):
        match=re.fullmatch(r'(MT_[A-Za-z]+)/Imgs/(exp([0-9]+)_num_([0-9]+))\.jpg',path)
        if not match:continue
        category,stem,exp,num=match.groups();mask=path[:-4]+'.png'
        if category not in CATEGORIES or mask not in blobs:raise ValueError('Authorimage/mask pair missing')
        pairs.append({'source_image':path,'source_mask':mask,'category':category,'stem':stem,
            'illumination_filename_prefix':'exp'+exp,'filename_number':num,'split':'train',
            'image_git_blob_sha1':row['sha'],'mask_git_blob_sha1':blobs[mask]['sha'],
            'image_bytes':row['size'],'mask_bytes':blobs[mask]['size']})
    if len(pairs)!=1344 or dict(collections.Counter(v['category'] for v in pairs))!=EXPECTED or len(blobs)!=2688:raise ValueError('Exact1344pairs/categorycounts required')
    if len({v['source_image'].casefold() for v in pairs})!=1344:raise ValueError('Casefoldpath collision')
    return metadata,pairs


def protocol_value():
    metadata,pairs=selection()
    return {'name':'MTD_author_allTRAIN_round4','author_repository':'https://github.com/abin24/Magnetic-tile-defect-datasets.',
        'author_commit':COMMIT,'archive_url':URL,'repository_metadata_sha256':META_SHA,
        'source_script_sha256':sha(Path(__file__)),'selection_before_pixels':pairs,'counts':EXPECTED,
        'assignment':'ALL1344sourceimages TRAIN ONLY; nointernalMTDvalidation/benchmark/heldoutclaim',
        'group_limit':'Authorsphotographsame tile undermultipleillumination; exp prefix/filename timestamp isNOT verifiedphysicaltileID. Imageuniqueness doesnotprovephysicalindependence.',
        'rights':{'author_README':metadata['README'],'formal_license_file_present':any(v['path'].lower().startswith('license') for v in metadata['tree']['tree']),
            'scope':'Author welcomes dataset research use with citation. No formal commercial license established; research-only, no commercialdeployment claim.',
            'paper':'https://doi.org/10.1007/s00371-018-1588-5'},
        'bounds':{'archive_max_bytes':100*1024**2,'total_uncompressed_max_bytes':100*1024**2,'selected_original_bytes':51952426},
        'mask_QA':'Expectedpairednativegrayscalebinary0background/255foreground (1allowedifactualbinary); rejectunknowncodes/geometry/categorycontradiction pendingreview. No labeldilation/resize/crop.',
        'visual_QA':'Firstsource byfilename forfirst3availableexp prefixes percategory; actualRAW/publisherGT/overlay only, no predictedmask or modelworking claim.',
        'SDD2_reserved466_or_official_test_access':False,'NN_executed':False,'new_training_started':False}


def verify_protocol():
    value=json.loads(PROTOCOL.read_text())
    if value!=protocol_value():raise ValueError('Frozenauthorselection/source/rights changed')
    return value


def acquire(protocol):
    if not ARCHIVE.exists():
        temporary=ARCHIVE.with_suffix('.zip.partial')
        if temporary.exists():raise ValueError('Preserve partialdownload; review before separatelyversioned retry')
        with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'LocalResearchDatasetProvenance'}),timeout=90) as response,temporary.open('xb') as stream:
            total=0
            while chunk:=response.read(1024**2):
                total+=len(chunk)
                if total>protocol['bounds']['archive_max_bytes']:raise ValueError('Archiveexceedsdeclaredbound')
                stream.write(chunk)
        temporary.replace(ARCHIVE)
    if ARCHIVE.stat().st_size>protocol['bounds']['archive_max_bytes']:raise ValueError('Originalarchivebound exceeded')
    source=OUT/'originals';rows=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sum(v.file_size for v in archive.infolist())>protocol['bounds']['total_uncompressed_max_bytes']:raise ValueError('Uncompressedsize bound exceeded')
        members={}
        for entry in archive.infolist():
            name=PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafeauthorarchivepath')
            if len(name.parts)>1:
                relative='/'.join(name.parts[1:])
                if relative in members:raise ValueError('Duplicatearchivepath')
                members[relative]=entry
        for selected in protocol['selection_before_pixels']:
            row=dict(selected)
            for kind in ['image','mask']:
                relative=row['source_'+kind];entry=members[relative];data=archive.read(entry)
                blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                if blob!=row[kind+'_git_blob_sha1'] or len(data)!=row[kind+'_bytes']:raise ValueError('AuthorGitblobbytes differ')
                path=source/relative
                if not path.resolve().is_relative_to(source.resolve()):raise ValueError('Extractionpath escaped')
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    if path.read_bytes()!=data:raise ValueError('Preserve changedoriginal')
                else:
                    with path.open('xb') as stream:stream.write(data)
                row[kind]=path.relative_to(ROOT).as_posix();row[kind+'_sha256']=sha(path);row[kind+'_zip_crc32']=entry.CRC
            rows.append(row)
    write_new(OUT/'original-acquisition-manifest.json',{'protocol_sha256':sha(PROTOCOL),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
        'rows':rows,'all_original_git_blobs_verified':True,'all_selected_zip_CRCs_verified_by_decompression':True,'NN_executed':False})
    print(json.dumps({'original_pairs':len(rows),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE),'NN_executed':False},indent=2))


def qa(protocol):
    import cv2,numpy as np
    cv2.setNumThreads(1)
    acquisition=OUT/'original-acquisition-manifest.json';record=json.loads(acquisition.read_text())
    if record['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Acquisitionprotocol changed')
    rows=[];issues=[];examples=[];bycategory=collections.defaultdict(list)
    for old in record['rows']:
        for kind in ['image','mask']:
            if sha(ROOT/old[kind])!=old[kind+'_sha256']:raise ValueError('Originalsource changed')
        image=cv2.imread(str(ROOT/old['image']),cv2.IMREAD_UNCHANGED);mask=cv2.imread(str(ROOT/old['mask']),cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:raise ValueError('Authororiginal decode failed')
        codes=np.unique(mask).tolist();row=dict(old);problem=[]
        if mask.ndim!=2 or mask.dtype!=np.uint8 or not set(codes)<={0,1,255}:problem.append('unknown_mask_encoding')
        if image.shape[:2]!=mask.shape[:2]:problem.append('native_image_mask_geometry_differs')
        foreground=mask>0 if mask.ndim==2 else np.zeros(image.shape[:2],bool)
        if old['category']=='MT_Free' and foreground.any():problem.append('free_category_nonempty_mask')
        if old['category']!='MT_Free' and not foreground.any():problem.append('defect_category_empty_mask')
        gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        count,components,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
        ph=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:]
        bits=ph>np.median(ph);phash=sum(int(v)<<i for i,v in enumerate(bits))
        row.update(native_image_shape=list(image.shape),native_mask_shape=list(mask.shape),mask_codes=codes,
            mask_foreground_definition='mask>0 after originalbinarycode validation; no morphology',foreground_pixels=int(foreground.sum()),
            foreground_components=[{'bbox_xywh':[int(v) for v in s[:4]],'pixels':int(s[4])} for s in stats[1:]],
            native_image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),native_gray_phash63=f'{phash:016x}',
            native_mask_pixel_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),QA_issues=problem,training_asset_valid=not problem)
        rows.append(row);bycategory[old['category']].append(row)
        if problem:issues.append({'source':old['source_image'],'issues':problem})
    # Fixed real source/GT triptychs: firstsource byfilename perfirst3exp/category.
    for category in CATEGORIES:
        selected=bycategory[category];prefixes=sorted({v['illumination_filename_prefix'] for v in selected},key=lambda v:int(v[3:]))[:3]
        for index,prefix in enumerate(prefixes):examples.append({'display_sheet':index,'row':next(v for v in selected if v['illumination_filename_prefix']==prefix)})
    visual=[]
    for sheet in range(3):
        chosen=[v['row'] for v in examples if v['display_sheet']==sheet];canvas=np.full((len(chosen)*218,900,3),25,np.uint8)
        for index,row in enumerate(chosen):
            raw=cv2.imread(str(ROOT/row['image']));mask=cv2.imread(str(ROOT/row['mask']),0)>0;gt=cv2.cvtColor((mask.astype(np.uint8)*255),cv2.COLOR_GRAY2BGR);overlay=raw.copy()
            overlay[mask]=(overlay[mask]*.55+np.array([30,50,245])*.45).astype(np.uint8)
            for col,(array,label) in enumerate([(raw,'RAW'),(gt,'PUBLISHER GT'),(overlay,'GT OVERLAY - NOT MODEL')]):
                h,w=array.shape[:2];scale=min(296/w,182/h);thumb=cv2.resize(array,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
                top=index*218+32;left=col*300;canvas[top:top+thumb.shape[0],left:left+thumb.shape[1]]=thumb
                cv2.putText(canvas,label,(left+2,index*218+25),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1)
            cv2.putText(canvas,row['category']+'/'+row['stem'],(2,index*218+215),cv2.FONT_HERSHEY_SIMPLEX,.37,(230,230,230),1)
        path=OUT/f'fixed-publisher-GT-sheet-{sheet}.jpg'
        if path.exists():raise ValueError('Preserve firstGTQAvisual')
        if not cv2.imwrite(str(path),canvas):raise ValueError('GTcontactsheet failed')
        visual.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'sources':[r['source_image'] for r in chosen],'scope':'RealRAW/publisherGTonly; no modelprediction'})
    manifest={'protocol_sha256':sha(PROTOCOL),'acquisition_sha256':sha(acquisition),'all_images_TRAIN_only':True,'official_or_internal_MTD_test_claim':False,
        'physical_group_independence':'Unknown; repeatedtile/multipleillumination, exp/num filenames not verifieditem IDs.',
        'counts':{'images':len(rows),'positive_images':sum(r['foreground_pixels']>0 for r in rows),'normal_images':sum(r['foreground_pixels']==0 for r in rows),
                  'categories':dict(collections.Counter(r['category'] for r in rows))},
        'all_original_native_pixel_GT_QA_passed':not issues,'issues':issues,'rows':rows,'fixed_visual_QA':visual,
        'crosssource_similarity_screening':'Pendingseparatemetadata/TRAIN-only screening; noSDD2reserved/testdecode or oldcomparisonpixelreopening',
        'NN_executed':False,'rights_scope':protocol['rights']['scope']}
    write_new(OUT/'train-asset-manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['rows','fixed_visual_QA']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['record-protocol','acquire','qa'],required=True);args=parser.parse_args()
    if args.stage=='record-protocol':write_new(PROTOCOL,protocol_value());print('PROTOCOL',sha(PROTOCOL));return
    protocol=verify_protocol()
    (acquire if args.stage=='acquire' else qa)(protocol)


if __name__=='__main__':main()
