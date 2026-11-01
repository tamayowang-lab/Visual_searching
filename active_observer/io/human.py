"""Strict image/coordinate/split contracts. No dataset download is implied."""
from pathlib import Path
import hashlib,json
import numpy as np
from .image import load_image


def load_manifest(path: str | Path) -> dict:
    path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('coordinate_units') not in {'pixels','normalized'}: raise ValueError("Declare coordinate_units: pixels or normalized")
    if not isinstance(data.get('initial_fixation_included'),bool): raise ValueError("Declare initial_fixation_included")
    records=[];split_by_id={};split_by_hash={}
    for item in data['records']:
        rec=dict(item);image_path=(path.parent/item['image']).resolve();im=load_image(image_path);h,w=im.shape[:2]
        if [w,h]!=[item['width'],item['height']]: raise ValueError("Image dimensions disagree with fixation metadata")
        if item.get('split') not in {'train','validation','test'}: raise ValueError("Explicit train/validation/test split required")
        if not isinstance(item.get('task',''),str) or 'participant' not in item: raise ValueError("Task string and participant ID required")
        digest=hashlib.sha256(im.tobytes()+str(im.shape).encode()).hexdigest()
        for registry,key in [(split_by_id,item['image_id']),(split_by_hash,digest)]:
            if key in registry and registry[key]!=item['split']: raise ValueError("Image leakage across splits (ID or pixel hash)")
            registry[key]=item['split']
        fix=[]
        for f in item['fixations']:
            x,y=float(f['x']),float(f['y']);x=x*w if data['coordinate_units']=='normalized' else x;y=y*h if data['coordinate_units']=='normalized' else y
            duration=float(f['duration_ms'])
            if not np.isfinite([x,y,duration]).all() or not (0<=x<w and 0<=y<h) or duration<=0: raise ValueError("Invalid fixation coordinates/duration; no silent clipping")
            fix.append(dict(x=x,y=y,duration_ms=duration))
        if len(fix)<2: raise ValueError("At least two fixations required for evaluation")
        rec.update(image_path=str(image_path),fixations=fix,image_sha256=digest,initial_fixation_included=data['initial_fixation_included'])
        if rec.get('viewing_duration_ms') is not None and (not np.isfinite(rec['viewing_duration_ms']) or rec['viewing_duration_ms']<=0):raise ValueError('Invalid recorded viewing duration')
        if rec.get('depth'):rec['depth_path']=str((path.parent/rec['depth']).resolve())
        records.append(rec)
    if not records: raise ValueError("Empty manifest")
    return dict(metadata={k:v for k,v in data.items() if k!='records'},records=records)


def convert_coco(raw_path: str, image_manifest_path: str, out_path: str) -> dict:
    """COCO adapter requires audited display geometry; never guesses image padding."""
    source=json.loads(Path(raw_path).read_text(encoding='utf-8'));meta=json.loads(Path(image_manifest_path).read_text(encoding='utf-8'));lookup={x['name']:x for x in meta['images']};records=[]
    for t in source:
        if t['name'] not in lookup:continue
        im=lookup[t['name']]
        if len(t['X'])!=len(t['Y']) or len(t['X'])!=len(t['T']):raise ValueError("COCO coordinate/duration lengths differ")
        records.append(dict(image_id=t['name'],image=im['image'],width=im['width'],height=im['height'],split=im['split'],participant=str(t['subject']),task=t['task'],fixations=[dict(x=x,y=y,duration_ms=d) for x,y,d in zip(t['X'],t['Y'],t['T'])]))
    out=dict(schema_version=1,dataset_name='COCO-Search18 user-supplied records',coordinate_units=meta['coordinate_units'],initial_fixation_included=meta['initial_fixation_included'],geometry_notes=meta['geometry_notes'],records=records)
    Path(out_path).write_text(json.dumps(out,allow_nan=False), encoding='utf-8');return out
