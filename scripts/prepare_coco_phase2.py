"""Prepare a licensed, local COCO-Search18 mirror subset with explicit geometry/splits.

No dataset is redistributed. Requires an existing CCP_CSHL checkout containing
original ecc_0 images and the original COCO fixation JSON. No oracle boxes used.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np,cv2

def prepare(source,out):
    source=Path(source);out=Path(out);images=source/'ccp_search18_train_subset/ecc_0';raw=json.loads((source/'coco_search18/coco_search18_fixations_TP_train_split1.json').read_text(encoding='utf-8'));names=sorted([p.name for p in images.glob('*.jpg')],key=lambda n:hashlib.sha256(('rgb-observer-phase2:'+n).encode()).hexdigest())[:24]
    if len(names)<12:raise ValueError('At least 12 locally available original photographs required')
    splits={name:('train' if i<len(names)//2 else 'validation' if i<3*len(names)//4 else 'test') for i,name in enumerate(names)};(out/'images').mkdir(parents=True,exist_ok=True)
    geometry={}
    for name in names:
        image=cv2.imread(str(images/name));h,w=image.shape[:2]
        # Exact padding rule in the mirror's resize_img_COCO_Search helper.
        if w/h>1.6:px=0;py=int(round(((w*1050/1680)-h)//2))
        elif w/h<1.6:py=0;px=int(round(int(round((h*1680/1050)-w))//2))
        else:px=py=0
        padded=cv2.copyMakeBorder(image,py,py,px,px,cv2.BORDER_CONSTANT,value=[0,0,0]);resized=cv2.resize(padded,(640,400),interpolation=cv2.INTER_CUBIC);cv2.imwrite(str(out/'images'/(name+'.png')),resized)
        geometry[name]=dict(source_size=[w,h],padding_xy=[px,py],original_display=[1680,1050],inference_display=[640,400],coordinate_scale=640/1680,source_sha256=hashlib.sha256((images/name).read_bytes()).hexdigest())
    records=[];excluded={};bump=lambda reason:excluded.update({reason:excluded.get(reason,0)+1})
    for t in raw:
        if t['name'] not in splits:continue
        if t.get('correct')!=1:bump('incorrect_trial');continue
        if not all(0<=x<1680 and 0<=y<1050 and np.isfinite([x,y,d]).all() and d>0 for x,y,d in zip(t['X'],t['Y'],t['T'])):bump('invalid_coordinate_or_duration');continue
        rt=float(t.get('RT',sum(t['T'])));fix=[];remaining=rt
        for x,y,d in zip(t['X'],t['Y'],t['T']):
            if remaining<=0:break
            duration=min(float(d),remaining);fix.append(dict(x=x*640/1680,y=y*400/1050,duration_ms=duration));remaining-=duration
        if len(fix)<2:bump('fewer_than_two_fixations_before_response');continue
        records.append(dict(image_id=t['name'],image='images/'+t['name']+'.png',width=640,height=400,split=splits[t['name']],participant=str(t['subject']),task=t['task'],viewing_duration_ms=rt,condition='target-present',fixations=fix))
    meta=dict(schema_version=1,dataset_name='COCO-Search18 target-present exploratory internal split',synthetic=False,coordinate_units='pixels',initial_fixation_included=True,task='target-present categorical search',official_source_split='released training split; these are NOT official benchmark test results',split_policy='24 locally available originals ordered by SHA256(rgb-observer-phase2:filename); first 12 train, next 6 validation, final 6 test; no fitting in this phase',timing='Model budget conditioned on each recorded RT. Human durations truncated at RT using cumulative fixation time; saccade gaps unavailable. This does not predict response time.',geometry_notes='Mirror aspect-preserving black padding to 1680x1050, then uniform 640x400 rescale. Coordinates scaled identically. Original stimulus pipeline may differ; use official display assets for confirmation.',geometry_audit=geometry,source_repository='https://github.com/RosenholtzLab/CCP_CSHL',source_commit='401baae571e325eacb000f21783463d2343609ab',exclusions=excluded,records=records)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(images={s:sum(v==s for v in splits.values()) for s in ['train','validation','test']},trials={s:sum(t['split']==s for t in records) for s in ['train','validation','test']},excluded=excluded)));return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');a=p.parse_args();prepare(a.source,a.out)
