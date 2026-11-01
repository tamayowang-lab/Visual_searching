"""New scenes chosen by filename hash before viewing model performance."""
from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np

def prepare(source_list,old_manifest,raw_path,originals,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);old=json.loads(Path(old_manifest).read_text(encoding='utf-8'));seen={r['image_id'] for r in old['records']};raw=json.loads(Path(raw_path).read_text(encoding='utf-8'));available=json.loads(Path(source_list).read_text(encoding='utf-8'));eligible={t['name'] for t in raw if t.get('correct')==1}
    chosen=sorted([x for x in available if x['name'] not in seen and x['name'] in eligible],key=lambda x:hashlib.sha256(('phase4b-independent-v1:'+x['name']).encode()).hexdigest())[:12]
    (out/'download-list.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    if not all((Path(originals)/x['name']).exists() for x in chosen):print('Download listed originals first');return
    records=[];audit={};(out/'images').mkdir(exist_ok=True)
    for i,item in enumerate(chosen):
        name=item['name'];im=cv2.imread(str(Path(originals)/name));h,w=im.shape[:2];px=py=0
        if w/h>1.6:py=int(round(((w*1050/1680)-h)//2))
        elif w/h<1.6:px=int(round(int(round(h*1680/1050-w))//2))
        a=cv2.copyMakeBorder(im,py,py,px,px,cv2.BORDER_CONSTANT,value=[0,0,0]);cv2.imwrite(str(out/'images'/(name+'.png')),cv2.resize(a,(640,400),interpolation=cv2.INTER_CUBIC));bounds=[px*640/a.shape[1],py*400/a.shape[0],(px+w)*640/a.shape[1],(py+h)*400/a.shape[0]];audit[name]=dict(source_size=[w,h],padding_xy=[px,py],valid_stimulus_bounds=bounds,source_sha256=hashlib.sha256((Path(originals)/name).read_bytes()).hexdigest())
        for t in raw:
            if t['name']!=name or t.get('correct')!=1:continue
            if not all(0<=x<1680 and 0<=y<1050 and d>0 for x,y,d in zip(t['X'],t['Y'],t['T'])):continue
            remaining=t['RT'];fix=[]
            for x,y,d in zip(t['X'],t['Y'],t['T']):
                if remaining<=0:break
                duration=min(d,remaining);fix.append(dict(x=x*640/1680,y=y*400/1050,duration_ms=duration));remaining-=duration
            if len(fix)<2:continue
            records.append(dict(image_id=name,image='images/'+name+'.png',width=640,height=400,split='validation' if i<6 else 'test',participant=str(t['subject']),task=t['task'],viewing_duration_ms=t['RT'],fixations=fix,valid_stimulus_bounds=bounds))
    result=dict(schema_version=1,dataset_name='Phase 4B independent local mirror scenes',coordinate_units='pixels',initial_fixation_included=True,geometry_notes='Same audited mirror padding and 640x400 scale; published 54x35 degree geometry for evaluation only',geometry_audit=audit,selection='12 non-Phase2/3/4A scenes sorted by SHA256(phase4b-independent-v1:filename), six validation then six test; no performance filtering',records=records)
    (out/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print({s:sum(r['split']==s for r in records) for s in ['validation','test']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_list');p.add_argument('old_manifest');p.add_argument('raw');p.add_argument('originals');p.add_argument('out');a=p.parse_args();prepare(a.source_list,a.old_manifest,a.raw,a.originals,a.out)
