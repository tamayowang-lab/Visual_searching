"""Bundled deterministic RGB fixture; no downloaded imagery or human labels."""
from pathlib import Path
import cv2,numpy as np
r=Path(__file__).resolve().parents[1];image=np.full((240,360,3),[55,60,65],np.uint8)
cv2.rectangle(image,(35,40),(100,135),(220,45,35),-1);cv2.circle(image,(260,70),30,(35,170,80),-1);cv2.rectangle(image,(180,150),(320,195),(40,80,220),-1)
(r/'examples').mkdir(exist_ok=True);cv2.imwrite(str(r/'examples/synthetic.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
# Separate synthetic train image: distinct pixels, not a renamed test image.
train=np.flip(image,axis=1).copy();cv2.imwrite(str(r/'examples/synthetic-train.png'),cv2.cvtColor(train,cv2.COLOR_RGB2BGR))
relative=np.tile(np.linspace(0,1,360,dtype=np.float32),(240,1));np.savez_compressed(r/'examples/synthetic-depth.npz',depth=relative,quality=np.full((240,360),.4,np.float32))
import json
manifest=dict(schema_version=1,dataset_name='Synthetic engineering fixture — NOT human eye tracking',synthetic=True,coordinate_units='pixels',initial_fixation_included=True,geometry_notes='Zero-indexed synthetic 360x240 images; pixel-based foveation; invented scanpaths for pipeline checks only',records=[dict(image_id='synthetic-train',image='synthetic-train.png',width=360,height=240,task='',participant='fixture-only',split='train',fixations=[dict(x=180,y=120,duration_ms=250),dict(x=290,y=80,duration_ms=300)]),dict(image_id='synthetic',image='synthetic.png',depth='synthetic-depth.npz',depth_kind='inverse_depth',width=360,height=240,task='',participant='fixture-only',split='test',fixations=[dict(x=180,y=120,duration_ms=250),dict(x=65,y=80,duration_ms=300),dict(x=250,y=170,duration_ms=350),dict(x=260,y=70,duration_ms=250)])])
(r/'examples/synthetic-manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
