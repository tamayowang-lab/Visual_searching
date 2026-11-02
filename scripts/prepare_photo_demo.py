"""Recover the original Porto panorama and extract three real perspective views."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
import cv2
import numpy as np

URL='https://raw.githubusercontent.com/David-Ef/salient360Toolbox/main/data/stimuli/1_PortoRiverside.png'
VIEWS=[('porto-square','Porto · Riverside square',0.),('porto-promenade','Porto · Promenade',-55.),('porto-waterfront','Porto · Waterfront',65.)]


def perspective(image,yaw,width=640,height=400,fov=90.,pitch=-5.):
    focal=width/(2*np.tan(np.deg2rad(fov)/2));yy,xx=np.mgrid[:height,:width]
    rays=np.stack(((xx-(width-1)/2)/focal,-(yy-(height-1)/2)/focal,np.ones_like(xx)),axis=-1)
    a=np.deg2rad(pitch);p=np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]])
    a=np.deg2rad(yaw);r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    rays=rays@p.T@r.T;longitude=np.arctan2(rays[...,0],rays[...,2]);latitude=np.arctan2(rays[...,1],np.hypot(rays[...,0],rays[...,2]))
    h,w=image.shape[:2];mx=((longitude/(2*np.pi)+.5)*w).astype(np.float32);my=((.5-latitude/np.pi)*h).astype(np.float32)
    return cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',action='store_true');args=parser.parse_args()
    source=Path('.cache/stimuli/1_PortoRiverside.png');source.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        if not args.download:raise SystemExit('Use --download to recover the original photograph')
        urllib.request.urlretrieve(URL,source)
    image=cv2.imread(str(source));root=Path('examples/photos');root.mkdir(parents=True,exist_ok=True);records=[]
    for key,name,yaw in VIEWS:
        dest=root/(key+'.jpg');cv2.imwrite(str(dest),perspective(image,yaw),[cv2.IMWRITE_JPEG_QUALITY,95])
        records.append(dict(id=key,name=name,path=str(dest),yaw_degrees=yaw,fov_degrees=90,pitch_degrees=-5,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (root/'manifest.json').write_text(json.dumps(dict(source_url=URL,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_note='Same Porto panorama used by Gaze Lab; perspective projection only. Repository MIT license; stimulus-specific license not separately documented.',scenes=records),indent=2))
    print('Prepared three real Porto environment views; no object annotations were supplied.')

if __name__=='__main__':main()
