"""Download pinned CPU assets only with --download; verify both checkpoint hashes."""
from pathlib import Path
import argparse,hashlib,json,tarfile,urllib.request,shutil

SSD='https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
MIDAS='https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt'
REV='454597711a62eabcbf7d1e89f3fb9f569051ac9b'
HASHES={'ssdlite320_mobilenet_v3_large_coco-a79551df.pth':'a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2','midas_v21_small_256.pt':'70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--directory',default='.cache/models');a=p.parse_args();root=Path(a.directory);root.mkdir(parents=True,exist_ok=True)
    for url in [SSD,MIDAS]:
        name=url.rsplit('/',1)[1];dest=root/name
        if not dest.exists():
            if not a.download:raise SystemExit(f'Missing {name}; supply --download explicitly')
            urllib.request.urlretrieve(url,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=HASHES[name]:raise SystemExit('Checkpoint digest mismatch: '+name)
    source=root/('MiDaS-'+REV)
    if not (source/'midas/midas_net_custom.py').exists():
        if not a.download:raise SystemExit('Missing pinned MiDaS source; use --download')
        archive=root/'midas-source.tar.gz';urllib.request.urlretrieve(f'https://github.com/isl-org/MiDaS/archive/{REV}.tar.gz',archive)
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                target=(root/member.name).resolve()
                if root.resolve() not in target.parents or member.issym() or member.islnk():raise ValueError('Unsafe source archive entry')
            t.extractall(root,filter='data')
    spec=dict(objects=dict(provider='torchvision_ssdlite',checkpoint=str((root/'ssdlite320_mobilenet_v3_large_coco-a79551df.pth').resolve()),threshold=.35,context_scale=3.),depth=dict(provider='midas_small',checkpoint=str((root/'midas_v21_small_256.pt').resolve()),source=str(source.resolve()),revision=REV),cache_directory='.cache/observer')
    (root/'perception.json').write_text(json.dumps(spec,indent=2),encoding='utf-8');print(root/'perception.json')

if __name__=='__main__':main()
