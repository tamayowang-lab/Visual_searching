"""Lossless depth assets and provenance; serialization never runs inference."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from active_observer.perception.depth import array_hash


def depth_arrays(record,asset_root=None):
    if not record or not record.get('available'):return None
    if 'arrays' in record:
        arrays={name:np.array(item['values'],dtype=item['dtype']) for name,item in record['arrays'].items()}
    else:
        if asset_root is None:raise ValueError('Depth asset root is required')
        root=Path(asset_root).resolve();path=(root/record['asset']).resolve()
        if root not in path.parents:raise ValueError('Depth asset escapes recording folder')
        if not path.is_file():raise ValueError('Missing recorded depth asset: '+record['asset'])
        with np.load(path,allow_pickle=False) as data:
            arrays={name:data[name].copy() for name in ['inverse_depth','valid','quality']}
            if str(data['input_sha256'])!=record['input_sha256']:raise ValueError('Depth asset sensory hash mismatch')
            if str(data['frame_id'])!=record['frame_id']:raise ValueError('Depth frame ID mismatch')
            if json.loads(str(data['provenance_json']))!=record['metadata']:raise ValueError('Depth provenance mismatch')
            if json.loads(str(data['coordinate_contract_json']))!=record['coordinate_contract']:raise ValueError('Depth coordinate contract mismatch')
    for name,a in arrays.items():
        if array_hash(a)!=record['array_hashes'][name]:raise ValueError('Recorded depth array hash mismatch: '+name)
    return arrays


def save_run(result,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    saved=deepcopy(result)
    for f in saved['fixations']:
        record=f.get('depth')
        if not record or not record.get('available'):continue
        arrays=depth_arrays(record,path.parent)
        relative=f"depth/fixation-{f['step']:04d}.npz";dest=path.parent/relative;dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,**arrays,input_sha256=np.array(record['input_sha256']),frame_id=np.array(record['frame_id']),
                            provenance_json=np.array(json.dumps(record['metadata'],sort_keys=True,allow_nan=False)),
                            coordinate_contract_json=np.array(json.dumps(record['coordinate_contract'],sort_keys=True)))
        record.pop('arrays',None);record['asset']=relative
    path.write_text(json.dumps(saved,allow_nan=False,indent=2),encoding='utf-8')
    return saved
