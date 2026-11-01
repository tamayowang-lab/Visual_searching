"""One provider factory shared by single-image, preview and empirical evaluation."""
from pathlib import Path
import json,warnings
from .objects import ClassicalObjects
from .depth import CachedDepth,PrecomputedDepth,ConstantDepth,TransformersDepth

class ProviderFactory:
    def __init__(self):self.objects={};self.depths={}
    def get(self,config,record=None):
        spec=config.perception;cache=Path(spec.get('cache_directory','.cache/observer'));object_spec=spec.get('objects',{'provider':'classical'});depth_spec=spec.get('depth',{'provider':'none'})
        key=json.dumps(object_spec,sort_keys=True);name=object_spec.get('provider','classical')
        if not config.semantic_enabled:name='classical';key='classical'
        if key not in self.objects:
            if name=='classical':self.objects[key]=ClassicalObjects()
            elif name=='torchvision_ssdlite':
                from .pretrained import TorchvisionFoveal,CachedObjects
                args={k:v for k,v in object_spec.items() if k!='provider'};self.objects[key]=CachedObjects(TorchvisionFoveal(**args),cache/'objects')
            elif name=='transformers':
                from .objects import FovealTransformersObjects
                from .pretrained import CachedObjects
                self.objects[key]=CachedObjects(FovealTransformersObjects(**{k:v for k,v in object_spec.items() if k!='provider'}),cache/'objects')
            else:raise ValueError('Unknown object provider: '+name)
        depth=None
        if config.depth_enabled:
            if record and record.get('depth_path'):depth=PrecomputedDepth(record['depth_path'],kind=record.get('depth_kind','inverse_depth'))
            elif depth_spec.get('provider','none')!='none':
                dk=json.dumps(depth_spec,sort_keys=True)
                if dk not in self.depths:
                    args={k:v for k,v in depth_spec.items() if k!='provider'}
                    if depth_spec['provider']=='midas_small':
                        from .pretrained import MidasSmall
                        source=MidasSmall(**args)
                    elif depth_spec['provider']=='transformers':source=TransformersDepth(**args)
                    elif depth_spec['provider']=='constant':source=ConstantDepth()
                    elif depth_spec['provider']=='precomputed':source=PrecomputedDepth(**args)
                    else:raise ValueError('Unknown depth provider')
                    self.depths[dk]=CachedDepth(source,cache/'depth',dk)
                depth=self.depths[dk]
        return self.objects[key],depth
