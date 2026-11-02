"""Provider construction; providers never receive a privileged original image."""
from pathlib import Path
import json
from .depth import CachedDepth,TransformersDepth

class ProviderFactory:
    def __init__(self):self.objects={};self.depths={}

    def get_objects(self,config):
        spec=dict(config.objects);name=spec.pop('provider');key=json.dumps(config.objects,sort_keys=True)
        if key not in self.objects:
            from .pretrained import CachedObjects
            if name=='torchvision_ssdlite':
                from .pretrained import TorchvisionFoveal
                source=TorchvisionFoveal(**spec)
            elif name=='transformers':
                from .objects import FovealTransformersObjects
                source=FovealTransformersObjects(**spec)
            else:raise ValueError('Choose a centrally gated object provider')
            self.objects[key]=CachedObjects(source,Path(config.cache_directory)/'objects')
        return self.objects[key]

    def get_depth(self,config):
        spec=dict(config.depth);name=spec.pop('provider','none');depth=None
        if name!='none':
            key=json.dumps(config.depth,sort_keys=True)
            if key not in self.depths:
                if name=='midas_small':
                    from .pretrained import MidasSmall
                    source=MidasSmall(**spec)
                elif name=='transformers':source=TransformersDepth(**spec)
                else:raise ValueError('Depth must be inferred from current gaze-dependent RGB; precomputed full-scene depth is not a runtime provider')
                self.depths[key]=CachedDepth(source,Path(config.cache_directory)/'depth',key)
            depth=self.depths[key]
        return depth
