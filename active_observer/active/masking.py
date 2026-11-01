"""Explicit stimulus validity. Pixel darkness is never used to infer padding."""
import base64
import numpy as np

class StimulusMask:
    def __init__(self,width,height,mask=None,bounds=None):
        if mask is not None and bounds is not None:raise ValueError('Supply mask or bounds, not both')
        self.width=width;self.height=height;self.basis='all pixels explicitly assumed stimulus' if mask is None and bounds is None else 'explicit bitmap' if mask is not None else 'declared preprocessing rectangle'
        if bounds is not None:
            x1,y1,x2,y2=map(float,bounds)
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Invalid stimulus bounds')
            yy,xx=np.mgrid[:height,:width];mask=(xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)
        self.mask=np.ones((height,width),bool) if mask is None else np.asarray(mask,bool)
        if self.mask.shape!=(height,width) or not self.mask.any():raise ValueError('Nonempty aligned valid-stimulus mask required')
    def valid(self,x,y):return bool(0<=x<self.width and 0<=y<self.height and self.mask[int(y),int(x)])
    def metadata(self):return dict(basis=self.basis,width=self.width,height=self.height,valid_fraction=float(self.mask.mean()),packed_bits_base64=base64.b64encode(np.packbits(self.mask.ravel(),bitorder='little').tobytes()).decode(),bitorder='little')
