"""Actual gaze-dependent RGB input; no original-image handle reaches providers."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Observation:
    rgb: np.ndarray
    reliability: np.ndarray
    central: np.ndarray
    gaze: tuple[float,float]
    radius_px: float
    geometry: str = 'pixel-based foveation; no visual-angle calibration'


def observe(image,gaze,config):
    h,w=image.shape[:2];x,y=gaze
    if not (0<=x<w and 0<=y<h):raise ValueError('Off-screen gaze')
    radius=w*config.central_fraction
    yy,xx=np.mgrid[:h,:w];distance=np.hypot(xx-x,yy-y)
    central=distance<=radius
    reliability=np.clip(np.exp(-distance/(3*radius)),.05,1.)
    cw=min(w,config.peripheral_width);ch=max(1,round(h*cw/w))
    coarse=cv2.resize(image,(cw,ch),interpolation=cv2.INTER_AREA)
    peripheral=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_LINEAR)
    rgb=np.where(central[...,None],image,peripheral).astype(np.uint8)
    return Observation(rgb,reliability,central,gaze,radius)
