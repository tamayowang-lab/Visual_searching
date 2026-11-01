"""Generate actual blurred RGB input; no hidden full-resolution semantic input."""
from dataclasses import dataclass
import cv2
import numpy as np
from active_observer.config import Config

@dataclass
class Observation:
    rgb: np.ndarray
    reliability: np.ndarray
    central: np.ndarray
    gaze: tuple[float,float]
    radius_px: float
    geometry: str


def observe(image: np.ndarray, gaze: tuple[float,float], config: Config) -> Observation:
    h,w=image.shape[:2];x,y=gaze
    if not (0 <= x < w and 0 <= y < h): raise ValueError("Off-screen gaze")
    radius=config.central_degrees*config.pixels_per_degree if config.pixels_per_degree else w*config.central_fraction
    yy,xx=np.mgrid[:h,:w];distance=np.hypot(xx-x,yy-y)
    reliability=np.clip(np.exp(-distance/(3*radius)),.05,1.)
    central=distance <= radius
    mild=cv2.GaussianBlur(image,(0,0),max(1,w/100));strong=cv2.GaussianBlur(image,(0,0),max(2,w/30))
    alpha=np.exp(-.5*(distance/radius)**2)[...,None]
    outer=np.clip(distance/(5*radius),0,1)[...,None]
    blurred=mild*(1-outer)+strong*outer
    rgb=np.clip(image*alpha+blurred*(1-alpha),0,255).astype(np.uint8)
    return Observation(rgb,reliability,central,gaze,radius,"degrees via supplied pixels-per-degree" if config.pixels_per_degree else "pixel-based approximation; no visual-angle calibration")
