"""Zero-indexed image pixels: 0 <= x < W, 0 <= y < H."""
from dataclasses import dataclass
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None: raise ValueError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

@dataclass
class Camera:
    width: int
    height: int
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    def __post_init__(self):
        if self.width < 2 or self.height < 2: raise ValueError("Image must be at least 2x2")
        if (self.fx is None) != (self.fy is None): raise ValueError("Supply both fx and fy")
        self.assumption = "provided intrinsics; relative depth remains nonmetric" if self.fx is not None else "canonical 60-degree pseudo-camera for visualization only"
        if self.fx is None: self.fx = self.fy = self.width / (2*np.tan(np.deg2rad(30)))
        if not np.isfinite([self.fx,self.fy]).all() or min(self.fx,self.fy) <= 0: raise ValueError("Invalid intrinsics")
        self.cx = (self.width-1)/2 if self.cx is None else self.cx
        self.cy = (self.height-1)/2 if self.cy is None else self.cy

    def bearing(self, x: float, y: float) -> np.ndarray:
        if not (0 <= x < self.width and 0 <= y < self.height): raise ValueError("Off-screen fixation")
        v = np.array([(x-self.cx)/self.fx, (y-self.cy)/self.fy, 1.])
        return v/np.linalg.norm(v)

    def metadata(self):
        return dict(fx=self.fx,fy=self.fy,cx=self.cx,cy=self.cy,assumption=self.assumption,xyz_units="pseudo-3D relative units, never meters")
