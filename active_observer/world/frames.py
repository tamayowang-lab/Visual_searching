"""Explicit future pose boundary; current memory is never labeled world coordinates."""
from typing import Protocol
import numpy as np

class CameraPoseProvider(Protocol):
    def image_anchor_to_robot(self,time_ms: float) -> np.ndarray: ...

def transform_point(point,transform):
    matrix=np.asarray(transform,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):raise ValueError('Expected finite rigid homogeneous transform')
    rot=matrix[:3,:3]
    if not np.allclose(rot.T@rot,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rot),1):raise ValueError('Transform rotation must be rigid')
    return (matrix@np.r_[point,1])[:3].tolist()
