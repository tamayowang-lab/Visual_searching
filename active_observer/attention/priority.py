"""Dense priority over visual coordinates, using only sensory input and memory.

Array samples approximate a continuous field; they are not attention cells or
candidate groups. Maxima can occur anywhere, independently of the predefined candidate grid.
All scores are heuristic indices, never calibrated probabilities.
"""
import cv2
import numpy as np


class PriorityField:
    def __init__(self, width, height, valid, resolution=64, temperature=.08):
        self.width, self.height = width, height
        self.temperature = temperature
        self.w = min(width, resolution)
        self.h = max(1, round(height * self.w / width))
        self.yy, self.xx = np.mgrid[:self.h, :self.w]
        self.x = np.minimum(width - 1, (self.xx + .5) * width / self.w)
        self.y = np.minimum(height - 1, (self.yy + .5) * height / self.h)
        self.valid = valid[self.y.astype(int), self.x.astype(int)]
        if not self.valid.any():
            raise ValueError('Validity mask has no support at priority-field resolution')
        self.exposure = np.zeros((self.h, self.w))
        self.ior = np.zeros_like(self.exposure)

    def update(self, observation, dwell_ms):
        distance = np.hypot(self.x - observation.gaze[0], self.y - observation.gaze[1])
        central = distance <= observation.radius_px
        self.exposure += central * dwell_ms / 300
        self.ior *= np.exp(-dwell_ms / 1800)
        self.ior = np.maximum(self.ior, np.exp(-.5 * (distance / observation.radius_px)**2))

    def score(self, observation, world):
        rgb=cv2.resize(observation.rgb,(self.w,self.h),interpolation=cv2.INTER_AREA).astype(float)/255
        gray=rgb.mean(axis=2)
        contrast=np.linalg.norm(rgb-np.median(rgb.reshape(-1,3),axis=0),axis=2)
        gradient=np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0),cv2.Sobel(gray,cv2.CV_64F,0,1))
        support=np.clip(contrast*2+gradient,0,1)
        novelty=np.exp(-self.exposure)
        perceptual_need=novelty.copy();geometric_need=np.zeros_like(novelty)
        for memory in world.objects:
            kernel=np.exp(-.5*((self.x-memory.centroid[0])**2+(self.y-memory.centroid[1])**2)/max(1,observation.radius_px)**2)
            perceptual_need=np.maximum(perceptual_need,kernel*memory.identity_uncertainty)
            geometry=memory.reconstruction
            # Unknown hidden thickness cannot be resolved by staring at one image.
            # Retain its uncertainty in memory; only predict reducible surface gain.
            need=max(geometry.depth_uncertainty,max(geometry.dimension_uncertainty[:2]))
            need*=float(np.exp(-memory.observation_count/3))
            geometric_need=np.maximum(geometric_need,kernel*need)
        # This is a declared prediction proxy. It is never logged as actual improvement.
        gain=(.15+.85*support)*novelty+.35*geometric_need
        movement=np.hypot(self.x-observation.gaze[0],self.y-observation.gaze[1])/np.hypot(self.width,self.height)
        components=dict(salience=.55*np.clip(contrast+gradient,0,1)*novelty,
                        peripheral_structure=.25*support*novelty,novelty=.18*novelty,
                        perceptual_information_need=.12*perceptual_need,
                        reconstruction_information_need=.35*geometric_need,
                        estimated_reconstruction_gain=.9*gain,
                        inhibition_of_return=-.65*self.ior,movement_cost=-.18*movement)
        return dict(total=sum(components.values()),components=components,gain=gain,movement=movement)

    def choose(self, field, rng):
        # Softmax is a declared spatial choice rule, not a recognition probability.
        logits = np.where(self.valid, field['total'] / self.temperature, -np.inf)
        mass = np.exp(logits - np.max(logits))
        probability = mass / mass.sum()
        flat = int(rng.choice(probability.size, p=probability.ravel()))
        iy, ix = np.unravel_index(flat, probability.shape)
        x, y = float(self.x[iy, ix]), float(self.y[iy, ix])
        candidate=dict(x=x,y=y,utility=float(field['total'][iy,ix]),
                       estimated_information_gain=float(field['gain'][iy,ix]),
                       components={k:float(v[iy,ix]) for k,v in field['components'].items()},
                       field_index=[int(ix),int(iy)],selection_probability=float(probability[iy,ix]),temperature=self.temperature)
        return candidate

    def snapshot(self, field):
        def values(a):
            return np.round(a, 4).tolist()
        return dict(width=self.w, height=self.h, total=values(field['total']),
                    components={k:values(v) for k,v in field['components'].items()},
                    valid=self.valid.tolist(), coverage=values(1-np.exp(-self.exposure)),
                    coordinate_note='sample centers in image coordinates; numeric integration mesh; no region-center proposals')
