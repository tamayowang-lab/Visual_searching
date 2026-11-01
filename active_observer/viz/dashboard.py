"""Headless verification PNG; original/foveated observation are distinct panels."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from active_observer.config import Config
from .contracts import validate_result
from active_observer.perception.foveation import observe


def dashboard(image: np.ndarray,result: dict,path: str | Path,step: int | None=None) -> None:
    validate_result(image,result)
    fix=result['fixations'];step=len(fix)-1 if step is None else int(step)
    if not 0<=step<len(fix): raise ValueError("Step outside scanpath")
    f=fix[step];cfg=Config(**result['config']);obs=observe(image,(f['x_px'],f['y_px']),cfg)
    fig,axes=plt.subplots(2,3,figsize=(16,9));axes=axes.ravel();axes[0].imshow(image)
    xs=[p['x_px'] for p in fix[:step+1]];ys=[p['y_px'] for p in fix[:step+1]];axes[0].plot(xs,ys,'-o',c='#f2b134',lw=1,ms=5)
    for i,(x,y) in enumerate(zip(xs,ys)): axes[0].annotate(str(i),(x,y),fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.6,pad=1))
    axes[0].set_title('Original RGB reference + observed fixation history')
    axes[1].imshow(obs.rgb);axes[1].add_patch(plt.Circle(obs.gaze,obs.radius_px,fill=False,color='white'));axes[1].set_title(f"Actual foveated observation · step {step}")
    if result['depth']['available']:
        reference=result['depth']['reference'];d=np.array(reference['values'],dtype=float);seen=np.zeros(d.shape,bool)
        for previous in fix[:step+1]:seen |= observe(image,(previous['x_px'],previous['y_px']),cfg).central
        shown=np.ma.masked_where(~seen|~np.array(reference['valid']),d);depth_plot=axes[2].imshow(shown,cmap='viridis',vmin=0,vmax=1);fig.colorbar(depth_plot,ax=axes[2],label='Normalized inverse depth [0,1]',shrink=.7)
        axes[2].set_title('Observed relative inverse depth · unobserved masked')
    else:
        axes[2].text(.5,.5,'No relative depth provider',ha='center',va='center',transform=axes[2].transAxes);axes[2].set_title('Relative depth · never meters')
    heat=np.zeros(image.shape[:2],float)
    for p in fix[:step+1]: heat[min(heat.shape[0]-1,int(p['y_px'])),min(heat.shape[1]-1,int(p['x_px']))]+=p['duration_ms']
    axes[3].imshow(image,alpha=.4);axes[3].imshow(gaussian_filter(heat,max(2,image.shape[1]/45)),cmap='inferno',alpha=.7);axes[3].set_title('Accumulated dwell heatmap')
    axes[4].imshow(obs.rgb)
    for m in f['memory']['objects']:
        x,y,x2,y2=m['bbox'];color='lime' if m['state']=='observed' else 'orange'
        axes[4].add_patch(plt.Rectangle((x,y),x2-x,y2-y,fill=False,color=color,ls='-' if m['state']=='observed' else '--'))
        axes[4].text(x,y,m['id']+': '+max(m['probabilities'],key=m['probabilities'].get),fontsize=7,color=color)
    axes[4].set_title('Observed (green) / remembered (dashed) objects')
    terms=f['candidate_details'].get('utility_components',{});axes[5].barh(list(terms),list(terms.values()),color='#267c85');axes[5].axvline(0,color='gray',lw=1);axes[5].set_title(f"Arrival decision terms · {f['chosen_candidate']}")
    for a in axes[:5]: a.set_xticks([]);a.set_yticks([])
    fig.suptitle('RGB Active Observer · interpretable prototype, not human-validated',fontsize=15);fig.tight_layout();Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=140);plt.close(fig)
