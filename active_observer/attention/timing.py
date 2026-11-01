"""Separable stochastic race timing and bounded processing hold (seconds internally)."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass
class TimingParameters:
    dt_ms: float=20.
    threshold: float=1.
    urgency: float=4.
    priority_gain: float=2.
    diffusion: float=.2
    urgency_ramp: float=0.
    preparation_ms: float=30.
    hold_strength: float=.5
    sensory_tau_ms: float=250.
    safety_timeout_ms: float=3000.
    heuristic_scale: float=1.
    heuristic_sigma: float=.18
    selection_mode: str='softmax'
    execution_enabled: bool=True

    def validate(self):
        numeric=[v for k,v in asdict(self).items() if k not in {'selection_mode','execution_enabled'}]
        if not np.isfinite(numeric).all():raise ValueError('Timing parameters must be finite')
        if self.dt_ms<=0 or self.dt_ms>100 or self.threshold<=0 or self.urgency<=0 or self.priority_gain<0 or self.diffusion<0:raise ValueError('Invalid race parameters')
        if not 0<=self.hold_strength<=.9 or self.preparation_ms<0 or self.sensory_tau_ms<=0 or self.safety_timeout_ms<=self.preparation_ms:raise ValueError('Invalid hold/preparation/timeout')
        if self.heuristic_scale<=0 or self.heuristic_sigma<=0 or self.selection_mode not in {'softmax','race'}:raise ValueError('Invalid heuristic or spatial selection')
        return self

@dataclass
class ProcessingContext:
    initial_evidence: float=0.
    sensory_ceiling: float=.6
    tau_ms: float=250.
    quality_source: str='synthetic controlled input'

    def at(self,time_ms: float,normalization_tau_ms: float=250.) -> dict:
        ceiling=float(np.clip(self.sensory_ceiling,0,1));initial=float(np.clip(self.initial_evidence,0,ceiling));remaining=(ceiling-initial)*np.exp(-max(0,time_ms)/self.tau_ms)
        return dict(sensory_evidence=float(ceiling-remaining),sensory_ceiling=ceiling,acquisition_rate_per_second=float(remaining/(self.tau_ms/1000)),normalized_acquisition=float(np.clip(remaining*normalization_tau_ms/self.tau_ms,0,1)))


def relative_priorities(utilities):
    a=np.asarray(utilities,float)
    if a.ndim!=1 or len(a)==0 or not np.isfinite(a).all():raise ValueError('Finite nonempty utility vector required')
    span=np.ptp(a)
    return (a-a.min())/span if span>1e-12 else np.zeros_like(a)


def diffusion_increment(rng,shape,diffusion: float,dt_seconds: float):
    return diffusion*np.sqrt(dt_seconds)*rng.standard_normal(shape)


def simulate_timing(utilities,processing: ProcessingContext,variant: str,parameters: TimingParameters,rng,*,remaining_ms=10000.,heuristic_terms=None,initial=False,keep_trace=True):
    """Return first passage + preparation; execution is separate in the runner.

    Hold changes drift only; zero strength gives exactly the no-hold trajectory.
    Candidate utilities are onset snapshots of the unchanged spatial policy.
    """
    p=parameters.validate();priorities=relative_priorities(utilities);limit=min(float(remaining_ms),p.safety_timeout_ms)
    if limit<0 or not np.isfinite(limit):raise ValueError('Invalid observation budget')
    if variant not in {'heuristic_lognormal','accumulator','accumulator_hold'}:raise ValueError('Unknown timing variant')
    values=np.zeros(len(priorities));t=0.;crossing=None;winner=None;trace=[]
    def record(at,phase,hold,drift):
        if keep_trace:trace.append(dict(t_ms=float(at),phase=phase,accumulators=values.tolist(),hold=float(hold),drift_per_second=np.asarray(drift).tolist(),**processing.at(at,p.sensory_tau_ms)))
    initial_hold=p.hold_strength*processing.at(0.,p.sensory_tau_ms)['normalized_acquisition'] if variant=='accumulator_hold' else 0.
    initial_drift=(p.urgency+p.priority_gain*priorities)*(1-initial_hold) if variant!='heuristic_lognormal' else np.zeros_like(values)
    record(0.,'processing',initial_hold,initial_drift)
    planned=None
    if variant=='heuristic_lognormal':
        terms=heuristic_terms or {};mean=250. if initial else 180+160*terms.get('info',0)+100*terms.get('task',0)+80*terms.get('novelty',0)
        # Same V2 hard bounds retained as a named baseline, not imposed on races.
        planned=float(np.clip(mean*p.heuristic_scale if initial else rng.lognormal(np.log(mean*p.heuristic_scale)-.5*p.heuristic_sigma**2,p.heuristic_sigma),100,800))
        planned=max(planned,p.preparation_ms)
        crossing=max(0.,planned-p.preparation_ms)
        winner=None
    else:
        while t<limit-1e-9:
            dt=min(p.dt_ms,limit-t);state=processing.at(t,p.sensory_tau_ms);hold=p.hold_strength*state['normalized_acquisition'] if variant=='accumulator_hold' else 0.
            drift=(p.urgency+p.priority_gain*priorities+p.urgency_ramp*t/1000)*(1-hold)
            previous=values.copy();values=np.maximum(0,values+drift*(dt/1000)+diffusion_increment(rng,values.shape,p.diffusion,dt/1000));new_t=t+dt
            crossed=np.flatnonzero(values>=p.threshold)
            if crossed.size:
                fractions=(p.threshold-previous[crossed])/np.maximum(values[crossed]-previous[crossed],1e-12);j=int(np.argmin(fractions));fraction=float(np.clip(fractions[j],0,1));winner=int(crossed[j]);crossing=t+fraction*dt;values=previous+fraction*(values-previous);t=crossing;record(t,'threshold_crossing',hold,drift);break
            t=new_t;record(t,'processing',hold,drift)
        if crossing is not None:planned=crossing+p.preparation_ms
    completed=planned is not None and planned<=limit+1e-9
    dwell=float(planned if completed else limit)
    if variant=='heuristic_lognormal':
        if keep_trace:
            for at in np.arange(p.dt_ms,min(dwell,crossing)+1e-9,p.dt_ms):record(at,'processing',0.,np.zeros_like(values))
            if crossing<=dwell:record(crossing,'heuristic_release',0.,np.zeros_like(values))
    if crossing is not None and crossing<dwell:
        if keep_trace:
            for at in np.arange(crossing+p.dt_ms,dwell,p.dt_ms):record(at,'motor_preparation',0.,np.zeros_like(values))
        record(dwell,'saccade_ready' if completed else 'budget_end',0.,np.zeros_like(values))
    elif not completed:record(dwell,'budget_end',0.,np.zeros_like(values))
    return dict(variant=variant,duration_ms=dwell,decision_time_ms=float(crossing) if crossing is not None and crossing<=limit else None,motor_preparation_ms=float(dwell-crossing) if crossing is not None and crossing<=dwell else 0.,planned_preparation_ms=p.preparation_ms,completed=bool(completed),right_censored=not completed,winner_index=winner,threshold=p.threshold,priorities=priorities.tolist(),trace=trace,safety_timeout=bool(not completed and remaining_ms>p.safety_timeout_ms),processing_context=asdict(processing),selection_mode=p.selection_mode)


def saccade_duration_ms(amplitude_px: float,parameters: TimingParameters,pixels_per_degree=None):
    if not parameters.execution_enabled or amplitude_px<1:return 0.
    value=18+2.2*amplitude_px/pixels_per_degree if pixels_per_degree else 20+.04*amplitude_px
    return float(np.clip(value,10,90))
