"""Interpretable expected-value stopping. Inputs are model observations, never GT.

Scores are heuristic evidence indices, not calibrated presence/absence
probabilities. Correctness is evaluated by a separate module.
"""
from dataclasses import asdict
import numpy as np
from active_observer.active.perception import overlap
from .options import StoppingOptions

class StopController:
    def __init__(self,mode,target,options=None):
        self.mode=mode;self.target=target;self.options=(options or StoppingOptions()).validate();self.history=[(0.,0.)];self.streak=0;self.evidence=[]
    def assess(self,world,observation,detections,candidates,perception,mask,elapsed_ms,dwell_ms):
        o=self.options;memory=world.snapshot();objects=memory['objects'];regions=[r for r in memory['regions'] if mask.valid(r['x'],r['y'])];coverage=float(np.mean([r['coverage'] for r in regions])) if regions else 0.
        information=float(sum(m['confidence']*(1-m['identity_uncertainty']) for m in objects));object_uncertainty=float(np.mean([1-m['confidence'] for m in objects])) if objects else 1.
        hypotheses=perception.hypotheses if perception else [];unresolved=float(np.mean([p['status']!='recognized' for p in hypotheses])) if hypotheses else 0.
        cutoff=elapsed_ms-o.recent_window_ms;past=next(((t,v) for t,v in self.history if t>=cutoff),self.history[-1]);rate=max(0.,information-past[1])/max(.001,(elapsed_ms-past[0])/1000);self.history.append((float(elapsed_ms),information));self.history=[a for a in self.history if a[0]>=cutoff] or self.history[-1:]
        values=[]
        for c in candidates:
            support=c.get('perceptual_support',c['terms'].get('salience',0.));reduction=c.get('expected_uncertainty_reduction',0.);gain=float(reduction*support*np.clip(c['terms'].get('novelty',1.),0,1)) if world.config.information_enabled else 0.
            frontier=.15*(1-coverage);task=c['terms'].get('task',0.) if self.mode=='search' else 0.;cost=o.observation_cost+o.movement_cost*c['terms'].get('move',0.);value=gain+frontier+task-cost;values.append(dict(candidate=c['id'],expected_gain=gain,unobserved_frontier_value=frontier,task_evidence_value=task,cost=cost,net_value=value))
        best=max(values,key=lambda a:a['net_value']);net=float(best['net_value']);local=[]
        if self.mode=='search':
            for d in detections:
                cx=(d.bbox[0]+d.bbox[2])/2;cy=(d.bbox[1]+d.bbox[3])/2
                if d.source!='semantic' or np.hypot(cx-observation.gaze[0],cy-observation.gaze[1])>observation.radius_px:continue
                score=float(d.confidence*d.probabilities.get(self.target,0.))
                if score>=o.possible_target_score:
                    item=dict(bbox=list(d.bbox),score=score,elapsed_ms=float(elapsed_ms),exposure_ms=float(dwell_ms));local.append(item)
            # At most one observation of a given target hypothesis per fixation.
            if local:self.evidence.append(max(local,key=lambda a:a['score']))
        consistent=[]
        if self.evidence:
            anchor=self.evidence[-1];consistent=[e for e in self.evidence if overlap(e['bbox'],anchor['bbox'])>.1 and e['score']>=o.verification_score]
        verified=len(consistent)>=o.verification_observations and sum(e['exposure_ms'] for e in consistent)>=o.verification_exposure_ms
        low=net<=o.expected_value_threshold and rate<=o.recent_gain_rate_threshold and elapsed_ms>=o.minimum_elapsed_ms;self.streak=self.streak+1 if low else 0;stable=self.streak>=o.persistence;decision='continue';outcome=None;reason='Expected value or recent acquisition still supports observation'
        if self.mode=='search' and verified:decision='stop';outcome='target_verified';reason='Repeated consistent central semantic evidence exceeds the declared verification rule'
        elif stable:
            if self.mode=='explore' and coverage>=o.exploration_min_coverage:decision='stop';outcome='exploration_complete';reason='Low expected value and recent gain persist after minimum observed coverage'
            elif self.mode=='search':
                decision='stop';absence=o.allow_absence and coverage>=o.absence_min_coverage and unresolved<=o.absence_max_unresolved_fraction and not self.evidence;outcome='target_judged_absent' if absence else 'unresolved';reason='Coverage and low residual value support a heuristic absence judgment' if absence else 'Low observation value without sufficient evidence to verify target or justify absence'
        return dict(behavior_mode=self.mode,elapsed_ms=float(elapsed_ms),estimated_memory_information=information,information_is_correctness=False,spatial_coverage=coverage,object_observed_count=len(objects),remaining_spatial_uncertainty=1-coverage,remaining_object_uncertainty=object_uncertainty,unresolved_proto_fraction=unresolved,recent_information_gain_rate=rate,expected_value_of_observation=net,expected_value_components=best,all_candidate_observation_values=values,observation_cost=o.observation_cost,low_value_streak=self.streak,persistence_required=o.persistence,target_status='verified' if verified else 'possible' if self.evidence else 'not_observed',target_evidence_count=len(self.evidence),target_verification_observations=len(consistent),target_verification_exposure_ms=sum(e['exposure_ms'] for e in consistent),target_verification_support=max([e['score'] for e in consistent],default=0.),presence_probability=None,absence_probability=None,probability_note='No calibrated presence/absence probability is defined',decision=decision,terminal_outcome=outcome,administrative_timeout=False,reason=reason,parameters=asdict(o))
