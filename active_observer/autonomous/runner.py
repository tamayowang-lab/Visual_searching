"""One causal autonomous trajectory; observation duration is only a replay view."""
from dataclasses import replace, asdict
from copy import deepcopy
import hashlib
import numpy as np
import cv2
from active_observer.active.perception import ActivePerception
from active_observer.active.options import ActiveOptions
from active_observer.active.masking import StimulusMask
from active_observer.attention.timing import TimingParameters, ProcessingContext, simulate_timing, saccade_duration_ms
from active_observer.experiment.options import ExperimentOptions, StoppingOptions
from active_observer.experiment.stopping import StopController
from active_observer.io.image import Camera
from active_observer.perception.foveation import observe
from active_observer.perception.factory import ProviderFactory
from active_observer.world.model import World, decay_without_observation
from .field import PriorityField


def run_autonomous(image, cfg, *, image_id='image', target='', start=None,
                   object_provider=None, depth_provider=None, camera=None,
                   provider_factory=None, include_depth_reference=True,
                   valid_mask=None, stimulus_bounds=None, frame_callback=None):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')
    options = ExperimentOptions(**cfg.experiment).validate()
    target = target if options.behavior_mode == 'search' else ''
    if options.behavior_mode == 'search' and not target:
        raise ValueError('Search requires a target')
    h, w = image.shape[:2]
    camera = camera or Camera(w, h)
    if (camera.width, camera.height) != (w, h):
        raise ValueError('Camera/image dimensions differ')
    gaze = tuple(start) if start is not None else ((w-1)/2, (h-1)/2)
    mask = StimulusMask(w, h, valid_mask, stimulus_bounds)
    if not mask.valid(*gaze):
        raise ValueError('Starting fixation is outside the valid stimulus')
    # Legacy grid, counts, timing variants and ablation flags cannot control this path.
    stop_options = StoppingOptions(**{'allow_absence':False, **cfg.stopping}).validate()
    cfg = replace(cfg, experiment=asdict(options), stopping=asdict(stop_options), depth_enabled=False, timing_model='heuristic_lognormal', timing={**cfg.timing,'selection_mode':'softmax'}, evidence_model='bounded', memory_enabled=True, semantic_enabled=True, information_enabled=True)
    world = World(w, h, cfg, spatial_grid=False)
    field = PriorityField(w, h, mask.mask, temperature=cfg.spatial_temperature)
    if object_provider is None:
        object_provider, _ = (provider_factory or ProviderFactory()).get(replace(cfg, depth_enabled=False))
    peripheral = ActivePerception(object_provider, ActiveOptions(peripheral_guidance=True, coarse_width=32), mask, cfg.perception.get('cache_directory','.cache/observer')+'/peripheral-real')
    controller = StopController(options.behavior_mode, target, stop_options)
    rng = np.random.default_rng(cfg.seed)
    timing_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    params = TimingParameters(**{**cfg.timing, 'selection_mode':'softmax'}).validate()
    fixations, saccades = [], []
    elapsed, pending = 0., None
    termination = None
    while elapsed < options.administrative_timeout_ms:
        before = deepcopy(world.snapshot())
        observation = observe(image, gaze, cfg, strict_periphery=True)
        detections = object_provider.detect(observation)
        coarse = cv2.resize(cv2.resize(observation.rgb, (min(32,w), max(1,round(h*min(32,w)/w))), interpolation=cv2.INTER_AREA), (w,h), interpolation=cv2.INTER_LINEAR)
        cues, cue_source = peripheral.coarse_semantics(coarse, observation)
        # Providers receive only the actual observation; unlabeled structural evidence stays
        # distinct from centrally gated semantic detections in memory and the UI.
        onset_field = field.score(observation, world, target, cues)
        timing = simulate_timing([float(onset_field['total'].max())], ProcessingContext(),
                                 'heuristic_lognormal', params, timing_rng,
                                 remaining_ms=options.administrative_timeout_ms-elapsed,
                                 heuristic_terms=pending['terms'] if pending else None,
                                 initial=pending is None, keep_trace=False)
        dwell = timing['duration_ms']
        onset = elapsed
        world.update(detections, observation, dwell)
        field.update(observation, dwell)
        elapsed += dwell
        priority = field.score(observation, world, target, cues)
        selected = field.choose(priority, rng)
        coverage = float((1-np.exp(-field.exposure))[field.valid].mean())
        # Assess the most useful possible observation, not just the movement-biased winner.
        gain_map = priority['gain'] + priority['task']
        gy, gx = np.unravel_index(np.argmax(np.where(field.valid, gain_map, -np.inf)), gain_map.shape)
        best = dict(selected, id='maximum-remaining-value',
                    expected_uncertainty_reduction=float(priority['gain'][gy,gx]),
                    perceptual_support=1., terms=dict(novelty=1., task=float(priority['task'][gy,gx]), move=float(priority['movement'][gy,gx])))
        stop = controller.assess(world, observation, detections, [best], None, mask, elapsed, dwell,
                                 spatial_coverage=coverage, frontier_value=0.)
        stop['threshold'] = controller.options.expected_value_threshold
        timeout = not timing['completed'] or elapsed >= options.administrative_timeout_ms
        if timeout:
            stop.update(decision='stop', terminal_outcome='administrative_timeout', administrative_timeout=True,
                        reason='Administrative elapsed-time safeguard reached; cognitive completion is unknown')
        # No step-count safeguard: only elapsed-time administrative protection.
        if stop['decision'] == 'stop':
            outcome = stop['terminal_outcome']
            termination = dict(action='stop', outcome=outcome, reason=stop['reason'], elapsed_ms=elapsed,
                               administrative_timeout=timeout,
                               cognitive_completion=outcome in {'target_verified','exploration_complete','target_judged_absent'})
        trace = dict(memory_before=before, memory_after=deepcopy(world.snapshot()),
                     observation=dict(rgb_sha256=hashlib.sha256(observation.rgb.tobytes()).hexdigest(),
                                      central_radius_px=observation.radius_px,
                                      peripheral_cues=deepcopy(cues), peripheral_source=cue_source,
                                      detections=[dict(bbox=d.bbox, probabilities=d.probabilities, confidence=d.confidence, source=d.source) for d in detections]),
                     priority=field.snapshot(priority), selected_next=None if termination else selected,
                     selection=dict(action='stop' if termination else 'fixate', location=None if termination else [selected['x'], selected['y']],
                                    reason=stop['reason'] if termination else f'Seeded softmax over the dense priority field; temperature {cfg.spatial_temperature}; independent lognormal timing'),
                     stopping=stop)
        fixation = dict(step=len(fixations), x_px=float(gaze[0]), y_px=float(gaze[1]),
                        duration_ms=dwell, fixation_onset_ms=onset, fixation_offset_ms=elapsed,
                        memory=deepcopy(world.snapshot()), decision_trace=trace, timing=timing,
                        right_censored=not timing['completed'], saccade_duration_ms=0., next_fixation_onset_ms=None,
                        chosen_candidate=pending['id'] if pending else 'supplied-start',
                        candidate_details=pending or {},
                        utility_terms=pending['terms'] if pending else {}, incoming_choice=deepcopy(pending))
        fixations.append(fixation)
        if frame_callback:
            frame_callback(fixation, observation)
        if termination:
            break
        distance = float(np.hypot(selected['x']-gaze[0], selected['y']-gaze[1]))
        execution = saccade_duration_ms(distance, params, cfg.pixels_per_degree)
        observed = min(execution, options.administrative_timeout_ms-elapsed)
        fixation.update(saccade_duration_ms=execution, next_fixation_onset_ms=elapsed+execution,
                        next_fixation_xy=[selected['x'],selected['y']])
        saccades.append(dict(onset_ms=elapsed, duration_ms=execution, observed_duration_ms=observed,
                             from_xy=list(gaze), to_xy=[selected['x'],selected['y']]))
        elapsed += observed
        decay_without_observation(world, observed)
        field.ior *= np.exp(-observed/1800)
        if elapsed >= options.administrative_timeout_ms:
            termination = dict(action='stop', outcome='administrative_timeout', reason='Administrative timeout during saccade',
                               elapsed_ms=elapsed, administrative_timeout=True, cognitive_completion=False)
            trace['administrative_termination_after_saccade'] = termination.copy()
            break
        gaze = (selected['x'], selected['y'])
        pending = selected
    return dict(schema_version=6, model='autonomous-dense-field', image_id=image_id,
                width=w, height=h, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), config=cfg.to_dict(), target=target,
                experimental_control=dict(behavior_mode=options.behavior_mode, observation_condition='autonomous_source_trajectory'),
                fixations=fixations, saccades=saccades, duration_ms=elapsed,
                termination=termination, stop_reason=termination['reason'],
                perception=object_provider.metadata() if hasattr(object_provider,'metadata') else dict(provider=type(object_provider).__name__),
                validity=mask.metadata(), camera=camera.metadata(), depth=dict(available=False, reference=None),
                assumptions=['Heuristic evidence and value indices, not calibrated probabilities',
                             'Strict central RGB and 32-pixel-wide peripheral RGB; no remote semantic bank',
                             'Legacy fixation count, time budget, grid and phase flags are ignored',
                             'Memory commits at fixation offset; replay must not reveal future commits'])
