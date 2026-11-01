"""Fit/data boundary tests do not substitute synthetic scores for the real study."""
import importlib.util
from pathlib import Path


def module():
    p=Path(__file__).resolve().parents[1]/'scripts/compare_timing.py';s=importlib.util.spec_from_file_location('timing_comparison',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_test_labels_never_enter_fit(tmp_path,monkeypatch):
    m=module();seen=[]
    def scoring(contexts,variant,parameters,repetitions=6):
        seen.extend(c['split'] for c in contexts)
        return dict(mean=10.,image_se=1.,image_losses={'a':10.},rows=[])
    monkeypatch.setattr(m,'score_contexts',scoring)
    c=[dict(split='train',image_id='train-only'),dict(split='validation',image_id='val-only'),dict(split='test',image_id='test-forbidden',duration_ms=99999)]
    result=m.fit_models(c,tmp_path/'fit.json');assert 'test' not in seen;assert result['selected_variant']=='heuristic_lognormal';assert result['training_images']==['train-only']
