"""Regression for Windows cp1252 defaults: UTF-8 must be explicit at every boundary."""
import ast
import pytest
from pathlib import Path
import numpy as np
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.viz.html import write_html


def test_inspector_roundtrips_under_cp1252_default(tmp_path,monkeypatch):
    original=Path.open
    def windows_open(self,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
        # Emulate Windows' default without depending on the test host locale.
        if 'b' not in mode and encoding in (None,'locale'):encoding='cp1252'
        return original(self,mode,buffering,encoding,errors,newline)
    monkeypatch.setattr(Path,'open',windows_open)
    with pytest.raises(UnicodeDecodeError):
        Path(__file__).resolve().parents[1].joinpath('active_observer/viz/viewer.html').read_text()
    image=np.zeros((20,30,3),np.uint8)
    result=run(image,Config(max_fixations=1),image_id='café-视觉',target='椅子')
    path=tmp_path/'inspection.html';write_html(image,result,path)
    content=path.read_bytes().decode('utf-8')
    assert '← Previous' in content
    assert '<meta charset="utf-8">' in content
    assert '__OBSERVER_DATA__' not in content


def test_all_text_boundaries_declare_encoding():
    root=Path(__file__).resolve().parents[1]
    for directory in ['active_observer','scripts']:
        for path in (root/directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'read_text','write_text','to_csv'}:
                    assert any(k.arg=='encoding' for k in node.keywords),f'{path}:{node.lineno}'


def test_simple_run_does_not_import_pandas():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c','import active_observer.cli; import sys; assert "pandas" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
