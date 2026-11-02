"""Real environment photographs with actual gaze-dependent semantic perception."""
from pathlib import Path
import argparse
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.viz.autonomous import image_uri, write_demo

PHOTO_ROOT = Path(__file__).resolve().parents[2] / 'examples' / 'photos'


def scenes():
    return json.loads((PHOTO_ROOT/'manifest.json').read_text(encoding='utf-8'))['scenes']


def scene_image(scene):
    record = next((r for r in scenes() if r['id']==scene), None)
    if record is None:raise ValueError('Unknown environment photograph')
    image = cv2.imread(str(PHOTO_ROOT/Path(record['path']).name))
    if image is None:raise ValueError('Photograph is missing; run scripts/prepare_photo_demo.py --download')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def demo_config(seed=42, mode='explore'):
    return Config(seed=seed, central_fraction=.12, depth_enabled=False,
                  experiment=dict(behavior_mode=mode, observation_condition='until_finished',
                                  administrative_timeout_ms=20000.))


def make_entry(scene, mode, seed, asset_directory=None, provider_factory=None):
    record = next(r for r in scenes() if r['id']==scene)
    name, target = record['name'], record['target']
    image = scene_image(scene)
    frames = []
    def asset(rgb):
        if asset_directory is None:return image_uri(rgb)
        import hashlib
        directory=Path(asset_directory);(directory/'frames').mkdir(parents=True,exist_ok=True)
        filename='frames/'+hashlib.sha256(rgb.tobytes()).hexdigest()+'.png'
        dest=directory/filename
        if not dest.exists():cv2.imwrite(str(dest),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        return filename
    def capture(fixation, observation):
        frames.append(asset(observation.rgb))
    result = run(image, demo_config(seed, mode), image_id=scene, target=target, frame_callback=capture, provider_factory=provider_factory)
    return dict(scene=scene, name=name, mode=mode, seed=seed, image=asset(image), frames=frames, result=result)


def build(out, seeds):
    from active_observer.perception.factory import ProviderFactory
    factory=ProviderFactory()
    entries = [make_entry(record['id'], mode, seed, out, factory) for record in scenes()
               for mode in ['explore', 'search'] for seed in seeds]
    write_demo(dict(entries=entries, launch_command='python -m active_observer.autonomous.demo --serve\nOpen http://localhost:8000/', regeneration=f'python -m active_observer.autonomous.demo --out {out} --seeds '+ ' '.join(map(str,seeds))), Path(out)/'index.html', external=True)
    summary = [dict(scene=e['scene'], mode=e['mode'], seed=e['seed'], fixations=len(e['result']['fixations']),
                    duration_ms=round(e['result']['duration_ms']), termination=e['result']['termination']) for e in entries]
    (Path(out)/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def serve(out, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(Path(out).resolve()), **kwargs)
        def do_GET(self):
            query = urlparse(self.path)
            if query.path != '/api/run':
                return super().do_GET()
            try:
                args = parse_qs(query.query)
                scene, mode, seed = args['scene'][0], args['mode'][0], int(args['seed'][0])
                if mode not in {'explore','search'} or not 0 <= seed <= 2**32-1:
                    raise ValueError('Invalid mode or seed')
                payload = json.dumps(make_entry(scene, mode, seed), allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, KeyError, StopIteration) as exc:
                self.send_error(400, str(exc))
    print(f'Observer demonstration: http://localhost:{port}/ (Python regeneration enabled)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/autonomous')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    build(args.out, args.seeds)
    if args.serve:
        serve(args.out, args.port)


if __name__ == '__main__':
    main()
