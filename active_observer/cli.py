"""One photograph, one observer, optional causal replay inspection."""
import argparse,json
from pathlib import Path
from .config import Config
from .io.image import load_image
from .simulation.runner import run
from .simulation.replay import inspect_at
from .simulation.recording import save_run
from .viz.replay import write_replay


def main():
    parser=argparse.ArgumentParser(prog='active-observer')
    parser.add_argument('image',help='RGB photograph of a real environment')
    parser.add_argument('--config');parser.add_argument('--seed',type=int)
    parser.add_argument('--out',default='outputs/reconstruction')
    parser.add_argument('--inspect-seconds',type=float,help='Inspect a prefix; does not change autonomous execution')
    args=parser.parse_args();cfg=Config.load(args.config)
    if args.seed is not None:cfg.seed=args.seed
    image=load_image(args.image);result=run(image,cfg,image_id=Path(args.image).stem)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    save_run(result,out/'run.json')
    write_replay(image,result,out/'inspector.html')
    if args.inspect_seconds is not None:
        (out/'inspection.json').write_text(json.dumps(inspect_at(result,args.inspect_seconds*1000),allow_nan=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),termination=result['termination'],capabilities=result['capabilities'])))

if __name__=='__main__':main()
