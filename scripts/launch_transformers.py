#!/usr/bin/env python3
"""Launch fixed and AWN Multi30K experiments on separate GPU pairs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-gpus', default='0,1')
    parser.add_argument('--awn-gpus', default='2,3')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpu_pairs = [args.fixed_gpus.split(','), args.awn_gpus.split(',')]
    gpu_pairs = [[gpu.strip() for gpu in pair] for pair in gpu_pairs]
    if any(len(pair) != 2 or len(set(pair)) != 2 or not all(g.isdigit() for g in pair)
           for pair in gpu_pairs):
        parser.error('Specify two distinct GPU indices per experiment, e.g. 0,1.')
    if set(gpu_pairs[0]) & set(gpu_pairs[1]):
        parser.error('Fixed and AWN GPU pairs must not overlap.')

    root = Path(__file__).resolve().parents[1]
    log_dir = root / 'RESULTS/GRID/TRANSFORMER' / f'launch_{os.getpid()}'
    env = dict(os.environ, RAY_ENABLE_UV_RUN_RUNTIME_ENV='0')
    processes = []
    with tempfile.TemporaryDirectory(prefix='awn-transformers-') as temporary:
        try:
            for kind, gpus in zip(('fixed', 'awn'), gpu_pairs):
                source = root / f'EXP_CONFIGS/GRID/TRANSFORMER/Multi30K/transformer_{kind}.yml'
                config = yaml.safe_load(source.read_text())
                config['resources']['max_gpus'] = 2
                config['resources']['gpus_subset'] = ','.join(gpus)
                path = Path(temporary) / f'{kind}.yml'
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                print(f'{kind}: GPUs {",".join(gpus)}, max_gpus=2', flush=True)
                if args.dry_run:
                    if kind == 'awn':
                        print(f"awn: theta_prior_scale={config['grid']['theta_prior_scale']}")
                    continue
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f'{kind}.log'
                with log_path.open('w') as output:
                    processes.append(subprocess.Popen(
                        [sys.executable, '-m', 'mlwiz.launch_experiment', '--config-file', str(path)],
                        cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT,
                    ))
                print(f'{kind}: PID {processes[-1].pid}, log {log_path}', flush=True)
            statuses = [process.wait() for process in processes]
            return int(any(statuses))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()


if __name__ == '__main__':
    sys.exit(main())
