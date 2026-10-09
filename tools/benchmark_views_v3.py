"""Synthetic P1 only: no encoder, optimization, network or cached acceptance."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import resource
import platform
import os
import socket
import sys
import statistics
import time
import tracemalloc

import numpy as np

from psdl2.data import load_export
from psdl2.schema import CropConfig, Frame
from psdl2.views import (ViewConfig, TOPOLOGY_POLICY, CONNECTION_SEMANTICS,
                        COMPARISON_BASIS, validate_pair)


def fixture(n, density):
    root = Path(__file__).resolve().parents[1]
    template = load_export(root/'tests/fixtures/review_roi_v1_export/review_rois.json').ready[0]
    channels = {k: np.zeros((n, n), np.uint8) for k in ('layout:0', 'layout:1', 'cnt', 'via:0')}
    channels['valid'] = np.ones((n, n), np.uint8)
    # Parallel full-frame lines 60 cells apart. CNT in guarded lanes cannot change
    # their anchor relation; every timing therefore runs both scales/both views.
    lines = (n//2-30, n//2+30)
    for k, col in zip(('layout:0', 'layout:1'), lines):
        channels[k][:, col] = 1
    rng = np.random.default_rng(2026)
    if density == .01:
        # Even-row/even-column lattice: every CNT is 4-neighbour isolated.
        channels['cnt'][::2, ::2] = rng.random(((n+1)//2, (n+1)//2)) < .04
    else:
        channels['cnt'][:] = rng.random((n, n)) < density
    for col in lines:
        channels['cnt'][:, col-5:col+6] = 0
    row = deepcopy(template.row); row['center_xy_um'] = [100+n*.025, 200+n*.025]
    frame = Frame(template.frame.frame_id, (100, 200), .05, (n, n))
    from dataclasses import asdict
    row['frame'] = {**row['frame'], **asdict(frame)}
    return replace(template, row=row, frame=frame, channels=channels)


def measure(n, density, coarse):
    roi = fixture(n, density)
    near = min(400, n); core = min(200, near)
    cfg = ViewConfig(CropConfig((near*.05,)*2, (n*.05,)*2, 'preserve_exported_layers_v1'),
                     .05, .2 if coarse else .05, 'nearest', 'area_coverage' if coarse else 'nearest',
                     (0, 0), (core*.05,)*2, TOPOLOGY_POLICY, CONNECTION_SEMANTICS, COMPARISON_BASIS, 3)
    crop_times, validation_times, total_times = [], [], []
    for i in range(7):
        start = time.perf_counter(); a, b = cfg.bundle(roi), cfg.bundle(roi)
        cropped = time.perf_counter()
        validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)
        done = time.perf_counter()
        if i >= 2:
            crop_times.append(cropped-start); validation_times.append(done-cropped); total_times.append(done-start)
    # Memory is a separate full invocation so tracing doesn't distort medians.
    tracemalloc.start(); a, b = cfg.bundle(roi), cfg.bundle(roi)
    validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    crop, validation = statistics.median(crop_times), statistics.median(validation_times)
    return dict(source_cells=n, near_cells=near, wide_cells=n//4 if coarse else n, core_cells=core,
                mode='area_coverage' if coarse else 'nearest', channels=5, seed=2026,
                requested_cnt_density=density, actual_cnt_density=float(roi.channels['cnt'].mean()),
                crop_seconds=crop, validation_seconds=validation, pair_seconds=statistics.median(total_times),
                tracemalloc_peak_mib=peak/2**20, process_maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024),
                bound_seconds=max(2, 10*crop), passed=validation <= max(2, 10*crop))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def denied(*args, **kwargs): raise AssertionError('benchmark network forbidden')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    import torch
    torch.set_num_threads(1)
    rows = [measure(n, density, coarse) for density in (.01, .3)
            for coarse in (False, True) for n in (500, 1000, 2000)]
    growth = []
    for density in (.01, .3):
        for mode in ('nearest', 'area_coverage'):
            group = [r for r in rows if r['requested_cnt_density'] == density and r['mode'] == mode]
            for a, b in zip(group, group[1:]):
                ratio = b['validation_seconds']/a['validation_seconds']
                growth.append(dict(density=density, mode=mode, from_cells=a['source_cells'], to_cells=b['source_cells'],
                                   ratio=ratio, passed=ratio <= 6))
    result = dict(synthetic_only=True, platform=platform.platform(), machine=platform.machine(),
                  python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                  thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, threads=1, warmups=2, repeats=5, rows=rows, growth=growth,
                  passed=all(r['passed'] for r in rows+growth))
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
