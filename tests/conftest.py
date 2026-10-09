"""Hand-defined original layouts; all validation tests forbid socket connections."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('network is forbidden during input-contract validation')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


FIXTURE = Path(__file__).parent / 'fixtures/review_roi_v1_export'


@pytest.fixture
def source(tmp_path):
    # Use the actual producer structure, with separate hand-specified crop masks.
    payload = json.loads((FIXTURE / 'review_rois.json').read_text())
    row = next(row for row in payload['rois'] if row['status'] == 'ready')
    payload['rois'] = [row]
    payload['excluded_records'] = []
    payload['assets_root'] = '.'
    row['center_xy_um'] = [102, 202]
    row['frame']['shape_yx'] = [8, 8]
    row['failure_type'] = 'vertical_short'
    row['endpoints'][0]['actual_layer'] = 'M1'
    row['endpoints'][1]['actual_layer'] = 'M3'
    row['analysis_layer'] = 'M3'
    for entry, ep in zip(row['source_assets']['layouts'], row['endpoints']):
        entry['endpoint'] = ep.copy()
    a = np.zeros((8, 8), np.uint8); a[2, 2] = 1
    b = np.zeros((8, 8), np.uint8); b[2, 5] = 1
    cnt = np.zeros((8, 8), np.uint8); cnt[3, 4] = 1
    valid = np.ones((8, 8), np.uint8); valid[0, 0] = 0
    refs = [entry['asset'] for entry in row['source_assets']['layouts']] + [row['source_assets']['cnt'], row['source_assets']['valid']]
    for ref, mask in zip(refs, (a, b, cnt, valid)):
        path = tmp_path / ref['path']; path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)
        ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    def write():
        path = tmp_path / 'review_rois.json'
        path.write_text(json.dumps(payload))
        return path
    return payload, write, tmp_path
