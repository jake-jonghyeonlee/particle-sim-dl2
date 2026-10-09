"""File-contract synthetic execution; structural checks only, no ranking-quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from psdl2.checkpoint import read_checkpoint
from psdl2.data import load_export
from psdl2.train import read_config, train

ROOT = Path(__file__).parents[1]


def synthetic_exchange(root):
    """Six ready ROIs, three product lineages, two hand-written local structures."""
    template = json.loads((ROOT/'tests/fixtures/review_roi_v1_export/review_rois.json').read_text())
    ready = next(r for r in template['rois'] if r['status'] == 'ready')
    payload = {**template, 'producer_commit': '0'*40, 'assets_root': '.',
               'rois': [], 'excluded_records': [], 'status': 'available'}
    lineages, assignments = {}, {}
    for product in range(3):
        pid = f'SYNTHETIC_{product}'; lineages[pid] = f'lineage_{product}'
        for structure in range(2):
            rid = f'roi_{product}_{structure}'
            row = deepcopy(ready)
            row.update(roi_id=rid, product_id=pid, case_id='SYNTHETIC_CASE', frame_id=f'frame_{rid}',
                       source_version='synthetic_e2e_original_v1', center_xy_um=[102, 202])
            row['source_identity'].update(product_id=pid, case_id=row['case_id'], frame_id=row['frame_id'])
            row['frame'].update(frame_id=row['frame_id'], shape_yx=[8, 8])
            row['endpoints'][0]['actual_layer'] = 'M1'; row['endpoints'][1]['actual_layer'] = 'M3'
            row['failure_type'] = 'vertical_short'; row['analysis_layer'] = 'M3'
            for layout, endpoint in zip(row['source_assets']['layouts'], row['endpoints']):
                layout['endpoint'] = endpoint.copy()
            a = np.zeros((8, 8), np.uint8); b = np.zeros_like(a); cnt = np.zeros_like(a)
            if structure == 0:
                a[2, 2] = 1; b[2, 5] = 1; cnt[3, 4] = 1
            else:
                a[2:4, 2:4] = 1; b[4:6, 4:6] = 1; cnt[3:5, 4] = 1
            refs = [layout['asset'] for layout in row['source_assets']['layouts']] + [
                row['source_assets']['cnt'], row['source_assets']['valid']]
            for channel, (ref, values) in enumerate(zip(refs, (a, b, cnt, np.ones_like(a)))):
                path = root/'assets'/f'{rid}_{channel}.npy'; path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
                ref.update(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            payload['rois'].append(row)
            assignments[rid] = ('train', 'validation', 'test')[product]
    payload['source_manifest_hash'] = hashlib.sha256(json.dumps(payload['rois'], sort_keys=True).encode()).hexdigest()
    path = root/'review_rois.json'; path.write_text(json.dumps(payload))
    # Every train/index/search stage imports this actual file. No ready ROI objects
    # are fabricated in memory to bypass the consumer's validation.
    imported = load_export(path)
    assert len(imported.ready) == 6 and not imported.blocked
    config = read_config(ROOT/'configs/synthetic.example.json')
    for name, value in [('lineages', lineages), ('split', assignments), ('config', config)]:
        (root/f'{name}.json').write_text(json.dumps(value))
    return path, imported, config, lineages, assignments


def test_default_one_epoch_executes_exactly_one_step_and_atomic_checkpoint(tmp_path, monkeypatch):
    _, export, config, lineages, assignments = synthetic_exchange(tmp_path)
    # Smallest valid training split is two locations; batch size two gives one step.
    assert config['epochs'] == 1 and config['batch_size'] == 2
    steps, replacements = [], []
    original_step = torch.optim.Adam.step
    def step(optimizer, *args, **kwargs):
        steps.append(1)
        assert len(steps) <= 1, 'default suite must not exceed one optimizer step'
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    import psdl2.checkpoint as checkpoint
    original_replace = checkpoint.os.replace
    def replace(source, target):
        replacements.append((Path(source), Path(target)))
        return original_replace(source, target)
    monkeypatch.setattr(checkpoint.os, 'replace', replace)
    path = tmp_path/'checkpoint.pt'
    result = train(export.ready, config, path, lineages=lineages, assignments=assignments)
    assert steps == [1] and len(result['history']) == 1 and result['history'][0]['epoch'] == 1
    assert all(np.isfinite(result['history'][0][key]) for key in ('total', 'alignment', 'variance', 'covariance', 'supervised'))
    assert len(replacements) == 1 and replacements[0][1] == path
    assert replacements[0][0].parent == path.parent and not replacements[0][0].exists()
    raw = torch.load(path, weights_only=True)
    checked = read_checkpoint(path, raw['config'])
    assert checked['epoch'] == 1
    assert all(state['step'].item() == 1 for state in checked['optimizer']['state'].values())
    assert checked['config']['labels'] is None


@pytest.mark.training
def test_opt_in_real_cli_train_resume_index_search_evaluate(tmp_path, capsys, record_property):
    started = time.perf_counter()
    path, export, config, _, _ = synthetic_exchange(tmp_path)
    from psdl2.train import main as train_main
    from psdl2.index import main as index_main
    from psdl2.search import main as search_main
    from psdl2.evaluate import main as evaluate_main
    config['epochs'] = 2; config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    checkpoint_path = tmp_path/'checkpoint.pt'
    args = [str(path), '--config', str(config_path), '--out', str(checkpoint_path),
            '--lineages', str(tmp_path/'lineages.json'), '--split', str(tmp_path/'split.json')]
    assert '--labels' not in args
    assert train_main(args) == 0
    initial = json.loads(capsys.readouterr().out)
    assert [entry['epoch'] for entry in initial['history']] == [1, 2]
    for entry in initial['history']:
        assert all(np.isfinite(entry[key]) for key in ('total', 'alignment', 'variance', 'covariance', 'supervised'))
        assert entry['negative_pair_count'] == 0 and entry['supervised_known_pair_count'] == 0
    assert torch.load(checkpoint_path, weights_only=True)['epoch'] == 2
    # Extend only the epoch budget; all model/data/view/loss fingerprints remain fixed.
    config['epochs'] = 4; config_path.write_text(json.dumps(config))
    assert train_main(args+['--resume', str(checkpoint_path)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert [entry['epoch'] for entry in resumed['history']] == [3, 4]
    assert resumed['config_fingerprint'] == initial['config_fingerprint']
    raw = torch.load(checkpoint_path, weights_only=True)
    assert raw['epoch'] == 4 and all(state['step'].item() == 4 for state in raw['optimizer']['state'].values())
    for entry in resumed['history']:
        assert all(np.isfinite(entry[key]) for key in ('total', 'alignment', 'variance', 'covariance', 'supervised'))
    assert raw['target_epochs'] == 4
    before = checkpoint_path.read_bytes()
    assert train_main(args+['--resume', str(checkpoint_path)]) == 2
    assert 'resume_target_epochs_must_exceed_saved_epoch' in capsys.readouterr().err
    assert checkpoint_path.read_bytes() == before
    changed = deepcopy(config); changed['learning_rate'] *= 2
    config_path.write_text(json.dumps(changed))
    assert train_main(args+['--resume', str(checkpoint_path)]) == 2
    assert 'incompatible_config_fingerprint' in capsys.readouterr().err
    assert checkpoint_path.read_bytes() == before
    index_path = tmp_path/'index.json'
    assert index_main([str(path), '--checkpoint', str(checkpoint_path), '--out', str(index_path)]) == 0
    capsys.readouterr(); index = json.loads(index_path.read_text())
    assert len(index['records']) == 6 and not index['rejected']
    expected_fields = {'roi_id', 'source_identity', 'frame', 'case_id', 'valid_coverage', 'source_assets', 'source_versions'}
    assert all(expected_fields <= record.keys() for record in index['records'])
    cohort = tmp_path/'cohort.json'; cohort.write_text('["SYNTHETIC_1"]')
    result_path = tmp_path/'result.json'
    search_args = [str(path), '--roi-id', 'roi_0_0', '--checkpoint', str(checkpoint_path),
                   '--index', str(index_path), '--cohort', str(cohort), '--topk', '6', '--out', str(result_path)]
    assert search_main(search_args) == 0
    capsys.readouterr(); primary_only = json.loads(result_path.read_text())
    assert len(primary_only['primary']) == 2 and primary_only['supplementary'] == []
    assert search_main(search_args+['--expand']) == 0
    capsys.readouterr(); expanded = json.loads(result_path.read_text())
    assert expanded['primary'] == primary_only['primary'] and len(expanded['supplementary']) == 3
    for scope in ('primary', 'supplementary'):
        for row in expanded[scope]:
            assert expected_fields | {'score', 'search_scope'} <= row.keys()
            assert row['search_scope'] == scope and np.isfinite(row['score'])
    results = tmp_path/'results.json'; results.write_text(json.dumps([expanded]))
    assert evaluate_main([str(results), '--k', '2']) == 3
    metrics = json.loads(capsys.readouterr().out)
    assert metrics['status'] == 'unavailable' and metrics['recall_at_k'] is None
    assert metrics['reason'] == 'correspondence_labels_not_supplied'
    elapsed = time.perf_counter()-started
    record_property('wall_time_seconds', elapsed)
    print(f'Opt-in synthetic CLI chain wall time: {elapsed:.3f}s')
