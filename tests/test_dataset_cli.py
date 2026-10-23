"""train / index / search CLIs over psdl2_dataset_v1, and the unchanged single-export path."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from psdl2.index import main as index_main
from psdl2.search import main as search_main
from psdl2.train import main as train_main
from test_dataset import ENDPOINTS, derivation, make_export, source_for, write_dataset
from test_end_to_end import synthetic_exchange

ROOT = Path(__file__).parents[1]
BASELINE = 'f85e8c0'


def build_world(root, *, derivations=True, roles=('data', 'data', 'data'), epochs=1):
    """Three single-lineage products (train / validation / test) as one dataset."""
    root = Path(root)
    sources, lineages, split = [], {}, {}
    for i, product in enumerate(('P0', 'P1', 'P2')):
        export = make_export(root, f'prod{i}', product=product, shift=i % 2, role=roles[i])
        extra = {}
        if derivations:
            value = derivation(endpoints=[ENDPOINTS[0], {**ENDPOINTS[1], 'role': roles[i]}])
            extra['mask_derivation'] = value
        sources.append(source_for(export, root, **extra))
        lineages[product] = f'lineage_{i}'
        for structure in (0, 1):
            split[f'{product}_{structure}'] = ('train', 'validation', 'test')[i]
    manifest = write_dataset(root, sources)
    config = json.loads((ROOT / 'configs/synthetic.example.json').read_text())['config']
    config['epochs'] = epochs
    for name, value in (('lineages', lineages), ('split', split), ('config', config)):
        (root / f'{name}.json').write_text(json.dumps(value))
    (root / 'cohort.json').write_text(json.dumps(['P1', 'P2']))
    return manifest


def train_args(root, manifest, out, *extra):
    return ['--dataset', str(manifest), '--config', str(root / 'config.json'), '--out', str(out),
            '--lineages', str(root / 'lineages.json'), '--split', str(root / 'split.json'), '--no-progress', *extra]


def error_of(capsys):
    return json.loads(capsys.readouterr().err)['error']


def test_cli_requires_exactly_one_of_export_or_dataset(tmp_path, capsys):
    manifest = build_world(tmp_path)
    export = tmp_path / 'prod0' / 'review_rois.json'
    base = train_args(tmp_path, manifest, tmp_path / 'c.pt')
    both = ['--config', str(tmp_path / 'config.json'), '--out', str(tmp_path / 'c.pt'), '--lineages', str(tmp_path / 'lineages.json'),
            '--split', str(tmp_path / 'split.json'), '--dataset', str(manifest), str(export)]
    none = base[2:]
    assert train_main(both) == 2 and error_of(capsys) == 'exactly_one_of_export_or_dataset_required'
    assert train_main(none) == 2 and error_of(capsys) == 'exactly_one_of_export_or_dataset_required'
    assert train_main(base + ['--assets-root', str(tmp_path)]) == 2 and error_of(capsys) == 'assets_root_not_allowed_with_dataset'
    assert not (tmp_path / 'c.pt').exists()
    assert train_main(base) == 0; capsys.readouterr()
    ckpt = tmp_path / 'c.pt'
    out = tmp_path / 'i.json'
    assert index_main([str(export), '--dataset', str(manifest), '--checkpoint', str(ckpt), '--out', str(out)]) == 2
    assert error_of(capsys) == 'exactly_one_of_export_or_dataset_required'
    assert index_main(['--checkpoint', str(ckpt), '--out', str(out)]) == 2
    assert error_of(capsys) == 'exactly_one_of_export_or_dataset_required'
    assert not out.exists()
    common = ['--roi-id', 'P0_0', '--checkpoint', str(ckpt), '--index', str(out), '--cohort', str(tmp_path / 'cohort.json'),
              '--topk', '3', '--out', str(tmp_path / 'r.json')]
    assert search_main([str(export), '--query-dataset', str(manifest)] + common) == 2
    assert error_of(capsys) == 'exactly_one_of_export_or_query_dataset_required'
    assert search_main(common) == 2 and error_of(capsys) == 'exactly_one_of_export_or_query_dataset_required'


def run_chain(root, manifest, capsys, *, epochs=1):
    ckpt, index = root / 'c.pt', root / 'index.json'
    assert train_main(train_args(root, manifest, ckpt)) == 0
    trained = json.loads(capsys.readouterr().out)
    assert index_main(['--dataset', str(manifest), '--checkpoint', str(ckpt), '--out', str(index)]) == 0
    capsys.readouterr()
    return ckpt, index, trained


def search_for(root, query_manifest, ckpt, index, roi, capsys, *extra, out='r.json'):
    code = search_main(['--query-dataset', str(query_manifest), '--roi-id', roi, '--checkpoint', str(ckpt), '--index', str(index),
                        '--cohort', str(root / 'cohort.json'), '--topk', '6', '--out', str(root / out), *extra])
    capsys.readouterr()
    return code, json.loads((root / out).read_text()) if (root / out).exists() else None


def test_dataset_train_index_search_cli_with_derivation_matrix(tmp_path, capsys):
    manifest = build_world(tmp_path)
    ckpt, index_path, trained = run_chain(tmp_path, manifest, capsys)
    raw = torch.load(ckpt, weights_only=True)
    # Dataset-only fingerprint content: the manifest, export hashes and the normalized derivation.
    dataset = raw['config']['dataset']
    assert dataset['schema_version'] == 'psdl2_dataset_v1' and len(dataset['sources']) == 3
    assert all(len(s['export_sha256']) == 64 and s['mask_derivation']['rule_version'] == 'between_stack_order_v1'
               for s in dataset['sources'])
    index = json.loads(index_path.read_text())
    assert 'producer_versions' not in index and len(index['records']) == 6
    assert [s['export'] for s in index['sources']] == [f'prod{i}/review_rois.json' for i in range(3)]
    assert all(s['export_sha256'] and s['versions']['schema_version'] == 'review_roi_v1' for s in index['sources'])
    assert all(r['case_signature']['mask_derivation']['removed_layers'] == ['M2'] for r in index['records'])
    # Real search CLI: the query dataset is a manifest around the (unchanged) query export.
    query_dir = tmp_path / 'query'; query_dir.mkdir()
    qexport = tmp_path / 'prod0' / 'review_rois.json'
    same = write_dataset(query_dir, [source_for(qexport, query_dir, mask_derivation=derivation())], 'same.json')
    code, result = search_for(tmp_path, same, ckpt, index_path, 'P0_0', capsys)
    assert code == 0 and result['status'] == 'available' and not result['rejected']
    assert {r['roi_id'] for r in result['primary']} == {'P1_0', 'P1_1', 'P2_0', 'P2_1'}
    assert 'mask_derivation_comparison' not in result
    # A different removed-layer list excludes every candidate, with the reason recorded.
    other = write_dataset(query_dir, [source_for(qexport, query_dir, mask_derivation=derivation(removed=('M2', 'M9')))], 'other.json')
    code, result = search_for(tmp_path, other, ckpt, index_path, 'P0_0', capsys)
    assert code == 3 and result['status'] == 'unavailable' and not result['primary']
    assert {r['reason'] for r in result['rejected'] if r['roi_id'].startswith(('P1', 'P2'))} == {'mask_derivation_mismatch'}
    # Override: compared nothing, and the result says so.
    code, result = search_for(tmp_path, other, ckpt, index_path, 'P0_0', capsys, '--ignore-mask-derivation')
    assert code == 0 and result['mask_derivation_comparison'] == 'ignored' and len(result['primary']) == 4
    # Query without a derivation against a derivation index: unknown, never guessed.
    plain = write_dataset(query_dir, [source_for(qexport, query_dir)], 'plain.json')
    code, result = search_for(tmp_path, plain, ckpt, index_path, 'P0_0', capsys)
    assert code == 3 and {r['reason'] for r in result['rejected'] if r['roi_id'].startswith(('P1', 'P2'))} == {'mask_derivation_unknown'}


def test_changed_dataset_content_rejects_resume_without_touching_the_checkpoint(tmp_path, capsys):
    manifest = build_world(tmp_path, epochs=1)
    ckpt = tmp_path / 'c.pt'
    assert train_main(train_args(tmp_path, manifest, ckpt)) == 0; capsys.readouterr()
    config = json.loads((tmp_path / 'config.json').read_text()); config['epochs'] = 2
    (tmp_path / 'config.json').write_text(json.dumps(config))
    before = ckpt.read_bytes()
    # Unchanged dataset resumes fine on a copy of the checkpoint.
    resumed = tmp_path / 'resumed.pt'; shutil.copy(ckpt, resumed)
    assert train_main(train_args(tmp_path, manifest, resumed, '--resume', str(resumed))) == 0; capsys.readouterr()
    assert torch.load(resumed, weights_only=True)['epoch'] == 2
    payload = json.loads(manifest.read_text())
    payload['sources'][0]['mask_derivation']['cases'][0]['removed_layers'] = ['M2', 'M4']
    manifest.write_text(json.dumps(payload))
    assert train_main(train_args(tmp_path, manifest, ckpt, '--resume', str(ckpt))) == 2
    assert error_of(capsys) == 'incompatible_config_fingerprint'
    assert ckpt.read_bytes() == before
    # A different export hash (content change) is rejected the same way, before any training.
    payload['sources'][0]['export_sha256'] = '1' * 64
    manifest.write_text(json.dumps(payload))
    assert train_main(train_args(tmp_path, manifest, ckpt, '--resume', str(ckpt))) == 2
    assert error_of(capsys) == 'export_sha256_mismatch' and ckpt.read_bytes() == before


def test_unknown_role_is_rejected_without_growing_the_checkpoint_vocabulary(tmp_path, capsys):
    # Third product (test split) uses a role the checkpoint vocabulary does not have.
    manifest = build_world(tmp_path, roles=('data', 'data', 'sensor'))
    ckpt = tmp_path / 'c.pt'
    assert train_main(train_args(tmp_path, manifest, ckpt)) == 0; capsys.readouterr()
    raw = torch.load(ckpt, weights_only=True)
    assert raw['config']['model']['roles'] == ['gate', 'data'] and raw['config']['model']['layers'] == ['M1', 'M3']
    index_path = tmp_path / 'index.json'
    assert index_main(['--dataset', str(manifest), '--checkpoint', str(ckpt), '--out', str(index_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['n_indexed'] == 4 and summary['n_rejected'] == 2
    index = json.loads(index_path.read_text())
    assert {r['roi_id'] for r in index['rejected']} == {'P2_0', 'P2_1'}
    assert all('unknown_role_or_actual_layer' in r['reason'] for r in index['rejected'])
    assert torch.load(ckpt, weights_only=True)['config']['model']['roles'] == ['gate', 'data']
    # Unknown role in the training split fails the run and writes nothing.
    bad = tmp_path / 'bad'; bad.mkdir()
    manifest = build_world(bad, roles=('sensor', 'data', 'data'))
    assert train_main(train_args(bad, manifest, bad / 'c.pt')) == 2
    assert 'unknown_role_or_actual_layer' in error_of(capsys) and not (bad / 'c.pt').exists()


# ---- single-export path is unchanged -------------------------------------------------------------

def run_module(src, module, args):
    env = {**os.environ, 'PYTHONPATH': str(src)}
    return subprocess.run([sys.executable, '-m', module, *map(str, args)], env=env, capture_output=True, text=True)


def legacy_chain(src, fixture, out, checkpoint_for_resume=None):
    """train -> index -> search (-> resume) with the CLIs of ``src`` on one fixture.

    Outputs are compared byte for byte. No wall-clock field exists in any of them: the only timing
    (progress lines with elapsed_s) goes to stderr and is switched off with --no-progress, so
    nothing is excluded from the comparison.
    """
    out.mkdir()
    config = json.loads((fixture / 'config.json').read_text()); config['epochs'] = 2
    (fixture / 'config2.json').write_text(json.dumps(config))
    export = fixture / 'review_rois.json'
    steps = {}
    r = run_module(src, 'psdl2.train', [export, '--config', fixture / 'config2.json', '--out', out / 'c.pt', '--lineages',
                                         fixture / 'lineages.json', '--split', fixture / 'split.json', '--no-progress'])
    assert r.returncode == 0, r.stderr
    steps['train.stdout'] = r.stdout
    raw = torch.load(out / 'c.pt', weights_only=True)
    steps['fingerprint'] = json.dumps({'fingerprint': raw['fingerprint'], 'model': raw['model_fingerprint'],
                                       'config': raw['config']}, sort_keys=True)
    r = run_module(src, 'psdl2.index', [export, '--checkpoint', out / 'c.pt', '--out', out / 'index.json'])
    assert r.returncode == 0, r.stderr
    steps['index.stdout'] = r.stdout; steps['index.json'] = (out / 'index.json').read_text()
    (fixture / 'cohort.json').write_text('["SYNTHETIC_1"]')
    r = run_module(src, 'psdl2.search', [export, '--roi-id', 'roi_0_0', '--checkpoint', out / 'c.pt', '--index', out / 'index.json',
                                          '--cohort', fixture / 'cohort.json', '--topk', '6', '--expand', '--out', out / 'search.json'])
    assert r.returncode == 0, r.stderr
    steps['search.stdout'] = r.stdout; steps['search.json'] = (out / 'search.json').read_text()
    if checkpoint_for_resume is not None:
        config['epochs'] = 3; (fixture / 'config3.json').write_text(json.dumps(config))
        r = run_module(src, 'psdl2.train', [export, '--config', fixture / 'config3.json', '--out', out / 'resumed.pt', '--lineages',
                                             fixture / 'lineages.json', '--split', fixture / 'split.json', '--no-progress',
                                             '--resume', checkpoint_for_resume])
        assert r.returncode == 0, r.stderr
        steps['resume.stdout'] = r.stdout
        steps['resumed.fingerprint'] = torch.load(out / 'resumed.pt', weights_only=True)['fingerprint']
    return steps


def test_single_export_outputs_are_byte_identical_to_the_baseline_commit(tmp_path):
    archive = subprocess.run(['git', '-C', str(ROOT), 'archive', BASELINE, 'src'], capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f'baseline commit {BASELINE} is not available in this checkout')
    old = tmp_path / 'old'; old.mkdir()
    subprocess.run(['tar', '-x', '-C', str(old)], input=archive.stdout, check=True)
    fixture = tmp_path / 'fixture'; fixture.mkdir()
    synthetic_exchange(fixture)
    before = legacy_chain(old / 'src', fixture, tmp_path / 'out_old')
    # Resume a checkpoint written by the baseline code with the baseline and with the current code.
    seed = tmp_path / 'out_old' / 'c.pt'
    before = {**before, **{k: v for k, v in legacy_chain(old / 'src', fixture, tmp_path / 'out_old_r', seed).items()
                           if k.startswith('resume')}}
    after = legacy_chain(ROOT / 'src', fixture, tmp_path / 'out_new', seed)
    for key in before:
        assert before[key] == after[key], key
    assert before['resume.stdout'] == after['resume.stdout']
    # And the new fields really are absent from the legacy shapes.
    index = json.loads(after['index.json'])
    assert 'producer_versions' in index and 'sources' not in index
    assert all('mask_derivation' not in r['case_signature'] for r in index['records'])
    assert 'dataset' not in json.loads(after['fingerprint'])['config']
    assert 'mask_derivation_comparison' not in json.loads(after['search.json'])
