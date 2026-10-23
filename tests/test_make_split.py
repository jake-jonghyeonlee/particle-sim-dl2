"""psdl2.make_split: lineage-assigned split.json, leakage, atomic publication, and train acceptance."""
import json
import os
from pathlib import Path

import pytest

from dataclasses import asdict

from psdl2.make_split import build_split, main as make_split_main
from psdl2.schema import ContractError
from psdl2.split import record_for_roi, validate_split
from psdl2.train import main as train_main
from test_dataset_cli import build_world, error_of, train_args
from test_dataset_patches import load, patched, views


def run(root, manifest, out, *extra, lineages='lineages.json', assign='assign.json'):
    root = Path(root)
    return make_split_main(['--dataset', str(manifest), '--config', str(root/'config.json'), '--lineages', str(root/lineages),
                            '--assign', str(root/assign), '--out', str(out), *extra])


def full_config(tmp_path, patch_config, **view_changes):
    """A complete psdl2_run_v1 config whose views are the ones the patch set was generated with."""
    from test_dataset_cli import ROOT
    config = json.loads((ROOT/'configs/synthetic.example.json').read_text())['config']
    config['views'] = {**json.loads(patch_config.read_text())['config']['views'], **view_changes}
    (tmp_path/'config.json').write_text(json.dumps(config))


def world(tmp_path):
    manifest = build_world(tmp_path)
    (tmp_path/'assign.json').write_text(json.dumps({'lineage_0': 'train', 'lineage_1': 'validation', 'lineage_2': 'test'}))
    return manifest


def test_output_equals_the_split_train_consumes_and_trains(tmp_path, capsys):
    manifest = world(tmp_path)
    out = tmp_path/'made.json'
    assert run(tmp_path, manifest, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['records'] == 6 and summary['split_counts'] == {'train': 2, 'validation': 2, 'test': 2}
    # Same mapping the hand-written split.json of the existing CLI tests carries.
    assert json.loads(out.read_text()) == json.loads((tmp_path/'split.json').read_text())
    assert not [p for p in tmp_path.iterdir() if p.name.endswith('.tmp')]
    args = train_args(tmp_path, manifest, tmp_path/'c.pt')
    args[args.index('--split')+1] = str(out)
    assert train_main(args) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained['status'] == 'completed' and sum(e['optimizer_steps'] for e in trained['epoch_history']) >= 1


def test_patch_set_dataset_end_to_end(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    full_config(tmp_path, config)
    (tmp_path/'lineages.json').write_text(json.dumps({'T': 'tmpl_lineage'}))
    (tmp_path/'assign.json').write_text(json.dumps({'tmpl_lineage': 'train'}))
    out = tmp_path/'split.json'
    assert run(tmp_path, manifest, out) == 0
    ids = sorted(json.loads(out.read_text()))
    assert len(ids) > 2 and ids == sorted(r.row['roi_id'] for r in load(manifest).ready)
    assert set(json.loads(out.read_text()).values()) == {'train'}


def test_patch_set_views_are_checked(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    full_config(tmp_path, config, jitter_xy_um=[.2, .2])
    (tmp_path/'lineages.json').write_text(json.dumps({'T': 'l'})); (tmp_path/'assign.json').write_text(json.dumps({'l': 'train'}))
    assert run(tmp_path, manifest, tmp_path/'s.json') == 2
    assert 'patch_set_view_config_mismatch' in error_of(capsys)
    assert not (tmp_path/'s.json').exists()


def test_far_apart_patches_of_one_template_cannot_straddle_train_and_test(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    rois = list(load(manifest).ready)
    centres = sorted(rois, key=lambda r: r.center_xy_um)
    near, far = centres[0], centres[-1]
    assert far.center_xy_um != near.center_xy_um
    vc = views('native')
    records = [record_for_roi(r, 'one_template', vc) for r in rois]
    split = {r.row['roi_id']: 'train' for r in rois}
    split[far.row['roi_id']] = 'test'
    with pytest.raises(ContractError, match='split_leakage'):
        validate_split(records, split)
    # The CLI maps by product lineage, so every patch of the template lands in one split.
    full_config(tmp_path, config)
    (tmp_path/'lineages.json').write_text(json.dumps({'T': 'one_template'}))
    (tmp_path/'assign.json').write_text(json.dumps({'one_template': 'test'}))
    assert run(tmp_path, manifest, tmp_path/'s.json') == 0
    assert set(json.loads((tmp_path/'s.json').read_text()).values()) == {'test'}


def test_two_products_sharing_a_lineage_are_one_split_unit(tmp_path, capsys):
    manifest = world(tmp_path)
    (tmp_path/'lineages.json').write_text(json.dumps({'P0': 'x', 'P1': 'x', 'P2': 'y'}))
    (tmp_path/'assign.json').write_text(json.dumps({'x': 'train', 'y': 'test'}))
    assert run(tmp_path, manifest, tmp_path/'s.json') == 0
    values = json.loads((tmp_path/'s.json').read_text())
    assert values == {'P0_0': 'train', 'P0_1': 'train', 'P1_0': 'train', 'P1_1': 'train', 'P2_0': 'test', 'P2_1': 'test'}


def test_unassigned_lineage_and_product_without_lineage(tmp_path, capsys):
    manifest = world(tmp_path)
    out = tmp_path/'s.json'
    (tmp_path/'assign.json').write_text(json.dumps({'lineage_0': 'train', 'lineage_1': 'validation'}))
    assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'lineage_unassigned: lineage_2'
    (tmp_path/'assign.json').write_text(json.dumps({'lineage_0': 'train', 'lineage_1': 'validation', 'lineage_2': 'dev'}))
    assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'invalid_split_name: lineage_2'
    (tmp_path/'assign.json').write_text(json.dumps({'lineage_0': 'train', 'lineage_1': 'validation', 'lineage_2': 'test'}))
    (tmp_path/'lineages.json').write_text(json.dumps({'P0': 'lineage_0', 'P1': 'lineage_1'}))
    assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'product_without_lineage: P2'
    assert not out.exists() and not [p for p in tmp_path.iterdir() if p.name.endswith('.tmp')]


def test_existing_output_needs_overwrite_and_failed_replace_keeps_old_bytes(tmp_path, capsys, monkeypatch):
    manifest = world(tmp_path)
    out = tmp_path/'s.json'
    out.write_bytes(b'old bytes')
    assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'output_exists_use_overwrite'
    assert out.read_bytes() == b'old bytes'
    real = os.replace
    def boom(src, dst):
        raise OSError('interrupted before replace')
    monkeypatch.setattr(os, 'replace', boom)
    assert run(tmp_path, manifest, out, '--overwrite') == 2 and 'interrupted' in error_of(capsys)
    assert out.read_bytes() == b'old bytes'
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.s.json')]
    monkeypatch.setattr(os, 'replace', real)
    assert run(tmp_path, manifest, out, '--overwrite') == 0
    capsys.readouterr()
    assert json.loads(out.read_text())['P0_0'] == 'train'


def test_unrelated_temp_files_are_left_alone(tmp_path, capsys, monkeypatch):
    manifest = world(tmp_path)
    foreign = tmp_path/'.s.json.foreign.tmp'
    foreign.write_bytes(b'someone else')
    monkeypatch.setattr(os, 'replace', lambda a, b: (_ for _ in ()).throw(OSError('x')))
    assert run(tmp_path, manifest, tmp_path/'s.json') == 2
    assert foreign.read_bytes() == b'someone else'


def test_invalid_lineage_id_and_non_object_inputs_and_no_ready_rois(tmp_path, capsys):
    manifest = world(tmp_path)
    out = tmp_path/'s.json'
    for bad in ('', 5, None, ['lineage_2']):
        (tmp_path/'lineages.json').write_text(json.dumps({'P0': 'lineage_0', 'P1': 'lineage_1', 'P2': bad}))
        assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'invalid_lineage_id: P2'
    for name, other in (('lineages', 'assign'), ('assign', 'lineages')):
        for bad in ([], 'text', 3):
            good = {'lineages': {'P0': 'lineage_0', 'P1': 'lineage_1', 'P2': 'lineage_2'},
                    'assign': {'lineage_0': 'train', 'lineage_1': 'validation', 'lineage_2': 'test'}}
            (tmp_path/'lineages.json').write_text(json.dumps(good['lineages'])); (tmp_path/'assign.json').write_text(json.dumps(good['assign']))
            (tmp_path/f'{name}.json').write_text(json.dumps(bad))
            assert run(tmp_path, manifest, out) == 2 and error_of(capsys) == 'lineages_and_assign_objects_required'
    assert not out.exists()
    with pytest.raises(ContractError, match='no_ready_rois_to_split'):
        build_split([], views('native'), {}, {})
