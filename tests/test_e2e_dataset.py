"""End to end, as a user runs it: patch sets -> dataset -> make_split -> train -> index -> search -> eval_position.

Three synthetic products in three lineages (train / validation / test). Everything goes through the CLIs' ``main(argv)``.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from psdl2.eval_position import main as eval_main
from psdl2.index import main as index_main
from psdl2.make_split import main as make_split_main
from psdl2.patches import main as patches_main
from psdl2.search import main as search_main
from psdl2.train import main as train_main
from test_dataset import ROOT, derivation, source_for, write_dataset
from test_patches import make_template, views

PRODUCTS = ('T0', 'T1', 'T2')
LINEAGE = {'T0': 'line_train', 'T1': 'line_validation', 'T2': 'line_test'}
ASSIGN = {'line_train': 'train', 'line_validation': 'validation', 'line_test': 'test'}
PER_CASE, GRID_STEP = 6, 2.0
# Grid step 2 um on the 15 um frame: 8x8 = 64 points, 5x5 of them keep the 4.2 um wide window inside the frame (x, y in 104..112).
# Layout 0 occupies x 105..106, layout 1 x 107..108 (rows y 203..212): the core anchor needs x in {106, 108}, and both layouts in
# the 3 um near window leaves only x = 106. Five y values remain, hence 5 patches per product.
EXPECTED_GRID_PATCHES = 5


def cli(main, capsys, *argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def ok(main, capsys, *argv):
    code, out, err = cli(main, capsys, *argv)
    assert code == 0, err
    return json.loads(out) if out.strip() else None


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('e2e')
    config = json.loads((ROOT/'configs/synthetic.example.json').read_text())['config']
    from dataclasses import asdict
    config['views'] = json.loads(json.dumps(asdict(views('native'))))
    (root/'config.json').write_text(json.dumps(config))
    exports = {p: make_template(root, f'prod_{p}', product=p, rows=[{'rid': f'{p}_case0'}]) for p in PRODUCTS}
    # T1 was derived with an extra removed layer, so it must never be compatible with the T2 query.
    derivations = {'T0': derivation(), 'T1': derivation(removed=('M2', 'M9')), 'T2': derivation()}
    plain = write_dataset(root, [source_for(exports[p], root, mask_derivation=derivations[p]) for p in PRODUCTS], 'plain.json')
    (root/'lineages.json').write_text(json.dumps(LINEAGE))
    (root/'assign.json').write_text(json.dumps(ASSIGN))
    (root/'cohort.json').write_text(json.dumps(['T0', 'T1']))
    return root, exports, derivations, plain


def generate(capsys, root, plain, name, mode, source, *extra):
    base = ('--seed', 3, '--per-case', PER_CASE, '--min-spacing-um', 1) if mode == 'random' else ('--grid-step-um', GRID_STEP)
    return ok(patches_main, capsys, '--dataset', plain, '--source', source, '--config', root/'config.json', '--mode', mode,
              '--out', root/name, *base, *extra)


def manifest_with(root, exports, derivations, name, prefix):
    sources = [source_for(exports[p], root, patch_sets=[f'{prefix}_{p}.json'], mask_derivation=derivations[p]) for p in PRODUCTS]
    return write_dataset(root, sources, name)


def test_patch_to_position_evaluation_chain(world, capsys):
    root, exports, derivations, plain = world

    # 1. random patches for training, grid patches for the index, one set per product
    for i, product in enumerate(PRODUCTS):
        random_report = generate(capsys, root, plain, f'random_{product}.json', 'random', str(i))
        assert random_report['n_patches'] == PER_CASE and random_report['n_shortage_cases'] == 0
    grids = {p: generate(capsys, root, plain, f'grid_{p}.json', 'grid', str(i)) for i, p in enumerate(PRODUCTS)}
    assert all(g['n_patches'] == EXPECTED_GRID_PATCHES for g in grids.values())

    # 2. dataset manifests (template rows are replaced by their patches), split, train
    train_manifest = manifest_with(root, exports, derivations, 'train_dataset.json', 'random')
    index_manifest = manifest_with(root, exports, derivations, 'index_dataset.json', 'grid')
    summary = ok(make_split_main, capsys, '--dataset', train_manifest, '--config', root/'config.json', '--lineages', root/'lineages.json',
                 '--assign', root/'assign.json', '--out', root/'split.json')
    assert summary['records'] == 3*PER_CASE and summary['split_counts'] == {'train': PER_CASE, 'validation': PER_CASE, 'test': PER_CASE}
    split = json.loads((root/'split.json').read_text())
    assert {split[r] for r in split} == {'train', 'validation', 'test'}
    trained = ok(train_main, capsys, '--dataset', train_manifest, '--config', root/'config.json', '--out', root/'model.pt',
                 '--lineages', root/'lineages.json', '--split', root/'split.json', '--no-progress')
    assert trained['status'] == 'completed' and sum(e['optimizer_steps'] for e in trained['epoch_history']) >= 1
    raw = torch.load(root/'model.pt', weights_only=True)
    assert len(raw['config']['dataset']['sources']) == 3

    # 3. grid index of all three products; patch provenance and derivation are kept per record
    ok(index_main, capsys, '--dataset', index_manifest, '--checkpoint', root/'model.pt', '--out', root/'index.json')
    index = json.loads((root/'index.json').read_text())
    assert len(index['records']) == 3*EXPECTED_GRID_PATCHES and not index['rejected']
    assert all(r['provenance']['patch']['mode'] == 'grid' for r in index['records'])
    removed = {r['product_id']: r['case_signature']['mask_derivation']['removed_layers'] for r in index['records']}
    assert removed == {'T0': ['M2'], 'T1': ['M2', 'M9'], 'T2': ['M2']}

    # 4. query: the template export of the held-out (test) lineage, wrapped with its own derivation
    query_manifest = write_dataset(root, [source_for(exports['T2'], root, mask_derivation=derivations['T2'])], 'query.json')
    assert LINEAGE['T2'] == 'line_test' and ASSIGN[LINEAGE['T2']] == 'test'
    assert not [r for r in split if r.startswith('T2') and split[r] != 'test']
    code, out, err = cli(search_main, capsys, '--query-dataset', query_manifest, '--roi-id', 'T2_case0', '--checkpoint', root/'model.pt',
                         '--index', root/'index.json', '--cohort', root/'cohort.json', '--topk', 100, '--out', root/'search.json')
    assert code == 0, err
    result = json.loads((root/'search.json').read_text())
    assert result['query']['roi_id'] == 'T2_case0' and result['query']['product_id'] == 'T2'
    assert {r['product_id'] for r in result['primary']} == {'T0'} and len(result['primary']) == EXPECTED_GRID_PATCHES
    t1_rejected = [r for r in result['rejected'] if r['roi_id'] in {x['roi_id'] for x in index['records'] if x['product_id'] == 'T1'}]
    assert len(t1_rejected) == EXPECTED_GRID_PATCHES and {r['reason'] for r in t1_rejected} == {'mask_derivation_mismatch'}

    # 5. position evaluation against roi_correspondence_v1 labels: a known grid point plus a (0.3, 0.4) offset = 0.5 um away
    t0_points = [tuple(r['center_xy_um']) for r in result['primary']]
    assert sorted(t0_points) == [(106.0, 204.0 + 2*k) for k in range(5)]    # the five grid points derived above
    anchor = sorted(t0_points)[0]
    truth_xy = [anchor[0]+0.3, anchor[1]+0.4]
    frame = result['primary'][0]['frame']['frame_id']
    labels = root/'labels.json'
    labels.write_text(json.dumps({'schema_version': 'roi_correspondence_v1', 'pairs': [
        {'pair_id': 'p0', 'status': 'ready', 'a': {'roi_id': 'T2_case0'}, 'b': {'product_id': 'T0', 'frame_id': frame, 'center_xy_um': truth_xy}}]}))
    ok(eval_main, capsys, '--search', root/'search.json', '--correspondence', labels, '--hit-tolerance-um', 0.6, '--k', 100, '--out', root/'eval.json')
    evaluation = json.loads((root/'eval.json').read_text())
    assert evaluation['summary']['n_evaluable'] == 1 and evaluation['summary']['hit_rate_at_100'] == 1.0
    row = evaluation['queries'][0]
    known = [c for c in row['candidates'] if c['hit']]
    assert len(known) == 1 and known[0]['distance_um'] == pytest.approx(0.5)
    assert tuple(known[0]['center_xy_um']) == anchor
    assert row['first_hit_rank'] is not None and all(c['reason'] is None for c in row['candidates'])
    # The same labels do not match anything inside a tolerance below the true offset.
    ok_miss = cli(eval_main, capsys, '--search', root/'search.json', '--correspondence', labels, '--hit-tolerance-um', 0.4, '--k', 100,
                  '--out', root/'eval_miss.json')
    assert ok_miss[0] == 0 and json.loads((root/'eval_miss.json').read_text())['summary']['hit_rate_at_100'] == 0.0

    # 6. the derivation check is the only reason T1 was excluded
    code, out, err = cli(search_main, capsys, '--query-dataset', query_manifest, '--roi-id', 'T2_case0', '--checkpoint', root/'model.pt',
                         '--index', root/'index.json', '--cohort', root/'cohort.json', '--topk', 100, '--out', root/'search_ignored.json',
                         '--ignore-mask-derivation')
    assert code == 0, err
    ignored = json.loads((root/'search_ignored.json').read_text())
    assert ignored['mask_derivation_comparison'] == 'ignored' and {r['product_id'] for r in ignored['primary']} == {'T0', 'T1'}
