"""load_dataset patch-set expansion: admission, tamper detection, views check and the train/index CLI chain."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from psdl2.data import load_dataset, load_export, patch_roi_id, patch_set_content_hash
from psdl2.index import main as index_main
from psdl2.patches import main as patches_main
from psdl2.schema import ContractError
from psdl2.train import main as train_main, read_config
from psdl2.views import ViewConfig, preflight
from test_dataset import ENDPOINTS, ROOT, derivation, failure, source_for, write_dataset
from test_patches import bars, make_template, read_set, views, write_config
from test_dataset_cli import error_of, train_args


def generate(tmp_path, capsys, manifest, config, name, *extra, source='0', mode='random'):
    args = ('--seed', '5', '--per-case', '6', '--min-spacing-um', '1') if mode == 'random' else ('--grid-step-um', '2')
    code = patches_main(['--dataset', str(manifest), '--source', source, '--config', str(config), '--mode', mode,
                         '--out', str(tmp_path/name), *args, *extra])
    assert code == 0, capsys.readouterr().err
    capsys.readouterr()
    return tmp_path/name


def patched(tmp_path, capsys, *, rows=None, vc='native', derivations=True, **template):
    export = make_template(tmp_path, rows=rows, **template)
    plain = write_dataset(tmp_path, [source_for(export, tmp_path)], 'plain.json')
    config = write_config(tmp_path, views(vc))
    ps = generate(tmp_path, capsys, plain, config, 'ps.json')
    extra = {'mask_derivation': derivation()} if derivations else {}
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json'], **extra)])
    return export, manifest, config, ps


def rewrite(path, mutate, rehash=True):
    document = json.loads(Path(path).read_text())
    mutate(document)
    if rehash:
        document['header']['content_sha256'] = patch_set_content_hash(document)
    Path(path).write_text(json.dumps(document))


def load(manifest, vc='native'):
    return load_dataset(manifest, expected_views=views(vc))


def test_admitted_ids_counts_and_template_rows_excluded(tmp_path, capsys, monkeypatch):
    export, manifest, config, ps = patched(tmp_path, capsys)
    document = read_set(ps)
    elsewhere = tmp_path/'elsewhere'; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    dataset = load(manifest)
    template_id = load_export(export).ready[0].row['roi_id']
    assert [r.row['roi_id'] for r in dataset.ready] == [p['roi_id'] for p in document['patches']] and len(dataset.ready) == 6
    assert template_id not in {r.row['roi_id'] for r in dataset.ready} and not dataset.blocked
    set_sha = hashlib.sha256(ps.read_bytes()).hexdigest()
    for roi, patch in zip(dataset.ready, document['patches']):
        assert roi.center_xy_um == tuple(patch['center_xy_um']) and roi.row['source_kind'] == 'manual'
        meta = {'set_sha256': set_sha, 'template_roi_id': template_id, 'mode': 'random', 'generator_version': 'psdl2_patches_v1'}
        assert roi.row['provenance']['patch'] == meta == roi.patch
        assert roi.row['provenance']['position_policy'] == 'explicit_manual_point'     # template provenance is kept
        assert roi.mask_derivation['removed_layers'] == ['M2'] and roi.row['source_assets'] == load_export(export).ready[0].row['source_assets']
        assert preflight(roi, views('native'))['status'] == 'passed'
    assert dataset.fingerprint['sources'][0]['patch_sets'] == [{'path': 'ps.json', 'sha256': set_sha}]
    assert dataset.sources[0]['patch_sets'] == [{'path': 'ps.json', 'sha256': set_sha}]
    # The template row itself is also what a set-less source admits.
    assert [r.row['roi_id'] for r in load_dataset(tmp_path/'plain.json').ready] == [template_id]


def test_expanded_rows_share_the_templates_decoded_arrays(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    dataset = load(manifest)
    first = dataset.ready[0]
    for other in dataset.ready[1:]:
        assert other.channels.keys() == first.channels.keys() and other.channels is not first.channels
        assert all(other.channels[k] is first.channels[k] for k in first.channels)
        assert all(not other.channels[k].flags.writeable for k in other.channels)
    # One template decoded once: 4 files, 4 references, regardless of 6 patches.
    assert dataset.asset_memory['decoded_assets'] == 4 and dataset.asset_memory['asset_references'] == 4
    assert dataset.asset_memory['decoded_bytes'] == 4*900


def test_mixed_sources_and_two_sets_for_one_source(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    other = make_template(tmp_path, 'other', product='U', rows=[{'rid': 'U_case0'}])
    grid = generate(tmp_path, capsys, tmp_path/'plain.json', config, 'grid.json', mode='grid')
    two = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json', 'grid.json']), source_for(other, tmp_path)], 'two.json')
    dataset = load(two)
    n_grid = len(read_set(grid)['patches'])
    assert len(dataset.ready) == 6 + n_grid + 1 and dataset.ready[-1].row['roi_id'] == 'U_case0'
    assert {r.row['provenance']['patch']['mode'] for r in dataset.ready[:-1]} == {'random', 'grid'}
    assert 'patch' not in dataset.ready[-1].row['provenance'] and dataset.ready[-1].patch is None
    assert dataset.sources[1].get('patch_sets') is None


def test_stale_patch_set_when_the_export_changed(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    payload = json.loads(export.read_text()); payload['source_manifest_hash'] = '1'*64
    export.write_text(json.dumps(payload))
    fresh = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json'])], 'fresh.json')
    assert failure(lambda: load(fresh)).startswith('stale_patch_set')


def test_patch_field_tamper_is_a_patch_id_mismatch_even_when_the_hash_is_forged(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    original = ps.read_text()
    def move(document): document['patches'][0]['center_xy_um'][0] += 0.5
    def flip(document): document['patches'][0]['mode'] = 'grid'
    def rename(document): document['patches'][0]['roi_id'] = 'patch_' + '0'*24
    def retarget(document): document['patches'][0]['template_roi_id'] += '_x'
    for mutate, message in ((move, 'patch_id_mismatch'), (flip, 'patch_id_mismatch'), (rename, 'patch_id_mismatch'),
                            (retarget, 'patch_template_not_ready')):
        for rehash in (False, True):
            ps.write_text(original)
            rewrite(ps, mutate, rehash)
            assert failure(lambda: load(manifest)).startswith(message), (mutate.__name__, rehash)
    ps.write_text(original)
    rewrite(ps, lambda d: d['patches'][1].update(center_xy_um=[float('nan'), 1.0]), rehash=False)
    assert failure(lambda: load(manifest)).startswith('patch_center_not_finite')
    ps.write_text(original)
    rewrite(ps, lambda d: d['patches'][1].update(extra=1), rehash=False)
    assert failure(lambda: load(manifest)) == 'patch_record_invalid'
    ps.write_text(original)
    assert len(load(manifest).ready) == 6


def test_template_must_be_a_ready_manual_row(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys, rows=[{'rid': 'T_manual'}, {'rid': 'T_region', 'kind': 'region'}])
    def retarget(document):
        patch = document['patches'][0]
        patch['template_roi_id'] = 'T_region'
        patch['roi_id'] = patch_roi_id('T_region', patch['center_xy_um'], patch['mode'], document['header']['generator_version'])
    rewrite(ps, retarget)
    assert failure(lambda: load(manifest)).startswith('patch_template_not_manual')


def test_header_tamper_is_a_hash_mismatch_and_a_forged_header_changes_the_fingerprint(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    honest = load(manifest).fingerprint
    original = ps.read_text()
    for mutate in (lambda d: d['header']['arguments'].update(seed=99), lambda d: d['header'].update(mode='grid'),
                   lambda d: d['header']['cases'][0].update(requested=7), lambda d: d['header'].update(content_sha256='0'*64)):
        ps.write_text(original); rewrite(ps, mutate, rehash=False)
        assert failure(lambda: load(manifest)).startswith('patch_set_hash_mismatch')
    ps.write_text(original); rewrite(ps, lambda d: d['header']['arguments'].update(seed=99))
    forged = load(manifest)    # consistent hash: loads, but the file is different, so the dataset identity is
    assert forged.fingerprint != honest and [r.row['roi_id'] for r in forged.ready] == [r.row['roi_id'] for r in load(manifest).ready]


def test_duplicate_and_overlapping_patch_sets_are_errors(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    twice = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json', 'ps.json'])], 'twice.json')
    assert failure(lambda: load(twice)).startswith('duplicate_patch_set')
    shutil.copy(ps, tmp_path/'copy.json')
    copy = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json', 'copy.json'])], 'copy.json.manifest')
    assert failure(lambda: load(copy)).startswith('duplicate_patch_set')
    # Same patches, different file bytes: no silent de-duplication.
    shutil.copy(ps, tmp_path/'overlap.json'); rewrite(tmp_path/'overlap.json', lambda d: d['header']['arguments'].update(seed=6))
    overlap = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json', 'overlap.json'])], 'overlap.manifest')
    assert failure(lambda: load(overlap)).startswith('duplicate_roi_id_across_sources')
    # The same patch set under two sources of identical export collides as well.
    again = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json']),
                                     source_for(export, tmp_path, patch_sets=['overlap.json'])], 'again.manifest')
    assert failure(lambda: load(again)).startswith('duplicate_roi_id_across_sources')
    # Duplicate ids inside one set.
    shutil.copy(ps, tmp_path/'dup.json'); rewrite(tmp_path/'dup.json', lambda d: d['patches'].append(dict(d['patches'][0])))
    inner = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['dup.json'])], 'inner.manifest')
    assert failure(lambda: load(inner)).startswith('duplicate_roi_id_across_sources')


@pytest.mark.parametrize('patch_source_first', [True, False])
def test_patch_ids_are_unique_against_every_other_source(tmp_path, capsys, patch_source_first):
    export, manifest, config, ps = patched(tmp_path, capsys)
    pid = read_set(ps)['patches'][0]['roi_id']
    clash = make_template(tmp_path, 'clash', product='U', rows=[{'rid': pid}])
    sources = [source_for(export, tmp_path, patch_sets=['ps.json']), source_for(clash, tmp_path)]
    manifest2 = write_dataset(tmp_path, sources if patch_source_first else sources[::-1], 'clash.json')
    assert failure(lambda: load(manifest2)).startswith('duplicate_roi_id_across_sources')


def test_views_must_match_and_cannot_be_skipped(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    assert failure(lambda: load_dataset(manifest)) == 'patch_set_views_unchecked'
    assert failure(lambda: load_dataset(manifest, expected_views=None)) == 'patch_set_views_unchecked'
    assert failure(lambda: load(manifest, 'area')).startswith('patch_set_view_config_mismatch')
    assert failure(lambda: load_dataset(manifest, expected_views=object())) == 'expected_views_must_be_view_config'
    assert len(load(manifest).ready) == 6


def test_missing_and_malformed_patch_sets(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys)
    missing = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['nope.json'])], 'missing.json')
    assert failure(lambda: load(missing)).startswith('patch_set_missing')
    original = ps.read_text()
    rewrite(ps, lambda d: d.update(schema_version='psdl2_patch_set_v9'), rehash=False)
    assert failure(lambda: load(manifest)) == 'unknown_patch_set_schema'
    ps.write_text(original); rewrite(ps, lambda d: d.update(extra=1), rehash=False)
    assert failure(lambda: load(manifest)) == 'patch_set_envelope_invalid'
    ps.write_text(original); rewrite(ps, lambda d: d['header'].pop('views_fingerprint'), rehash=False)
    assert failure(lambda: load(manifest)) == 'nonempty_identifier_required'
    bad = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets='ps.json')], 'bad.json')
    assert failure(lambda: load(bad)) == 'patch_sets_list_required'


def test_a_forged_patch_outside_the_frame_or_the_valid_domain_is_rejected(tmp_path, capsys):
    holes = ((3, 5, 3, 5),)
    export, manifest, config, ps = patched(tmp_path, capsys, rows=[{'arrays': bars((30, 30), holes)}])
    original = ps.read_text()
    def place(x, y):
        def mutate(document):
            patch = document['patches'][0]
            patch['center_xy_um'] = [x, y]
            patch['roi_id'] = patch_roi_id(patch['template_roi_id'], (x, y), patch['mode'], document['header']['generator_version'])
        return mutate
    rewrite(ps, place(101.75, 201.75))      # cell (3, 3) is a hole in the template's valid mask
    assert failure(lambda: load(manifest)).startswith('patch_position_outside_valid_domain')
    ps.write_text(original); rewrite(ps, place(99.0, 201.75))
    assert failure(lambda: load(manifest)).startswith('patch_position_outside_source_frame')
    ps.write_text(original); rewrite(ps, place(115.0, 201.75))       # exactly the far edge is outside
    assert failure(lambda: load(manifest)).startswith('patch_position_outside_source_frame')


# ---- train / index chain ---------------------------------------------------------------------------------------

def chain(tmp_path, capsys):
    """Random + grid patch sets over one template, with the synthetic run config used for generation and training."""
    export = make_template(tmp_path, rows=[{'rid': 'T_case0'}])
    config = json.loads((ROOT/'configs/synthetic.example.json').read_text())['config']
    (tmp_path/'config.json').write_text(json.dumps(config))
    plain = write_dataset(tmp_path, [source_for(export, tmp_path)], 'plain.json')
    random_set = generate(tmp_path, capsys, plain, tmp_path/'config.json', 'random.json')
    grid_set = generate(tmp_path, capsys, plain, tmp_path/'config.json', 'grid.json', mode='grid')
    train_manifest = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['random.json'], mask_derivation=derivation())], 'train.json')
    index_manifest = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['grid.json'], mask_derivation=derivation())], 'index.json.manifest')
    ids = [p['roi_id'] for p in read_set(random_set)['patches']]
    (tmp_path/'lineages.json').write_text(json.dumps({'T': 'lineage_T'}))
    (tmp_path/'split.json').write_text(json.dumps({i: 'train' for i in ids}))
    return export, train_manifest, index_manifest, ids, read_set(grid_set)


def test_generated_patches_train_index_and_carry_provenance_to_the_index(tmp_path, capsys):
    export, train_manifest, index_manifest, ids, grid = chain(tmp_path, capsys)
    ckpt = tmp_path/'c.pt'
    assert train_main(train_args(tmp_path, train_manifest, ckpt)) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained['status'] == 'completed' and sum(e['optimizer_steps'] for e in trained['epoch_history']) >= 1
    assert sorted(trained['split']['train']) == sorted(ids)
    raw = torch.load(ckpt, weights_only=True)
    entry = raw['config']['dataset']['sources'][0]['patch_sets'][0]
    assert entry['path'] == 'random.json' and entry['sha256'] == hashlib.sha256((tmp_path/'random.json').read_bytes()).hexdigest()
    out = tmp_path/'index.json'
    assert index_main(['--dataset', str(index_manifest), '--checkpoint', str(ckpt), '--out', str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['n_indexed'] == len(grid['patches']) and summary['n_rejected'] == 0
    index = json.loads(out.read_text())
    set_sha = hashlib.sha256((tmp_path/'grid.json').read_bytes()).hexdigest()
    assert {r['roi_id'] for r in index['records']} == {p['roi_id'] for p in grid['patches']}
    for record in index['records']:
        assert record['provenance'] == {'patch': {'set_sha256': set_sha, 'template_roi_id': 'T_case0', 'mode': 'grid',
                                                  'generator_version': 'psdl2_patches_v1'}}
        assert record['case_signature']['mask_derivation']['removed_layers'] == ['M2']
    assert index['view_fingerprint'] == grid['header']['views_fingerprint']
    assert index['sources'][0]['patch_sets'] == [{'path': 'grid.json', 'sha256': set_sha}]
    # The unchanged single-export path never writes patch provenance.
    legacy = tmp_path/'legacy.json'
    assert index_main([str(export), '--checkpoint', str(ckpt), '--out', str(legacy)]) in (0, 3)
    capsys.readouterr()
    assert all('provenance' not in r for r in json.loads(legacy.read_text())['records'])


def test_patch_set_views_must_match_the_checkpoint_and_a_forged_header_blocks_resume(tmp_path, capsys):
    export, train_manifest, index_manifest, ids, grid = chain(tmp_path, capsys)
    ckpt = tmp_path/'c.pt'
    assert train_main(train_args(tmp_path, train_manifest, ckpt)) == 0; capsys.readouterr()
    config = json.loads((tmp_path/'config.json').read_text()); config['epochs'] = 2
    (tmp_path/'config.json').write_text(json.dumps(config))
    before = ckpt.read_bytes()
    resumed = tmp_path/'resumed.pt'; shutil.copy(ckpt, resumed)
    assert train_main(train_args(tmp_path, train_manifest, resumed, '--resume', str(resumed))) == 0; capsys.readouterr()
    # Different views in the run config: the patch set's views fingerprint no longer matches.
    changed = json.loads(json.dumps(config)); changed['views']['jitter_xy_um'] = [0.2, 0.2]
    other = tmp_path/'other.json'; other.write_text(json.dumps(changed))
    args = train_args(tmp_path, train_manifest, tmp_path/'o.pt'); args[args.index('--config')+1] = str(other)
    assert train_main(args) == 2 and error_of(capsys).startswith('patch_set_view_config_mismatch') and not (tmp_path/'o.pt').exists()
    # Raw header tamper: refused at load.
    original = (tmp_path/'random.json').read_text()
    rewrite(tmp_path/'random.json', lambda d: d['header']['arguments'].update(seed=77), rehash=False)
    assert train_main(train_args(tmp_path, train_manifest, ckpt, '--resume', str(ckpt))) == 2
    assert error_of(capsys).startswith('patch_set_hash_mismatch') and ckpt.read_bytes() == before
    # Consistently re-hashed header: loads, but the dataset identity changed, so resume is refused untouched.
    (tmp_path/'random.json').write_text(original)
    rewrite(tmp_path/'random.json', lambda d: d['header']['arguments'].update(seed=77))
    assert train_main(train_args(tmp_path, train_manifest, ckpt, '--resume', str(ckpt))) == 2
    assert error_of(capsys) == 'incompatible_config_fingerprint' and ckpt.read_bytes() == before
    # The unchanged set resumes.
    (tmp_path/'random.json').write_text(original)
    assert train_main(train_args(tmp_path, train_manifest, resumed.with_name('r2.pt'), '--resume', str(ckpt))) == 0
    capsys.readouterr()


def test_index_with_other_views_than_the_patch_set_is_refused(tmp_path, capsys):
    export, train_manifest, index_manifest, ids, grid = chain(tmp_path, capsys)
    config = json.loads((tmp_path/'config.json').read_text()); config['views']['jitter_xy_um'] = [0.2, 0.2]
    (tmp_path/'config.json').write_text(json.dumps(config))
    free = make_template(tmp_path, 'free', rows=[{'rid': 'F1'}, {'rid': 'F2'}])
    free_manifest = write_dataset(tmp_path, [source_for(free, tmp_path)], 'free.json')
    (tmp_path/'split.json').write_text(json.dumps({'F1': 'train', 'F2': 'train'}))
    ckpt = tmp_path/'other.pt'
    assert train_main(train_args(tmp_path, free_manifest, ckpt)) == 0; capsys.readouterr()
    out = tmp_path/'index.json'
    assert index_main(['--dataset', str(index_manifest), '--checkpoint', str(ckpt), '--out', str(out)]) == 2
    assert json.loads(capsys.readouterr().err)['error'].startswith('patch_set_view_config_mismatch') and not out.exists()


def test_template_ids_of_a_patched_source_stay_reserved_for_other_sources(tmp_path, capsys):
    export, manifest, config, ps = patched(tmp_path, capsys, derivations=False)
    template_id = load_export(export).ready[0].row['roi_id']
    other = make_template(tmp_path, name='other', product='U', rows=[{'rid': template_id}])
    two = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json']), source_for(other, tmp_path)], 'two.json')
    with pytest.raises(ContractError, match='duplicate_roi_id_across_sources'):
        load(two)
