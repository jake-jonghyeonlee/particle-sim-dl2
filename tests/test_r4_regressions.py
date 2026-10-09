"""R4 reproductions with explicit geometry/connectivity expectations."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest
import torch

from psdl2.crop import build_bundle
from psdl2.data import load_export
from psdl2.schema import ContractError, CropConfig, Frame
from psdl2.views import ViewConfig, generate_pair, validate_pair, TOPOLOGY_POLICY
from test_views import view_config
from test_model import model_config
from test_losses import loss_config


def hand_roi(source, masks, *, origin=(100, 200), pitch=.5, center=(150, 250)):
    _, write, _ = source; original = load_export(write()).ready[0]
    shape = masks[0].shape
    channels = {'layout:0': masks[0], 'layout:1': masks[1],
                'cnt': np.zeros(shape, np.uint8), 'valid': np.ones(shape, np.uint8)}
    row = deepcopy(original.row); row['center_xy_um'] = list(center)
    row['frame'].update(origin_xy_um=list(origin), pixel_pitch_um=pitch, shape_yx=list(shape))
    return replace(original, row=row, frame=Frame(original.frame.frame_id, origin, pitch, shape), channels=channels)


def test_B1_review_half_cell_periodic_lines(source):
    lines = np.zeros((400, 400), np.uint8); lines[:, ::4] = 1
    roi = hand_roi(source, (lines, lines.T), pitch=.05, center=(103.075, 210.025))
    bundle = build_bundle(roi, CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1'))
    # (103.075-100)/.05=61.5; subtract half-width 20 and add .5 -> first index 42.
    # Exact ties select the higher cell. Source line columns 44,48,... give 2,6,... .
    assert bundle.near.metadata['source_indices_xy'][0] == list(range(42, 82))
    assert bundle.near.metadata['source_indices_xy'][1] == list(range(181, 221))
    np.testing.assert_array_equal(np.flatnonzero(bundle.near.channels['layout:0'][0]), np.arange(2, 40, 4))
    assert bundle.near.metadata['boundary_tie_rule'] == 'higher_source_index_half_open'


def test_B1_origins_and_280_centres_unit_step_sweep(source):
    lines = np.zeros((400, 400), np.uint8); lines[:, ::4] = 1
    # Cell edge f=0 -> start=n-20; half cell f=.5 -> start=n-19 (higher-cell tie);
    # arbitrary f=.37 -> start=n-20. These expectations are stated independently.
    for origin in ((0, 0), (100, 200), (-13.275, 27.125)):
        for n in range(60, 340):
            for fraction, start in ((Fraction(0), n-20), (Fraction(1, 2), n-19), (Fraction(37, 100), n-20)):
                center = (float(Fraction(str(origin[0]))+(n+fraction)/20),
                          float(Fraction(str(origin[1]))+Fraction(401, 40)))
                roi = hand_roi(source, (lines, lines.T), origin=origin, pitch=.05, center=center)
                near = build_bundle(roi, CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1')).near
                assert near.metadata['source_indices_xy'][0] == list(range(start, start+40))
                assert np.all(np.diff(near.metadata['source_indices_xy'][1]) == 1)
                assert np.all(np.diff(np.flatnonzero(near.channels['layout:0'][0])) == 4)


def test_B1_review_area_fraction_and_frame_edge_oracles(source):
    a = np.zeros((2, 10), np.uint8); a[0, 5] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), center=(101.75, 200.5))
    cfg = CropConfig((4, 1), (4, 1), 'preserve_exported_layers_v1')
    patch = build_bundle(roi, cfg, near_output_pitch_um=1, wide_output_pitch_um=1,
                         near_resampling_mode='area_coverage').near
    # Source cell [102.5,103) × [200,200.5) straddles output boundary 102.75.
    np.testing.assert_array_equal(patch.channels['layout:0'], [[0, 0, .125, .125]])
    empty = np.zeros((2, 8), np.uint8)
    edge = hand_roi(source, (empty, empty), center=(103.5, 200.5))
    patch = build_bundle(edge, cfg, near_output_pitch_um=1, wide_output_pitch_um=1,
                         near_resampling_mode='area_coverage').near
    np.testing.assert_array_equal(patch.channels['valid'], [[1, 1, .5, 0]])


def stripes(source):
    horizontal = np.zeros((200, 200), np.uint8)
    horizontal[::20, :] = 1; horizontal[1::20, :] = 1
    return hand_roi(source, (horizontal, horizontal.T.copy()))


def stripe_config(jitter):
    return ViewConfig(CropConfig((20, 20), (40, 40), 'preserve_exported_layers_v1'),
                      .5, .5, 'nearest', 'nearest', (jitter, jitter), (10, 10), TOPOLOGY_POLICY)


@pytest.mark.parametrize('jitter', [0, .1, 1.])
def test_B2_review_frame_crossing_stripes_180_pairs_and_loss(source, jitter):
    from psdl2.model import Encoder
    from psdl2.train import paired_forward
    roi = stripes(source); rng = np.random.default_rng(2026)
    # Each horizontal/vertical core segment connects through a straight line
    # within either window. Cutting its remote ends cannot change that relation.
    # Hence all 9 centres × 20 draws must be accepted, for every stated jitter.
    accepted = 0
    for x in (125, 150, 175):
        for y in (225, 250, 275):
            row = deepcopy(roi.row); row['center_xy_um'] = [x, y]
            local = replace(roi, row=row)
            for _ in range(20):
                assert generate_pair(local, stripe_config(jitter), rng)[2]['status'] == 'accepted'
                accepted += 1
    assert accepted == 180
    other_row = deepcopy(roi.row); other_row['roi_id'] = 'other_location'
    result, rejected = paired_forward(Encoder(model_config()), [roi, replace(roi, row=other_row)],
                                      stripe_config(jitter), loss_config(), rng)
    assert not rejected and torch.isfinite(result['total'])
    assert result['accepted_pair_count'] == 2 and result['rejected_pair_count'] == 0


@pytest.mark.parametrize('jitter', [.3, 1.])
def test_B2_review_isolated_cells_edge_changes_are_accepted(source, jitter):
    rng = np.random.default_rng(2026)
    a = (rng.random((200, 200)) < .02).astype(np.uint8)
    b = (rng.random((200, 200)) < .02).astype(np.uint8)
    roi = hand_roi(source, (a, b)); accepted = 0
    # Isolated cells in the core retain the same single-cell connectivity, while
    # isolated cells entering/leaving remote window edges are irrelevant to it.
    for x in (125, 150, 175):
        for y in (225, 250, 275):
            row = deepcopy(roi.row); row['center_xy_um'] = [x, y]
            for _ in range(20):
                generate_pair(replace(roi, row=row), stripe_config(jitter), rng); accepted += 1
    assert accepted == 180


def test_B2_bridge_outside_core_lost_to_window_is_rejected(source):
    a = np.zeros((20, 20), np.uint8)
    # Two core cells (row10,col8/11) connect via a U above the core. A downward
    # shifted near window excludes only its top bridge; core content stays fixed.
    a[5:11, 8] = 1; a[5:11, 11] = 1; a[5, 8:12] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), pitch=1, center=(110, 210))
    cfg = ViewConfig(CropConfig((10, 10), (14, 14), 'preserve_exported_layers_v1'),
                     1, 1, 'nearest', 'nearest', (0, 2), (4, 4), TOPOLOGY_POLICY)
    first = cfg.bundle(roi); first.metadata['original_center_xy_um'] = [110, 210]
    row = deepcopy(roi.row); row['center_xy_um'] = [110, 212]
    second = cfg.bundle(replace(roi, row=row)); second.metadata['original_center_xy_um'] = [110, 210]
    with pytest.raises(ContractError, match='core_connection_lost_between_views'):
        validate_pair(roi, first, second, (4, 4), TOPOLOGY_POLICY)


def test_B2_coarse_grid_removed_bridge_and_merge_limitation(source):
    a = np.zeros((20, 20), np.uint8); a[10, 6:14] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), pitch=1, center=(110, 210))
    cfg = ViewConfig(CropConfig((12, 12), (16, 16), 'preserve_exported_layers_v1'),
                     2, 2, 'area_coverage', 'area_coverage', (0, 0), (8, 8), TOPOLOGY_POLICY)
    first, second, _ = generate_pair(roi, cfg, np.random.default_rng(1))
    channels = dict(second.near.channels); mask = channels['layout:0'].copy(); mask[3, 3] = 0
    channels['layout:0'] = mask; second = replace(second, near=replace(second.near, channels=channels))
    with pytest.raises(ContractError, match='resampled_core_marked_content_lost|resampled_core_connection_lost'):
        validate_pair(roi, first, second, (8, 8), TOPOLOGY_POLICY)


def test_B2_nonzero_jitter_crosses_cell_and_identical_metadata(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    from psdl2.train import read_config
    from pathlib import Path
    cfg = ViewConfig.from_dict(read_config(Path(__file__).parents[1]/'configs/synthetic.example.json')['views'])
    class Offsets:
        def __init__(self, offsets): self.offsets = iter(offsets)
        def uniform(self, low, high): return np.array(next(self.offsets))
    first, second, evidence = generate_pair(roi, cfg, Offsets([[-.3, -.3], [.3, .3]]))
    assert evidence['identical_views']['near'] is False
    assert not np.array_equal(first.near.channels['layout:0'], second.near.channels['layout:0'])
    _, _, identical = generate_pair(roi, replace(cfg, jitter_xy_um=(.1, .1)), Offsets([[-.1, -.1], [.1, .1]]))
    assert identical['identical_views']['near'] is True


def test_N1_unversioned_export_requires_explicit_acceptance_and_records_it(source, capsys):
    from psdl2.data import main
    doc, write, _ = source; doc['producer_commit'] = None
    doc['producer_commit_reason'] = 'producer_not_in_git_checkout'; path = write()
    with pytest.raises(ContractError, match='producer_commit_required'): load_export(path)
    accepted = load_export(path, allow_unversioned_producer=True)
    assert accepted.ready[0].versions['unversioned_producer_accepted'] is True
    assert accepted.versions['producer_commit'] is None
    assert main([str(path), '--allow-unversioned-producer']) == 0
    assert json.loads(capsys.readouterr().out)['unversioned_producer_accepted'] is True
    with pytest.raises(ContractError, match='producer_commit_mismatch'):
        load_export(path, allow_unversioned_producer=True, expected_producer_commit='a'*40)


def test_N3_known_different_counts_and_numpy_loss_reproduction():
    from psdl2.losses import objective
    rng = np.random.default_rng(2026)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    config = loss_config()
    labels = torch.full((5, 5), -1)
    labels[0, 1] = labels[1, 0] = 1; labels[1, 2] = labels[2, 1] = 0
    result = objective(torch.tensor(a), torch.tensor(b), config, relations=labels)
    assert result['alignment_pair_indices'] == [[i, i] for i in range(5)]
    assert result['negative_pair_count'] == sum(i != j for i, j in result['alignment_pair_indices'])
    assert result['supervised_known_pair_count'] == 2 and result['supervised_different_pair_count'] == 1
    alignment = ((a-b)**2).mean(); variance, covariance = [], []
    for values in (a, b):
        cov = np.cov(values, rowvar=False, ddof=1)
        variance.append(np.maximum(0, 2-np.sqrt(np.diag(cov)+.25)).mean())
        covariance.append(((cov-np.diag(np.diag(cov)))**2).sum()/3)
    assert result['alignment'].item() == pytest.approx(alignment)
    assert result['variance'].item() == pytest.approx(np.mean(variance))
    assert result['covariance'].item() == pytest.approx(np.mean(covariance))
    assert result['total'].item() == pytest.approx(alignment+np.mean(variance)+np.mean(covariance)+result['supervised'].item())
    # Permuting view 2 only changes alignment; no hidden cross-location term is added.
    permuted = objective(torch.tensor(a), torch.tensor(b[[1, 2, 3, 4, 0]]), config)
    assert permuted['variance'].item() == pytest.approx(result['variance'].item())
    assert permuted['covariance'].item() == pytest.approx(result['covariance'].item())
    assert permuted['alignment'].item() != pytest.approx(alignment)


@pytest.mark.parametrize('change', ['reverse', 'rename'])
def test_N4_endpoint_order_and_names_do_not_restrict_case_search(source, change):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    query, records = fixed_records(source); candidate = records[0]
    signature = candidate['case_signature']
    for key in ('endpoints', 'cnt_binding'):
        if change == 'reverse': signature[key].reverse()
        else:
            for endpoint in signature[key]: endpoint['electrode_id'] = 'renamed_'+endpoint['electrode_id']
    result = search(query, make_index([candidate]), {'A'}, topk=1)
    assert result['primary'][0]['roi_id'] == candidate['roi_id']
    assert result['rejected'] == []


def test_N5_outside_cohort_rejections_are_hidden_without_expansion(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    query, records = fixed_records(source); outside = records[-1]
    for key in ('endpoints', 'cnt_binding'): outside['case_signature'][key][0]['role'] = 'incompatible'
    assert search(query, make_index(records), {'A'}, topk=3)['rejected'] == []
    assert search(query, make_index(records), {'A'}, topk=3, expand=True)['rejected'] == [
        {'roi_id': 'B', 'reason': 'case_or_CNT_binding_incompatible'}]


def test_N6_rounded_scores_tie_by_id(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    query, records = fixed_records(source)
    records[0]['embedding'] = [1, .0001]; records[1]['embedding'] = [1, .0002]
    result = search(query, make_index(records[:2]), {'A'}, topk=2)
    assert [(row['roi_id'], row['score']) for row in result['primary']] == [('A1', 1), ('A2', 1)]


def test_N10_split_filters_train_queries_and_missing_positives_are_reported(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    from psdl2.evaluate import evaluate
    query, records = fixed_records(source)
    result = search(query, make_index(records[:2]), {'A'}, topk=2)
    train_result = deepcopy(result); train_result['query']['roi_id'] = 'train_query'
    labels = [{'a': 'query', 'b': 'A1', 'relation': 'similar'},
              {'a': 'query', 'b': 'not_indexed', 'relation': 'similar'},
              {'a': 'train_query', 'b': 'A2', 'relation': 'similar'}]
    measured = evaluate([train_result, result], k=1, labels=labels,
                        assignments={'query': 'test', 'train_query': 'train'}, partition='test')
    assert measured['excluded_queries'] == ['train_query'] and measured['n_evaluable_queries'] == 1
    assert measured['recall_at_k'] == 1
    assert measured['positives_missing_from_index'] == [{'roi_id': 'query', 'missing_positive_roi_ids': ['not_indexed']}]


def test_N11_training_rejects_label_id_absent_from_export(source, tmp_path):
    from psdl2.train import train, read_config
    from pathlib import Path
    _, write, _ = source; roi = load_export(write()).ready[0]
    config = read_config(Path(__file__).parents[1]/'configs/synthetic.example.json')
    with pytest.raises(ContractError, match='label_roi_ids_absent_from_export: typo'):
        train([roi], config, tmp_path/'never.pt', lineages={}, assignments={},
              labels=[{'a': roi.row['roi_id'], 'b': 'typo', 'relation': 'similar'}])
    assert not (tmp_path/'never.pt').exists()


def test_N13_directory_is_fsynced_after_replace(tmp_path, monkeypatch):
    import os
    import psdl2.checkpoint as checkpoint
    events = []; original_replace = os.replace; original_fsync = os.fsync
    def replace_file(a, b): events.append('replace'); return original_replace(a, b)
    def fsync(fd):
        import stat
        events.append('directory_fsync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file_fsync')
        return original_fsync(fd)
    monkeypatch.setattr(checkpoint.os, 'replace', replace_file); monkeypatch.setattr(checkpoint.os, 'fsync', fsync)
    checkpoint.atomic_torch_save({'test': 1}, tmp_path/'state.pt')
    assert events == ['file_fsync', 'replace', 'directory_fsync']


def test_B2_rejection_counts_in_effective_batch_error(source):
    from psdl2.train import paired_forward
    from psdl2.model import Encoder
    _, write, _ = source; roi = load_export(write()).ready[0]
    row = deepcopy(roi.row); row['roi_id'] = 'second'
    channels = dict(roi.channels); invalid = channels['valid'].copy(); invalid[3, 3] = 0; channels['valid'] = invalid
    bad = replace(roi, channels=channels)
    with pytest.raises(ContractError) as caught:
        paired_forward(Encoder(model_config()), [bad, replace(bad, row=row)], view_config(), loss_config(), np.random.default_rng(1))
    payload = json.loads(str(caught.value).split(': ', 1)[1])
    assert payload == {'accepted_pair_count': 0, 'rejected_pair_count': 2, 'reasons': {'common_core_not_observed': 2}}


@pytest.mark.parametrize('change', ['blocked_to_ready', 'identity', 'origin', 'axis', 'uppercase_hash', 'cnt_missing', 'parent_path', 'tensor_policy_missing', 'valid_polarity'])
def test_review_importer_reproductions_remain_fail_closed(source, change):
    doc, write, _ = source; row = doc['rois'][0]
    if change == 'blocked_to_ready':
        doc['excluded_records'] = [{'roi_id': row['roi_id'], 'reasons': ['source_shape_required']}]
    elif change == 'identity': row['source_identity']['product_id'] = 'wrong'
    elif change == 'origin': row['frame']['origin_xy_um'] = [1000, 2000]
    elif change == 'axis': row['frame']['axis_order'] = 'xy'
    elif change == 'uppercase_hash': row['source_assets']['cnt']['sha256'] = row['source_assets']['cnt']['sha256'].upper()
    elif change == 'cnt_missing': row['source_assets']['cnt']['observed'] = False
    elif change == 'parent_path': row['source_assets']['cnt']['path'] = 'assets/../assets/cnt.npy'
    elif change == 'tensor_policy_missing': del doc['tensor_input_policy']
    else: row['source_assets']['valid']['value_semantics'] = 'binary_zero_is_marked'
    if change == 'tensor_policy_missing':
        with pytest.raises(ContractError): load_export(write())
    else:
        assert not load_export(write()).ready


def test_review_all_unknown_loss_and_no_batchnorm():
    from psdl2.losses import objective
    from psdl2.model import Encoder
    a = torch.tensor([[0., 1.], [2., 3.]])
    unknown = objective(a, a, loss_config(), relations=torch.full((2, 2), -1))
    baseline = objective(a, a, loss_config())
    assert unknown['supervised'].item() == 0 and unknown['supervised_known_pair_count'] == 0
    assert unknown['supervised_different_pair_count'] == 0
    torch.testing.assert_close(unknown['total'], baseline['total'])
    assert not any(isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
                   for module in Encoder(model_config()).modules())


def test_review_search_is_independent_of_index_iteration_order(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    query, records = fixed_records(source)
    normal = search(query, make_index(records), {'A'}, topk=3, expand=True)
    reverse = search(query, make_index(records[::-1]), {'A'}, topk=3, expand=True)
    assert normal['primary'] == reverse['primary'] and normal['supplementary'] == reverse['supplementary']


def test_topology_policy_is_required_in_config_and_outputs(source):
    from dataclasses import asdict
    _, write, _ = source; roi = load_export(write()).ready[0]
    raw = asdict(view_config()); del raw['topology_policy']
    with pytest.raises(ContractError, match='explicit_complete'): ViewConfig.from_dict(raw)
    a, b, evidence = generate_pair(roi, view_config(), np.random.default_rng(1))
    assert a.metadata['topology_policy'] == b.metadata['pair_acceptance']['topology_policy'] == evidence['topology_policy'] == TOPOLOGY_POLICY


def test_B2_coarse_connection_lost_outside_core_but_core_content_unchanged(source):
    a = np.zeros((20, 20), np.uint8)
    a[5:12, 6] = 1; a[5:12, 13] = 1; a[5, 6:14] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), pitch=1, center=(110, 210))
    cfg = ViewConfig(CropConfig((14, 14), (18, 18), 'preserve_exported_layers_v1'),
                     2, 2, 'area_coverage', 'area_coverage', (0, 0), (8, 4), TOPOLOGY_POLICY)
    first, second, _ = generate_pair(roi, cfg, np.random.default_rng(1))
    channels = dict(second.near.channels); mask = channels['layout:0'].copy(); mask[1, 3] = 0
    channels['layout:0'] = mask; second = replace(second, near=replace(second.near, channels=channels))
    # The removed coarse cell is above the core (y205..207 vs core y208..212).
    # Both core stems remain marked, but their only route is broken.
    with pytest.raises(ContractError, match='resampled_core_connection_lost'):
        validate_pair(roi, first, second, (8, 4), TOPOLOGY_POLICY)


def test_N8_coarse_merging_is_explicitly_not_checked(source):
    a = np.zeros((20, 20), np.uint8); a[10, 8] = 1; a[10, 10] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), pitch=1, center=(110, 210))
    cfg = ViewConfig(CropConfig((8, 8), (8, 8), 'preserve_exported_layers_v1'),
                     4, 4, 'area_coverage', 'area_coverage', (0, 0), (8, 8), TOPOLOGY_POLICY)
    # Two separate native cells become adjacent marked coarse cells. This is a
    # merge, not a loss of native connection; the declared policy permits it.
    assert generate_pair(roi, cfg, np.random.default_rng(1))[2]['status'] == 'accepted'


def test_review_non_square_crop_axes_padding_and_shared_center(source):
    a = np.zeros((6, 10), np.uint8); b = np.zeros_like(a)
    a[1, 1] = 1; b[4, 8] = 1
    roi = hand_roi(source, (a, b), center=(102.5, 201.5))
    bundle = build_bundle(roi, CropConfig((4, 2), (8, 4), 'preserve_exported_layers_v1'))
    expected_a = np.zeros((4, 8), np.uint8); expected_a[0, 0] = 1
    expected_b = np.zeros((4, 8), np.uint8); expected_b[3, 7] = 1
    np.testing.assert_array_equal(bundle.near.channels['layout:0'], expected_a)
    np.testing.assert_array_equal(bundle.near.channels['layout:1'], expected_b)
    valid = np.zeros((8, 16), np.uint8); valid[1:7, 3:13] = 1
    np.testing.assert_array_equal(bundle.wide.channels['valid'], valid)
    assert bundle.near.metadata['center_xy_um'] == bundle.wide.metadata['center_xy_um'] == [102.5, 201.5]


def test_N7_low_level_pitch_defaults_do_not_change_physical_extent(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    cfg = CropConfig((2, 2), (4, 4), 'preserve_exported_layers_v1')
    native = build_bundle(roi, cfg)
    coarser = build_bundle(roi, cfg, near_output_pitch_um=1, wide_output_pitch_um=1)
    assert native.near.grid.shape_yx == (4, 4) and coarser.near.grid.shape_yx == (2, 2)
    assert native.near.metadata['bbox_xyxy_um'] == coarser.near.metadata['bbox_xyxy_um'] == [101, 201, 103, 203]
