"""Addendum §6: each expected outcome follows the stated hand-built graph."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from psdl2.schema import ContractError, CropConfig
from psdl2.views import (ViewConfig, ViewRejection, TOPOLOGY_POLICY, CONNECTION_SEMANTICS,
                        COMPARISON_BASIS, generate_pair, validate_pair, joint_labels, preflight)
from test_r4_regressions import hand_roi, stripes, stripe_config
from test_views import view_config


def recipe(*, near=14, wide=30, core=6, jitter=(0, 3), pitch=1, coarse=False, attempts=3):
    return ViewConfig(CropConfig((near,)*2, (wide,)*2, 'preserve_exported_layers_v1'),
                      pitch, pitch, 'area_coverage' if coarse else 'nearest',
                      'area_coverage' if coarse else 'nearest', jitter, (core,)*2,
                      TOPOLOGY_POLICY, CONNECTION_SEMANTICS, COMPARISON_BASIS, attempts)


class Offsets:
    def __init__(self, *offsets): self.values = iter(offsets)
    def uniform(self, low, high): return np.asarray(next(self.values))


def bridge_roi(source, channel='cnt'):
    a = np.zeros((40, 40), np.uint8); b = a.copy()
    a[:, 17] = 1; b[:, 22] = 1
    roi = hand_roi(source, (a, b), pitch=1, center=(120, 220))
    channels = dict(roi.channels); channels[channel] = np.zeros_like(a)
    channels[channel][14, 17:23] = 1
    return replace(roi, channels=channels)


def pair_at(roi, cfg, first=(0, 0), second=(0, 0)):
    bundles = []
    for offset in (first, second):
        row = deepcopy(roi.row); row['center_xy_um'] = (np.array(roi.center_xy_um)+offset).tolist()
        bundle = cfg.bundle(replace(roi, row=row))
        bundle.metadata['original_center_xy_um'] = list(roi.center_xy_um)
        bundles.append(bundle)
    return bundles


def same_native_core(roi, a, b, cfg):
    # In this B3 oracle the core source rows/columns are exactly 17..22.
    for bundle in (a, b):
        patch = bundle.near; sx, sy = patch.metadata['source_indices_xy']
        x, y = np.array([sx.index(i) for i in range(17, 23)]), np.array([sy.index(i) for i in range(17, 23)])
        for key in roi.channels:
            np.testing.assert_array_equal(patch.channels[key][np.ix_(y, x)], roi.channels[key][17:23, 17:23])


@pytest.mark.parametrize('channel', ['cnt', 'via:0'])
def test_B3_natural_bridge_crop_exact_reason(source, channel):
    roi = bridge_roi(source, channel); cfg = recipe()
    a, b = pair_at(roi, cfg, second=(0, 3)); same_native_core(roi, a, b, cfg)
    # Near1 rows13..26 contains row14 bridge; near2 rows16..29 excludes it.
    # The two core layout lines go from one joint component to two.
    assert a.near.channels[channel].sum() == 6 and b.near.channels[channel].sum() == 0
    with pytest.raises(ViewRejection, match='^core_joint_connectivity_changed$'):
        validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)


@pytest.mark.parametrize('channel', ['cnt', 'via:0'])
def test_B3_actual_native_patch_bridge_removed(source, channel):
    roi = bridge_roi(source, channel); cfg = recipe(jitter=(0, 0)); a, b = pair_at(roi, cfg)
    channels = dict(b.near.channels); channels[channel] = np.zeros_like(channels[channel])
    b = replace(b, near=replace(b.near, channels=channels)); same_native_core(roi, a, b, cfg)
    # Source still has the bridge. Actual output loses the only layout-layout path.
    with pytest.raises(ViewRejection, match='^core_joint_connectivity_changed$'):
        validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)


def test_M7_both_views_lose_bridge_accepted_between_views_only(source):
    roi = bridge_roi(source); cfg = recipe()
    a, b = pair_at(roi, cfg, (0, 3), (0, 2.5)); same_native_core(roi, a, b, cfg)
    # Both near graphs have two components; both wide graphs retain one.
    evidence = validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)
    assert evidence['status'] == 'accepted'
    assert evidence['comparison_basis'] == 'between_views_only'
    assert evidence['connection_semantics'] == 'global_contact_proxy_unverified'


def test_crossing_layouts_require_contact_not_direct_layout_edge():
    masks = np.zeros((5, 7, 7), bool)
    masks[0, 3, :] = True; masks[1, :, 3] = True
    labels = joint_labels(masks, [0, 1])
    # Coincident XY in different layouts is not a physical/contact edge.
    assert labels[0, 3, 3] != labels[1, 3, 3]
    masks[2, 3, 3] = True; labels = joint_labels(masks, [0, 1])
    assert labels[0, 3, 3] == labels[1, 3, 3] == labels[2, 3, 3]
    # CNT/VIA and VIA/VIA have no direct edge without a marked layout mediator.
    masks[:] = False; masks[2:, 3, 3] = True
    labels = joint_labels(masks, [0, 1]); assert len(np.unique(labels[2:, 3, 3])) == 3


def test_dense_cnt_boundary_band_without_layout_contacts_is_allowed(source):
    roi = bridge_roi(source); channels = dict(roi.channels)
    cnt = np.zeros((40, 40), np.uint8)
    cnt[13:17, 13:16] = 1; cnt[13:17, 24:27] = 1
    channels['cnt'] = cnt; roi = replace(roi, channels=channels)
    cfg = recipe(); a, b = pair_at(roi, cfg, second=(0, 3)); same_native_core(roi, a, b, cfg)
    # Dense patches at the window boundary never touch x17/x22 layout lines.
    assert a.near.channels['cnt'].sum() > b.near.channels['cnt'].sum()
    assert validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)['status'] == 'accepted'


@pytest.mark.parametrize('jitter', [0, .1, .3, 1])
def test_two_percent_lattice_is_truly_isolated_and_accepted(source, jitter):
    rng = np.random.default_rng(2026)
    masks = []
    for _ in range(2):
        a = np.zeros((200, 200), np.uint8); a[::2, ::2] = rng.random((100, 100)) < .08
        # An even/even lattice cannot have a horizontal or vertical neighbour.
        assert not np.any(a[:-1] & a[1:]) and not np.any(a[:, :-1] & a[:, 1:])
        masks.append(a)
    roi = hand_roi(source, masks); cfg = stripe_config(jitter)
    for x in (125, 150, 175):
        for y in (225, 250, 275):
            row = deepcopy(roi.row); row['center_xy_um'] = [x, y]
            for _ in range(20):
                assert generate_pair(replace(roi, row=row), cfg, rng)[2]['status'] == 'accepted'


def test_coarse_cnt_bridge_deleted_outside_core_exact_reason(source):
    # Native bridge row14 maps to coarse row2, whose y214..216 support is outside
    # the fixed core [217,223); removing it leaves both vertical core stems.
    roi = bridge_roi(source); cfg = recipe(near=20, wide=32, pitch=2, coarse=True, jitter=(0, 0))
    a, b = pair_at(roi, cfg)
    assert validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)['status'] == 'accepted'
    changed = dict(b.near.channels); cnt = changed['cnt'].copy(); cnt[2, :] = 0; changed['cnt'] = cnt
    b = replace(b, near=replace(b.near, channels=changed))
    # The core output rows3..6 retain all channel values; this is not content loss.
    for key in roi.channels:
        np.testing.assert_array_equal(a.near.channels[key][3:7], b.near.channels[key][3:7])
    with pytest.raises(ViewRejection, match='^resampled_core_joint_relation_changed$'):
        validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)


def test_coarse_native_separation_becomes_spurious_merge_rejected(source):
    a = np.zeros((40, 40), np.uint8); a[18, 18] = a[18, 21] = 1
    roi = hand_roi(source, (a, np.zeros_like(a)), pitch=1, center=(120, 220))
    cfg = recipe(near=16, wide=32, core=8, pitch=4, coarse=True, jitter=(0, 0))
    first, second = pair_at(roi, cfg)
    for key in roi.channels:
        np.testing.assert_array_equal(first.near.channels[key], second.near.channels[key])
    # Native cells are separated by two empty columns; adjacent coarse cells
    # acquire support and join their anchors. Identical coarse views still reject.
    with pytest.raises(ViewRejection, match='^resampled_core_joint_relation_changed$'):
        validate_pair(roi, first, second, cfg.core_size_xy_um, TOPOLOGY_POLICY)


def test_fractional_coverage_differences_with_same_joint_partition_allowed(source):
    roi = bridge_roi(source); cfg = recipe(near=20, wide=32, pitch=2, coarse=True, jitter=(.2, .2))
    a, b = pair_at(roi, cfg, (-.2, -.2), (.2, .2))
    # Parallel lines stay separate except the same CNT bridge in both windows.
    assert not np.array_equal(a.near.channels['layout:0'], b.near.channels['layout:0'])
    assert validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)['status'] == 'accepted'


@pytest.mark.parametrize('change,reason', [('valid', 'common_core_not_observed'),
                                         ('cnt', 'core_marked_content_changed'),
                                         ('empty', 'no_core_layout_anchor')])
def test_core_observation_content_and_empty_anchor_are_distinct(source, change, reason):
    roi = bridge_roi(source); channels = dict(roi.channels); channels['cnt'] = channels['cnt'].copy()
    channels['cnt'][20, 20] = 1; roi = replace(roi, channels=channels)
    cfg = recipe(jitter=(0, 0)); a, b = pair_at(roi, cfg)
    if change == 'valid':
        channels = dict(roi.channels); channels['valid'] = channels['valid'].copy(); channels['valid'][20, 20] = 0
        roi = replace(roi, channels=channels); a, b = pair_at(roi, cfg)
    elif change == 'empty':
        channels = dict(roi.channels)
        for key in ('layout:0', 'layout:1'): channels[key] = np.zeros_like(channels[key])
        roi = replace(roi, channels=channels); a, b = pair_at(roi, cfg)
    else:
        channels = dict(b.near.channels); cnt = channels['cnt'].copy(); cnt[7, 7] = 0; channels['cnt'] = cnt
        b = replace(b, near=replace(b.near, channels=channels))
    with pytest.raises(ViewRejection, match='^'+reason+'$'):
        validate_pair(roi, a, b, cfg.core_size_xy_um, TOPOLOGY_POLICY)


def test_M1_preflight_exact_extrema_and_M2_coarse_nearest(source):
    roi = stripes(source); cfg = stripe_config(4.9)
    assert preflight(roi, cfg)['status'] == 'passed'  # aligned core10 fits exact indices
    bad = replace(cfg, core_size_xy_um=(10.2, 10.2))
    with pytest.raises(ContractError, match='^view_preflight_core_not_represented_at_jitter_extrema$'):
        preflight(roi, bad)
    with pytest.raises(ContractError, match='coarse_nearest_not_supported_for_views'):
        preflight(roi, replace(cfg, wide_pitch_um=1))
    assert preflight(roi, replace(cfg, wide_pitch_um=1, wide_resampling_mode='area_coverage'))['status'] == 'passed'
    channels = dict(roi.channels); valid = channels['valid'].copy(); valid[100, 100] = 0; channels['valid'] = valid
    with pytest.raises(ContractError, match='^view_preflight_common_core_not_observed$'):
        preflight(replace(roi, channels=channels), cfg)


def test_M4_analysis_layer_strict_identity_but_search_compatible(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    query, records = fixed_records(source); candidate = deepcopy(query)
    candidate['roi_id'] = 'other_roi'; candidate['case_signature']['analysis_layer'] = 'M1'
    result = search(query, make_index([candidate]), {query['product_id']}, topk=1)
    assert result['primary'][0]['same_physical_case'] is False
    assert result['rejected'] == []


@pytest.mark.parametrize('field', ['topology_policy', 'connection_semantics', 'comparison_basis', 'max_pair_attempts'])
def test_all_v3_policy_settings_required_no_defaults(field):
    raw = asdict(view_config()); del raw[field]
    with pytest.raises(ContractError, match='^explicit_complete_view_config_required$'):
        ViewConfig.from_dict(raw)


def test_v2_checkpoints_and_index_records_fail_closed(source, tmp_path):
    from test_checkpoint import checkpoint_config
    from test_model import model_config
    from psdl2.model import Encoder
    from psdl2.checkpoint import save_checkpoint, read_checkpoint
    from test_search import fixed_records, make_index
    from psdl2.search import validate_index
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters())
    config = checkpoint_config(); config['views']['topology_policy'] = 'common_core_connectivity_v2'
    path = tmp_path/'v2.pt'; save_checkpoint(path, model, optimizer, config, epoch=1)
    with pytest.raises(ContractError, match='^explicit_supported_topology_policy_required$'):
        read_checkpoint(path, config)
    _, records = fixed_records(source); records[0]['topology_policy'] = 'common_core_connectivity_v2'
    with pytest.raises(ContractError, match='^incompatible_candidate_view_policy$'):
        validate_index(make_index(records))


def test_M3_changing_other_location_preserves_own_alignment():
    from psdl2.losses import objective
    from test_losses import loss_config
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[0., 2.], [3., 5.]])
    original = objective(a, b, loss_config())
    b[1] = torch.tensor([50., -50.]); changed = objective(a, b, loss_config())
    # Location A squared errors [1,0] average .5, independent of B's values.
    assert original['alignment_per_location'][0].item() == changed['alignment_per_location'][0].item() == .5
    assert original['alignment_per_location'][1].item() != changed['alignment_per_location'][1].item()


def test_bounded_retry_success_skip_and_non_view_errors(source, monkeypatch):
    import psdl2.train as module
    from test_model import model_config
    from test_losses import loss_config
    from psdl2.model import Encoder
    _, write, _ = source
    from psdl2.data import load_export
    roi = load_export(write()).ready[0]; row = deepcopy(roi.row); row['roi_id'] = 'second'
    other = replace(roi, row=row); cfg = view_config(jitter=(0, 0)); bundle = cfg.bundle(roi)
    draws = []; schedule = iter([True, False, False, True])
    def scheduled(roi, cfg, rng):
        draws.append(roi.row['roi_id']); rng.random()
        if not next(schedule): raise ViewRejection('core_joint_connectivity_changed')
        return bundle, bundle, {}
    monkeypatch.setattr(module, 'generate_pair', scheduled)
    result, rejected = module.paired_forward(Encoder(model_config()), [roi, other], cfg, loss_config(), np.random.default_rng(3))
    assert result['attempted_pairs'] == 4 and result['retry_attempts'] == 2
    assert result['accepted_pairs_used'] == result['accepted_pairs_generated'] == 2
    assert result['rejected_attempts_by_reason'] == {'core_joint_connectivity_changed': 2}
    assert draws == [roi.row['roi_id'], 'second', 'second', 'second']
    schedule = iter([True, False, False, False]); draws.clear()
    result, _ = module.paired_forward(Encoder(model_config()), [roi, other], cfg, loss_config(), np.random.default_rng(3))
    assert result['accepted_pairs_used'] == 0 and result['accepted_pairs_generated'] == 1 and result['skipped_batches'] == 1
    def forged(*args): raise ContractError('two_views_must_share_original_location_and_bindings')
    monkeypatch.setattr(module, 'generate_pair', forged)
    with pytest.raises(ContractError, match='^two_views_must_share_original_location_and_bindings$'):
        module.paired_forward(Encoder(model_config()), [roi, other], cfg, loss_config(), np.random.default_rng(3))


def test_array_joint_graph_matches_independent_small_breadth_first_oracle():
    # Only the tiny test oracle uses tuples; production graph labels are arrays.
    rng = np.random.default_rng(43)
    for _ in range(20):
        masks = rng.random((4, 8, 9)) < .3
        labels = joint_labels(masks, [0, 1]); expected = np.zeros_like(labels)
        label = 0
        for c, y, x in np.argwhere(masks):
            if expected[c, y, x]: continue
            label += 1; todo = [(c, y, x)]; expected[c, y, x] = label
            while todo:
                k, r, u = todo.pop()
                neighbours = [(k, r-1, u), (k, r+1, u), (k, r, u-1), (k, r, u+1)]
                neighbours += [(other, r, u) for other in range(4) if (k < 2) != (other < 2)]
                for other, v, w in neighbours:
                    if 0 <= v < 8 and 0 <= w < 9 and masks[other, v, w] and not expected[other, v, w]:
                        expected[other, v, w] = label; todo.append((other, v, w))
        # Check full component partitions via first occurrences, not AxA matrices.
        from psdl2.views import _partition
        np.testing.assert_array_equal(_partition(labels[masks]), _partition(expected[masks]))


def lightweight_training_setup(tmp_path, monkeypatch):
    """Real scheduling/checkpoint paths, with no real encoder or optimizer steps."""
    import psdl2.train as module
    from test_end_to_end import synthetic_exchange
    _, export, config, lineages, _ = synthetic_exchange(tmp_path)
    assignments = {r.row['roi_id']: 'train' for r in export.ready}
    config.update(batch_size=2)
    class TinyEncoder(torch.nn.Module):
        def __init__(self, config): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, bundles):
            # Finite, different location vectors suffice to execute the real loss.
            values = torch.arange(len(bundles)*2, dtype=torch.float32).reshape(-1, 2)
            return {'projection': values*self.weight}
    class NoStepOptimizer:
        def __init__(self, parameters, lr): self.parameters = list(parameters); self.steps = 0
        def zero_grad(self, **kwargs):
            for p in self.parameters: p.grad = None
        def step(self): self.steps += 1  # Scheduling spy, no parameter update.
        def state_dict(self): return {'steps': self.steps}
        def load_state_dict(self, state): self.steps = state['steps']
    original_rng = module.np.random.default_rng
    class FixedBatches:
        def __init__(self, seed): self.rng = original_rng(seed)
        @property
        def bit_generator(self): return self.rng.bit_generator
        def permutation(self, n): self.rng.permutation(n); return np.arange(n)
        def random(self): return self.rng.random()
    monkeypatch.setattr(module, 'Encoder', TinyEncoder)
    monkeypatch.setattr(module.torch.optim, 'Adam', NoStepOptimizer)
    monkeypatch.setattr(module.np.random, 'default_rng', FixedBatches)
    def scheduled(roi, cfg, rng):
        value = rng.random()
        # First batch: one success plus a ROI rejected on all 3 tries. Others
        # accepted; the draw is included in evidence to prove retry order resumes.
        if roi.row['roi_id'] == 'roi_0_1':
            raise ViewRejection('core_joint_connectivity_changed')
        bundle = cfg.bundle(roi); return bundle, bundle, {'draw': value}
    monkeypatch.setattr(module, 'generate_pair', scheduled)
    return module, export.ready, config, lineages, assignments


def test_retry_skip_training_continues_and_resume_order_bit_exact_without_training(tmp_path, monkeypatch):
    module, rois, config, lineages, assignments = lightweight_training_setup(tmp_path, monkeypatch)
    config['epochs'] = 2; resumed_path = tmp_path/'resume.pt'
    initial = module.train(rois, config, resumed_path, lineages=lineages, assignments=assignments)
    # Batch0: 4 attempts/2 retries, one generated but unused. Next batches: 4 used, 2 steps.
    for epoch in initial['epoch_history']:
        assert {k: epoch[k] for k in ('attempted_pairs', 'retry_attempts', 'accepted_pairs_generated',
                                     'accepted_pairs_used', 'skipped_batches', 'optimizer_steps')} == dict(
            attempted_pairs=8, retry_attempts=2, accepted_pairs_generated=5, accepted_pairs_used=4,
            skipped_batches=1, optimizer_steps=2)
        assert epoch['rejected_attempts_by_reason'] == {'core_joint_connectivity_changed': 3}
        assert epoch['batches'][0]['skipped_batches'] == 1
    config['epochs'] = 4
    resumed = module.train(rois, config, resumed_path, lineages=lineages, assignments=assignments, resume=resumed_path)
    continuous_path = tmp_path/'continuous.pt'
    continuous = module.train(rois, config, continuous_path, lineages=lineages, assignments=assignments)
    assert resumed['epoch_history'] == continuous['epoch_history']
    assert resumed['history'] == continuous['history'][6:]
    a, b = (torch.load(p, weights_only=True) for p in (resumed_path, continuous_path))
    assert a['optimizer'] == b['optimizer'] == {'steps': 8}
    assert a['model_fingerprint'] == b['model_fingerprint']
    assert a['rng']['views'] == b['rng']['views']
    assert a['epoch_history'] == b['epoch_history']
    torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'], rtol=0, atol=0)
    assert a['rng']['python'] == b['rng']['python'] and a['rng']['numpy'] == b['rng']['numpy']


def test_no_usable_epoch_preserves_completed_checkpoint_and_reports_counts(tmp_path, monkeypatch):
    module, rois, config, lineages, assignments = lightweight_training_setup(tmp_path, monkeypatch)
    path = tmp_path/'last_completed.pt'
    module.train(rois, config, path, lineages=lineages, assignments=assignments)
    before = path.read_bytes()
    def rejected(*args): raise ViewRejection('core_joint_connectivity_changed')
    monkeypatch.setattr(module, 'generate_pair', rejected); config['epochs'] = 2
    with pytest.raises(ContractError, match='^no_usable_batch_in_epoch: ') as error:
        module.train(rois, config, path, lineages=lineages, assignments=assignments, resume=path)
    counts = error.value.statistics
    assert counts['epoch'] == 2 and counts['optimizer_steps'] == 0 and counts['skipped_batches'] == 3
    assert counts['attempted_pairs'] == 18 and counts['retry_attempts'] == 12
    assert counts['rejected_attempts_by_reason'] == {'core_joint_connectivity_changed': 18}
    assert path.read_bytes() == before and torch.load(path, weights_only=True)['epoch'] == 1


def test_preflight_failure_before_optimizer_constructed(source, tmp_path, monkeypatch):
    import psdl2.train as module
    from pathlib import Path
    roi = stripes(source); row = deepcopy(roi.row); row['roi_id'] = 'other'
    rois = [roi, replace(roi, row=row)]
    config = module.read_config(Path(__file__).parents[1]/'configs/synthetic.example.json')
    config['views'] = asdict(replace(stripe_config(4.9), core_size_xy_um=(10.2, 10.2)))
    def forbidden(*args, **kwargs): pytest.fail('preflight must precede optimizer construction/steps')
    monkeypatch.setattr(module.torch.optim, 'Adam', forbidden)
    with pytest.raises(ContractError, match='^view_preflight_core_not_represented_at_jitter_extrema$'):
        module.train(rois, config, tmp_path/'never.pt', lineages={roi.row['product_id']: 'same'},
                     assignments={r.row['roi_id']: 'train' for r in rois})
    assert not (tmp_path/'never.pt').exists()


def test_retry_setting_is_checkpoint_fingerprint_bound(tmp_path):
    from psdl2.checkpoint import save_checkpoint, read_checkpoint
    from test_checkpoint import checkpoint_config
    from psdl2.model import Encoder
    from test_model import model_config
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters())
    config = checkpoint_config(); path = tmp_path/'state.pt'; save_checkpoint(path, model, optimizer, config, epoch=1)
    changed = deepcopy(config); changed['views']['max_pair_attempts'] += 1
    with pytest.raises(ContractError, match='^incompatible_config_fingerprint$'):
        read_checkpoint(path, changed)


@pytest.mark.parametrize('attempts', [0, -1, True, 1.5])
def test_max_pair_attempts_is_explicit_positive_integer(attempts):
    with pytest.raises(ContractError, match='^explicit_positive_max_pair_attempts_required$'):
        replace(view_config(), max_pair_attempts=attempts)


def test_M5_via_count_not_endpoint_inference_and_M6_missing_commit_opt_in(source):
    from test_search import fixed_records, make_index
    from psdl2.search import search
    from psdl2.data import load_export
    query, records = fixed_records(source)
    query['case_signature']['via_count'] = 1; records[0]['case_signature']['via_count'] = 3
    assert search(query, make_index(records[:1]), {'A'}, topk=1)['primary'][0]['roi_id'] == records[0]['roi_id']
    doc, write, _ = source; del doc['producer_commit']; doc['producer_commit_reason'] = 'synthetic_unversioned'
    with pytest.raises(ContractError, match='^producer_commit_required$'): load_export(write())
    assert load_export(write(), allow_unversioned_producer=True).versions['unversioned_producer_accepted'] is True


def test_frame_outside_core_and_duplicate_location_fail_closed(source):
    from psdl2.train import paired_forward
    from psdl2.model import Encoder
    from test_model import model_config
    from test_losses import loss_config
    roi = stripes(source); row = deepcopy(roi.row); row['center_xy_um'] = [100, 200]
    with pytest.raises(ContractError, match='^view_preflight_common_core_not_observed$'):
        preflight(replace(roi, row=row), stripe_config(0))
    with pytest.raises(ContractError, match='^batch_requires_distinct_original_locations$'):
        paired_forward(Encoder(model_config()), [roi, roi], stripe_config(0), loss_config(), np.random.default_rng(2))


def test_legacy_container_versions_and_top_level_policy_not_migrated(source, tmp_path):
    from test_checkpoint import checkpoint_config
    from test_model import model_config
    from psdl2.model import Encoder
    from psdl2.checkpoint import save_checkpoint, read_checkpoint
    from test_search import fixed_records, make_index
    from psdl2.search import validate_index
    config = checkpoint_config(); model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path/'legacy.pt'; save_checkpoint(path, model, optimizer, config, epoch=1)
    raw = torch.load(path, weights_only=True); raw['schema_version'] = 'psdl2_checkpoint_v1'; torch.save(raw, path)
    with pytest.raises(ContractError, match='^incompatible_checkpoint_schema$'): read_checkpoint(path, config)
    _, records = fixed_records(source); index = make_index(records)
    index['schema_version'] = 'psdl2_index_v1'
    with pytest.raises(ContractError, match='^unknown_index_schema$'): validate_index(index)
    index['schema_version'] = 'psdl2_index_v2'; index['topology_policy'] = 'common_core_connectivity_v2'
    with pytest.raises(ContractError, match='^incompatible_index_view_policy$'): validate_index(index)
