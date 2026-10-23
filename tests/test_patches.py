"""psdl2.patches: generator-only location conditions, sampling, shortage and atomic publication."""
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from psdl2.crop import rational
from psdl2.data import load_dataset, load_export, patch_roi_id, patch_set_content_hash, views_fingerprint
from psdl2.patches import FUNNEL_STAGES, main as patches_main
from psdl2.schema import CropConfig
from psdl2.views import ViewConfig, preflight
from test_dataset import ROOT, source_for, write_dataset
from test_views import view_config

STRINGS = ('common_core_connectivity_v3', 'global_contact_proxy_unverified', 'between_views_only')


def views(name):
    if name == 'native':
        return view_config()
    if name == 'nonintegral':
        return ViewConfig(CropConfig((3.0, 3.0), (3.9, 3.9), 'preserve_exported_layers_v1'), .3, .3, 'nearest', 'nearest',
                          (.1, .1), (1.8, 1.8), *STRINGS, 3)
    if name == 'area':
        return ViewConfig(CropConfig((3, 3), (4, 4), 'preserve_exported_layers_v1'), .5, 1.0, 'nearest', 'area_coverage',
                          (.1, .1), (2, 2), *STRINGS, 3)
    raise KeyError(name)


FRAMES = {'native': ((100.0, 200.0), .5), 'nonintegral': ((100.15, 200.05), .3), 'area': ((100.0, 200.0), .5)}


def bars(shape=(30, 30), valid_holes=()):
    h, w = shape
    l0 = np.zeros(shape, np.uint8); l1 = np.zeros(shape, np.uint8)
    l0[6:24, 10:12] = 1; l1[6:24, 14:16] = 1
    cnt = np.zeros(shape, np.uint8); cnt[6:24, 12:14] = 1
    valid = np.ones(shape, np.uint8)
    for y0, y1, x0, x1 in valid_holes:
        valid[y0:y1, x0:x1] = 0
    return {'l0': l0, 'l1': l1, 'cnt': cnt, 'valid': valid}


def make_template(root, name='tmpl', *, rows=None, shape=(30, 30), origin=(100.0, 200.0), pitch=.5, product='T'):
    """Hand-written manual template export (one ready row per entry of ``rows``) with assets/*.npy."""
    folder = Path(root) / name
    template = json.loads((ROOT / 'tests/fixtures/review_roi_v1_export/review_rois.json').read_text())
    ready = next(r for r in template['rois'] if r['status'] == 'ready')
    payload = {**template, 'producer_commit': '0' * 40, 'assets_root': '.', 'rois': [], 'excluded_records': [], 'status': 'available'}
    for spec in rows or [{}]:
        rid = spec.get('rid', f'{product}_case0')
        row = json.loads(json.dumps(ready))
        centre = [origin[0]+(shape[1]//2+.5)*pitch, origin[1]+(shape[0]//2+.5)*pitch]
        row.update(roi_id=rid, product_id=product, case_id=spec.get('case', 'CASE'), frame_id=f'frame_{rid}',
                   source_version='synthetic_patch_v1', center_xy_um=[float(centre[0]), float(centre[1])],
                   failure_type=spec.get('failure_type', 'vertical_short'))
        row['source_identity'].update(product_id=product, case_id=row['case_id'], frame_id=row['frame_id'])
        row['frame'].update(frame_id=row['frame_id'], shape_yx=list(shape), origin_xy_um=list(origin), pixel_pitch_um=pitch)
        row['endpoints'] = [{'electrode_id': 'A', 'role': 'gate', 'actual_layer': 'M1'},
                            {'electrode_id': 'B', 'role': spec.get('role', 'data'), 'actual_layer': 'M3'}]
        row['analysis_layer'] = 'M3'
        if spec.get('kind') == 'region':
            half = pitch*4
            row.update(source_kind='electrode_count_region', source_region_id='R', region_bbox_um=[
                centre[0]-half, centre[1]-half, centre[0]+half, centre[1]+half])
            row['provenance'] = {'coverage': {'source_coverage_status': 'available'}}
        for layout, endpoint in zip(row['source_assets']['layouts'], row['endpoints']):
            layout['endpoint'] = endpoint.copy()
        arrays = spec.get('arrays') or bars(shape)
        refs = [layout['asset'] for layout in row['source_assets']['layouts']] + [row['source_assets']['cnt'], row['source_assets']['valid']]
        for channel, (ref, key) in enumerate(zip(refs, ('l0', 'l1', 'cnt', 'valid'))):
            path = folder / 'assets' / f'{rid}_{channel}.npy'; path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arrays[key])
            ref.update(path=str(path.relative_to(folder)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        payload['rois'].append(row)
    payload['source_manifest_hash'] = hashlib.sha256(json.dumps(payload['rois'], sort_keys=True).encode()).hexdigest()
    path = folder / 'review_rois.json'
    path.write_text(json.dumps(payload))
    return path


def write_config(root, vc, name='run.json'):
    path = Path(root) / name
    path.write_text(json.dumps({'config': {'views': asdict(vc)}}))
    return path


def world(tmp_path, variant='native', **kwargs):
    origin, pitch = FRAMES[variant]
    export = make_template(tmp_path, origin=origin, pitch=pitch, **kwargs)
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path)])
    return export, manifest, write_config(tmp_path, views(variant))


def run(capsys, manifest, config, out, *extra, mode='random'):
    code = patches_main(['--dataset', str(manifest), '--source', '0', '--config', str(config), '--mode', mode,
                         '--out', str(out), *extra])
    captured = capsys.readouterr()
    return (code, json.loads(captured.out) if captured.out.strip() else None,
            json.loads(captured.err) if captured.err.strip() else None)


RANDOM_ALL = ('--seed', '1', '--per-case', '100000', '--min-spacing-um', '0', '--allow-shortage')


# ---- independent brute-force oracle -------------------------------------------------------------------

def oracle(roi, vc, *, repeat=None, centres=None, spacing=None):
    """Per-candidate Fraction arithmetic over scanned cells; shares nothing with the generator's vectorised path."""
    f = roi.frame
    o = [rational(v) for v in f.origin_xy_um]; p = rational(f.pixel_pitch_um); h, w = f.shape_yx
    count = (w, h)
    if centres is None:
        centres = [(float(o[0]+(i+Fraction(1, 2))*p), float(o[1]+(j+Fraction(1, 2))*p)) for j in range(h) for i in range(w)]
    sizes = (vc.crop.wide_size_xy_um, vc.crop.near_size_xy_um)
    layouts = (roi.channels['layout:0'] > 0, roi.channels['layout:1'] > 0)

    def window(c, size, a):
        return o[a]+p*0, rational(c)-rational(size)/2, rational(c)+rational(size)/2

    def cells(c, size, a, clip=True):
        left, right = rational(c)-rational(size)/2, rational(c)+rational(size)/2
        return [k for k in range(count[a]) if o[a]+k*p < right and o[a]+(k+1)*p > left]

    stages = {'candidates': len(centres)}
    alive = list(centres)

    def keep(name, test):
        nonlocal alive
        new = [c for c in alive if test(c)]
        stages[name] = len(alive)-len(new); alive = new

    keep('rejected_repeat_period', lambda c: repeat is None or all(rational(c[a])-o[a] < rational(repeat[a]) for a in (0, 1)))

    def in_window(c):
        for a in (0, 1):
            for size in sizes:
                half = rational(size[a])/2+rational(vc.jitter_xy_um[a])
                if rational(c[a])-half < o[a] or rational(c[a])+half > o[a]+count[a]*p:
                    return False
        for a in (0, 1):
            if rational(c[a])-rational(vc.core_size_xy_um[a])/2 < o[a] or rational(c[a])+rational(vc.core_size_xy_um[a])/2 > o[a]+count[a]*p:
                return False
        return True
    keep('rejected_window', in_window)

    def core_cells(c):
        return [cells(c[a], vc.core_size_xy_um[a], a) for a in (0, 1)]
    keep('rejected_valid', lambda c: all(roi.channels['valid'][np.ix_(core_cells(c)[1], core_cells(c)[0])].ravel() == 1))
    keep('rejected_anchor', lambda c: any(layout[np.ix_(core_cells(c)[1], core_cells(c)[0])].any() for layout in layouts))

    def near_both(c):
        xs, ys = (cells(c[a], vc.crop.near_size_xy_um[a], a) for a in (0, 1))
        return all(layout[np.ix_(ys, xs)].any() for layout in layouts)
    keep('rejected_q1', near_both)

    survivors = []
    for c in alive:
        probe = replace(roi, row={**roi.row, 'center_xy_um': [c[0], c[1]]})
        try:
            preflight(probe, vc)
            survivors.append(c)
        except ValueError:
            pass
    stages['rejected_preflight'] = len(alive)-len(survivors)
    if spacing is not None:
        chosen = []
        for c in survivors:
            if all((rational(c[0])-rational(d[0]))**2+(rational(c[1])-rational(d[1]))**2 >= rational(spacing)**2 for d in chosen):
                chosen.append(c)
        survivors = chosen
    return stages, sorted(survivors, key=lambda c: (c[1], c[0]))


def read_set(path):
    return json.loads(Path(path).read_text())


def centres_of(document, template=None):
    return sorted(((p['center_xy_um'][0], p['center_xy_um'][1]) for p in document['patches']
                   if template is None or p['template_roi_id'] == template), key=lambda c: (c[1], c[0]))


# ---- location conditions ---------------------------------------------------------------------------------

@pytest.mark.parametrize('variant', ['native', 'nonintegral', 'area'])
def test_all_location_conditions_equal_the_independent_oracle(tmp_path, capsys, variant):
    holes = ((14, 17, 12, 14),)
    export, manifest, config = world(tmp_path, variant, rows=[{'arrays': bars((30, 30), holes)}])
    roi = load_export(export).ready[0]
    stages, expected = oracle(roi, views(variant))
    assert expected and stages['rejected_window'] > 0 and stages['rejected_valid'] > 0 and stages['rejected_anchor'] > 0 and stages['rejected_q1'] > 0
    code, out, err = run(capsys, manifest, config, tmp_path/'ps.json', *RANDOM_ALL)
    assert code == 0 and err is None
    document = read_set(tmp_path/'ps.json')
    assert centres_of(document) == expected
    funnel = document['header']['cases'][0]['funnel']
    assert {k: funnel[k] for k in stages} == stages and funnel['output'] == len(expected)
    assert funnel['rejected_spacing'] == 0 and funnel['rejected_preflight'] == stages['rejected_preflight']
    # Origin/pitch relative coordinates: the origin is not zero and the pitch (0.3) is not an integer for two variants.
    assert all(c[0] > FRAMES[variant][0][0] and c[1] > FRAMES[variant][0][1] for c in expected)


def test_astra_counterexample_window_outside_frame_is_rejected_by_the_window_rule(tmp_path, capsys):
    vc = ViewConfig(CropConfig((6, 6), (20, 20), 'preserve_exported_layers_v1'), 1.0, 1.0, 'nearest', 'nearest',
                    (1.0, 1.0), (4, 4), *STRINGS, 3)
    ones = {k: np.ones((100, 100), np.uint8) for k in ('l0', 'l1', 'cnt', 'valid')}
    export = make_template(tmp_path, origin=(0.0, 0.0), pitch=1.0, shape=(100, 100), rows=[{'arrays': ones}])
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path)])
    config = write_config(tmp_path, vc)
    roi = load_export(export).ready[0]
    probe = replace(roi, row={**roi.row, 'center_xy_um': [5.0, 5.0]})
    assert preflight(probe, vc)['status'] == 'passed'   # the unchanged preflight accepts the padded window
    code, out, err = run(capsys, manifest, config, tmp_path/'grid.json', '--grid-step-um', '5', mode='grid')
    assert code == 0
    centres = {tuple(c) for c in map(tuple, centres_of(read_set(tmp_path/'grid.json')))}
    assert (5.0, 5.0) not in centres and (10.0, 10.0) not in centres and (15.0, 15.0) in centres
    assert min(c[0] for c in centres) == 15.0 and max(c[0] for c in centres) == 85.0   # 20/2 + 1 = 11 <= x <= 89
    funnel = out['cases'][0]['funnel']
    assert funnel['rejected_window'] > 0 and funnel['rejected_preflight'] == 0


def test_zero_layout_and_empty_core_with_both_layouts_near_are_rejected(tmp_path, capsys):
    empty = bars(); empty['l0'][:] = 0; empty['l1'][:] = 0
    export, manifest, config = world(tmp_path, rows=[{'rid': 'T_zero', 'arrays': empty}])
    code, out, err = run(capsys, manifest, config, tmp_path/'zero.json', '--seed', '1', '--per-case', '1', '--min-spacing-um', '0')
    assert code == 2 and err['error'] == 'patch_shortage'
    funnel = err['cases'][0]['funnel']
    assert funnel['output'] == 0 and funnel['rejected_anchor'] > 0 and funnel['rejected_q1'] == 0 and not (tmp_path/'zero.json').exists()
    # Two single-cell marks 3 um (6 cells) apart: the near window centred on column 13 overlaps columns 10..16 and
    # holds both marks, while the 2 um core (columns 11..15) holds neither.
    arrays = bars(); arrays['l0'][:] = 0; arrays['l1'][:] = 0
    arrays['l0'][15, 10] = 1; arrays['l1'][15, 16] = 1
    (tmp_path/'b').mkdir()
    export2, manifest2, config2 = world(tmp_path/'b', rows=[{'rid': 'T_near', 'arrays': arrays}])
    roi = load_export(export2).ready[0]
    stages, expected = oracle(roi, views('native'))
    assert expected == [] and stages['rejected_anchor'] > 0
    # The candidate whose near window holds both marks exists and passes the window and valid stages.
    near_cols, core_cols = slice(10, 17), slice(11, 16)    # centre (106.75, 207.75) = column 13, row 15
    assert arrays['l0'][15, near_cols].any() and arrays['l1'][15, near_cols].any()
    assert not (arrays['l0'][13:18, core_cols].any() or arrays['l1'][13:18, core_cols].any())
    assert oracle(roi, views('native'), centres=[(106.75, 207.75)])[0]['rejected_anchor'] == 1
    code, out, err = run(capsys, manifest2, config2, tmp_path/'b'/'near.json', '--seed', '1', '--per-case', '1', '--min-spacing-um', '0')
    assert code == 2 and {k: err['cases'][0]['funnel'][k] for k in stages} == stages
    # The same marks 2 um apart share a core: anchor and Q1 both pass for centres between them.
    arrays['l1'][15, 16] = 0; arrays['l1'][15, 14] = 1
    c = tmp_path/'c'; c.mkdir()
    export3, manifest3, config3 = world(c, rows=[{'rid': 'T_ok', 'arrays': arrays}])
    stages3, expected3 = oracle(load_export(export3).ready[0], views('native'))
    assert expected3
    assert run(capsys, manifest3, config3, c/'ok.json', *RANDOM_ALL)[0] == 0 and centres_of(read_set(c/'ok.json')) == expected3


def test_boundary_equality_is_allowed_and_one_step_beyond_is_not(tmp_path, capsys):
    # wide 4 + jitter .25: the first cell centre 2.25 um from the origin sits exactly on the window rule.
    vc = replace(view_config(jitter=(.25, .25)))
    origin, pitch = (100.0, 200.0), .5
    ones = {k: np.ones((30, 30), np.uint8) for k in ('l0', 'l1', 'cnt', 'valid')}
    export = make_template(tmp_path, origin=origin, pitch=pitch, rows=[{'arrays': ones}])
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path)])
    config = write_config(tmp_path, vc)
    assert run(capsys, manifest, config, tmp_path/'t.json', *RANDOM_ALL)[0] == 0
    xs = sorted({c[0] for c in centres_of(read_set(tmp_path/'t.json'))})
    assert xs[0] == 102.25 and xs[-1] == 100.0+15.0-2.25
    config2 = write_config(tmp_path, view_config(jitter=(.26, .26)), 'run2.json')
    assert run(capsys, manifest, config2, tmp_path/'t2.json', *RANDOM_ALL)[0] == 0
    xs2 = sorted({c[0] for c in centres_of(read_set(tmp_path/'t2.json'))})
    assert xs2[0] == 102.75 and xs2[-1] == 100.0+15.0-2.75


def test_generated_patches_pass_preflight_of_the_run_config_and_legacy_padded_windows_stay_accepted(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    assert run(capsys, manifest, config, tmp_path/'ps.json', '--seed', '3', '--per-case', '7', '--min-spacing-um', '1')[0] == 0
    manifest_ps = write_dataset(tmp_path, [source_for(export, tmp_path, patch_sets=['ps.json'])], 'with.json')
    dataset = load_dataset(manifest_ps, expected_views=views('native'))
    assert len(dataset.ready) == 7 and all(preflight(r, views('native'))['status'] == 'passed' for r in dataset.ready)
    # Legacy: a template whose centre sits near the frame edge has a wide window outside the frame; preflight still passes.
    roi = load_export(export).ready[0]
    edge = replace(roi, row={**roi.row, 'center_xy_um': [100.0+1.25, 200.0+1.25]})
    assert preflight(edge, views('native'))['status'] == 'passed'
    _, expected = oracle(roi, views('native'), centres=[(101.25, 201.25)])
    assert expected == []   # ...but the generator never emits such a location


# ---- sampling ------------------------------------------------------------------------------------------------

def test_seed_decides_the_choice_and_bytes_are_deterministic(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    args = ('--per-case', '5', '--min-spacing-um', '0.5')
    outs = {}
    for name, seed in (('a', '1'), ('b', '1'), ('c', '2')):
        assert run(capsys, manifest, config, tmp_path/f'{name}.json', '--seed', seed, *args)[0] == 0
        outs[name] = (tmp_path/f'{name}.json').read_bytes()
    assert outs['a'] == outs['b'] and outs['a'] != outs['c']
    # Whole-file bytes always differ (the seed is in the header); the chosen locations themselves must differ.
    assert centres_of(json.loads(outs['a'])) == centres_of(json.loads(outs['b']))
    assert centres_of(json.loads(outs['a'])) != centres_of(json.loads(outs['c']))
    assert ({p['roi_id'] for p in json.loads(outs['a'])['patches']} != {p['roi_id'] for p in json.loads(outs['c'])['patches']})
    document = read_set(tmp_path/'a.json')
    assert len(document['patches']) == 5 and len({p['roi_id'] for p in document['patches']}) == 5
    assert document['header']['content_sha256'] == patch_set_content_hash(document)
    assert document['header']['views_fingerprint'] == views_fingerprint(views('native'))
    for patch in document['patches']:
        assert patch['roi_id'] == patch_roi_id(patch['template_roi_id'], patch['center_xy_um'], 'random', document['header']['generator_version'])
    # Exact formula, spelled out.
    first = document['patches'][0]
    expected = 'patch_'+hashlib.sha256(json.dumps([first['template_roi_id'], first['center_xy_um'], 'random', 'psdl2_patches_v1'],
                                                   sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    assert first['roi_id'] == expected


def test_min_spacing_is_exact_and_greedy_per_case(tmp_path, capsys):
    export, manifest, config = world(tmp_path, rows=[{'rid': 'T_a'}, {'rid': 'T_b'}])
    roi = load_export(export).ready[0]
    assert run(capsys, manifest, config, tmp_path/'s.json', '--seed', '4', '--per-case', '1000', '--min-spacing-um', '1.5', '--allow-shortage')[0] == 0
    document = read_set(tmp_path/'s.json')
    for case in ('T_a', 'T_b'):
        points = centres_of(document, case)
        assert len(points) > 3
        for i, a in enumerate(points):
            for b in points[i+1:]:
                assert (Fraction(str(a[0]))-Fraction(str(b[0])))**2+(Fraction(str(a[1]))-Fraction(str(b[1])))**2 >= Fraction(3, 2)**2
    # Cases are independent: both cases see the same candidate domain and spacing.
    stats = {c['template_roi_id']: c['funnel'] for c in document['header']['cases']}
    assert stats['T_a']['candidates'] == stats['T_b']['candidates'] == 900
    # Distance exactly equal to the spacing is allowed: two cells apart (1 um) with spacing 1.
    assert run(capsys, manifest, config, tmp_path/'t.json', '--seed', '4', '--per-case', '1000', '--min-spacing-um', '1', '--allow-shortage')[0] == 0
    pts = centres_of(read_set(tmp_path/'t.json'), 'T_a')
    dists = [((a[0]-b[0])**2+(a[1]-b[1])**2)**.5 for i, a in enumerate(pts) for b in pts[i+1:]]
    assert min(dists) == 1.0   # the tie itself occurs (cell centres are exact binary fractions here), so '<' vs '<=' matters


def test_repeat_period_limits_random_candidates_to_the_first_period_cell(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    roi = load_export(export).ready[0]
    assert run(capsys, manifest, config, tmp_path/'r.json', *RANDOM_ALL, '--repeat-period-um', '9', '7')[0] == 0
    points = centres_of(read_set(tmp_path/'r.json'))
    assert points and all(c[0] < 109.0 and c[1] < 207.0 for c in points)
    stages, expected = oracle(roi, views('native'), repeat=(9, 7))
    assert points == expected
    assert read_set(tmp_path/'r.json')['header']['arguments']['repeat_period_um'] == [9.0, 7.0]


# ---- shortage and publication -------------------------------------------------------------------------------

def shortage_cases():
    full = bars()
    lonely = np.zeros((30, 30), np.uint8); lonely[15, 15] = 1   # only the template's own centre cell is valid
    return {
        'too_many': (full, ('--per-case', '100000', '--min-spacing-um', '0')),
        'spacing': (full, ('--per-case', '200', '--min-spacing-um', '4')),
        'repeat': (full, ('--per-case', '100', '--min-spacing-um', '0', '--repeat-period-um', '3')),
        'all_invalid': ({**full, 'valid': lonely}, ('--per-case', '1', '--min-spacing-um', '0')),
        'empty_layout': ({**full, 'l0': np.zeros((30, 30), np.uint8), 'l1': np.zeros((30, 30), np.uint8)},
                         ('--per-case', '1', '--min-spacing-um', '0')),
    }


@pytest.mark.parametrize('name', ['too_many', 'spacing', 'repeat', 'all_invalid', 'empty_layout'])
def test_every_shortage_variant_exits_2_with_exact_funnel_and_no_file(tmp_path, capsys, name):
    arrays, extra = shortage_cases()[name]
    export = make_template(tmp_path, rows=[{'arrays': arrays}])
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path)])
    config = write_config(tmp_path, views('native'))
    out = tmp_path/'out.json'
    sentinel = b'{"old": "bytes"}'
    out.write_bytes(sentinel)
    code, stdout, err = run(capsys, manifest, config, out, '--seed', '1', '--overwrite', *extra)
    assert code == 2 and stdout is None and err['error'] == 'patch_shortage'
    assert out.read_bytes() == sentinel and sorted(p.name for p in tmp_path.iterdir() if p.name.startswith('.')) == []
    funnel = err['cases'][0]['funnel']
    repeat = None
    if '--repeat-period-um' in extra:
        value = float(extra[extra.index('--repeat-period-um')+1]); repeat = (value, value)
    spacing = float(extra[extra.index('--min-spacing-um')+1])
    stages, expected = oracle(load_export(export).ready[0], views('native'), repeat=repeat, spacing=spacing)
    assert {k: funnel[k] for k in stages} == stages and funnel['output'] == len(expected)
    assert err['cases'][0]['status'] == 'shortage' and funnel['output'] < err['cases'][0]['requested']
    if name == 'spacing':
        assert funnel['rejected_spacing'] > 0 and funnel['output'] > 0
    if name == 'all_invalid':
        assert funnel['rejected_valid'] > 0 and funnel['output'] == 0
    if name == 'empty_layout':
        assert funnel['rejected_anchor'] > 0 and funnel['output'] == 0
    if name == 'repeat':
        assert funnel['rejected_repeat_period'] > 0
    # --allow-shortage publishes the same result and records the shortage; exit code 0 (nothing at all is still an error).
    code, stdout, err = run(capsys, manifest, config, out, '--seed', '1', '--overwrite', '--allow-shortage', *extra)
    if funnel['output'] == 0:
        assert code == 2 and err['error'] == 'no_patches_generated' and out.read_bytes() == sentinel
    else:
        assert code == 0 and read_set(out)['header']['cases'][0]['status'] == 'shortage' and stdout['n_shortage_cases'] == 1


def test_existing_output_needs_overwrite_and_retry_gives_the_same_bytes(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    out = tmp_path/'ps.json'
    args = ('--seed', '9', '--per-case', '6', '--min-spacing-um', '1')
    assert run(capsys, manifest, config, out, *args)[0] == 0
    first = out.read_bytes()
    code, _, err = run(capsys, manifest, config, out, *args)
    assert code == 2 and err['error'] == 'output_exists_use_overwrite' and out.read_bytes() == first
    assert run(capsys, manifest, config, out, *args, '--overwrite')[0] == 0 and out.read_bytes() == first
    assert len({p['roi_id'] for p in read_set(out)['patches']}) == 6
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.')] == []


def test_interruption_before_replace_keeps_old_bytes_and_removes_only_its_own_temp(tmp_path, capsys, monkeypatch):
    export, manifest, config = world(tmp_path)
    out = tmp_path/'ps.json'
    other = tmp_path/'.ps.json.foreign.tmp'; other.write_text('someone else')
    args = ('--seed', '9', '--per-case', '6', '--min-spacing-um', '1', '--overwrite')
    assert run(capsys, manifest, config, out, *args)[0] == 0
    old = out.read_bytes()

    def boom(src, dst):
        raise KeyboardInterrupt()
    monkeypatch.setattr(os, 'replace', boom)
    with pytest.raises(KeyboardInterrupt):
        patches_main(['--dataset', str(manifest), '--source', '0', '--config', str(config), '--mode', 'random', '--out', str(out),
                      '--seed', '10', '--per-case', '6', '--min-spacing-um', '1', '--overwrite'])
    monkeypatch.undo()
    capsys.readouterr()
    assert out.read_bytes() == old and other.read_text() == 'someone else'
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith('.')) == ['.ps.json.foreign.tmp']


def test_argument_validation_has_no_defaults_and_mode_specific_arguments(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    cases = [
        (('--per-case', '3', '--min-spacing-um', '1'), 'explicit_seed_required'),
        (('--seed', '1', '--min-spacing-um', '1'), 'explicit_per_case_required'),
        (('--seed', '1', '--per-case', '3'), 'explicit_min_spacing_um_required'),
        (('--seed', '1', '--per-case', '0', '--min-spacing-um', '1'), 'per_case_must_be_positive_integer'),
        (('--seed', '1', '--per-case', '3', '--min-spacing-um', '-1'), 'min_spacing_must_be_nonnegative'),
        (('--seed', '1', '--per-case', '3', '--min-spacing-um', '1', '--grid-step-um', '2'), 'grid_step_not_allowed_with_random'),
    ]
    for extra, message in cases:
        assert run(capsys, manifest, config, tmp_path/'x.json', *extra)[2]['error'] == message
    for extra, message in [((), 'explicit_grid_step_um_required'), (('--grid-step-um', '2', '--seed', '1'), 'random_arguments_not_allowed_with_grid'),
                           (('--grid-step-um', '2', '--min-spacing-um', '1'), 'random_arguments_not_allowed_with_grid'),
                           (('--grid-step-um', '2', '--repeat-period-um', '4'), 'random_arguments_not_allowed_with_grid'),
                           (('--grid-step-um', '0'), 'positive_number_required')]:
        assert run(capsys, manifest, config, tmp_path/'x.json', *extra, mode='grid')[2]['error'] == message
    assert not (tmp_path/'x.json').exists()
    assert run(capsys, manifest, config, tmp_path/'x.json', '--grid-step-um', '2', '--coverage-labels', 'x', mode='random')[2]['error'] in (
        'grid_step_not_allowed_with_random', 'explicit_seed_required')
    stale = write_dataset(tmp_path, [{**source_for(export, tmp_path), 'export_sha256': '0'*64}], 'stale.json')
    assert run(capsys, stale, config, tmp_path/'x.json', '--grid-step-um', '2', mode='grid')[2]['error'] == 'export_sha256_mismatch'


def test_non_manual_ready_rows_are_not_templates(tmp_path, capsys):
    export, manifest, config = world(tmp_path, rows=[{'rid': 'T_manual'}, {'rid': 'T_region', 'kind': 'region'}])
    assert {r.row['roi_id']: r.row['source_kind'] for r in load_export(export).ready} == {
        'T_manual': 'manual', 'T_region': 'electrode_count_region'}
    assert run(capsys, manifest, config, tmp_path/'p.json', '--seed', '1', '--per-case', '3', '--min-spacing-um', '1')[0] == 0
    document = read_set(tmp_path/'p.json')
    assert {p['template_roi_id'] for p in document['patches']} == {'T_manual'}
    assert document['header']['ignored_template_rows'] == [{'roi_id': 'T_region', 'reason': 'not_manual'}]


# ---- grid -------------------------------------------------------------------------------------------------------

def test_grid_is_anchored_at_the_frame_origin_without_spacing_or_repeat(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    roi = load_export(export).ready[0]
    code, out, err = run(capsys, manifest, config, tmp_path/'g.json', '--grid-step-um', '1.5', mode='grid')
    assert code == 0
    document = read_set(tmp_path/'g.json')
    assert document['header']['mode'] == 'grid' and document['header']['arguments']['grid_step_um'] == 1.5
    lattice = [(100.0+i*1.5, 200.0+j*1.5) for j in range(11) for i in range(11)]
    stages, expected = oracle(roi, views('native'), centres=lattice)
    assert expected and centres_of(document) == expected
    assert document['header']['cases'][0]['funnel']['candidates'] == len(lattice) and out['n_patches'] == len(expected)
    assert all(p['mode'] == 'grid' for p in document['patches'])
    # Grid ids are different from random ids at the same centre (mode is part of the id).
    p = document['patches'][0]
    assert p['roi_id'] == patch_roi_id(p['template_roi_id'], p['center_xy_um'], 'grid', 'psdl2_patches_v1')
    assert p['roi_id'] != patch_roi_id(p['template_roi_id'], p['center_xy_um'], 'random', 'psdl2_patches_v1')


def test_grid_coverage_report_gives_nearest_grid_distance_per_label(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    row = load_export(export).ready[0].row
    labels = {'schema_version': 'roi_correspondence_v1', 'pairs': [
        {'pair_id': 'p1', 'status': 'ready', 'a': {'roi_id': 'Q1'}, 'b': {'product_id': row['product_id'], 'frame_id': row['frame_id'],
                                                                           'center_xy_um': [106.0, 207.0], 'roi_id': None}},
        {'pair_id': 'p2', 'status': 'ready', 'a': {'roi_id': 'Q2'}, 'b': {'product_id': 'OTHER', 'frame_id': 'f', 'center_xy_um': [1, 1]}}]}
    (tmp_path/'labels.json').write_text(json.dumps(labels))
    code, out, err = run(capsys, manifest, config, tmp_path/'g.json', '--grid-step-um', '1.5', '--coverage-labels', str(tmp_path/'labels.json'), mode='grid')
    assert code == 0 and len(out['coverage']) == 1
    entry = out['coverage'][0]
    points = centres_of(read_set(tmp_path/'g.json'))
    assert entry['pair_id'] == 'p1' and entry['nearest_grid_distance_um'] == pytest.approx(min(((106-x)**2+(207-y)**2)**.5 for x, y in points))
    assert read_set(tmp_path/'g.json')['header']['content_sha256']


def test_legacy_single_export_training_with_a_window_outside_the_frame_is_unchanged(tmp_path, capsys):
    from test_dataset import make_export
    export = make_export(tmp_path, 'legacy', product='L')
    config = json.loads((ROOT/'configs/synthetic.example.json').read_text())['config']
    vc = ViewConfig.from_dict(config['views'])
    roi = load_export(export).ready[0]
    # The 8x8 frame is only 4 um wide: the 4.6 um jittered wide window around (102, 202) leaves it on every side.
    assert preflight(roi, vc)['status'] == 'passed'
    assert oracle(roi, vc, centres=[roi.center_xy_um])[0]['rejected_window'] == 1
    for name, value in (('config', config), ('lineages', {'L': 'lineage'}), ('split', {'L_0': 'train', 'L_1': 'train'})):
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    from psdl2.train import main as train_main
    assert train_main([str(export), '--config', str(tmp_path/'config.json'), '--out', str(tmp_path/'c.pt'),
                       '--lineages', str(tmp_path/'lineages.json'), '--split', str(tmp_path/'split.json'), '--no-progress']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


# ---- T2-fix ---------------------------------------------------------------------------------------------------

def preflight_only_world(tmp_path):
    # near 3 / core 2 / jitter .5 / nearest over all-ones layers: every window-valid candidate passes valid, anchor and Q1,
    # so the unchanged preflight is the only stage that rejects them (and it rejects all of them).
    vc = ViewConfig(CropConfig((3, 3), (4, 4), 'preserve_exported_layers_v1'), .5, .5, 'nearest', 'nearest', (.5, .5), (2, 2), *STRINGS, 3)
    ones = {k: np.ones((30, 30), np.uint8) for k in ('l0', 'l1', 'cnt', 'valid')}
    export = make_template(tmp_path, rows=[{'arrays': ones}])
    manifest = write_dataset(tmp_path, [source_for(export, tmp_path)])
    return vc, export, manifest, write_config(tmp_path, vc)


def test_preflight_is_the_only_rejecting_stage_for_random_candidates(tmp_path, capsys):
    vc, export, manifest, config = preflight_only_world(tmp_path)
    stages, expected = oracle(load_export(export).ready[0], vc)
    assert stages['rejected_preflight'] == 400 and expected == []
    assert all(stages[k] == 0 for k in ('rejected_valid', 'rejected_anchor', 'rejected_q1'))
    code, out, err = run(capsys, manifest, config, tmp_path/'p.json', *RANDOM_ALL)
    assert code == 2 and err['error'] == 'no_patches_generated' and not (tmp_path/'p.json').exists()
    funnel = err['cases'][0]['funnel']
    assert funnel['rejected_preflight'] == 400 and funnel['output'] == 0
    assert funnel['preflight_reasons'] == {'view_preflight_core_not_represented_at_jitter_extrema': 400}


def test_preflight_is_the_only_rejecting_stage_for_grid_candidates(tmp_path, capsys):
    vc, export, manifest, config = preflight_only_world(tmp_path)
    roi = load_export(export).ready[0]
    lattice = [(100.0+i*.25, 200.0+j*.25) for j in range(61) for i in range(61)]
    stages, expected = oracle(roi, vc, centres=lattice)
    assert stages['rejected_preflight'] > 0 and expected      # cell-edge lattice points survive, cell-centre ones do not
    code, out, err = run(capsys, manifest, config, tmp_path/'g.json', '--grid-step-um', '.25', mode='grid')
    assert code == 0
    document = read_set(tmp_path/'g.json')
    funnel = document['header']['cases'][0]['funnel']
    assert centres_of(document) == expected and funnel['rejected_preflight'] == stages['rejected_preflight'] > 0
    assert all(funnel[k] == 0 for k in ('rejected_valid', 'rejected_anchor', 'rejected_q1'))


def test_repeat_period_is_half_open_at_a_cell_centre(tmp_path, capsys):
    ones = {k: np.ones((30, 30), np.uint8) for k in ('l0', 'l1', 'cnt', 'valid')}
    export, manifest, config = world(tmp_path, rows=[{'arrays': ones}])
    # Columns/rows with centre offsets 2.25 (first window-valid cell) and 2.75: a period of 2.75 excludes 2.75 itself.
    assert run(capsys, manifest, config, tmp_path/'r.json', *RANDOM_ALL, '--repeat-period-um', '2.75')[0] == 0
    document = read_set(tmp_path/'r.json')
    assert centres_of(document) == [(102.25, 202.25)]
    assert document['header']['cases'][0]['funnel']['rejected_repeat_period'] > 0


def test_grid_case_without_output_is_empty_and_reported(tmp_path, capsys):
    empty = bars(); empty['l0'][:] = 0; empty['l1'][:] = 0
    export, manifest, config = world(tmp_path, rows=[{'rid': 'T_a'}, {'rid': 'T_zero', 'arrays': empty}])
    code, out, err = run(capsys, manifest, config, tmp_path/'g.json', '--grid-step-um', '1.5', mode='grid')
    assert code == 0
    status = {c['template_roi_id']: c['status'] for c in read_set(tmp_path/'g.json')['header']['cases']}
    assert status == {'T_a': 'complete', 'T_zero': 'empty'}
    assert {c['template_roi_id']: c['status'] for c in out['cases']} == status


def test_published_file_follows_the_umask(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    previous = os.umask(0o022)
    try:
        assert run(capsys, manifest, config, tmp_path/'m.json', '--seed', '1', '--per-case', '2', '--min-spacing-um', '1')[0] == 0
    finally:
        os.umask(previous)
    assert (tmp_path/'m.json').stat().st_mode & 0o777 == 0o644


def test_random_funnel_stages_sum_to_candidates_with_an_explicit_unexamined_count(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    assert run(capsys, manifest, config, tmp_path/'f.json', '--seed', '1', '--per-case', '3', '--min-spacing-um', '1')[0] == 0
    funnel = read_set(tmp_path/'f.json')['header']['cases'][0]['funnel']
    stages = [k for k in FUNNEL_STAGES if k not in ('candidates', 'output')]
    assert funnel['unexamined'] > 0 and funnel['output'] == 3
    assert funnel['candidates'] == sum(funnel[k] for k in stages)+funnel['output']+funnel['unexamined']
    assert funnel['examined'] == funnel['output']+funnel['rejected_preflight']+funnel['rejected_spacing']
    assert run(capsys, manifest, config, tmp_path/'g.json', '--grid-step-um', '1.5', mode='grid')[0] == 0
    grid = read_set(tmp_path/'g.json')['header']['cases'][0]['funnel']
    assert grid['unexamined'] == 0 and grid['candidates'] == sum(grid[k] for k in stages)+grid['output']


def test_non_object_manifest_is_a_contract_error_not_a_traceback(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    manifest.write_text('[]')
    code, out, err = run(capsys, manifest, config, tmp_path/'x.json', '--seed', '1', '--per-case', '1', '--min-spacing-um', '1')
    assert code == 2 and err == {'error': 'unknown_dataset_schema'}


def test_header_records_numpy_version_inside_the_content_hash_but_not_the_ids(tmp_path, capsys):
    export, manifest, config = world(tmp_path)
    assert run(capsys, manifest, config, tmp_path/'n.json', '--seed', '1', '--per-case', '2', '--min-spacing-um', '1')[0] == 0
    document = read_set(tmp_path/'n.json')
    assert document['header']['numpy_version'] == np.__version__
    other = json.loads(json.dumps(document)); other['header']['numpy_version'] = '0.0.0'
    assert patch_set_content_hash(other) != patch_set_content_hash(document) == document['header']['content_sha256']
    assert all(p['roi_id'] == patch_roi_id(p['template_roi_id'], p['center_xy_um'], 'random', 'psdl2_patches_v1') for p in document['patches'])


def test_zero_total_patches_with_allow_shortage_still_exits_2_and_the_help_says_so(tmp_path, capsys):
    empty = bars(); empty['l0'][:] = 0; empty['l1'][:] = 0
    export, manifest, config = world(tmp_path, rows=[{'arrays': empty}])
    code, out, err = run(capsys, manifest, config, tmp_path/'z.json', *RANDOM_ALL)
    assert code == 2 and err['error'] == 'no_patches_generated' and not (tmp_path/'z.json').exists()
    with pytest.raises(SystemExit):
        patches_main(['--help'])
    assert 'no_patches_generated' in ' '.join(capsys.readouterr().out.split())
