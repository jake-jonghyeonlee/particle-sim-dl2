"""Offline patch-location generator over one template export (psdl2_patch_set_v1).

A patch set lists physical centres on the frame of a template export row; it never copies masks.
The generator-only location conditions below are stricter than ``views.preflight`` (which stays
unchanged and is still run on every kept location with the real run config):

1. whole-window: centre +- (window/2 + jitter) lies inside the frame (equality allowed);
2. the whole core window is valid;
3. the core holds at least one marked cell of an involved layout (anchor);
4. each endpoint layout has a marked cell in the unjittered near window (original grid, no threshold).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .crop import rational
from .data import PATCH_SET_SCHEMA, _join, load_export, patch_roi_id, patch_set_content_hash, views_fingerprint
from .schema import ContractError, number
from .views import ViewConfig, preflight

GENERATOR_VERSION = 'psdl2_patches_v1'
FUNNEL_STAGES = ('candidates', 'rejected_repeat_period', 'rejected_window', 'rejected_valid', 'rejected_anchor',
                 'rejected_q1', 'rejected_preflight', 'rejected_spacing', 'output')


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _exact(value):
    """The rational the rest of the code base sees for a stored float."""
    return rational(float(value))


def _axis(centres, origin, pitch, count, wide, near, core, jitter):
    """Per-axis exact tests for every centre: window ok, core index range, near index range."""
    n = len(centres)
    window_ok = np.zeros(n, bool); core_ok = np.zeros(n, bool)
    core_lo = np.zeros(n, np.int64); core_hi = np.zeros(n, np.int64)
    near_lo = np.zeros(n, np.int64); near_hi = np.zeros(n, np.int64)
    low, high = origin, origin+count*pitch
    for k, c in enumerate(centres):
        window_ok[k] = all(c-(w/2+jitter) >= low and c+(w/2+jitter) <= high for w in (wide, near))
        left, right = c-core/2, c+core/2
        first, stop = int((left-origin)//pitch), _ceil((right-origin)/pitch)
        if left >= low and right <= high and first < stop:
            core_ok[k], core_lo[k], core_hi[k] = True, first, stop
        left, right = c-near/2, c+near/2
        first, stop = max(0, int((left-origin)//pitch)), min(count, _ceil((right-origin)/pitch))
        if first < stop:
            near_lo[k], near_hi[k] = first, stop
    return window_ok, core_ok, core_lo, core_hi, near_lo, near_hi


def _integral(mask):
    table = np.zeros((mask.shape[0]+1, mask.shape[1]+1), np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return table


def _rect(table, y0, y1, x0, x1):
    """Sum over rows [y0, y1) x columns [x0, x1); y* shape (ny, 1), x* shape (1, nx)."""
    return table[y1, x1]-table[y0, x1]-table[y1, x0]+table[y0, x0]


def _screen(roi, config, xs, ys, repeat):
    """Candidate grid (len(ys) x len(xs)) -> alive mask and sequential rejection counts."""
    frame = roi.frame
    o_x, o_y = (rational(v) for v in frame.origin_xy_um)
    p = rational(frame.pixel_pitch_um)
    h, w = frame.shape_yx
    funnel = {'candidates': len(xs)*len(ys)}
    alive = np.ones((len(ys), len(xs)), bool)

    def take(name, keep):
        nonlocal alive
        new = alive & keep
        funnel[name] = int(alive.sum()-new.sum()); alive = new

    if repeat is None:
        funnel['rejected_repeat_period'] = 0
    else:
        inside_x = np.array([c-o_x < repeat[0] for c in xs]); inside_y = np.array([c-o_y < repeat[1] for c in ys])
        take('rejected_repeat_period', inside_y[:, None] & inside_x[None, :])
    sizes = [tuple(rational(v) for v in getattr(config.crop, name)) for name in ('wide_size_xy_um', 'near_size_xy_um')]
    core = tuple(rational(v) for v in config.core_size_xy_um)
    jitter = tuple(rational(v) for v in config.jitter_xy_um)
    ax = _axis(xs, o_x, p, w, sizes[0][0], sizes[1][0], core[0], jitter[0])
    ay = _axis(ys, o_y, p, h, sizes[0][1], sizes[1][1], core[1], jitter[1])
    take('rejected_window', (ay[0] & ay[1])[:, None] & (ax[0] & ax[1])[None, :])
    # Out-of-range indices of rejected axes are clipped; those cells are already dead.
    cy0, cy1 = (np.clip(a, 0, h)[:, None] for a in (ay[2], ay[3])); cx0, cx1 = (np.clip(a, 0, w)[None, :] for a in (ax[2], ax[3]))
    area = (cy1-cy0)*(cx1-cx0)
    take('rejected_valid', _rect(_integral(roi.channels['valid'] == 1), cy0, cy1, cx0, cx1) == area)
    layouts = [roi.channels['layout:0'] > 0, roi.channels['layout:1'] > 0]
    take('rejected_anchor', _rect(_integral(layouts[0] | layouts[1]), cy0, cy1, cx0, cx1) > 0)
    ny0, ny1 = ay[4][:, None], ay[5][:, None]; nx0, nx1 = ax[4][None, :], ax[5][None, :]
    take('rejected_q1', (_rect(_integral(layouts[0]), ny0, ny1, nx0, nx1) > 0) &
         (_rect(_integral(layouts[1]), ny0, ny1, nx0, nx1) > 0))
    return alive, funnel


def _preflight_ok(roi, config, center):
    probe = replace(roi, row={**roi.row, 'center_xy_um': [center[0], center[1]]})
    try:
        preflight(probe, config)
    except ContractError as exc:
        return str(exc)
    return None


def _cell_axis(origin, pitch, count):
    return [float(origin+(i+Fraction(1, 2))*pitch) for i in range(count)]


def _case_random(roi, config, args):
    frame = roi.frame
    o, p = (tuple(rational(v) for v in frame.origin_xy_um), rational(frame.pixel_pitch_um))
    fx, fy = _cell_axis(o[0], p, frame.shape_yx[1]), _cell_axis(o[1], p, frame.shape_yx[0])
    repeat = None if args['repeat_period_um'] is None else tuple(rational(v) for v in args['repeat_period_um'])
    alive, funnel = _screen(roi, config, [_exact(v) for v in fx], [_exact(v) for v in fy], repeat)
    survivors = np.flatnonzero(alive.ravel())
    seq = [int(args['seed']), int(hashlib.sha256(roi.row['roi_id'].encode()).hexdigest()[:8], 16)]
    order = survivors[np.random.default_rng(seq).permutation(len(survivors))]
    # Spacing is exact on the lattice: distance^2 = p^2 (di^2 + dj^2) < spacing^2.
    limit = (rational(args['min_spacing_um'])/p)**2
    reach = max(1, _ceil(rational(args['min_spacing_um'])/p))
    buckets, chosen, examined = {}, [], 0
    reasons, rejected_preflight, rejected_spacing = {}, 0, 0
    nx = len(fx)
    for flat in order:
        if len(chosen) == args['per_case']:
            break
        examined += 1
        j, i = divmod(int(flat), nx)
        reason = _preflight_ok(roi, config, (fx[i], fy[j]))
        if reason is not None:
            rejected_preflight += 1; reasons[reason] = reasons.get(reason, 0)+1
            continue
        near = False
        for bi in range(i//reach-1, i//reach+2):
            for bj in range(j//reach-1, j//reach+2):
                for (ci, cj) in buckets.get((bi, bj), ()):
                    if (i-ci)**2+(j-cj)**2 < limit:
                        near = True
        if near:
            rejected_spacing += 1
            continue
        buckets.setdefault((i//reach, j//reach), []).append((i, j)); chosen.append((fx[i], fy[j]))
    funnel.update(rejected_preflight=rejected_preflight, rejected_spacing=rejected_spacing, output=len(chosen),
                  examined=examined, unexamined=len(survivors)-examined, preflight_reasons=dict(sorted(reasons.items())))
    return sorted(chosen, key=lambda c: (c[1], c[0])), funnel


def _case_grid(roi, config, args):
    frame = roi.frame
    o = tuple(rational(v) for v in frame.origin_xy_um)
    p, step = rational(frame.pixel_pitch_um), rational(args['grid_step_um'])
    points = []
    for axis, count in ((0, frame.shape_yx[1]), (1, frame.shape_yx[0])):
        extent = count*p; k = int(extent//step)
        points.append([float(o[axis]+i*step) for i in range(k+1)])
    fx, fy = points
    alive, funnel = _screen(roi, config, [_exact(v) for v in fx], [_exact(v) for v in fy], None)
    chosen, reasons, rejected = [], {}, 0
    for j, i in zip(*np.nonzero(alive)):
        reason = _preflight_ok(roi, config, (fx[i], fy[j]))
        if reason is None:
            chosen.append((fx[i], fy[j]))
        else:
            rejected += 1; reasons[reason] = reasons.get(reason, 0)+1
    funnel.update(rejected_preflight=rejected, rejected_spacing=0, output=len(chosen), examined=int(alive.sum()),
                  unexamined=0, preflight_reasons=dict(sorted(reasons.items())))
    return chosen, funnel


def generate(rois, config, args):
    """Return (header cases, patches) for the ready manual template rows; pure and deterministic."""
    cases, patches = [], []
    for roi in sorted(rois, key=lambda r: r.row['roi_id']):
        centres, funnel = (_case_random if args['mode'] == 'random' else _case_grid)(roi, config, args)
        status = 'complete'
        if args['mode'] == 'random' and len(centres) < args['per_case']:
            status = 'shortage'
        elif not centres:
            status = 'empty'
        cases.append({'template_roi_id': roi.row['roi_id'],
                      'case': {'failure_type': roi.row['failure_type'], 'endpoints': roi.row['endpoints']},
                      'frame_id': roi.row['frame_id'], 'product_id': roi.row['product_id'],
                      'status': status, 'requested': args['per_case'], 'funnel': funnel})
        for cx, cy in centres:
            patches.append({'template_roi_id': roi.row['roi_id'], 'center_xy_um': [cx, cy], 'mode': args['mode'],
                            'roi_id': patch_roi_id(roi.row['roi_id'], (cx, cy), args['mode'], GENERATOR_VERSION)})
    return cases, patches


def build_document(source, export_sha256, config_views, args, rois, ignored):
    cases, patches = generate(rois, config_views, args)
    header = {'generator_version': GENERATOR_VERSION, 'numpy_version': np.__version__, 'mode': args['mode'],
              'template_export': {'path': source['export'], 'sha256': export_sha256},
              'arguments': args, 'views_fingerprint': views_fingerprint(config_views), 'views': asdict(config_views),
              'cases': cases, 'ignored_template_rows': ignored}
    document = {'schema_version': PATCH_SET_SCHEMA, 'header': header, 'patches': patches}
    header['content_sha256'] = patch_set_content_hash(document)
    return document


def serialize(document):
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def publish(path, data, overwrite):
    """Same-folder temp file -> fsync -> os.replace. Only this call's own temp file is ever removed."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ContractError('output_exists_use_overwrite')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        mask = os.umask(0); os.umask(mask)
        os.chmod(temporary, 0o666 & ~mask)   # mkstemp creates 0600; published sets follow the umask like any other file
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    try:
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass  # directory fsync is unavailable on some platforms; the file itself is already durable


def _select_source(manifest_path, selector):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ContractError('unknown_dataset_schema')
    sources = manifest.get('sources')
    if manifest.get('schema_version') != 'psdl2_dataset_v1' or not isinstance(sources, list) or not sources:
        raise ContractError('unknown_dataset_schema')
    base = Path(manifest_path).parent
    if selector.isdigit():
        if int(selector) >= len(sources):
            raise ContractError('source_index_out_of_range')
        return base, sources[int(selector)]
    wanted = Path(selector).resolve()
    matches = [s for s in sources if selector == s.get('export') or _join(base, s['export']).resolve() == wanted]
    if len(matches) != 1:
        raise ContractError('source_not_found_or_ambiguous')
    return base, matches[0]


def _arguments(ns):
    random_only = {'per_case': ns.per_case, 'min_spacing_um': ns.min_spacing_um, 'repeat_period_um': ns.repeat_period_um,
                   'seed': ns.seed}
    if ns.mode == 'random':
        if ns.grid_step_um is not None:
            raise ContractError('grid_step_not_allowed_with_random')
        for name in ('per_case', 'min_spacing_um', 'seed'):
            if random_only[name] is None:
                raise ContractError('explicit_%s_required' % name)
        if type(ns.seed) is not int or ns.seed < 0:
            raise ContractError('seed_must_be_nonnegative_integer')
        if type(ns.per_case) is not int or ns.per_case < 1:
            raise ContractError('per_case_must_be_positive_integer')
        spacing = number(ns.min_spacing_um)
        if spacing < 0:
            raise ContractError('min_spacing_must_be_nonnegative')
        repeat = None
        if ns.repeat_period_um is not None:
            if len(ns.repeat_period_um) not in (1, 2):
                raise ContractError('repeat_period_needs_one_or_two_values')
            repeat = [number(v, positive=True) for v in ns.repeat_period_um]
            repeat = repeat*2 if len(repeat) == 1 else repeat
        return {'mode': 'random', 'seed': ns.seed, 'per_case': ns.per_case, 'min_spacing_um': spacing,
                'repeat_period_um': repeat, 'grid_step_um': None, 'allow_shortage': bool(ns.allow_shortage)}
    if any(v is not None for v in random_only.values()):
        raise ContractError('random_arguments_not_allowed_with_grid')
    if ns.grid_step_um is None:
        raise ContractError('explicit_grid_step_um_required')
    if ns.allow_shortage:
        raise ContractError('allow_shortage_not_allowed_with_grid')
    return {'mode': 'grid', 'seed': None, 'per_case': None, 'min_spacing_um': None, 'repeat_period_um': None,
            'grid_step_um': number(ns.grid_step_um, positive=True), 'allow_shortage': False}


def coverage_report(path, rois, document):
    """Distance from every labelled position to the nearest emitted grid point of its product/frame."""
    from .eval_position import read_correspondence
    frames = {(r.row['product_id'], r.row['frame_id']): r.row['roi_id'] for r in rois}
    by_template = {}
    for patch in document['patches']:
        by_template.setdefault(patch['template_roi_id'], []).append(patch['center_xy_um'])
    rows = []
    for query, truths in sorted(read_correspondence(path).items()):
        for truth in truths:
            template = frames.get((truth['product_id'], truth['frame_id']))
            if template is None:
                continue
            points = by_template.get(template, [])
            nearest = min((((truth['center_xy_um'][0]-x)**2+(truth['center_xy_um'][1]-y)**2)**.5 for x, y in points), default=None)
            rows.append({'query_roi_id': query, 'pair_id': truth['pair_id'], 'product_id': truth['product_id'],
                         'frame_id': truth['frame_id'], 'center_xy_um': truth['center_xy_um'],
                         'nearest_grid_distance_um': nearest})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate an offline psdl2_patch_set_v1 for one template export; print JSON.')
    parser.add_argument('--dataset', required=True, type=Path, help='psdl2_dataset_v1 manifest naming the template source')
    parser.add_argument('--source', required=True, help='source index in the manifest, its export entry or the export path')
    parser.add_argument('--config', required=True, type=Path, help='run config; only its views are used')
    parser.add_argument('--mode', required=True, choices=('random', 'grid'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int); parser.add_argument('--per-case', type=int)
    parser.add_argument('--min-spacing-um', type=float); parser.add_argument('--repeat-period-um', type=float, nargs='+')
    parser.add_argument('--grid-step-um', type=float)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-shortage', action='store_true', help='write a patch set with short cases (recorded in the header); a set with zero patches in total is still '
                        'refused (exit 2, no_patches_generated)')
    parser.add_argument('--coverage-labels', type=Path, help='roi_correspondence_v1; grid mode reports nearest-grid distances')
    parser.add_argument('--allow-unversioned-producer', action='store_true')
    parser.add_argument('--max-asset-memory-mb', type=float)
    ns = parser.parse_args(argv)
    try:
        args = _arguments(ns)
        if ns.coverage_labels is not None and args['mode'] != 'grid':
            raise ContractError('coverage_labels_need_grid_mode')
        if ns.out.exists() and not ns.overwrite:
            raise ContractError('output_exists_use_overwrite')
        config_doc = json.loads(ns.config.read_text())
        views = ViewConfig.from_dict(config_doc.get('config', config_doc)['views'])
        base, source = _select_source(ns.dataset, ns.source)
        export_path = _join(base, source['export'])
        digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
        if digest != source.get('export_sha256'):
            raise ContractError('export_sha256_mismatch')
        loaded = load_export(export_path, assets_root=None if source.get('assets_root') is None else _join(base, source['assets_root']),
                             allow_unversioned_producer=ns.allow_unversioned_producer,
                             max_asset_memory_mb=ns.max_asset_memory_mb)
        templates = [r for r in loaded.ready if r.row['source_kind'] == 'manual']
        ignored = [{'roi_id': r.row['roi_id'], 'reason': 'not_manual'} for r in loaded.ready if r.row['source_kind'] != 'manual']
        if not templates:
            raise ContractError('no_ready_manual_template_rows')
        document = build_document(source, digest, views, args, templates, ignored)
        cases = document['header']['cases']
        short = [c for c in cases if c['status'] == 'shortage']
        report = {'cases': [{k: c[k] for k in ('template_roi_id', 'status', 'requested', 'funnel')} for c in cases]}
        if short and not args['allow_shortage']:
            print(json.dumps({'error': 'patch_shortage', **report}, allow_nan=False), file=sys.stderr)
            return 2
        if not document['patches']:
            print(json.dumps({'error': 'no_patches_generated', **report}, allow_nan=False), file=sys.stderr)
            return 2
        data = serialize(document)
        publish(ns.out, data, ns.overwrite)
        report.update(status='written', out=str(ns.out), mode=args['mode'], n_patches=len(document['patches']),
                      n_shortage_cases=len(short), content_sha256=document['header']['content_sha256'],
                      file_sha256=hashlib.sha256(data).hexdigest())
        if ns.coverage_labels is not None:
            report['coverage'] = coverage_report(ns.coverage_labels, templates, document)
        print(json.dumps(report, allow_nan=False)); return 0
    except (ContractError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
