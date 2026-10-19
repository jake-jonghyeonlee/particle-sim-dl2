"""Offline review_roi_v1 validation; producer files are the sole exchange interface."""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

from .schema import ContractError, Frame, IDENTITY_FIELDS, SCHEMA_VERSION, safe_path, text, xy


@dataclass(frozen=True)
class ROI:
    row: dict
    frame: Frame
    channels: dict[str, np.ndarray]
    channel_bindings: dict[str, dict]
    versions: dict

    @property
    def center_xy_um(self):
        return tuple(self.row['center_xy_um'])

    @property
    def source_identity(self):
        return tuple(self.row['source_identity'][k] for k in IDENTITY_FIELDS)


@dataclass(frozen=True)
class Export:
    ready: tuple[ROI, ...]
    blocked: tuple[dict, ...]
    excluded_records: tuple[dict, ...]
    versions: dict
    asset_memory: dict | None = None

    def summary(self):
        return {**self.versions, 'n_ready': len(self.ready), 'n_blocked': len(self.blocked),
                'asset_memory': self.asset_memory,
                'ready_roi_ids': [roi.row['roi_id'] for roi in self.ready],
                'blocked': [{key: row[key] for key in ('roi_id', 'status', 'reasons', 'source_identity')}
                            for row in self.blocked], 'excluded_records': list(self.excluded_records)}


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ContractError('sha256_required')
    return value


class AssetMemoryLimit(ContractError):
    """Resource guard over the whole export; never downgraded to a blocked row."""


class AssetCache:
    """Decoded masks shared by every ROI that references the same verified file.

    Keyed by resolved path, declared hash, value semantics and frame shape, so many
    ROIs over one full-frame layer hold one read-only array instead of one each.
    """

    def __init__(self, max_bytes=None):
        self.max_bytes, self.arrays, self.references = max_bytes, {}, 0

    @property
    def decoded_bytes(self):
        return sum(arr.nbytes for arr in self.arrays.values())

    def report(self):
        return {'decoded_assets': len(self.arrays), 'asset_references': self.references,
                'decoded_bytes': self.decoded_bytes, 'max_bytes': self.max_bytes}


def read_asset(root, ref, frame, cache=None):
    # Reference checks run in the same order with or without a cache, so blocked reasons do not change.
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256', 'observed', 'value_semantics'}:
        raise ContractError('unsupported_asset_reference')
    if ref['observed'] is not True:
        raise ContractError('missing_observation')
    if ref['value_semantics'] not in ('binary', 'binary_zero_is_marked'):
        raise ContractError('unsupported_value_semantics')
    path = safe_path(root, ref['path'])
    key = None
    if cache is not None:
        cache.references += 1
        if isinstance(ref['sha256'], str):
            # Only a reference whose path and declared hash were already verified can hit.
            key = (str(path), ref['sha256'], ref['value_semantics'], tuple(frame.shape_yx))
            if key in cache.arrays:
                return cache.arrays[key]
    # Decode the same bytes whose checksum was checked.
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _hash(ref['sha256']):
        raise ContractError('asset_hash_mismatch')
    if cache is not None and cache.max_bytes is not None:
        # Counts the retained uint8 masks only; transient decode buffers are extra. Checked before the
        # format/shape/binary checks, so at the limit boundary an otherwise invalid file rejects the export.
        projected = cache.decoded_bytes + int(np.prod(frame.shape_yx))
        if projected > cache.max_bytes:
            raise AssetMemoryLimit(f'asset_memory_limit_exceeded: decoded {projected} bytes > limit {cache.max_bytes}')
    if path.suffix.lower() == '.npy':
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    elif path.suffix.lower() in ('.png', '.bmp', '.tif', '.tiff'):
        with Image.open(io.BytesIO(data)) as image:
            if image.mode not in ('1', 'L', 'I', 'I;16', 'I;16B', 'I;16L', 'F'):
                raise ContractError('single_channel_numeric_image_required')
            arr = np.array(image)
    else:
        raise ContractError('unsupported_asset_format')
    if arr.ndim != 2 or arr.shape != frame.shape_yx:
        raise ContractError('asset_frame_shape_mismatch')
    if arr.dtype.kind not in 'biuf' or not np.isfinite(arr).all():
        raise ContractError('exact_binary_required')
    unique = set(np.unique(arr).tolist())
    if not unique <= {0, 1} and not (arr.dtype.kind in 'iu' and unique <= {0, 255}):
        raise ContractError('exact_binary_required')
    mask = arr == 0 if ref['value_semantics'] == 'binary_zero_is_marked' else arr != 0
    mask = mask.astype(np.uint8)
    mask.flags.writeable = False
    if key is not None:
        cache.arrays[key] = mask
    return mask


def _endpoint(ep):
    if not isinstance(ep, dict) or set(ep) != {'electrode_id', 'role', 'actual_layer'}:
        raise ContractError('endpoint_binding_required')
    return tuple(text(ep[k]) for k in ('electrode_id', 'role', 'actual_layer'))


def _ready(row, root, versions, cache=None):
    allowed = {'product_id', 'design_version', 'case_id', 'source_identity', 'source_kind',
               'source_region_id', 'selection_policy_version', 'frame', 'frame_id',
               'center_xy_um', 'region_bbox_um', 'endpoints', 'failure_type', 'analysis_layer',
               'source_assets', 'source_version', 'provenance', 'crop_recipe', 'status',
               'reasons', 'roi_id'}
    if set(row) - allowed:
        raise ContractError('unsupported_roi_field_original_input_only')
    if row['reasons']:
        raise ContractError('ready_row_has_blocking_reasons')
    text(row['roi_id'])
    identity = row['source_identity']
    if set(identity) != set(IDENTITY_FIELDS):
        raise ContractError('composite_source_identity_required')
    for key in IDENTITY_FIELDS:
        text(identity[key])
    for key in ('product_id', 'design_version', 'case_id', 'frame_id'):
        if row[key] != identity[key]:
            raise ContractError('composite_source_identity_mismatch')
    frame = Frame.from_dict(row['frame'])
    if frame.frame_id != identity['frame_id']:
        raise ContractError('source_frame_identity_mismatch')
    center = xy(row['center_xy_um'])
    cx, cy = frame.um_to_pixel(center)
    if not (0 <= cx < frame.shape_yx[1] and 0 <= cy < frame.shape_yx[0]):
        raise ContractError('position_outside_source_frame')
    if row['source_kind'] not in ('manual', 'electrode_count_region'):
        raise ContractError('unsupported_roi_source_kind')
    if row['source_kind'] == 'manual' and (row['source_region_id'] is not None or row['region_bbox_um'] is not None):
        raise ContractError('invalid_manual_region')
    if row['source_kind'] == 'electrode_count_region':
        text(row['source_region_id'])
        bbox = row['region_bbox_um']
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ContractError('physical_region_bbox_required')
        from .schema import number
        x0, y0, x1, y1 = (number(v) for v in bbox)
        if x1 <= x0 or y1 <= y0 or center != ((x0 + x1) / 2, (y0 + y1) / 2):
            raise ContractError('region_center_mismatch')
        coverage = row['provenance'].get('coverage', {})
        if coverage.get('source_coverage_status') != 'available':
            raise ContractError('map_coverage_incomplete')
    text(row['selection_policy_version'])
    text(row['source_version'])
    if row['failure_type'] not in ('lateral_short', 'vertical_short'):
        raise ContractError('unsupported_failure_type')
    endpoints = row['endpoints']
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ContractError('two_bound_endpoints_required')
    bindings = [_endpoint(ep) for ep in endpoints]
    if len(set(bindings)) != 2:
        raise ContractError('duplicate_endpoint')
    text(row['analysis_layer'])
    if row['analysis_layer'] not in {ep['actual_layer'] for ep in endpoints}:
        raise ContractError('analysis_layer_binding_mismatch')
    assets = row['source_assets']
    if not isinstance(assets, dict) or set(assets) != {'layouts', 'valid', 'cnt', 'via'}:
        raise ContractError('original_layout_CNT_VIA_valid_only')
    layouts = assets['layouts']
    if not isinstance(layouts, list) or len(layouts) != 2:
        raise ContractError('two_original_layouts_required')
    channels, channel_bindings = {}, {}
    for i, (entry, endpoint) in enumerate(zip(layouts, endpoints)):
        if set(entry) != {'endpoint', 'asset'} or entry['endpoint'] != endpoint:
            raise ContractError('layout_endpoint_binding_mismatch')
        key = f'layout:{i}'
        channels[key] = read_asset(root, entry['asset'], frame, cache)
        channel_bindings[key] = deepcopy(endpoint)
    for key in ('cnt', 'valid'):
        channels[key] = read_asset(root, assets[key], frame, cache)
    if not isinstance(assets['via'], list):
        raise ContractError('via_reference_list_required')
    for i, ref in enumerate(assets['via']):
        channels[f'via:{i}'] = read_asset(root, ref, frame, cache)
    if not channels['valid'][int(cy), int(cx)]:
        raise ContractError('position_outside_valid_domain')
    return ROI(deepcopy(row), frame, channels, channel_bindings, deepcopy(versions))


def load_export(path, *, assets_root=None, expected_producer_commit=None, allow_unversioned_producer=False,
                max_asset_memory_mb=None):
    """Invalid ready rows become blocked; envelope errors reject the export.

    assets_root explicitly relocates the unchanged producer references. Without it,
    use the producer's original root. Asset references must stay inside that root.
    Decoded masks are shared across ROIs; ``max_asset_memory_mb`` rejects the whole
    export before decoding would exceed it.
    """
    if max_asset_memory_mb is not None and (isinstance(max_asset_memory_mb, bool) or not isinstance(max_asset_memory_mb, (int, float))
                                            or not math.isfinite(max_asset_memory_mb) or not 0 < max_asset_memory_mb <= 2**40):
        raise ContractError('max_asset_memory_mb_must_be_positive')
    cache = AssetCache(None if max_asset_memory_mb is None else int(max_asset_memory_mb * 1024 * 1024))
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ContractError('unknown_schema_version')
    commit = payload.get('producer_commit')
    unversioned = commit is None
    if unversioned:
        if allow_unversioned_producer is not True or not isinstance(payload.get('producer_commit_reason'), str) or not payload['producer_commit_reason'].strip():
            raise ContractError('producer_commit_required')
    elif not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ContractError('producer_commit_required')
    if expected_producer_commit is not None and commit != expected_producer_commit:
        raise ContractError('producer_commit_mismatch')
    versions = {'schema_version': SCHEMA_VERSION, 'producer_commit': commit,
                'source_manifest_hash': _hash(payload['source_manifest_hash']),
                'unversioned_producer_accepted': unversioned,
                'producer_commit_reason': payload.get('producer_commit_reason')}
    if payload.get('tensor_input_policy') != 'original_layout_CNT_VIA_valid_only':
        raise ContractError('unsupported_tensor_input_policy')
    if assets_root is not None:
        root = Path(assets_root).resolve()
    else:
        root = Path(text(payload['assets_root']))
        if not root.is_absolute():
            root = path.parent / root
    if not root.is_dir():
        raise ContractError('assets_root_missing')
    rows = payload['rois']
    excluded = payload['excluded_records']
    if not isinstance(rows, list) or not isinstance(excluded, list):
        raise ContractError('roi_and_excluded_lists_required')
    excluded_ids = set()
    for record in excluded:
        if not isinstance(record, dict) or not isinstance(record.get('reasons'), list) or not record['reasons']:
            raise ContractError('excluded_record_reasons_required')
        if any(not isinstance(reason, str) or not reason for reason in record['reasons']):
            raise ContractError('invalid_excluded_reason')
        if record.get('roi_id') is not None:
            excluded_ids.add(text(record['roi_id']))
    seen, ready, blocked = set(), [], []
    for original in rows:
        row = deepcopy(original)
        rid = text(row['roi_id'])
        if rid in seen:
            raise ContractError('duplicate_roi_id')
        seen.add(rid)
        if row.get('status') not in ('ready', 'blocked') or not isinstance(row.get('reasons'), list):
            raise ContractError('roi_status_and_reasons_required')
        if any(not isinstance(reason, str) or not reason for reason in row['reasons']):
            raise ContractError('invalid_blocking_reason')
        if row['status'] == 'blocked':
            if not row['reasons']:
                raise ContractError('blocked_reason_required')
            blocked.append(row)
            continue
        try:
            if rid in excluded_ids:
                raise ContractError('excluded_record_not_candidate')
            ready.append(_ready(row, root, versions, cache))
        except AssetMemoryLimit:
            raise
        except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
            row['status'] = 'blocked'
            row['reasons'].append(str(exc) if isinstance(exc, ContractError) else 'invalid_original_source_record')
            blocked.append(row)
    return Export(tuple(ready), tuple(blocked), tuple(deepcopy(excluded)), versions, cache.report())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate original review_roi_v1 exports offline; print JSON.')
    parser.add_argument('export', type=Path)
    parser.add_argument('--assets-root', type=Path, help='Explicit relocation root for original referenced assets')
    parser.add_argument('--expected-producer-commit', help='Require this exact producer revision')
    parser.add_argument('--allow-unversioned-producer', action='store_true', help='Explicitly accept a null producer commit with a declared reason')
    parser.add_argument('--max-asset-memory-mb', type=float, help='Reject the export if retained decoded masks would exceed this (transient decode buffers are extra)')
    args = parser.parse_args(argv)
    try:
        result = load_export(args.export, assets_root=args.assets_root,
                             expected_producer_commit=args.expected_producer_commit,
                             allow_unversioned_producer=args.allow_unversioned_producer,
                             max_asset_memory_mb=args.max_asset_memory_mb)
        print(json.dumps(result.summary(), allow_nan=False))
        return 3 if result.blocked or result.excluded_records else 0
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
