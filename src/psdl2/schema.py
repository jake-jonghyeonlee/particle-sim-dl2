"""Explicit geometry and recipe contracts for original-layout ROI consumers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PureWindowsPath

SCHEMA_VERSION = 'review_roi_v1'
IDENTITY_FIELDS = ('product_id', 'design_version', 'source_id', 'process_step_id',
                   'exposure_id', 'case_id', 'frame_id', 'source_contract_version')


class ContractError(ValueError):
    """Input cannot be used as an observed original-layout candidate."""


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError('finite_number_required')
    if positive and value <= 0:
        raise ContractError('positive_number_required')
    return float(value)


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractError('nonempty_identifier_required')
    return value


def xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ContractError('physical_xy_required')
    return tuple(number(v) for v in value)


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ContractError('unsafe_asset_path')
    p = Path(relative)
    if p.is_absolute() or PureWindowsPath(relative).drive or '..' in p.parts:
        raise ContractError('unsafe_asset_path')
    root = Path(root).resolve()
    resolved = (root / p).resolve()
    try:
        resolved.relative_to(root)  # Path.is_relative_to는 3.9+
    except ValueError:
        raise ContractError('missing_or_escaping_asset_path') from None
    if not resolved.is_file():
        raise ContractError('missing_or_escaping_asset_path')
    return resolved


@dataclass(frozen=True)
class Frame:
    frame_id: str
    origin_xy_um: tuple[float, float]
    pixel_pitch_um: float
    shape_yx: tuple[int, int]

    def __post_init__(self):
        text(self.frame_id)
        object.__setattr__(self, 'origin_xy_um', xy(self.origin_xy_um))
        object.__setattr__(self, 'pixel_pitch_um', number(self.pixel_pitch_um, positive=True))
        if len(self.shape_yx) != 2 or any(type(v) is not int or v <= 0 for v in self.shape_yx):
            raise ContractError('positive_integer_shape_required')
        object.__setattr__(self, 'shape_yx', tuple(self.shape_yx))

    @classmethod
    def from_dict(cls, raw):
        if (raw.get('length_unit'), raw.get('axis_order'), raw.get('origin_convention')) != (
                'um', 'yx', 'top_left_cell_edge'):
            raise ContractError('unsupported_frame_convention')
        return cls(raw['frame_id'], raw['origin_xy_um'], raw['pixel_pitch_um'], raw['shape_yx'])

    def pixel_to_um(self, pixel_xy):
        """Continuous cell-edge coordinates (pixel centres use i + 0.5)."""
        return tuple(o + p * self.pixel_pitch_um for o, p in zip(self.origin_xy_um, xy(pixel_xy)))

    def um_to_pixel(self, physical_xy):
        return tuple((p - o) / self.pixel_pitch_um for o, p in zip(self.origin_xy_um, xy(physical_xy)))


@dataclass(frozen=True)
class CropConfig:
    near_size_xy_um: tuple[float, float]
    wide_size_xy_um: tuple[float, float]
    intermediate_layer_policy: str
    output_pitch_um: float | None = None
    required_layers: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ('near_size_xy_um', 'wide_size_xy_um'):
            size = xy(getattr(self, key))
            if any(v <= 0 for v in size):
                raise ContractError('positive_window_size_required')
            object.__setattr__(self, key, size)
        if any(w < n for n, w in zip(self.near_size_xy_um, self.wide_size_xy_um)):
            raise ContractError('wide_must_contain_near')
        if self.intermediate_layer_policy not in ('preserve_exported_layers_v1', 'require_listed_layers_v1'):
            raise ContractError('explicit_supported_intermediate_layer_policy_required')
        if not isinstance(self.required_layers, (list, tuple)):
            raise ContractError('required_layers_list_required')
        layers = tuple(text(v) for v in self.required_layers)
        if len(set(layers)) != len(layers):
            raise ContractError('duplicate_required_layer')
        object.__setattr__(self, 'required_layers', layers)
        if self.intermediate_layer_policy == 'require_listed_layers_v1' and not layers:
            raise ContractError('explicit_required_layers_required')
        if self.output_pitch_um is not None:
            object.__setattr__(self, 'output_pitch_um', number(self.output_pitch_um, positive=True))

    @classmethod
    def from_dict(cls, raw):
        required = {'near_size_xy_um', 'wide_size_xy_um', 'intermediate_layer_policy'}
        if not required <= raw.keys():
            raise ContractError('window_sizes_and_intermediate_policy_required')
        if set(raw) - required - {'output_pitch_um', 'required_layers'}:
            raise ContractError('unknown_crop_config_field')
        return cls(**raw)
