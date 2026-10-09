"""Small local-only shared CNN over an electrode-layer set and two physical scales."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .schema import ContractError


@dataclass(frozen=True)
class ModelConfig:
    roles: tuple[str, ...]
    layers: tuple[str, ...]
    width: int
    feature_dim: int
    projection_dim: int

    def __post_init__(self):
        for name in ('roles', 'layers'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                raise ContractError('explicit_unique_role_layer_vocabulary_required')
            object.__setattr__(self, name, tuple(values))
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 2
               for name in ('width', 'feature_dim', 'projection_dim')):
            raise ContractError('model_dimensions_at_least_two_required')

    def to_dict(self):
        return asdict(self)


def _tensor(mask, device):
    return torch.tensor(np.array(mask, copy=True), dtype=torch.float32, device=device)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ContractError('model_config_required')
        self.config = config
        w = config.width
        self.cnn = nn.Sequential(nn.Conv2d(5, w, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.role = nn.Embedding(len(config.roles), w)
        self.layer = nn.Embedding(len(config.layers), w)
        self.element = nn.Sequential(nn.Linear(6*w, config.feature_dim), nn.GELU())
        self.combine = nn.Linear(2*config.feature_dim, config.feature_dim)
        self.project = nn.Sequential(nn.Linear(config.feature_dim, config.feature_dim), nn.GELU(),
                                     nn.Linear(config.feature_dim, config.projection_dim))

    def _patch(self, patch, bindings):
        device = next(self.parameters()).device
        channels = patch.channels
        if set(channels) - {'cnt', 'valid'} - set(bindings) - {k for k in channels if k.startswith('via:')}:
            raise ContractError('only_original_model_channels_allowed')
        if not bindings or not {'cnt', 'valid'} <= channels.keys():
            raise ContractError('bound_layout_CNT_valid_required')
        tensors = {key: _tensor(mask, device) for key, mask in channels.items()}
        if any(t.shape != tuple(patch.grid.shape_yx) or not torch.isfinite(t).all()
               or torch.any((t < 0) | (t > 1)) for t in tensors.values()):
            raise ContractError('invalid_original_channel')
        valid = tensors['valid']
        if valid.sum() <= 0:
            raise ContractError('all_padding_input')
        via = torch.zeros_like(valid)
        for key in tensors:
            if key.startswith('via:'):
                via = torch.maximum(via, tensors[key])
        records, roles, layers = [], [], []
        for key, binding in bindings.items():
            if key not in tensors:
                raise ContractError('missing_bound_layout')
            try:
                roles.append(self.config.roles.index(binding['role']))
                layers.append(self.config.layers.index(binding['actual_layer']))
            except ValueError as exc:
                raise ContractError('unknown_role_or_actual_layer') from exc
            rest = torch.zeros_like(valid)
            for other in bindings:
                if other != key:
                    rest = torch.maximum(rest, tensors[other])
            records.append(torch.stack([tensors[key]*valid, rest*valid, tensors['cnt']*valid,
                                        via*valid, valid]))
        encoded = self.cnn(torch.stack(records))
        conditioned = torch.cat([encoded, self.role(torch.tensor(roles, device=device)),
                                 self.layer(torch.tensor(layers, device=device))], dim=1)
        return self.element(conditioned).mean(dim=0)

    def forward(self, bundles):
        if not isinstance(bundles, (list, tuple)) or not bundles:
            raise ContractError('nonempty_bundle_batch_required')
        features = []
        for bundle in bundles:
            bindings = bundle.metadata['channel_bindings']
            near = self._patch(bundle.near, bindings)
            wide = self._patch(bundle.wide, bindings)
            features.append(self.combine(torch.cat([near, wide])))
        raw = torch.stack(features)
        return {'features': raw, 'embedding': F.normalize(raw, dim=1), 'projection': self.project(raw)}

    def load_local_weights(self, path):
        path = Path(path)
        if not path.is_file():
            raise ContractError('local_weights_file_required')
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
