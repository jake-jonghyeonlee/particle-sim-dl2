from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from psdl2.data import load_export
from psdl2.model import Encoder, ModelConfig
from psdl2.schema import ContractError
from psdl2.train import paired_forward
from test_losses import loss_config
from test_views import view_config


def model_config():
    return ModelConfig(('gate', 'data'), ('M1', 'M3'), 4, 8, 4)


def test_label_free_view_loss_backward_without_training(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    row = deepcopy(roi.row); row['roi_id'] = 'location_B'
    model = Encoder(model_config())
    losses, rejected = paired_forward(model, [roi, replace(roi, row=row)], view_config(), loss_config(),
                                      np.random.default_rng(1))
    assert not rejected and losses['negative_pair_count'] == 0
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_variable_channel_set_permutation_and_conditioning(source):
    _, write, _ = source; roi = load_export(write()).ready[0]
    bundle = view_config().bundle(roi)
    torch.manual_seed(2); model = Encoder(model_config()).eval()
    baseline = model([bundle])
    metadata = deepcopy(bundle.metadata)
    metadata['channel_bindings'] = dict(reversed(list(metadata['channel_bindings'].items())))
    torch.testing.assert_close(model([replace(bundle, metadata=metadata)])['embedding'], baseline['embedding'])
    metadata['channel_bindings']['layout:0']['role'] = 'data'
    assert not torch.allclose(model([replace(bundle, metadata=metadata)])['features'], baseline['features'])
    # Add a third observed electrode-layer channel; shared encoder needs no fixed set count.
    metadata = deepcopy(bundle.metadata)
    metadata['channel_bindings']['layout:2'] = {'electrode_id': 'C', 'role': 'gate', 'actual_layer': 'M3'}
    near = replace(bundle.near, channels={**bundle.near.channels, 'layout:2': bundle.near.channels['layout:0']})
    wide = replace(bundle.wide, channels={**bundle.wide.channels, 'layout:2': bundle.wide.channels['layout:0']})
    output = model([replace(bundle, metadata=metadata, near=near, wide=wide)])
    assert output['embedding'].shape == (1, 8) and output['projection'].shape == (1, 4)
    assert torch.linalg.vector_norm(output['embedding']).item() == pytest.approx(1)


def test_padding_unknown_vocabulary_local_weights_and_forbidden_channels(source, tmp_path):
    _, write, _ = source; roi = load_export(write()).ready[0]
    bundle = view_config().bundle(roi); model = Encoder(model_config())
    empty = replace(bundle.near, channels={**bundle.near.channels, 'valid': np.zeros(bundle.near.grid.shape_yx)})
    with pytest.raises(ContractError, match='all_padding'): model([replace(bundle, near=empty)])
    bad = replace(bundle.near, channels={**bundle.near.channels, 'hotspot': np.zeros(bundle.near.grid.shape_yx)})
    with pytest.raises(ContractError, match='only_original'): model([replace(bundle, near=bad)])
    metadata = deepcopy(bundle.metadata); metadata['channel_bindings']['layout:0']['actual_layer'] = 'M2'
    with pytest.raises(ContractError, match='unknown_role'): model([replace(bundle, metadata=metadata)])
    torch.save(model.state_dict(), tmp_path/'weights.pt')
    copied = Encoder(model_config()); copied.load_local_weights(tmp_path/'weights.pt')
    torch.testing.assert_close(copied([bundle])['embedding'], model([bundle])['embedding'])
    with pytest.raises(ContractError, match='local_weights'): copied.load_local_weights(tmp_path/'missing')
