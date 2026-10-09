import pytest

from psdl2.schema import ContractError, CropConfig, Frame, safe_path


def test_nonzero_origin_round_trip():
    f = Frame('F', (100, 200), .5, (8, 10))
    assert f.pixel_to_um((4, 2)) == (102, 201)
    assert f.um_to_pixel((102, 201)) == (4, 2)
    assert f.pixel_to_um((4.5, 2.5)) == (102.25, 201.25)
    assert f.um_to_pixel((4, 2)) == (-192, -396)  # an ROI index is not physical XY


@pytest.mark.parametrize('missing', ['near_size_xy_um', 'wide_size_xy_um', 'intermediate_layer_policy'])
def test_required_recipe_has_no_defaults(missing):
    config = {'near_size_xy_um': [2, 2], 'wide_size_xy_um': [4, 4],
              'intermediate_layer_policy': 'preserve_exported_layers_v1'}
    del config[missing]
    with pytest.raises(ContractError, match='required'):
        CropConfig.from_dict(config)


@pytest.mark.parametrize('value', [True, 0, -1, float('nan'), float('inf')])
def test_invalid_pitch(value):
    with pytest.raises(ContractError):
        Frame('F', (100, 200), value, (8, 10))


@pytest.mark.parametrize('policy', [None, '', 'auto', 'remove_intermediate_v1'])
def test_no_inferred_or_removed_intermediate_policy(policy):
    with pytest.raises(ContractError):
        CropConfig((2, 2), (4, 4), policy)


def test_layer_list_required_and_versioned():
    with pytest.raises(ContractError, match='required_layers'):
        CropConfig((2, 2), (4, 4), 'require_listed_layers_v1')
    with pytest.raises(ContractError):
        CropConfig((2, 2), (1, 1), 'preserve_exported_layers_v1')


@pytest.mark.parametrize('path', ['/tmp/a.npy', '../a.npy', 'assets/../../a.npy', 'C:/a.npy', 'a\\b.npy'])
def test_unsafe_paths(tmp_path, path):
    with pytest.raises(ContractError):
        safe_path(tmp_path, path)


def test_symlink_escape(tmp_path):
    inside = tmp_path / 'root'; inside.mkdir()
    outside = tmp_path / 'outside.npy'; outside.write_bytes(b'x')
    (inside / 'link.npy').symlink_to(outside)
    with pytest.raises(ContractError, match='escaping'):
        safe_path(inside, 'link.npy')
