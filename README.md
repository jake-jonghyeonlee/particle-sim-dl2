# psdl2

T10 provides offline `review_roi_v1` validation and original multi-layer near/wide
ROI bundles. It preserves physical coordinates, endpoint roles and actual layers,
CNT/VIA masks, valid coverage and source versions. It has no model, loss, training
or search implementation yet (T11).

Use Python 3.12. Runtime/test pins match the existing DL1 environment where shared.
Torch is installed for T11; T10 uses NumPy and Pillow only.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m psdl2.data --help
.venv/bin/python -m psdl2.data tests/fixtures/review_roi_v1_export/review_rois.json \
  --assets-root tests/fixtures/review_roi_v1_export \
  --expected-producer-commit f5e86006d72faf16c3bb747051325fbba4b0f9fd
```

Validation prints a JSON summary. Exit 0 means every row is usable, 3 means
blocked/excluded records remain, and 2 means the envelope or input is invalid.
The included producer fixture returns 3: one ready manual row and three blocked
rows. The export is unchanged; `--assets-root` explicitly relocates its copied
references. No runtime import from the producer or DL1 is used.

Both window sizes in µm and an explicit versioned intermediate-layer policy are
required. There are no package defaults. Load a caller-supplied recipe as follows:

```python
import json
from psdl2.data import load_export
from psdl2.schema import CropConfig
from psdl2.crop import build_bundle

recipe = CropConfig.from_dict(json.load(open("recipe.json")))
export = load_export("review_rois.json", assets_root="original_assets")
bundle = build_bundle(export.ready[0], recipe)
```

The only example sizes are in tests and the clearly synthetic
[example config](tests/fixtures/synthetic_crop_config.json). Real window sizes and
the intermediate-layer decision remain pending user configuration. See
[the data contract](docs/data_contract.md) for sampling, policy and observation rules.

```sh
.venv/bin/python -m pytest -p no:warnings tests/test_schema.py tests/test_crop.py tests/test_roi_import.py
```

All tests block socket connections. Validation is synthetic; real-data acceptance,
the independent coordinator review, T11 and exchange integration T12 remain pending.
