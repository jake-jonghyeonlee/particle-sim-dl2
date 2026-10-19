# psdl2

> 한국어 사용 안내: 전체 절차와 입력·출력 형식은 **[docs/USAGE.md](docs/USAGE.md)** 에 정리되어 있습니다.
>
> Windows 폐쇄망 GPU(GTX 1660 Super) 설치·실행·문제 해결은 **[docs/USAGE.md §2.1](docs/USAGE.md#21-gpu-환경-windows-폐쇄망-gtx-1660-super)** 을 따르십시오.

T10 provides offline `review_roi_v1` validation and original multi-layer near/wide
ROI bundles. It preserves physical coordinates, endpoint roles and actual layers,
CNT/VIA masks, valid coverage and source versions. T11 adds a small encoder with
role/layer conditioning, set aggregation and label-free paired-view
training with alignment/variance/covariance regularisation, optional correspondence
labels, safe splits, resumable local checkpoints, cohort search and evaluation.

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
required. Neither sizes nor policies have defaults; low-level source-pitch/nearest
compatibility defaults are specified in docs/training.md and never change the
explicit physical window extent. Load a caller-supplied recipe as follows:

```python
import json
from psdl2.data import load_export
from psdl2.schema import CropConfig
from psdl2.crop import build_bundle

recipe = CropConfig.from_dict(json.load(open("recipe.json")))
export = load_export("review_rois.json", assets_root="original_assets")
bundle = build_bundle(export.ready[0], recipe)
```

T10 example sizes are in tests and the clearly synthetic
[example config](tests/fixtures/synthetic_crop_config.json). Real window sizes and
the intermediate-layer decision remain pending user configuration. See
[the data contract](docs/data_contract.md) for sampling, policy and observation rules.

```sh
.venv/bin/python -m pytest -p no:warnings tests/test_schema.py tests/test_crop.py tests/test_roi_import.py
```

All tests block socket connections. Validation is synthetic; real-data acceptance,
independent candidate review and exchange integration T12 remain pending.


T11 requires explicit real-run windows, both view pitches/resampling modes,
physical jitter/common core, a versioned intermediate-layer policy and explicit
`common_core_connectivity_v3` topology policy, explicit
`connection_semantics=global_contact_proxy_unverified`,
`comparison_basis=between_views_only`, and positive `max_pair_attempts`. Search
requires an explicit DL1 cohort file; optional correspondence labels never block
label-free training. G4, G6 and real performance acceptance remain pending.
Examples under `configs/` are marked synthetic and are not recommendations.

See [training and CLI instructions](docs/training.md) and
[validation evidence](docs/validation.md). These CLIs work offline:

```sh
.venv/bin/python -m psdl2.train --help
.venv/bin/python -m psdl2.index --help
.venv/bin/python -m psdl2.search --help
.venv/bin/python -m psdl2.evaluate --help
.venv/bin/python -m pytest -p no:warnings
```

Default tests include forward/loss/gradient and recovery checks plus one bounded
synthetic optimiser step. The full synthetic train/resume/index/search/evaluate
chain requires `--run-training`; no real-data training or search-quality measurement was run. Search
keeps cohort-only `primary` results separate from optional `supplementary`
results; unavailable candidates never become risk zero. Embedding similarity is
not spatial registration or evidence of cause. Quantitative CA remains outside
DL crops, and the regularisers are not claimed to remove false negatives.


```sh
# Full synthetic execution smoke, without labels or ranking-quality assertions:
.venv/bin/python -m pytest -p no:warnings --run-training
```

Resume may extend the total epoch target while retaining the model/data/view/loss
fingerprint. Changes to semantic configuration remain incompatible.


R4 fixes use exact rational crop boundaries (ties take the higher source cell) and
compare observed core connectivity, so long electrodes and jitter crossing native
cells are supported. Identical near/wide pairs are explicitly marked in metadata.
See [R4 finding status and regression evidence](docs/review/R4_FIXES.md).

Producer revision validation defaults to hexadecimal-format checking; use
`--expected-producer-commit` on validation to pin an approved revision. A null
revision fails unless `--allow-unversioned-producer` is explicitly chosen, and
that acceptance/reason is recorded. The closed-network procedure is in
[the data contract](docs/data_contract.md).

Search case matching ignores electrode names/order and preserves original names
in output; scores are rounded before sorting, so very close values can tie by ID.
Evaluation requires `--split` and `--partition`, excludes other query partitions
and separately reports known positives absent from the index. Low-level defaults,
coarse merge rejection, zero/sub-cell jitter equality and caller-owned lineage
mapping are detailed in [training.md](docs/training.md).


V3 compares joint layout/CNT/VIA anchor partitions between views at each scale;
coarse area views also preserve their own native reference partition, rejecting
both lost connections and spurious merges. The contact graph is an unverified
geometry proxy, never proof of physical connectivity. Recipe/frame preflight
runs before the first optimizer step. Typed view rejections receive bounded
retries; small effective batches skip, while a wholly unusable epoch fails with
counts and preserves the last completed checkpoint. Retry/skip histories resume
exactly. V2-policy checkpoint/index inputs fail closed without migration.

The array graph benchmark and finding-by-finding evidence are in
[R4B_FIXES.md](docs/review/R4B_FIXES.md). G4/G6, actual connection semantics,
real performance and independent candidate review remain pending.
