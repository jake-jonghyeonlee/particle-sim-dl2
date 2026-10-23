# Original ROI bundle contract

`psdl2.data.load_export` reads the producer's `review_roi_v1`. Query and database
candidate ROIs use this same interface. It verifies the schema version, a complete
40-character producer commit (optionally pinned by `expected_producer_commit`),
SHA256 source-manifest identifier, composite source identity, frame conventions,
endpoint bindings and every referenced asset's bytes/hash. It checks agreement
of product, design, case and frame with the composite identity, preserving source,
process step, exposure and source-contract version. Hashes identify the declared
sources; no unavailable numeric source manifest is silently reconstructed.

Envelope errors reject the export. Invalid ready rows become blocked with
consumer validation reasons. Producer-blocked rows and excluded records retain
their full metadata/reasons, never load assets and never enter `Export.ready`.
An ID in excluded records cannot be admitted even if its row claims ready.
`build_bundle` requires an admitted ready `ROI`; blocked dictionaries fail.
Unknown ROI fields and non-original asset kinds (including hotspot/CA/Fab) fail.
Map selection metadata stays in the imported row, outside crop channels.

The producer's `assets_root` locates originals. An explicit `assets_root` argument
relocates relative references without changing the export. Relative producer roots resolve
against the export file; an explicit relocation root resolves against the caller’s
working directory. References reject absolute/Windows paths, `..`, backslash
paths, missing files and symlink escapes. Only referenced originals are loaded;
files in the directory are not automatically collected. Bytes are decoded from
the same in-memory data used for SHA256 checking.

Assets are 2D NPY (no pickle), or numeric single-channel PNG/BMP/TIFF matching the exact source frame shape. Palette and colour images are rejected rather than treating palette indices as physical black/white values.
Only exact 0/1 masks or integer 0/255 masks are allowed. No threshold or polarity
inference occurs. `binary` means nonzero is marked;
`binary_zero_is_marked` means zero/black is marked. Other semantics fail.
`observed` must be true. Observed empty masks remain arrays of zeros; a missing
observation blocks the ROI and is never replaced with an empty array.

Each of the two layout channels retains its electrode/role/actual-layer binding.
CNT and every VIA use the same frame. `review_roi_v1` VIA entries are bare asset
references, not connection-endpoint records: bundle metadata preserves them and
case endpoints but explicitly records `exported_masks_only_no_inferred_via_endpoints`.
It does not infer a VIA's physical endpoint coordinates or declare connectivity
complete from a mask alone. An empty exported VIA list is retained verbatim.

## Required crop recipe and intermediate layers

`CropConfig` requires `near_size_xy_um`, `wide_size_xy_um` and
`intermediate_layer_policy`. Sizes are positive (x width, y height); wide must
contain near. No near/wide or policy default exists. Numeric examples are confined
to synthetic fixtures/tests. The policy is explicit even for lateral cases:

- `preserve_exported_layers_v1`: retain all original layouts the producer supplied;
  make no claim that unexported intermediate layers are observed.
- `require_listed_layers_v1`: also require a nonempty explicit `required_layers`
  list, otherwise fail; every listed layer must be observed in exported layouts.

Neither policy removes intermediate pixels, reconstructs simulator flattening,
infers numeric layer order from a name, or invents missing layers. In this producer
version the two layouts bind the case endpoints. A non-adjacent pair therefore
retains its actual endpoint layers; asking for an unexported middle layer fails.
Including additional intermediate layouts needs a future agreed producer contract.
These are explicit caller choices, not a decision resolving the user's open G4
intermediate-layer question. Policy version, layer list and retained layers are
recorded in bundle metadata.

## Coordinates and common sampling

Source and crop coordinates use x/y physical µm, NumPy y/x array order, a top-left
cell-edge origin and half-open bounds. Continuous source pixel coordinates obey
`physical_xy = source_origin_xy_um + pixel_xy * source_pitch_um`. Pixel centres
are `(column + 0.5, row + 0.5)`. `Frame.pixel_to_um` and `um_to_pixel` are inverses;
ROI centre values are physical positions, never raw row/column indices.

Both patches use the exact same supplied physical centre. For each window:
`crop_origin_xy_um = centre_xy_um - size_xy_um / 2`;
`crop_pixel_xy = (physical_xy - crop_origin_xy_um) / output_pitch_um`.
`output_pitch_um` is optional; omission uses the source pitch, without supplying
any window-size default. Each size must be an integer multiple of output pitch;
nonintegral sizes fail rather than silently rounding physical extent. Before
resampling the native size in pixels is recorded as `size_um / source_pitch_um`.
Changing source pitch changes native pixel count, not the requested physical area.

`physical_cell_centres_nearest_exact_v2` samples the source cell containing each output
cell's physical centre (floor of the source coordinate). Exact cell-boundary ties
go to the higher source index. Indices are computed with rational arithmetic
on the declared decimal numbers, not by floating-point floor. The same exact
overlap endpoints determine `physical_cell_area_coverage_exact_v2` weights;
only final coverage weights are converted to floats. The same x/y index vectors apply to every layout,
CNT, VIA and valid channel. No channel gets its own centre or transform. An
unaligned physical centre can require sampling even at unchanged pitch; metadata
marks this as resampled. Nearest-cell sampling preserves binary channel values
but can lose sub-grid structure; it is not an area-conserving measurement.

Outside-frame cells are padded zero in every channel, with valid zero. An
in-frame invalid cell retains source mask values but has valid zero. Thus mask=0
with valid=1 means observed empty, while mask=0 with valid=0 is not evidence of
empty space. Coverage records valid/total/outside-frame/observed-invalid counts
and valid fraction on the sampled grid. Crop geometry, source frame, source assets
and hashes, versions, endpoint/connection evidence and the explicit recipe are
recorded alongside arrays. No quantitative CA may be measured on these crops;
`quantitative_CA_permitted=false` is recorded for every patch. Original numeric
measurement remains the producer's responsibility.

## Offline CLI and limits

`python -m psdl2.data --help` requires no export, assets or network. Validation
prints JSON and exits 0 (all ready), 3 (blocked/excluded records), or 2 (invalid
input). The module imports neither Torch nor producer/DL1 code; dependencies
are installed ahead of time using the pinned local environment snapshot.

T10 tests use an actual producer export plus hand-calculated crop oracles and
socket-blocking fixtures. Model, training, augmentation validity, query/index
search and CA measurement are outside this implementation. Real recipe selection,
real-data acceptance, independent re-review and T12 remain pending; T11 model
and training features are documented separately in training.md.


## Producer revision in the closed network

By default a producer revision is checked for 40-character lowercase hexadecimal
format. That is an identity/format check, not a declaration that every such
revision has been compatibility tested. Use `expected_producer_commit` or
`psdl2.data --expected-producer-commit` to pin the exact approved producer revision;
carry that revision and the original source/asset hashes into the handoff record.
The supplied fixture proves compatibility only for its recorded producer commit.

A copied/installed producer outside Git may export `producer_commit=null` with a
reason. Preferred procedure: retain the reviewed producer code revision when
packaging it and have the producer explicitly record that revision. If it cannot
be recorded, a caller must explicitly use `allow_unversioned_producer=True` or
`--allow-unversioned-producer` (data/train/index/search). Without that choice the
export fails closed. A nonempty reason is required, and outputs/checkpoints
record the null revision, reason and `unversioned_producer_accepted=true`.
This flag does not bypass schema/identity/hash/observation checks or imply a
compatibility PASS. It cannot satisfy an explicit expected commit pin.

Only declared asset kind/binding and bytes/hash can be validated. A file named
`hotspot.npy` falsely declared as an original layout cannot be identified by
filename or binary appearance; correct original-layout provenance remains the
producer's responsibility. Hotspot/CA/Fab fields themselves are rejected.


## V3 view graph and identity contract

Training view policy is explicitly `common_core_connectivity_v3`, with required
`connection_semantics=global_contact_proxy_unverified`,
`comparison_basis=between_views_only`, and positive integer `max_pair_attempts`.
These settings have no package defaults and bind checkpoint/view fingerprints.
See [training.md](training.md) for exact graph edges, anchors, rejection reasons,
recipe/frame preflight, bounded retries and skip counters. Core content is checked
for every non-valid channel. Only the involved endpoint layouts, CNT and individual
VIA channels form the graph. Intermediate layout layers retain their content and
metadata but do not invent additional case endpoints.

Global CNT and endpoint-free VIA can contact any involved layout at the same XY;
no layer-specific endpoint limits can be justified by this export. The graph is
an unverified geometry proxy, not actual net connectivity. Two jittered views are
compared with each other, never with the un-jittered centre graph. Area views must
preserve their own native joint partition, including separation of anchors.
Fractional valid coverage remains fractional and only supplies observed support;
every core source cell itself still requires full observation.

Strict `same_physical_case` includes `analysis_layer`. Search compatibility uses
unordered role/actual-layer endpoints, failure type and CNT/VIA presence and
intentionally does not require the same analysis layer or VIA count. A missing
`producer_commit` key remains equivalent to null: explicit unversioned opt-in,
nonempty reason and any commit-pin checks still apply; no admission rule changed.
Legacy v2-policy checkpoint/index inputs are rejected, with no automatic migration.


## Processed masks and `mask_derivation` (Q8)

`1.png` / `2.png` of a case folder are **case-processed masks**, not raw per-layer layout.
For a vertical case the upper electrode's mask has the layers lying *between* the two endpoint
layers removed (`between_stack_order_v1`: every `layer_stack` layer whose `stack_order` is strictly
between the endpoints); the lower electrode is unchanged. A lateral case removes nothing. CNT is
the union of the DL1 `cno/*.png` of the product, shared by all cases; VIA is not used (empty list;
layers connected through VIA are already one electrode). The export does not mark these masks as
processed, because the producer cannot know it: the fact is a user premise, recorded here and in
`mask_derivation`. Existing exports are never rewritten to carry it.

Two products are comparable only when their masks were derived the same way, so the derivation
travels with the data: dataset file -> `ROI.mask_derivation` (`src/psdl2/data.py:334` parse, `:562-570` attach) ->
`case_signature['mask_derivation']` of each index record -> `search` (`src/psdl2/search.py:140-146`).
An ROI without a derivation is *unknown*, which differs from a *known empty* one (lateral).

## `psdl2_dataset_v1` — several exports as one input

```json
{"schema_version": "psdl2_dataset_v1",
 "sources": [{"export": "prodA/roi/review_rois.json", "export_sha256": "…64 hex…",
              "assets_root": "prodA/data", "patch_sets": ["patches/random_A.json"],
              "mask_derivation": "prodA/roi/mask_derivation.json"}]}
```

Exact keys: top level `schema_version`, `sources`; per source `export` and `export_sha256` required,
`assets_root`, `patch_sets` (list of paths), `mask_derivation` (inline object or path) optional.
Relative paths resolve from the manifest's folder, never the working directory. Each source goes
through the unchanged `load_export` checks; the export bytes must match `export_sha256`.
`load_dataset` is `src/psdl2/data.py:502`. A source with non-empty `patch_sets` admits **only its
patch rows**; its template rows are not admitted (their ids stay reserved for uniqueness).

| Error code | Meaning |
|---|---|
| `unknown_dataset_schema` | not an object / wrong `schema_version` |
| `unsupported_dataset_field` | extra top-level key |
| `dataset_sources_required` | `sources` missing, not a list or empty |
| `unsupported_dataset_source_field` | source lacks `export`/`export_sha256` or has an unknown key |
| `patch_sets_list_required` | `patch_sets` is not a list of strings |
| `patch_set_views_unchecked` | patch sets present but the caller passed no views (`expected_views=None`) |
| `export_sha256_mismatch` | export bytes differ from the manifest |
| `duplicate_roi_id_across_sources: <id>` | an `roi_id` (ready or blocked, template or patch) occurs twice in the dataset |
| `duplicate_patch_set: <path>` | two listed patch-set files with identical bytes (sha256), including the same file listed twice |
| `mask_derivation_case_missing: <roi_id>` | a ready ROI has no derivation entry (only checked when the source has a derivation) |
| `mask_derivation_case_unreferenced: <failure_type> [endpoints]` | an entry matches no ROI (ready or blocked) of the source |
| `cache_and_max_asset_memory_mb_exclusive` | `load_export` given both a shared cache and its own limit (programming error) |
| `asset_memory_limit_exceeded: …` | see the memory note below |

## `psdl2_mask_derivation_v1`

`{"schema_version", "rule_version", "cases": [{"case": {"failure_type", "endpoints": [{electrode_id, role, actual_layer} x2]}, "processed_endpoint", "removed_layers"}]}`
(`parse_mask_derivation`, `src/psdl2/data.py:334`). A case is matched to ROIs by `failure_type` and the
**set** of endpoints (order does not matter). `removed_layers` is stored sorted and unique;
`processed_endpoint` is an `electrode_id` of that case, or `null` for lateral cases (with `removed_layers: []`).
Per ROI the record kept is `{schema_version, rule_version, processed_endpoint: {role, actual_layer} | null, removed_layers}`:
the processed endpoint is compared by `(role, actual_layer)`, not by electrode name, because names differ between products.

| Error code | Meaning |
|---|---|
| `mask_derivation_envelope_invalid` | not an object with exactly the three keys |
| `unknown_mask_derivation_schema` | wrong `schema_version` |
| `mask_derivation_cases_list_required` | `cases` not a list |
| `mask_derivation_entry_invalid` | entry keys are not `case`, `processed_endpoint`, `removed_layers` |
| `mask_derivation_case_invalid` | `case` keys are not `failure_type`, `endpoints` |
| `mask_derivation_failure_type_invalid` | unknown `failure_type` |
| `mask_derivation_two_endpoints_required` | `endpoints` is not a list of two |
| `mask_derivation_duplicate_endpoint` | the two endpoints are identical |
| `mask_derivation_removed_layers_list_required` | `removed_layers` not a list |
| `mask_derivation_removed_layers_duplicate` | repeated layer name |
| `mask_derivation_removed_layer_is_endpoint_layer` | a removed layer is one of the case's endpoint layers |
| `mask_derivation_lateral_must_be_empty` | lateral case with a processed endpoint or removed layers |
| `mask_derivation_processed_endpoint_required` | non-lateral case without a string `processed_endpoint` |
| `mask_derivation_processed_endpoint_invalid` | it does not name exactly one endpoint |
| `mask_derivation_duplicate_case` | two entries with the same failure type and endpoint set |
| `invalid_record_mask_derivation` | an *index record's* derivation is malformed (`src/psdl2/search.py:29`): wrong keys, unsorted/duplicate layers, `processed_endpoint` null for a non-lateral case, or not one of the record's endpoints |

## `psdl2_patch_set_v1` — generated positions

Written by `python -m psdl2.patches` (`src/psdl2/patches.py:325`); it does not copy the export. Layout:
`{"schema_version", "header": {generator_version, numpy_version, mode, template_export: {path, sha256}, arguments, views_fingerprint, views, cases[], ignored_template_rows[], content_sha256}, "patches": [{template_roi_id, center_xy_um, mode, roi_id}]}`.
`roi_id = "patch_" + sha256(canonical JSON [template_roi_id, center_xy_um, mode, generator_version])[:24]`;
`content_sha256` is the hash of the canonical document without that field (generation arguments are covered by it, not by the ids).
`header.cases[]` records per template row the status (`complete`, `shortage`, `empty`) and the funnel
(candidates -> repeat period -> window -> valid -> anchor -> Q1 -> preflight -> spacing -> output; `unexamined` counts candidates a random run did not need to test).
Same inputs, arguments and seed give the same bytes. An expanded row shares the template's decoded channel arrays and differs only in
`center_xy_um`, `roi_id` and `provenance.patch = {set_sha256, template_roi_id, mode, generator_version}`; `set_sha256` is the hash of the patch-set file bytes.

| Error code (`load_dataset`, `src/psdl2/data.py:427-498`) | Meaning |
|---|---|
| `patch_set_missing: <path>` | file cannot be read |
| `patch_set_envelope_invalid` | wrong top-level/header structure |
| `unknown_patch_set_schema` | wrong `schema_version` |
| `stale_patch_set: <path>` | its template sha256 differs from the source's `export_sha256` |
| `patch_record_invalid` | a patch entry has other keys / wrong types |
| `patch_template_not_ready: <id>` / `patch_template_not_manual: <id>` | the template row is not a ready `manual` row |
| `patch_mode_invalid` | mode is not `random` or `grid` |
| `patch_center_not_finite: <id>` | centre is NaN/inf or malformed |
| `patch_id_mismatch: <id>` | the id is not the hash of its fields (tamper) |
| `patch_position_outside_source_frame: <id>` / `patch_position_outside_valid_domain: <id>` | the centre fails the loader's position/valid check (only reachable by forging a re-hashed file) |
| `patch_set_hash_mismatch: <path>` | `content_sha256` differs from the recomputed hash (header/argument tamper) |
| `patch_set_view_config_mismatch: <path>` | views fingerprint differs from the train config / index checkpoint views |
| `duplicate_roi_id_across_sources: <id>` | a patch id collides with another row (same set twice, overlapping sets) |

Generator-side errors (exit 2, JSON on stderr): `patch_shortage` (some case has fewer than N; no file written, per-case funnel in the message),
`no_patches_generated` (zero patches in total, also with `--allow-shortage`: an empty set is useless),
`no_ready_manual_template_rows`, `output_exists_use_overwrite`, `export_sha256_mismatch`, `source_index_out_of_range`,
`source_not_found_or_ambiguous`, `unknown_dataset_schema`, `explicit_<seed|per_case|min_spacing_um>_required`,
`explicit_grid_step_um_required`, `random_arguments_not_allowed_with_grid`, `grid_step_not_allowed_with_random`,
`allow_shortage_not_allowed_with_grid`, `coverage_labels_need_grid_mode`, `seed_must_be_nonnegative_integer`,
`per_case_must_be_positive_integer`, `min_spacing_must_be_nonnegative`, `repeat_period_needs_one_or_two_values`.
The patch set is published atomically (same-folder temp file, fsync, `os.replace`); an interrupted run leaves the old file unchanged.

## Memory limit and generator memory

`--max-asset-memory-mb` counts **retained decoded masks only** (`src/psdl2/data.py:100-118`): one read-only `uint8` array per distinct
(resolved path, hash, value semantics, frame shape). One cache serves every source of a dataset, so the limit applies to the **sum**, and
a file reachable by two paths is counted once. Patch rows reuse their template's arrays, so thousands of patches add nothing.
File bytes, decode buffers, embeddings, the index and the generator's working arrays are not counted.

The patch generator additionally builds integral images (about four `int64` maps) and `(ny, nx)` temporaries per template frame,
roughly 5 x 8 x height x width bytes at peak, which `--max-asset-memory-mb` does not cover. This is fine up to a few thousand cells per side;
for a much larger frame, leave headroom or generate from a coarser template.

## `no_patches_generated`

A generator run that finds no admissible position anywhere is an error, not an empty file: exit 2, JSON `{"error": "no_patches_generated", "cases": [...funnel...]}`
on stderr, nothing written, an existing output untouched. A shortage in only some cases is `patch_shortage` (exit 2) unless `--allow-shortage`
records it in the header and exits 0. Exit code 3 is never used, because the single-file runner treats it as "continue".
