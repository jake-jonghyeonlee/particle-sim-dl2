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

`physical_cell_centres_nearest_v1` samples the source cell containing each output
cell's physical centre (floor of the source coordinate). Exact cell-boundary ties
go to the higher source index. The same x/y index vectors apply to every layout,
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
real-data acceptance, independent review and T11/T12 remain pending.
