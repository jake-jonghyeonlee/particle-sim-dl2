# DL2 training and inference

T11 implements an offline baseline with a small shared 2D CNN, electrode-layer
set aggregation, near/wide feature combination, and a separate projection head.
Each electrode input carries its own layout, other-electrode union, CNT, VIA union
and valid channels. Learned role and actual-layer embeddings condition each set
element. The other-electrode union excludes the current element before union;
overlapping electrodes are retained. Set mean aggregation accepts variable set
sizes. Near/wide use the same encoder, but their physical grids may differ.
Features are normalised for cosine retrieval; loss regularisers use unnormalised
projections. This baseline uses no pretrained downloads. `Encoder.load_local_weights`
accepts only an existing local Torch state-dictionary file, read with
`weights_only=True`. The training CLI starts locally initialised weights unless
resuming a checkpoint.

These are implementation choices for synthetic validation, subject to the
coordinator's independent review and closed-network evaluation. They do not
establish real performance or resolve the user's G4/G6 decisions.

## Explicit configuration

Run configuration is `psdl2_run_v1`; see
[`configs/synthetic.example.json`](../configs/synthetic.example.json). Every number
there is synthetic, including architecture and loss coefficients. The optional
outer `synthetic_example_only`/`note` wrapper is documentation; the `config` object
is the actual configuration. The run-config path requires windows, both pitches, resampling modes, jitter, core
and policies explicitly; the low-level compatibility defaults are listed below. The user must supply:

- `views.crop.near_size_xy_um` and `wide_size_xy_um`, as physical x/y sizes.
- `views.near_pitch_um` and `wide_pitch_um`; sizes must divide exactly by pitch.
- `near_resampling_mode` and `wide_resampling_mode`, each `nearest` or
  `area_coverage`; area coverage requires output pitch at least source pitch.
- `jitter_xy_um`, maximum absolute physical jitter per axis; zero is an explicit
  choice. Two independent uniform offsets are drawn about the original centre.
- `core_size_xy_um`, the common observed core, fitting every possible near view.
- Explicit versioned `intermediate_layer_policy`, plus any `required_layers`.
- `topology_policy=common_core_connectivity_v3`,
  `connection_semantics=global_contact_proxy_unverified`, and
  `comparison_basis=between_views_only`; all three are required.
- Positive integer `max_pair_attempts`, including the first attempt, with no
  package default. The synthetic example's three attempts is not a recommendation.
- Role/layer vocabularies, model dimensions, loss settings, batch size, learning
  rate, epoch count and seed.

Crop geometry uses rational arithmetic on the declared decimal numbers
(`Fraction(str(value))`), including centre − half-window, output-cell centres,
source indices and area-overlap endpoints. `physical_cell_centres_nearest_exact_v2`
uses floor with exact boundary ties selecting the higher source index (half-open
source cells). At native pitch the index step is exactly one, including pitch
0.05. `physical_cell_area_coverage_exact_v2` computes exact source-cell overlaps
before conversion of coverage weights to floating-point arrays. Every channel
shares the index vectors/weights, and exact rational origins are recorded.
Quantitative CA is never measured on these model representations.

`generate_pair` calls T10 crops and records the explicit v3 policy and semantics
in config, checkpoint/view fingerprints and output records. The common core is
fixed about the original centre. Every source cell with positive-area overlap
with that half-open core must be observed and represented in both windows, with
identical marked content for **every non-valid channel**, including CNT and VIA.
Only marked involved-layout cells are graph anchors. An empty anchor set is
`no_core_layout_anchor`, rather than an admissible all-empty positive.

One joint graph uses 4-neighbour marked edges inside each layout/CNT/VIA channel.
Cross-channel edges exist only at coincident XY between an involved layout and a
CNT or VIA channel. There is no direct layout-layout, CNT-VIA or VIA-VIA edge.
No layer-specific CNT/VIA endpoint binding is inferred. This is the explicitly
unverified `global_contact_proxy_unverified` geometry assumption, not electrical
continuity, real nets, physical layer connectivity or evidence of a short.
Verified physical connectivity would need a new producer endpoint contract and
user gate. Actual-layer/role/case identity remains separate context.

The equivalence partition of core layout anchors through the window graph is
compared near-to-near and wide-to-wide. Remote clipping and disconnected exterior
components are allowed when the partition stays unchanged. A CNT/VIA bridge
retained by only one view rejects with `core_joint_connectivity_changed`.
**The un-jittered centre window is not a connectivity reference.** If both near
views lose the same bridge and their near partitions agree, they are accepted;
wide views independently need only agree with each other. This intentional
`comparison_basis=between_views_only` limits what positive-pair acceptance proves.

Native nearest uses actual patch pixels and exact source indices. Coarser views
require `area_coverage`; coarse nearest fails with
`coarse_nearest_not_supported_for_views`. Finer output pitches are unsupported for
these training views. Each area view has its own native reference graph over
source cells whose centres fall in its half-open window. Positive fractional
coverage represents observed support, not full validity. Source core cells must
all be fully observed, even when a coarse cell includes unobserved exterior area.
Every marked non-valid core source cell must remain supported in its output
channel. Missing support rejects with `resampled_core_marked_content_lost`.
The sampled joint anchor partition must equal that view's native partition:
both lost connections and spurious merges reject with
`resampled_core_joint_relation_changed`. Different fractional weights between
views are allowed if content support and partitions remain valid. This neither
preserves quantitative CA nor establishes exact physical topology.

`pair_acceptance.identical_views` records near/wide equality, including zero or
sub-cell jitter. The synthetic recipe now uses native near/wide nearest grids:
the old coarser recipe can create spurious anchor merges, which v3 correctly
rejects. Dedicated synthetic tests and the benchmark still exercise area grids.
No example specifies a real-run pitch or window recommendation.

Training preflight combines each train ROI/frame with the recipe before optimizer
construction. It checks integer output sizes, exact source-index bounds across
the entire jitter range at both scales, and frame/source-valid coverage of the
core. Core10.2/jitter4.9/near20 at source pitch .5 fails the index-bound check;
aligned core10 with the same jitter passes. There is no blanket extra-cell margin.

Only typed, named view rejections consume retries. Schema/identity/policy/file or
recipe errors fail immediately. Retry only the failed ROI in the same RNG stream,
preserve accepted pairs, and never duplicate a location to fill a batch. After
`max_pair_attempts`, omit that ROI. Two or more distinct accepted locations permit
one loss/optimizer step; smaller batches are skipped and later batches proceed.
An epoch without any optimizer steps fails `no_usable_batch_in_epoch` with
statistics and leaves the last completed checkpoint unchanged.

Batch and epoch histories record `attempted_pairs`, `retry_attempts`,
`rejected_attempts_by_reason`, `accepted_pairs_generated`, `accepted_pairs_used`,
`skipped_batches`, and `optimizer_steps`. Generated includes accepted pairs that
were discarded in a skipped batch; used does not. Legacy accepted/rejected counts
are also retained per batch. Completed epoch histories and RNG states are saved,
so continuation restores retry/skip order as well as optimizer state. CLI failure
JSON includes a `statistics` object for unusable epochs. Policy, graph semantics
and retry settings are fingerprinted. V2-policy checkpoints/indexes are rejected
without migration; checkpoint and index container schemas are now v2.

Low-level compatibility defaults: `CropConfig.output_pitch_um=None` uses source
pitch; `required_layers=()` adds no layer requirement for the explicitly chosen
preserve policy. `build_bundle` pitch overrides default to `None` (use the
explicit CropConfig pitch, or source pitch), and its two resampling modes default
to `nearest` for T10 compatibility. The private `_patch` also defaults to nearest.
No size, centre, jitter, intermediate-layer or topology-policy value is supplied
by these defaults. Output cell count follows explicit size divided by pitch;
physical extent is never changed by selecting source pitch. The T11 run-config
path supplies both pitches/modes explicitly and does not use these defaults.

## Objective and optional correspondence labels

Default training requires no labels. For two matched projection matrices `Z1,Z2`,
shape batch × feature dimension, the returned components are:

- Alignment: mean squared difference per location, then mean over locations.
- Variance: average across views/dimensions of
  `relu(variance_floor - sqrt(sample_variance + variance_epsilon))`.
- Covariance: average across views of the squared off-diagonal sample covariance
  sum divided by feature dimension.
- Supervised: zero unless optional known correspondence labels are supplied.

The total uses explicit weights. Sample covariance uses denominator batch−1;
at least two accepted view pairs are required. All three default weights and the
variance floor/epsilon are positive. Epsilon must be smaller than floor squared,
so constant projections incur a positive variance penalty. Logs return each
component, accepted/rejected counts and reasons, supervised known count and
`supervised_different_pair_count`. `alignment_pair_indices` records the actual
location-diagonal edges used to compute alignment; `negative_pair_count` counts
cross-location edges in that computation (zero for the default path). No other-location negative term or label-required
fallback exists. Distribution regularisers still couple batch statistics; this
implementation does not claim to remove the false-negative problem.

Optional labels use `roi_relations_v1`: `relations` is a list of
`{a: roi_id, b: roi_id, relation: similar|different|unknown}`.
See the synthetic labels file. Labels are symmetric; conflicting known labels,
self-relations and unknown schema are errors. Training validates every label ID
against the complete admitted export before batching; absent or blocked ROI IDs
are explicit errors rather than silent skips. Valid IDs outside the current batch
are naturally excluded from that batch only. Unknown pairs are excluded from
both the supervised numerator and denominator. Known pairs use BCE on cosine
logits of averaged view projections divided by the explicit temperature. No
known pairs means a zero optional term; it does not fail default training.
G6's real correspondence format/labels remain pending, so this is a versioned
local exchange proposal, not a claim to have real labels.

## Splits, execution and resume

The original ROI export, its assets, and sidecar files stay separate. Supply a
JSON `product_id -> lineage_id` mapping and a JSON
`roi_id -> train|validation|test` assignment for every admitted ROI. Lineage,
overlapping windows (including maximum jitter) in the same product/design/frame,
and augmentations of one ROI form transitive groups. A group crossing splits is
an error. All products in one lineage must stay together. Separate physical
frames are never registered by numerical coordinate equality. In the training
path, product-level lineage grouping already keeps all same-product ROIs together,
so its overlap guard is normally redundant. The generic split API additionally
checks overlaps/augmentation groups, but correct cross-product lineage mapping
is the caller’s responsibility; lineage is not inferred from shape or filenames.

Install the pinned requirements and the local package as described in README.
The following runs training only when the user explicitly invokes it:

```sh
.venv/bin/python -m psdl2.train review_rois.json --assets-root original_assets \
  --config run.json --lineages lineages.json --split split.json --out checkpoint.pt
# Optional: --labels relations.json
# Resume the exact configuration/data/split/supervision:
.venv/bin/python -m psdl2.train review_rois.json --assets-root original_assets \
  --config run.json --lineages lineages.json --split split.json \
  --out checkpoint.pt --resume checkpoint.pt
```

`--device cpu|cuda` (default `cpu`) selects the training/inference device for
`train`, `index` and `search`. An unavailable CUDA device is a contract error
(exit 2), never a silent CPU fallback. The device is not part of the
configuration fingerprint: a checkpoint written on one device loads and resumes
on the other. CUDA RNG state is stored only by CUDA runs, so a CPU checkpoint
resumes on a CPU-only machine. GPU arithmetic may differ from CPU in the last
digits; bitwise CPU/GPU equality is not claimed. The closed-network GPU venv
(`scripts/setup_gpu_venv.sh`) uses DL1's `torch 2.7.1+cu118`; the suite was run
on torch 2.7.1 (CPU build) and 2.14.1.

Decoded original masks are shared by every ROI that references the same file
(resolved path, declared hash, value semantics and frame shape) and stay
read-only. `--max-asset-memory-mb` (on `data`, `train`, `index`, `search`)
rejects the whole export before the retained decoded masks would exceed the limit;
it is a resource guard, not a blocked row. A reference is hash-verified before it is
counted, so a bad file stays a blocked row. Transient file bytes and decode buffers
are not counted; leave headroom below physical RAM. The limit is checked before
format/shape/binary decoding checks, so a hash-valid but otherwise invalid file
that arrives exactly at the limit rejects the export instead of blocking its row. Validation summaries report
`asset_memory`. Training prints DL1-format JSON progress lines on stderr
(`--no-progress` disables them); stdout keeps the single result JSON, and on
failure the error JSON is the last stderr line.

Only train-assigned rows are optimised. Rejected pairs are omitted and an effective
batch below two is skipped after bounded retries; only an epoch with no usable batch fails. Final singleton batches are merged into the
previous batch. No real training was run while implementing this package, and
the default suite has one explicitly bounded synthetic optimiser step. Multi-epoch
training and the complete CLI chain run only with `pytest --run-training`.

Checkpoints atomically replace one file via a flushed/fsynced temporary file,
then fsync the parent directory. This improves durability but is not an empirical
power-loss guarantee on every filesystem; CUDA and power-loss tests remain pending.
They store model, optimiser, completed epoch, run configuration, data/source hashes,
lineages/split, optional labels, Torch/NumPy/Python and view-generator RNG states.
The epoch target is stored separately from the semantic configuration fingerprint,
so a compatible resume may increase the total epoch budget. The requested target
must exceed the saved completed epoch; a no-op or backwards resume fails explicitly.
Changes to model/data/views/loss/learning-rate/split/labels still reject resume.
Resume rejects mismatched semantic configuration/schema; CUDA RNG state additionally
requires CUDA availability. Resume is at completed epoch boundaries; an
interrupted epoch replays from the last completed checkpoint. Exact weights have
a separate fingerprint; indexes bind both config and weight fingerprints, so an
index from another weight state is rejected even when architecture is unchanged.

## Index, cohort search and evaluation

```sh
.venv/bin/python -m psdl2.index candidates/review_rois.json \
  --assets-root candidates/original_assets --checkpoint checkpoint.pt --out index.json
.venv/bin/python -m psdl2.search query/review_rois.json \
  --assets-root query/original_assets --roi-id USER_ROI_ID \
  --checkpoint checkpoint.pt --index index.json --cohort dl1_cohort.json \
  --topk 10 --out result.json
# Optional: --expand adds a separate supplementary list.
.venv/bin/python -m psdl2.evaluate results.json --k 10 --split split.json --partition test
# Optional: --labels relations.json enables Recall@k and known-different retrieval checks.
```

The cohort file is a nonempty explicit JSON product-ID list from the chosen DL1
cohort. The primary list contains only cohort and case-compatible candidates.
Compatibility compares failure type, the unordered multiset of endpoint
(role, actual_layer), and connection kind (CNT, with VIA when exported).
Electrode names and endpoint order do not restrict cross-product matching;
original IDs, analysis layer and full bindings remain visible in the output.
`same_physical_case` additionally requires the same analysis layer along with
source/frame/centre/assets and full electrode identities. A different analysis
layer can remain search-compatible without being the same physical case. VIA
compatibility records presence, not VIA count or inferred endpoint binding.
It does not guess a pair from XY.
Coarser/finer transformations must share the checkpoint's view fingerprint.
Expansion ranks outside-cohort compatible candidates only in `supplementary`,
without changing primary membership or ordering. Scores are cosine in float64,
rounded to six decimals **before** sorting descending with ROI-ID ties ascending.
Differences below the rounding resolution can therefore become ID-ordered ties.
Cohort filtering precedes compatibility; out-of-cohort rejections appear only
when expansion is explicitly requested.
A physical-case equality flag also requires full source identity, source version,
frame, centre and original asset references. It does not assert spatial
registration across products or causal evidence. No candidates is unavailable,
with explicit reasons; no risk-zero value is generated.

Search result rows carry source identity, frame/case, source assets/versions,
centre, crop geometry, valid coverage, score and scope. Query itself is excluded.
CLI exit codes are 0 for available output, 3 for unavailable output, and 2 for
invalid input. Evaluation requires a split file and explicit partition and reports
excluded queries. It accepts a JSON list of search results and macro-averages
per-query Recall@k against known positives present in the index, on the primary list only;
supplementary never improves this metric. Labels absent or no known positives
means label metrics are unavailable, with null values. Positives absent from the
index are listed separately and excluded from the recall denominator; this may
raise Recall@k relative to all-labelled-positive recall, so missing-positive
coverage must be reviewed alongside the metric. Search records index ROI IDs;
older result files need `--index` to supply that set. Optional labels can include
known-different retrievals as counterexamples. Metrics describe only the supplied
labels; real performance remains pending.

## Patch generator conditions versus the unchanged preflight

`python -m psdl2.patches` chooses training/index positions by shape. Its location conditions are **generator-only**;
`views.preflight` (`src/psdl2/views.py:103-130`, called from `src/psdl2/train.py:89-90`) is unchanged and stays what it was.

| | Generator (`src/psdl2/patches.py:76-110`) | Preflight (`views.py:103-130`) |
|---|---|---|
| Whole window inside the frame | required: centre +/- (wide/2 + jitter) within the frame, equality with the edge allowed | **not checked**; cells outside the frame are contract padding (`data_contract.md`) |
| Core fully valid | required | not checked here (core cell representation only) |
| Layout anchor in core | required: a marked cell of either endpoint layout in the core | **not checked**; the anchor is tested when a view is built (`views.py:234-237`, `no_core_layout_anchor`) |
| Both endpoints in the near window (Q1) | required: each endpoint layout has a marked cell in the unjittered near window | not checked |
| Core cells representable at jitter extrema | by running `preflight` as the last condition | yes (`view_preflight_core_not_represented_at_jitter_extrema`, pitch rules) |

The generator conditions do not guarantee that every jitter pair passes topology; training keeps its bounded retries and rejection counters.
Existing single-export ROIs whose windows leave the frame still train as before (`tests/test_patches.py::test_legacy_single_export_training_with_a_window_outside_the_frame_is_unchanged`).
The grid/random candidate sets are exact (rational arithmetic on the stored floats), and a funnel per case is printed on shortage
(`candidates, rejected_repeat_period, rejected_window, rejected_valid, rejected_anchor, rejected_q1, rejected_preflight, rejected_spacing, output`).
`--repeat-period-um` and `--min-spacing-um` apply only to random patches; grid takes every admissible point and its step should be
at most the hit tolerance of the evaluation (`--coverage-labels` reports the actual nearest-grid distance per label).
A patch set records the views fingerprint it was generated for; `train`, `index`, `search` and `make_split` refuse it when their views differ
(`patch_set_view_config_mismatch`, `src/psdl2/data.py:497-498`).

## Dataset training and `make_split`

`train --dataset M`, `index --dataset M` and `search --query-dataset M` read a `psdl2_dataset_v1` manifest (`data_contract.md`).
Exactly one of the export argument and `--dataset` is accepted (`exactly_one_of_export_or_dataset_required`,
`src/psdl2/train.py:186`, `src/psdl2/index.py:114`, `exactly_one_of_export_or_query_dataset_required`, `src/psdl2/search.py:204`);
`--assets-root` is rejected with a dataset. In dataset mode only, the checkpoint fingerprint contains `dataset` (normalised manifest, export
hashes, patch-set file hashes, `mask_derivation`); a changed dataset rejects `--resume` (`incompatible_config_fingerprint`) and leaves the old
checkpoint untouched. Single-export runs produce byte-identical outputs to the baseline commit (`tests/test_dataset_cli.py`).
An index built from a dataset has `sources[]` (export path, sha256, versions, patch sets) instead of `producer_versions`, and each record
may carry `provenance.patch`.

`python -m psdl2.make_split --dataset M --config C --lineages L --assign A --out S [--overwrite]` (`src/psdl2/make_split.py:41`) maps
every ready `roi_id` through product -> lineage (`L`) -> split (`A`: lineage -> `train|validation|test`) and validates the result with
`validate_split` (so leakage is an error). Errors (exit 2, no output): `product_without_lineage: <product>`, `invalid_lineage_id: <product>`,
`lineage_unassigned: <lineage>`, `invalid_split_name: <lineage>`, `lineages_and_assign_objects_required`, `no_ready_rois_to_split`,
`output_exists_use_overwrite`, patch-set/dataset errors from `load_dataset`. The file is published atomically. All patches of one product share its
lineage, so they land in one split. `split.groups` was rewritten (hash unions for lineage/augmentation, per-frame sweep for overlap,
`src/psdl2/split.py:53`) with unchanged semantics, checked against a verbatim copy of the old implementation in `tests/test_split_groups_diff.py`;
20,000 patches take well under a second.

## `mask_derivation` search compatibility

The derivation of a query is not guessed from candidates: it comes from the query dataset manifest. `search` compares it with each candidate's
(`src/psdl2/search.py:140-146`) after the unchanged case-signature check:

| query | candidate | result |
|---|---|---|
| none | none | unchanged behaviour (single-export indexes) |
| present | present, equal (rule_version, processed endpoint `(role, actual_layer)`, removed layers) | compatible |
| present | present, different in any of those | rejected `mask_derivation_mismatch` |
| present or none | none or present (only one side has it) | rejected `mask_derivation_unknown` |
| any | any, with `--ignore-mask-derivation` | not compared; the result gets `"mask_derivation_comparison": "ignored"` |

Rejections appear in `rejected` of the result. If every other cohort candidate is excluded this way, `no_match_reasons` is
`no_mask_derivation_compatible_candidates` and the exit code is 3. `--ignore-mask-derivation` exists only on `search`; it is for
diagnosis (for example to see what would match across derivations), not a default.
