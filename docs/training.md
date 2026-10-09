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
- `topology_policy=common_core_connectivity_v2`; there is no implicit topology policy.
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

`generate_pair` calls T10 crops and records `common_core_connectivity_v2` in the
config, checkpoint/view fingerprint and outputs. Source cells intersecting the
physical common core must be observed in both windows, with identical marked
core content. For each involved layout/CNT/VIA channel, 4-neighbour connectivity
among core cells through the marked cells **inside each view's own window** must
be identical between jitter views at the same scale. Long wires may continue
beyond the window; remote clipped ends or disconnected components outside the
core are not rejection reasons. A bridge outside the core that only one window
retains changes the core relation and is rejected.

Native nearest output is mapped back by its exact source indices so modified or
lost core/bridge pixels are checked. Coarse/area-coverage output additionally must
keep every marked core anchor represented and preserve each native core
connection on the sampled grid. Positive area fractions mean marked presence;
source masks are never thresholded. Native core observation is checked exactly;
coarse cells may have partial valid coverage due to an unknown area **outside**
the core. That alone does not turn observed source core cells into missing data.
Coarse merging of distinct native shapes is deliberately not checked, and these
checks are not evidence of electrical connectivity or real performance.

`pair_acceptance.identical_views` reports near/wide array equality explicitly,
including zero jitter or sub-cell offsets that leave nearest arrays unchanged.
The synthetic config uses a jitter range capable of crossing a native cell;
this is a test example, not a real-run recommendation. Every batch and completed
epoch reports accepted/rejected counts and rejection reasons. Effective batches
below two fail with those counts/reasons in the error text.

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

Only train-assigned rows are optimised. Rejected pairs are omitted and an effective
batch below two fails explicitly. Final singleton batches are merged into the
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
original IDs, analysis layer and full bindings remain visible in the output. It does not guess a pair from XY.
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
