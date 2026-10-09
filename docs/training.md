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
is the actual configuration. There are no package defaults for windows, pitches,
jitter or intermediate-layer policy. The user must supply:

- `views.crop.near_size_xy_um` and `wide_size_xy_um`, as physical x/y sizes.
- `views.near_pitch_um` and `wide_pitch_um`; sizes must divide exactly by pitch.
- `near_resampling_mode` and `wide_resampling_mode`, each `nearest` or
  `area_coverage`; area coverage requires output pitch at least source pitch.
- `jitter_xy_um`, maximum absolute physical jitter per axis; zero is an explicit
  choice. Two independent uniform offsets are drawn about the original centre.
- `core_size_xy_um`, the common observed core, fitting every possible near view.
- Explicit versioned `intermediate_layer_policy`, plus any `required_layers`.
- Role/layer vocabularies, model dimensions, loss settings, batch size, learning
  rate, epoch count and seed.

The extension to T10 keeps old nearest sampling unchanged. Separate pitches are
keyword arguments of `build_bundle`. The `physical_cell_area_coverage_v1` mode
integrates source-cell overlap divided by output-cell area, with the identical
separable transform applied to each layout/CNT/VIA/valid channel. Thin lines can
therefore be partial coverage instead of disappearing. Outside-frame coverage is
zero. Coverage counts in this mode are fractional output-cell equivalents,
explicitly labelled in metadata. It is a model representation, never a
quantitative CA measurement. Transform versions and both output pitches are
recorded. Area coverage can preserve presence while merging fine topology; it
does not prove connectivity at coarse resolution.

`generate_pair` calls T10 crops for both jittered bundles and validates their
original ROI/source identity, centre and bindings. Every native cell intersecting
the common core must be observed. Every native 4-neighbour component of an
involved layout, CNT or VIA that touches the core must itself be fully observed and fit completely inside all
four patches and remain represented when sampled back at native cell centres.
Same-scale jitter views must also retain identical represented native pixel sets
for every involved component, including components outside the core. Loss of a
bridge/component or clipping rejects the pair. This deliberately
conservative policy can reject long wires extending outside near windows. The
recorded policy is `full_core_touching_native_components_v1`; no missing
observation is invented, and resampled mask presence is not a proof of electrical
connectivity. Real acceptance of view rules remains pending.

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
component, accepted positive count, rejected views, supervised known count, and
`negative_pair_count=0`. No other-location negative term or label-required
fallback exists. Distribution regularisers still couple batch statistics; this
implementation does not claim to remove the false-negative problem.

Optional labels use `roi_relations_v1`: `relations` is a list of
`{a: roi_id, b: roi_id, relation: similar|different|unknown}`.
See the synthetic labels file. Labels are symmetric; conflicting known labels,
self-relations and unknown schema are errors. Unknown pairs are excluded from
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
frames are never registered by numerical coordinate equality.

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

Checkpoints atomically replace one file via a flushed/fsynced temporary file.
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
.venv/bin/python -m psdl2.evaluate results.json --k 10
# Optional: --labels relations.json enables Recall@k and known-different retrieval checks.
```

The cohort file is a nonempty explicit JSON product-ID list from the chosen DL1
cohort. The primary list contains only cohort and case-compatible candidates.
Compatibility preserves failure type, ordered electrode/role/actual-layer binding,
analysis layer and exported CNT/VIA context. It does not guess a pair from XY.
Coarser/finer transformations must share the checkpoint's view fingerprint.
Expansion ranks outside-cohort compatible candidates only in `supplementary`,
without changing primary membership or ordering. Scores are cosine in float64,
rounded to six decimals, then sorted descending with ROI-ID ties ascending.
A physical-case equality flag also requires full source identity, source version,
frame, centre and original asset references. It does not assert spatial
registration across products or causal evidence. No candidates is unavailable,
with explicit reasons; no risk-zero value is generated.

Search result rows carry source identity, frame/case, source assets/versions,
centre, crop geometry, valid coverage, score and scope. Query itself is excluded.
CLI exit codes are 0 for available output, 3 for unavailable output, and 2 for
invalid input. Evaluation accepts a JSON list of search results. It macro-averages
per-query Recall@k against supplied known positives on the primary list only;
supplementary never improves this metric. Labels absent or no known positives
means label metrics are unavailable, with null values. Optional labels can include
known-different retrievals as counterexamples. Metrics describe only the supplied
labels; real performance remains pending.
