# R4 fixes and regression evidence

Baseline: `9849300e625b06e743a68c3bdd6a3e14efc6dfac`, clean tree, the report commit
on top of reviewed `c19a33d`. `R4_REVIEW.md` remains unchanged. This is an author
fix report, not an independent PASS. No findings are disputed.

| Finding | Status | Change / evidence |
|---|---|---|
| B1 | fixed | Rational decimal arithmetic for crop origin, source-cell indices and overlap endpoints; exact-boundary ties go to the higher source index; new nearest/area transform versions and rational origin metadata. `test_B1_review_half_cell_periodic_lines`, `test_B1_origins_and_280_centres_unit_step_sweep`, `test_B1_review_area_fraction_and_frame_edge_oracles`. |
| B2 | fixed | Required `common_core_connectivity_v2` compares observed source core content and the 4-neighbour connectivity equivalence relation through each window. Remote wire ends/components may leave the crop. Coarse output additionally checks existing connection/presence loss; merging is not checked. Stripe and isolated-cell reproductions, native/coarse external-bridge tests, observed-core/error-count and jitter equality tests. |
| N1 | fixed | Null revision requires explicit `allow_unversioned_producer` / `--allow-unversioned-producer`, nonempty reason, and records accepted status in source versions, results and checkpoints. Exact revision pin still rejects null. `test_N1_unversioned_export_requires_explicit_acceptance_and_records_it`. Closed-network procedure in data_contract.md. |
| N2 | documented | Default producer validation is 40-character lowercase hexadecimal format only; a tested revision must be explicitly pinned. A syntactically valid revision is not a compatibility approval. Existing `test_producer_revision_pin` and synthetic exchange revision illustrate this distinction. |
| N3 | fixed | Actual alignment pair indices compute the loss and its cross-location count; optional `supervised_different_pair_count` counts upper-triangular known-different relations separately. `test_N3_known_different_counts_and_numpy_loss_reproduction` independently computes default terms and perturbs row correspondence. |
| N4 | fixed | Compatibility uses unordered endpoint (role, actual_layer) multiset, failure type and exported connection kind. Names/order remain in provenance but no longer restrict retrieval. `test_N4_endpoint_order_and_names_do_not_restrict_case_search`, existing different-role/layer incompatibility and CNT-binding-validation tests. |
| N5 | fixed | Cohort filter precedes compatibility/self rejection. `test_N5_outside_cohort_rejections_are_hidden_without_expansion`. |
| N6 | documented | Float64 cosine is rounded to six decimals **before sorting**, so sub-resolution score differences tie by ROI ID. `test_N6_rounded_scores_tie_by_id`. |
| N7 | documented | Low-level source-pitch/nearest defaults and optional required-layer list are enumerated in training.md; all T11 physical recipe fields/policies remain explicit. These defaults change pixel count, never explicit physical extent. `test_N7_low_level_pitch_defaults_do_not_change_physical_extent`. |
| N8 | documented | Coarse merging of separate shapes is not checked and is not evidence of electrical connectivity. `test_N8_coarse_merging_is_explicitly_not_checked`; coarse bridge-loss tests separately prove actual connection loss is rejected. |
| N9 | documented | Zero jitter is an explicit allowed choice and can make alignment identically zero. Nonzero sub-cell jitter can also yield identical nearest arrays; `pair_acceptance.identical_views` records near/wide equality. `test_B2_nonzero_jitter_crosses_cell_and_identical_metadata`, zero-jitter stripes. |
| N10 | fixed | Evaluation requires split assignments and partition; excluded queries are reported. Known positives absent from the index are separately listed and excluded from Recall@k's denominator. Search output carries complete index IDs; older results may supply `--index`. `test_N10_split_filters_train_queries_and_missing_positives_are_reported`, updated offline/e2e CLI tests. |
| N11 | fixed | Training checks all label IDs against the complete admitted export before batching; missing/blocked IDs are errors. Only valid export IDs outside a particular batch are skipped in that batch. `test_N11_training_rejects_label_id_absent_from_export`. |
| N12 | documented | Training's product-level lineage grouping normally makes same-product overlap checks redundant; generic split API still handles overlaps/augmentations. Correct cross-product lineage mapping belongs to the caller; no automatic shape-based lineage inference is claimed. Existing split leakage/transitive tests. |
| N13 | fixed | File fsync → replace → parent-directory fsync, descriptor closed in finally. `test_N13_directory_is_fsynced_after_replace`; existing atomic-failure test still preserves prior data. No empirical power-loss guarantee or CUDA validation is claimed. |

## B1 reproductions

At origin (100,200), pitch 0.05, 400×400, 4-pixel-period lines and centre
(103.075,210.025), near x indices are **42 through 81**, y **181 through 220**.
Exact half-cell ties select the higher cell. Marked near columns are **2,6,...,38**,
so every gap is 4. The sweep covers three origins × 280 base cell positions ×
cell-edge/half-cell/arbitrary (0.37-cell) offsets, or 2520 crops; expected initial
indices are stated by hand and every native step must be one. Area-overlap
oracles `[0,0,.125,.125]` and valid `[1,1,.5,0]` reproduce the review's independent
checks. The non-square frame/window oracle is also covered explicitly.

The declared decimal values are represented by `Fraction(str(value))`; geometry
and selection decisions use these rationals. Only displayed geometry and final
area weights become floats. This cannot recover digits already lost by an
external caller before declaring its values, but source indices never acquire
nonunit native steps from floating-point floor.

## B2 reproductions and changed assertions

For 200×200, origin (100,200), pitch 0.5, horizontal/vertical 2-pixel lines every
20 pixels, near/wide/core of 20/40/10 µm, all **9 centres × 20 draws = 180 pairs**
are expected accepted at each jitter 0, 0.1 and 1.0. Reason: cutting remote ends
of a straight wire cannot change connectivity among its core cells. Finite
`paired_forward` loss is verified for each jitter. Two-percent isolated-cell
layouts similarly accept **180/180** pairs at jitter 0.3 and 1.0: irrelevant
edge cells entering/leaving a view do not change its observed core relation.

A U-shaped native route outside the core that only one window retains is still
rejected. Coarse output with a bridge cell removed outside the core is also
rejected while marked core anchors remain present. Missing core observation or
marked core content remains an error. Native graphs exclude unobserved cells;
coarse graph cells can have partial valid coverage caused by an unknown area
outside the core, while the native source core itself must be fully observed.
Coarse shape merges remain an explicit limitation.

Two old assertions in test_views.py were replaced:

- A frame-crossing line previously expected `core_component_clipped`; it now
  expects acceptance because its core connectivity remains unchanged.
- An isolated component outside the core disappearing at an edge previously
  expected rejection; it now expects acceptance because it neither changes
  core marked content nor a connection between core cells.

Other bridge tests now assert the precise new core-content/connection-loss
reasons. Transform-version assertions were updated to the exact-v2 versions.
These changes preserve the physical crop examples; they intentionally remove
old tests that defined the rejected B2 behaviour as correct.

The synthetic example uses an explicitly declared 0.3-µm jitter range, capable
of crossing a 0.5-µm native cell. Deterministic ±0.3 offsets yield distinct near
arrays; ±0.1 offsets yield identical near arrays, explicitly recorded. Config
requires the topology policy and its value enters checkpoint/view fingerprints
and output records. Old configs lacking the policy fail rather than silently
selecting it. Batch/epoch histories include counts and reasons; effective batch
errors contain the same counts.

## Other review reproductions

Importer tampering (blocked/excluded-to-ready, identity, origin outside frame,
axis order, uppercase SHA, unobserved CNT, parent traversal, missing tensor policy,
valid polarity) remains fail-closed. Filenames cannot prove original-layout kind;
that producer responsibility is documented. Random 5×3 loss terms are independently
computed with NumPy; permuting view correspondence changes only alignment, and
all-unknown supervision remains zero. No BatchNorm is present. Search ordering
is invariant to index iteration order. The opt-in chain now additionally checks
4-epoch uninterrupted training against 2+resume: model fingerprint, view RNG and
Torch RNG agree exactly.

Remaining: independent R4 re-review of the candidate, real G4/G6 configuration
and correspondence acceptance, real-data/CUDA/large-frame performance validation,
and T12 exchange integration. No real training, network access, cross-repository
changes, push or commit were performed.


## Final validation

- `.venv/bin/python -m pytest -p no:warnings`: **142 passed, 1 skipped in 9.31s**.
- `.venv/bin/python -m pytest -p no:warnings --run-training`: **143 passed in 9.62s**.

The default suite retains only its authorised one-step synthetic smoke; additional
training is behind the explicit option. All tests block sockets. The new sweep
and review reproductions run by default. `git diff --check` passes, and the
original review report has no diff. Changes remain uncommitted.
