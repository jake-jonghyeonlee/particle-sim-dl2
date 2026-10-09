# R4 re-review implementation evidence

Baseline: clean `main` at `20a2ca8`. Implemented adopted
`PLAN_v2_ADDENDUM_DL2_VIEWS.md` §§1–8, including both coordinator conditions.
Original R4 and R4_REREVIEW reports are unchanged. This is author evidence,
not an independent PASS. No findings are disputed.

## Findings and proving tests

| Finding | Status | Evidence |
|---|---|---|
| B3 | Fixed under adopted v3 proxy | `test_B3_natural_bridge_crop_exact_reason` and `test_B3_actual_native_patch_bridge_removed`, both CNT and VIA: original core contents asserted equal before exact `core_joint_connectivity_changed` rejection. |
| P1 | Fixed; synthetic bound and growth pass | Array horizontal-run nodes, integer labels, vectorized root hooking/pointer jumping; independent tiny BFS oracle in `test_array_joint_graph_matches_independent_small_breadth_first_oracle`. No per-cell Python tuple/set objects, per-anchor Fraction, or AxA production comparison. Full benchmark below. |
| M1 | Fixed by source-specific preflight | `test_M1_preflight_exact_extrema_and_M2_coarse_nearest`, `test_preflight_failure_before_optimizer_constructed`, `test_frame_outside_core_and_duplicate_location_fail_closed`: core10.2/jitter4.9 fails exact extrema; aligned core10 passes, frame/core-valid errors fail before optimizer. |
| M2 | Fixed by configuration rejection | Same preflight test and `test_coarse_nearest_is_configuration_error`: exact `coarse_nearest_not_supported_for_views` path; area alternative proceeds. Finer-than-source training grids also fail explicitly. |
| M3 | Evidence corrected; objective unchanged | Removed the diagonal `0 == 0` assertion. Retained NumPy arithmetic and view-row permutation tests, added `test_M3_changing_other_location_preserves_own_alignment`: A's alignment stays the hand value .5 when B changes. These, rather than a reported constant, are evidence for location-only alignment. Distribution regularizers still couple batch statistics. |
| M4 | Fixed | `test_M4_analysis_layer_strict_identity_but_search_compatible`: analysis layer alone differs, search includes the candidate while `same_physical_case=False`. View identity also requires analysis layer. |
| M5 | Documented; behavior retained | `test_M5_via_count_not_endpoint_inference_and_M6_missing_commit_opt_in`: VIA counts 1 and 3 compatible. Presence-only compatibility never infers endpoint bindings. |
| M6 | Documented; behavior retained | Same test: missing commit requires explicit opt-in with nonempty reason. Existing null/flag/pin tests retained. |
| M7 | Documented and fixed to adopted meaning | `test_M7_both_views_lose_bridge_accepted_between_views_only`: both near graphs disconnected, both wide connected; accepted. Required `comparison_basis=between_views_only` in config/fingerprint/output. Never compare connectivity with centre window. |
| M8 | Fixed | `test_bounded_retry_success_skip_and_non_view_errors`: attempts=4/retries=2/used=2 success, or generated=1/used=0/skip=1 exhaustion; schema identity errors propagate immediately. `test_retry_skip_training_continues_and_resume_order_bit_exact_without_training`: skipped batch followed by usable batches, complete counter/history and RNG/optimizer equivalence on 4 vs 2+resume. `test_no_usable_epoch_preserves_completed_checkpoint_and_reports_counts`: all-skipped epoch explicit error, last completed file unchanged. |

Additional v3 acceptance evidence in `tests/test_views_v3.py`:

- `test_crossing_layouts_require_contact_not_direct_layout_edge`: layout crossing
  remains two graph components without CNT/VIA; contact joins them. CNT–VIA and
  VIA–VIA do not directly connect. Tiny independent BFS checks the joint algorithm.
- `test_dense_cnt_boundary_band_without_layout_contacts_is_allowed`: entering
  dense exterior CNT never contacts anchors, so accepted; native core equality
  asserted first.
- Existing R4 stripes now cover jitter 0/.1/.3/1/4.9, each 9×20=180 accepted with
  finite `paired_forward` loss. Existing seed-2026 sparse fixture covers
  0/.1/.3/1. The dedicated 2%-expected-density lattice test first proves actual
  four-neighbour isolation, then checks 180 acceptances per jitter.
- `test_coarse_cnt_bridge_deleted_outside_core_exact_reason`: equal core output
  content, sampled bridge deletion changes native-reference anchor relation;
  exact `resampled_core_joint_relation_changed`.
- `test_coarse_native_separation_becomes_spurious_merge_rejected`: source anchors
  separated by empty columns become adjacent coarse support cells; exact same
  relation rejection. `test_fractional_coverage_differences_with_same_joint_partition_allowed`
  accepts different fractional coverage values when joint relations persist.
- Core observation, missing CNT content and empty layout anchors independently
  assert `common_core_not_observed`, `core_marked_content_changed`, and
  `no_core_layout_anchor`; connection errors are not interchangeable with these.
- Policy, contact semantics, comparison basis and attempts are required without
  defaults. Attempts reject bool/noninteger/nonpositive values. Retry changes
  invalidate the fingerprint. V2 policy and legacy checkpoint/index container
  versions reject without migration, including forged top-level index policy.

All connections use `global_contact_proxy_unverified`. Global CNT and endpoint-free
VIA contact every involved layout at coincident XY, without invented layer limits.
This is geometric support, not verified nets, physical continuity or short causes.
Exported intermediate layers have core content checked but are not invented case
endpoints. Each scale compares the two views independently. Each area view must
also preserve its own native joint partition, including anchor separation.

## Replaced assertions and recipes

- Former `test_bridge_removed_rejected_even_same_center` removed a pixel **inside
  the core**, so it proves content loss, not a bridge topology change. Renamed to
  `test_core_content_removed_rejected_even_same_center` and asserts only exact
  `core_marked_content_changed`. New B3 tests remove an exterior CNT/VIA bridge
  and explicitly prove identical core content first.
- Former `test_nearest_resampling_loss_of_native_bridge_rejected` used coarse
  nearest and a broad content/connection regex. Renamed to
  `test_coarse_nearest_is_configuration_error`, now asserting the configuration
  error only. Dedicated coarse area topology tests cover real bridge loss.
- Former `test_B2_coarse_grid_removed_bridge_and_merge_limitation` removed coarse
  **core content**, not an exterior bridge and not a merge. Renamed to
  `test_B2_coarse_core_content_removed` and asserts only exact
  `resampled_core_marked_content_lost`.
- Previous single-layout exterior U-bridge assertions now use the exact joint
  `core_joint_connectivity_changed` / `resampled_core_joint_relation_changed`
  names. Their content remains equal; new cross-channel tests add missing scope.
- The old coarse-merge acceptance is replaced with rejection because v3 preserves
  separation as well as connection. The synthetic run/example and routine
  view helper now use native wide nearest; old coarse example support could merge
  the fixture's diagonal layout/CNT cells. Coarse area coverage remains separately
  tested and benchmarked, without relaxing topology to force acceptance.
- Small effective batch assertions now check bounded retry/skip counters instead
  of immediate interruption. Whole-epoch failure alone is an error. End-to-end
  resume compares full saved epoch histories including per-batch order/counts.
- Existing native line clipping and irrelevant exterior component removal remain
  accepted. Source pitch and exact-v2 crop transforms are unchanged; T10 tests
  continue to cover their original API behavior.

## Synthetic P1 benchmark

Command (no network, no encoder or optimizer execution):

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python tools/benchmark_views_v3.py \
  --out docs/review/R4B_BENCHMARK.json
```

Machine: macOS 26.5.2, arm64; Python/NumPy/Torch versions and thread environment
are in [R4B_BENCHMARK.json](R4B_BENCHMARK.json). One thread, seed2026, two warmups
and five repeats; medians exclude warmups. Every invocation rebuilds two bundles
and validates both scales/views; no cached graph/acceptance. Five channels: two
parallel full-frame layouts, CNT, zero VIA and valid. 1% CNT is four-neighbour
isolated on an even/even lattice; 30% CNT is random, both guarded eleven source
columns around each layout to ensure full successful validation rather than an
early-exit timing. Measured 2000-cell densities are .009914 and .296984. Guarding
changes density slightly, preserves large component/edge workload, and is stated
explicitly; this measures synthetic graph cost, not real CNT contact frequency.

Source2000², near400², core200², wide2000² native or wide500² area:

| CNT | Wide grid | Crop two bundles (s) | Validate (s) | Full pair (s) | Traced peak MiB | Process maxrss MiB |
|---|---|---:|---:|---:|---:|---:|
| 1% isolated | 2000 native | 0.1175 | 0.1231 | 0.2403 | 273.41 | 622.92 |
| 1% isolated | 500 area | 0.1905 | 0.2206 | 0.4122 | 192.15 | 645.72 |
| 30% | 2000 native | 0.1178 | 0.1954 | 0.3134 | 287.94 | 647.05 |
| 30% | 500 area | 0.1904 | 0.2941 | 0.4840 | 206.67 | 651.09 |

All satisfy `validation <= max(2 seconds, 10 × crop)` (bound 2 seconds in these
four rows). 500/1000/2000 growth sweep passes for both densities/modes: largest
axis-doubling validation ratio **3.570**, below **6**. Memory is measured separately
with tracemalloc so timing medians are untraced; process maxrss is cumulative
process peak including imports and prior rows, not per-row incremental memory.
Full sweep values are retained in the JSON artifact. No real-scale acceptance
beyond these specified synthetic fixtures is claimed.

## Validation and pending work

Acceptance commands:

```sh
.venv/bin/python -m pytest -p no:warnings
.venv/bin/python -m pytest -p no:warnings --run-training
```

- Default: **184 passed, 1 skipped in 8.84s**.
- Opt-in `--run-training`: **185 passed in 9.13s**.

These include all existing T10 tests, socket-blocked CLI help/execution,
hand-calculated loss checks and the final v3 output-metadata assertions.
`git diff --check` passes; both existing review reports have zero diff.
Only the previously authorized default one-step smoke updates real optimizer
state. New multi-epoch retry/skip tests use a tiny encoder and no-step optimizer
spy; full real synthetic optimizer/CLI execution remains opt-in `--run-training`.
No real training, performance quality claim, external repository writes, remote,
push or commit. G4/G6, verified connection semantics/producer endpoint contract,
real layout frequencies and quality, CUDA, larger frames and T12 remain pending.
Fresh independent Opus review must inspect this exact candidate; the older FAIL
and partial verification are not reused as a PASS.
