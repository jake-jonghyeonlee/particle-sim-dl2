# Validation evidence and pending acceptance

Baseline: accepted T10 `836ee0f2a20a9b386373faf9de898d7ce0f8fb44`.
Dependencies are unchanged: Python 3.12, NumPy/Pillow/Torch/pytest pins in
`requirements.lock`. No producer or DL1 module imports, pretrained downloads,
network calls, remote creation or push occur. All tests use the existing
socket-blocking autouse fixture. The original producer export fixture remains
unchanged.

```sh
.venv/bin/python -m pytest -p no:warnings
```

The original T11 suite passed 104 tests without optimisation. The follow-up adds
one default single-step checkpoint smoke and an opt-in multi-epoch CLI chain;
all 64 T10 tests remain included. The suite covers:

- Hand-calculated near/wide area coverage of a one-native-cell thin line and a
  valid hole, plus recorded transform versions and padding coverage.
- Same-location jitter with common transforms, missing configuration errors,
  observed core checks, core connectivity equality and removed bridges at the
  same physical centre, including bridges outside the core. Remote clipping is
  allowed when it leaves core content/connectivity unchanged.
- Label-free views → encoder → finite loss → finite gradients, without training
  or labels; variable set sizes and order invariance, role/layer conditioning,
  invalid/all-padding channels and local-file weights.
- Explicit hand values for alignment/variance/covariance/supervised terms,
  collapse penalty, insufficient effective batches, no cross-location negative
  term, and unknown-logit invariance of the optional supervised term.
- Fixed embedding cosine/ranking/ties, cohort-only primary scope, a higher-scored
  supplementary candidate, incompatible role/layer/CNT context, unavailable
  no-candidate output, and hand-counted primary-only Recall@k.
- Transitive lineage, overlapping-window and augmentation groups; split leakage
  rejection and complete assignment requirements.
- Exact checkpoint embedding/optimiser/RNG round trip, configuration/schema and
  exact-weight fingerprints, and atomic-write failure preserving previous data.
  These recovery tests install synthetic optimiser states without steps; the
  separate bounded end-to-end smoke executes exactly one step by default.
- Socket-blocked module help and nontraining checkpoint → index → single-query
  search → optional-label evaluation, including stale transform rejection.

The CNN, conditioning/aggregation, objective coefficients, versioned common-core
connectivity policy and synthetic label schema are reviewable baseline choices.
Independent Opus review is required for this candidate; these tests are not an
independent PASS or real-data performance validation.

The user must choose real near/wide physical sizes, both output pitches and
resampling modes, jitter range/common core, intermediate-layer policy and any
required layers, cohort product list, lineage/split files, role/layer vocabulary,
training configuration and optional correspondence labels. G4 remains partly
pending; G6 real label/acceptance data remain pending. Area fractions are DL
representations and must never be used to measure quantitative CA.

Remaining work: independent candidate review, real recipe/view acceptance,
closed-network training and performance comparison, and T12 producer-consumer
results/handoff integration. Similarity is not registration or evidence of cause.
The default label-free path has no cross-location negative pair term; the effect
of batch distribution regularisation still needs evaluation, so no false-negative
or collapse-prevention performance claim is made from synthetic tests alone.


## Synthetic execution follow-up

```sh
.venv/bin/python -m pytest -p no:warnings
.venv/bin/python -m pytest -p no:warnings --run-training
```

`tests/test_end_to_end.py` writes six ready `review_roi_v1` rows with correct asset
hashes across three product lineages and two distinct hand-written structures.
Every stage reads the actual exchange through `load_export`. The default smoke
calls `train()` for one epoch/batch/optimiser step, observes atomic file replacement
and loads the written checkpoint. The opt-in test runs actual CLI `main(argv)`
entry points: two label-free epochs, resume to four total epochs, index, cohort
primary and supplementary search, and evaluation without labels. It asserts
finite histories, completed epochs and optimiser step counts, record fields,
fingerprint rejection, and unavailable label metrics. It asserts no ranking quality.
The socket-blocking fixture applies to the entire chain. The test records its
wall time as `wall_time_seconds` and prints it when capture is disabled.

One defect was exposed: increasing the epoch target from two to four previously
changed the full run fingerprint and made resume fail with
`incompatible_config_fingerprint`. The training compatibility fingerprint now
excludes only the epoch budget; checkpoints store `target_epochs` separately.
A resumed run continues from the saved epoch, while semantic changes (tested
with a changed learning rate) still reject without overwriting the checkpoint.
A target at or below the completed epoch is an explicit error instead of a
misleading successful no-op. Real training/performance and independent review
remain pending; this evidence is synthetic execution only.


Follow-up execution results:

- `.venv/bin/python -m pytest -p no:warnings`: **105 passed, 1 skipped in 1.22s**.
- `.venv/bin/python -m pytest -p no:warnings --run-training`: **106 passed in 1.25s**.
- Focused opt-in chain with capture disabled: **0.558s wall time**, including
  fixture setup and all CLI calls inside the test; no ranking quality is asserted.

The skipped default item states that multi-epoch synthetic training requires
explicit `--run-training`. The always-run smoke performs exactly one optimiser
step, as explicitly authorised by this follow-up task.


R4 changed crop/view acceptance and the operational notes; current status,
regression cases and final commands are recorded in [R4_FIXES.md](review/R4_FIXES.md).
The original R4 FAIL report is preserved unchanged.


R4-fix candidate validation: default **142 passed, 1 skipped in 9.31s**;
`--run-training` **143 passed in 9.62s**. Exact commands and per-finding
regressions are in the fix report linked above. Independent re-review remains
pending; these author-run checks do not replace it.


## Adopted v3 addendum / R4 re-review fixes

Baseline `20a2ca8`; complete finding statuses, replaced assertions, independent
hand-stated oracles and benchmark measurements are in
[R4B_FIXES.md](review/R4B_FIXES.md). Both original review reports stay unchanged.
V3 uses one joint layout/CNT/VIA contact graph at each scale, source-specific
preflight, mandatory bounded retries, counted batch skipping and completed-epoch
resume. The graph is explicitly `global_contact_proxy_unverified`, with
`comparison_basis=between_views_only`; neither observed similarity nor graph
acceptance establishes physical connectivity or real performance. Coarse area
views reject both lost anchor connections and spurious anchor merges. Coarse
nearest is rejected before training; v2-policy checkpoints/indexes are rejected
without migration.

The benchmark runs two warmups and five measured repeats with one CPU thread,
no graph-result cache, source2000² / near400² / core200² / native-wide2000² or
area-wide500², two layout channels + CNT + VIA + valid, seed2026. Both 1%-isolated
and 30%-random CNT fixtures pass the `max(2s,10×crop)` validation bound and the
500/1000/2000 growth bound. Validation medians are 0.1231/0.2206s (1% native/area)
and 0.1954/0.2941s (30% native/area); crop medians 0.1175/0.1905/0.1178/0.1904s;
full pair medians 0.2403/0.4122/0.3134/0.4840s. Largest growth ratio 3.570×,
traced peaks 192.15–287.94 MiB, cumulative process maxrss up to 651.09 MiB.
The fixture guards CNT away from layout contacts to time full successful
validation; actual density, memory meaning, platform, versions and full sweep
are recorded in [R4B_BENCHMARK.json](review/R4B_BENCHMARK.json). This measures
synthetic cost, not real topology frequency or training/search quality.

Default tests still perform only the authorized one real synthetic optimizer
step. New multi-epoch retry/skip recovery tests use a tiny encoder and no-step
optimizer spy; the actual synthetic optimizer/CLI chain remains opt-in. Tests
prove discarded accepted pairs are counted as generated but unused, schema and
identity failures are never retried, wholly unusable epochs keep the last
checkpoint, and retry/skip order plus all RNG/optimizer state reproduce on resume.
No real training was performed. User G4/G6 choices, producer endpoint semantics,
real closed-network validation, CUDA, larger frames, T12 and fresh independent
review remain pending.


Final v3 candidate commands: `.venv/bin/python -m pytest -p no:warnings`
→ **184 passed, 1 skipped in 8.84s**; the same command with `--run-training`
→ **185 passed in 9.13s**. All tests block sockets. `git diff --check` passes.
These checks are author evidence; fresh independent review remains pending.

## Position evaluation and template-matching baseline (PLAN_v3 T6, 2026-10-18)

`psdl2.eval_position` reads `roi_correspondence_v1` files written by
particle-sim-compare (side `a.roi_id` = query ROI, side `b` = labelled position in
the candidate product) and scores either `psdl2.search` outputs or
`psdl2.baseline_ncc` outputs with one metric: a query is a hit when a returned
candidate centre lies within `--hit-tolerance-um` of a labelled centre in the same
product and frame (top-1 hit rate, hit rate@k, top-1 distance). Queries without a
label or without candidates are `unavailable`, never misses. `psdl2.baseline_ncc`
is the non-learned reference: the query window (`--window-um`, no default) is cut
from the role/layer layout union of the query product and matched by FFT ZNCC
against the same role/layer union of the candidate product; endpoint maps are
summed and peaks become predicted centres. A structureless template is reported,
not guessed. Synthetic evidence: an exact translation is recovered with zero error
and score 1.0 (`tests/test_position_baseline.py`). Both tools read JSON only and
import no producer module. Real tolerance, window, labels and the DL2-versus-
baseline decision remain the user's in the closed network; tens of labels are a
decision aid, not a statistical claim.

### Batch mode and periodicity (PLAN_v4 T2, 2026-10-19)

`psdl2.baseline_ncc --all-ready` uses every ready ROI of the query export as a query
(no labels). Candidate FFTs are built once per role/layer and template shape and
shared; queries with identical template masks (SHA256 in `template_sha256`) are
matched once and listed in each other's `shared_with`. Per result `flags` report
`low_structure`, `window_clipped`, `score_gap` (rank 1 − rank 2) and, only when
`--ambiguity-margin` is given, `near_equal_peaks`. Scores order and label the review
sheet (`--sheet-dir`); nothing is thresholded, dropped or accepted by score.
`--self-check` matches the export against itself (and, like `--all-ready`, requires `--max-memory-mb`)
and exits 2 unless, for every available query, the accumulated score map the matcher really computed (read while it
is resident, independent of rank, top-k and suppression) is ~1 (>= 0.999) at the query's own window position, that position equals the one the
checker derives from the export frame and query centre, every emitted rank entry maps back from its map index to
the centre it reports (checker-owned geometry formula, tolerance 16 ULPs of the largest coordinate involved, nominally at least 1e-9 um, but a thousandth of the pixel pitch is the hard upper bound, so a pixel-scale error never hides; origins such as 1e7-1e8 um pass, but once one coordinate ULP exceeds pitch/1000 (about 1e12 um at pitch 0.1) a correct output can be refused, and a pixel shift lost to float64 rounding (about 1e16 um) cannot be detected), every emitted score equals the value the accumulated map holds at that entry's own emitted index (`primary[].map_score`, read from the resident map and not from the peak tuple, so transposed or swapped peaks fail), and rank 1 is not below the own-position value. A wrong template,
a corrupted map or a shifted output coordinate therefore fails. Periodic layouts
may rank another, equally scoring repeat first; the check neither depends on nor chooses one. Output JSON
records the SHA256 of both export files, the arguments and `implementation_revision` (`baseline_ncc_batch_v1.3`; a constant of the
matcher, bumped on any behavioural change, never a timestamp); a re-run is byte-identical.
Valid-domain holes: the query window's invalid pixels (`query_invalid_pixel_count`, flag `query_valid_hole`)
and each returned candidate window's (`primary[].invalid_pixel_count`, flag `candidate_valid_hole`; a pixel
counts when any ready ROI that contributes to that role/layer marks it invalid; ROIs bound only to other
layers never count) are reported and shown on the sheet; scores are unchanged
and nothing is accepted or rejected by them.
`--sheet-dir` must not exist yet or be empty. The path is walked once with directory file descriptors
(`O_DIRECTORY|O_NOFOLLOW` per component; user-planted symlinks anywhere in it are refused, only a root-owned link inside a
root-owned, not group/world-writable directory such as macOS `/tmp` is followed) and every create, no-clobber
publication (hard link) and rollback unlink is relative to that descriptor, so replacing the directory or an ancestor
afterwards cannot redirect writes or deletions. Rollback removes only entries this run created (device/inode
checked); after publishing, the path must still name the same directory, otherwise the run removes its files and exits 2
(`sheet_dir_replaced_during_write`). Files are planned first and created exclusively, so a
conflict exits 2 and never overwrites or follows a link. A hostile process with write access to the sheet directory
itself can still race an individual name; the guarantee is no write or delete outside that directory. Template hashes frame role,
layer, shape and mask length as JSON, so role/layer names containing `/` cannot collide.

`--max-memory-mb N` is required with `--all-ready` and `--self-check` (no default). Layers (role, actual layer, template
shape) are matched one at a time and their FFTs freed afterwards; only queries that span several layers
keep one summed score map until their last layer. Before any FFT the peak is estimated (one layer's FFT
pair and working arrays + running sums + sheet crops + the per-contributor-set validity maps and their two construction temporaries; mask unions already loaded are not counted); above N
the run exits 2 with the estimate, never degrading or truncating. The JSON reports
`memory_estimate_bytes`, `largest_layer_fft_bytes`, `validity_map_count` (distinct contributor sets, one bool map each) and `validity_map_bytes`. The maps grow with the number of distinct contributor sets (up to one per role/layer) and are resident for the whole run, so they count against the cap. Arithmetic stays float64.

Query chunking: when the whole batch's estimate exceeds N, the template groups (queries with an identical
template hash are one group and are never split, so they are still matched once) are cut into the largest
contiguous chunks, in roi_id order, whose own estimate fits N, and the chunks run one after another; only
peaks are kept between chunks. Each chunk recomputes the FFTs of the layers it touches, so more chunks
means more FFT work (`fft_recomputations`). Peaks, scores and row order do not depend on N; the JSON adds
`chunk_count` and `chunk_sizes` (groups per chunk). Every chunk estimate also carries the resident validity maps and their two construction temporaries. The run exits 2 only when one group's accumulator plus
one layer FFT unit (plus the sheet crops and validity maps) already exceeds N.

Product frames repeat a 2×2 unit, so one template can have several near-equal peaks
(one per repeat). `--topk` is the caller's value and should be at least the repeat
count; the tool returns all of them with `near_equal_peaks` and never picks one
automatically. The user chooses the repeat by looking at the sheet.

## Multi-product DB search (PLAN_v3 DB search, 2026-10-26)

Scope: `psdl2_search_batch_v1`, `psdl2_eval_position_multi_v1`, the ETX runners `psdl2_job_smoke_torch.py`, `psdl2_job_db_search.py` and `psdl2_job_eval.py`, and the documents that describe them
([PLAN_v3_db_search.md](plan/PLAN_v3_db_search.md)). All evidence below is author evidence on synthetic data; fresh independent review is separate.

- **Test count:** the full suite passes with 791 passed, 2 skipped, 0 failed (`.venv/bin/python -m pytest -p no:warnings`). This documentation change does not touch code, schema, bundles or tests.
- **Local reproduction of the ETX environment:** the runner tests use two Python 3.9 virtual environments built by `scripts/setup_etx_venvs.sh`: `venv39-notorch` (numpy and Pillow only) and `venv39-torch` (adds torch 2.1.1, CPU wheel). A missing venv makes the tests fail, not skip.
- **Byte equality with the earlier release:** the single-query `search` output, the single `--search`/`--baseline` evaluation output and the bundled `baseline_ncc` flow are compared byte for byte with commit `efc6dac` (`tests/test_search_batch.py`, `tests/test_eval_multi.py`, `tests/test_etx_runtime.py`).
- **Smoke flow (synthetic):** `psdl2_job_smoke_torch.py` runs `patches` (random, grid) → `make_split` → `train` (1 epoch) → `index` → `search --all-ready --exclude-query-product --per-product-topk 2` → `eval_position --search-batch`. It then checks that every query has candidates, that no candidate comes from the query product or the incompatible negative product, and that the evaluation output has schema `psdl2_eval_position_multi_v1` with three truth queries. The synthetic truth only proves that the evaluation runs to the end; it does not count hits.
- **Real-data runners:** `psdl2_job_db_search.py` and `psdl2_job_eval.py` run end to end on synthetic data (db_search in `venv39-torch`, then eval in `venv39-notorch`), including a non-ASCII and space-containing `BASE_DIR`, `-I`, another working directory and restart (`tests/test_etx_jobs.py`). Pre-run checks stop before any job on `FILL_ME` values, missing inputs and inconsistent manifests.
- **Not verified:**
  - A real Linux ETX run (file permissions, environment differences).
  - ETX actual package versions: reproduced locally with the user-reported values (Python 3.9.14, numpy 1.23.5, Pillow 10.1.0, torch 2.1.1+cu121); not run on the ETX server.
  - CUDA (the ETX build is cu121; every test runs on CPU).
  - Real-data size, memory and time. The envelope-size formula `Q × (8.5 KB + 0.15 KB × M + 9.8 KB × T)` comes from one synthetic measurement (Q=3, M=19, T=4: 153,194 bytes). The `M` coefficient is an estimate.
  - Ranking quality. A one-epoch synthetic model shows no ranking quality, and the synthetic truth makes no claim about hit rates.
  - The compare side: the "change point" export format is not fixed, and the pair-level behaviour of `roi_correspondence_v1` (request R3e) needs compare's confirmation.

