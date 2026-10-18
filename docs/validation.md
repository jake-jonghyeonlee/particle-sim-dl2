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

