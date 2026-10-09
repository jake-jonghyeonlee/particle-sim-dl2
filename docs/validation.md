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
  observed core checks, native component clipping, and removed bridges at the
  same physical centre. These are conservative structural acceptance checks.
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

The CNN, conditioning/aggregation, objective coefficients, conservative native
component policy and synthetic label schema are reviewable baseline choices.
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
