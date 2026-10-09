# Actual producer export, synthetic inputs

Producer commit: `f5e86006d72faf16c3bb747051325fbba4b0f9fd` in
`/Users/jonghyeonlee/git/particle-sim-compare`.

Exact command executed:

```sh
cd /Users/jonghyeonlee/git/particle-sim-compare && .venv/bin/python /Users/jonghyeonlee/git/particle-sim-dl2/tests/fixtures/review_roi_v1_export/generate.py
```

The preparation script asserts the producer commit, runs `load_dataset`,
`compute_electrode_maps(min_count=3)` and `export_review_rois` on the producer's
`tests/fixtures/review_roi_v1`, using its own environment. Selection includes all
map regions, a manual case-A ready point, and a manual point without case context.
Output was written outside that repository in a temporary directory, then copied
here byte-for-byte with its referenced assets. Export JSON was never hand-edited.
The original absolute `assets_root` is intentionally preserved: consumer tests
supply this fixture directory as an explicit relocation root.

Expected: ready manual A at (102.5,201.5), FRAME with origin (100,200), pitch 0.5;
two blocked map rows with `map_coverage_incomplete`, and one blocked manual row
with `case_context_required` and `source_shape_required`. Additional crop tests
construct separate hand-defined masks and expected pixel arrays.

`generate.py` is fixture provenance/preparation only. It is never imported by the
package or tests; no runtime dependency on the producer exists.
