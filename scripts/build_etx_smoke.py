#!/usr/bin/env python3
"""ETX(Linux 서버)용 torch smoke 실행기를 bundles/etx/psdl2_job_smoke_torch.py로 만든다.

실행기 하나만 올려 실행하면 합성 fixture(3제품 + 음성 제품 1개)를 BASE_DIR/psdl2_smoke 아래에 스스로 만들고
patches -> make_split -> train(1 epoch) -> index -> search --all-ready --exclude-query-product -> eval_position --search-batch 를 차례로 돈다.
외부 데이터가 필요 없다. 로그에서 단계별 시작·종료 코드와 python/numpy/Pillow/torch/cuda 판을 본다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_single_file import build  # noqa: E402

SMOKE = "psdl2_smoke"
PRODUCTS = ("P0", "P1", "P2")      # 서로 호환되는 제품(mask_derivation 이 같다)
NEGATIVE = "N"                     # 음성 제품: mask_derivation 이 달라 어떤 질의와도 호환되지 않는다(rejected 로만 나온다)
ALL_PRODUCTS = PRODUCTS + (NEGATIVE,)
PER_CASE, GRID_STEP = 6, 2.0

# 실행기 안의 DEVICE 이름을 가리키는 표지(작업 인자에서 문자열이 아니라 이름으로 쓴다)
DEVICE = object()

# tests/test_patches.py 의 views('native') 와 같은 값. build 때 하드코딩하고 시험이 psdl2.views 의 asdict 와 같은지 본다.
VIEWS = {"crop": {"near_size_xy_um": [3.0, 3.0], "wide_size_xy_um": [4.0, 4.0],
                  "intermediate_layer_policy": "preserve_exported_layers_v1", "output_pitch_um": None, "required_layers": []},
         "near_pitch_um": 0.5, "wide_pitch_um": 0.5, "near_resampling_mode": "nearest", "wide_resampling_mode": "nearest",
         "jitter_xy_um": [0.1, 0.1], "core_size_xy_um": [2.0, 2.0], "topology_policy": "common_core_connectivity_v3",
         "connection_semantics": "global_contact_proxy_unverified", "comparison_basis": "between_views_only", "max_pair_attempts": 3}


def _p(*parts):
    return "/".join((SMOKE,) + parts)


def _jobs():
    jobs = []
    for mode in ("random", "grid"):
        for index, product in enumerate(ALL_PRODUCTS):
            extra = ["--seed", "3", "--per-case", str(PER_CASE), "--min-spacing-um", "1"] if mode == "random" else \
                ["--grid-step-um", str(GRID_STEP)]
            jobs.append(("patches", ["--dataset", _p("plain.json"), "--source", str(index), "--config", _p("config.json"),
                                     "--mode", mode, "--out", _p("out", f"{mode}_{product}.json"), *extra, "--overwrite"]))
    jobs.append(("make_split", ["--dataset", _p("train_dataset.json"), "--config", _p("config.json"),
                                "--lineages", _p("lineages.json"), "--assign", _p("assign.json"),
                                "--out", _p("out", "split.json"), "--overwrite"]))
    jobs.append(("train", ["--dataset", _p("train_dataset.json"), "--config", _p("config.json"), "--out", _p("out", "model.pt"),
                           "--lineages", _p("lineages.json"), "--split", _p("out", "split.json"), "--no-progress",
                           "--device", DEVICE]))
    jobs.append(("index", ["--dataset", _p("index_dataset.json"), "--checkpoint", _p("out", "model.pt"),
                           "--out", _p("out", "index.json"), "--device", DEVICE]))
    jobs.append(("search", ["--query-dataset", _p("query.json"), "--all-ready", "--exclude-query-product",
                            "--checkpoint", _p("out", "model.pt"), "--index", _p("out", "index.json"),
                            "--cohort", _p("cohort.json"), "--topk", "4", "--per-product-topk", "2",
                            "--out", _p("out", "search_batch.json"), "--device", DEVICE]))
    jobs.append(("eval_position", ["--search-batch", _p("out", "search_batch.json"), "--correspondence", _p("correspondence.json"),
                                   "--hit-tolerance-um", "2", "--k", "4", "--out", _p("out", "eval_position.json")]))
    return jobs


def _jobs_source(jobs):
    def item(value):
        return "DEVICE" if value is DEVICE else repr(value)
    lines = ["JOBS = ["]
    for name, args in jobs:
        lines.append(f"    ({name!r}, [")
        index = 0
        while index < len(args):
            pair = (args[index] is not DEVICE and args[index].startswith("--") and index + 1 < len(args)
                    and (args[index + 1] is DEVICE or not args[index + 1].startswith("--")))
            chunk = args[index:index + 2] if pair else args[index:index + 1]
            lines.append("        " + " ".join(f"{item(a)}," for a in chunk))
            index += len(chunk)
        lines.append("    ]),")
    lines.append("]")
    return "\n".join(lines) + "\n"


HOOKS = r'''SMOKE_DIR = @@SMOKE@@
PRODUCTS = @@PRODUCTS@@
NEGATIVE = @@NEGATIVE@@
PER_CASE = @@PER_CASE@@
SMOKE_TOP = json.loads(@@TOP@@)
SMOKE_ROW = json.loads(@@ROW@@)
SMOKE_CONFIG = json.loads(@@CONFIG@@)
ENDPOINTS = [{"electrode_id": "A", "role": "gate", "actual_layer": "M1"}, {"electrode_id": "B", "role": "data", "actual_layer": "M3"}]


def _smoke_log(text):
    print(f"[psdl2_job] {time.strftime('%Y-%m-%d %H:%M:%S')} {text}", file=sys.stderr, flush=True)


def _derivation(removed):
    return {"schema_version": "psdl2_mask_derivation_v1", "rule_version": "between_stack_order_v1",
            "cases": [{"case": {"failure_type": "vertical_short", "endpoints": ENDPOINTS},
                       "processed_endpoint": "B", "removed_layers": list(removed)}]}


def _bars(shape, product):
    """제품마다 영상이 조금 다르다(P0..P2 는 막대 옆 점의 위치가 다르고, 음성 제품 N 은 막대가 가로로 누워 있다).
    그래서 동점이 없고, N 은 mask_derivation 검사를 꺼도 1위가 되지 못한다."""
    import numpy as np
    l0 = np.zeros(shape, np.uint8)
    l1 = np.zeros(shape, np.uint8)
    cnt = np.zeros(shape, np.uint8)
    if product == NEGATIVE:
        l0[10:12, 6:24] = 1
        l1[14:16, 6:24] = 1
        cnt[12:14, 6:24] = 1
    else:
        mark = PRODUCTS.index(product)
        l0[6:24, 10 - mark:12] = 1    # 첫 막대의 굵기가 제품마다 다르다(index 의 후보 수도 제품마다 조금 달라질 수 있다)
        l1[6:24, 14:16 + mark] = 1
        cnt[6:24, 12:14] = 1
    return [l0, l1, cnt, np.ones(shape, np.uint8)]


def _write_export(root, product):
    """합성 review_roi_v1 export 하나(준비된 ROI 한 줄 + assets/*.npy). 폴더 안 경로는 상대 경로다."""
    import numpy as np
    folder = root / "templates" / product
    shape, origin, pitch = (30, 30), (100.0, 200.0), 0.5
    rid = f"{product}_case0"
    row = json.loads(json.dumps(SMOKE_ROW))
    centre = [origin[0] + (shape[1] // 2 + 0.5) * pitch, origin[1] + (shape[0] // 2 + 0.5) * pitch]
    row.update(roi_id=rid, product_id=product, case_id="CASE", frame_id=f"frame_{rid}", source_version="synthetic_patch_v1",
               center_xy_um=[float(centre[0]), float(centre[1])], failure_type="vertical_short")
    row["source_identity"].update(product_id=product, case_id="CASE", frame_id=row["frame_id"])
    row["frame"].update(frame_id=row["frame_id"], shape_yx=list(shape), origin_xy_um=list(origin), pixel_pitch_um=pitch)
    row["endpoints"] = [dict(ep) for ep in ENDPOINTS]
    row["analysis_layer"] = "M3"
    for layout, endpoint in zip(row["source_assets"]["layouts"], row["endpoints"]):
        layout["endpoint"] = dict(endpoint)
    refs = [layout["asset"] for layout in row["source_assets"]["layouts"]] + [row["source_assets"]["cnt"], row["source_assets"]["valid"]]
    for channel, (ref, array) in enumerate(zip(refs, _bars(shape, product))):
        path = folder / "assets" / f"{rid}_{channel}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), array)
        ref.update(path=path.relative_to(folder).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    payload = dict(SMOKE_TOP, producer_commit="0" * 40, assets_root=".", rois=[row], excluded_records=[], status="available")
    payload["source_manifest_hash"] = hashlib.sha256(json.dumps(payload["rois"], sort_keys=True).encode()).hexdigest()
    target = folder / "review_rois.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _correspondence():
    """합성 정답(roi_correspondence_v1): 질의 Pi_case0 의 정답 위치는 다음 제품의 중심이다. 평가 단계가 끝까지 도는지만 본다."""
    centre = [100.0 + 15.5 * 0.5, 200.0 + 15.5 * 0.5]
    pairs = []
    for index, product in enumerate(PRODUCTS):
        other = PRODUCTS[(index + 1) % len(PRODUCTS)]
        pairs.append({"pair_id": f"S{index}",
                      "a": {"product_id": product, "design_version": "DESIGN", "frame_id": f"frame_{product}_case0",
                            "center_xy_um": centre, "analysis_layer": "M3", "roi_id": f"{product}_case0"},
                      "b": {"product_id": other, "design_version": "DESIGN", "frame_id": f"frame_{other}_case0",
                            "center_xy_um": centre, "analysis_layer": "M3"},
                      "source": {"kind": "manual", "labelled_by": "smoke", "labelled_at": "2026-10-26"},
                      "status": "ready", "reasons": []})
    return {"schema_version": "roi_correspondence_v1", "producer": {"kind": "manual", "version": "1"}, "pairs": pairs}


def _prepare(base):
    """합성 fixture 를 BASE_DIR/psdl2_smoke 에 만든다(제품 3 + 음성 제품 1). 다시 실행하면 같은 바이트로 덮어쓴다."""
    root = Path(base) / SMOKE_DIR
    (root / "out").mkdir(parents=True, exist_ok=True)
    _smoke_log(f"prepare synthetic fixture in {root}")
    derivations = {product: _derivation(("M2",)) for product in PRODUCTS}
    derivations[NEGATIVE] = _derivation(("M2", "M9"))   # 질의와 mask_derivation 이 달라 호환되지 않는다
    sources = {}
    for product in PRODUCTS + (NEGATIVE,):
        export = _write_export(root, product)
        sources[product] = {"export": export.relative_to(root).as_posix(),
                            "export_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
                            "mask_derivation": derivations[product]}
    order = PRODUCTS + (NEGATIVE,)

    def manifest(name, products, prefix=None):
        entries = []
        for product in products:
            entry = dict(sources[product])
            if prefix:
                entry["patch_sets"] = [f"out/{prefix}_{product}.json"]
            entries.append(entry)
        _dump(root / name, {"schema_version": "psdl2_dataset_v1", "sources": entries})

    manifest("plain.json", order)
    manifest("train_dataset.json", order, "random")
    manifest("index_dataset.json", order, "grid")
    manifest("query.json", PRODUCTS)
    _dump(root / "config.json", SMOKE_CONFIG)
    _dump(root / "lineages.json", {"P0": "line_train", "P1": "line_validation", "P2": "line_test", NEGATIVE: "line_negative"})
    _dump(root / "assign.json", {"line_train": "train", "line_validation": "validation", "line_test": "test", "line_negative": "train"})
    _dump(root / "cohort.json", list(order))
    _dump(root / "correspondence.json", _correspondence())
    for stale in (root / "out" / "eval_position.json", root / "out" / ("eval_position.json" + DONE_SUFFIX)):
        if stale.exists():   # 다시 실행하면 평가 산출물과 완료 표지를 새로 만든다(search 결과가 바이트 단위로 같다는 보장이 없다)
            stale.unlink()
    _smoke_log(f"fixture ready: {len(order)} products ({', '.join(order)}); negative product {NEGATIVE}")


def _check(base):
    """search 결과가 계약대로인지 본다: 질의마다 후보가 있고, 질의 제품·음성 제품 후보가 없다."""
    path = Path(base) / SMOKE_DIR / "out" / "search_batch.json"

    def bad(step, name, reason):
        _fail("smoke_check_failed", f"the output of step {step} ({name}) is not as expected: {reason}",
              f"{step} 단계의 출력 {name} 가 기대와 다르다: {reason}")

    if not path.is_file():
        bad("search", path.name, "the file is missing")
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch.get("schema_version") != "psdl2_search_batch_v1":
        bad("search", path.name, "schema_version")
    results, rankings = batch["results"], batch["rankings"]
    if len(results) != len(PRODUCTS) or len(rankings) != len(PRODUCTS):
        bad("search", path.name, f"expected {len(PRODUCTS)} queries, got {len(results)}")
    for result, ranking in zip(results, rankings):
        query_product = ranking["query_product_id"]
        if result["status"] != "available" or not ranking["candidates"]:
            bad("search", path.name, f"query {ranking['query_roi_id']} has no candidates")
        products = {row["product_id"] for row in ranking["candidates"]}
        if query_product in products or NEGATIVE in products:
            bad("search", path.name, f"query {ranking['query_roi_id']} got candidates of {sorted(products)}")
    evaluation = Path(base) / SMOKE_DIR / "out" / "eval_position.json"
    if not evaluation.is_file():
        bad("eval_position", evaluation.name, "the file is missing")
    summary = json.loads(evaluation.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "psdl2_eval_position_multi_v1" or summary["summary"]["n_queries_with_truth"] != len(PRODUCTS):
        bad("eval_position", evaluation.name, "wrong schema or query count")
    _smoke_log(f"eval check OK: {summary['summary']['n_hit_at_k']} of {len(PRODUCTS)} queries hit within the top 4")
    _smoke_log("smoke check OK: " + ", ".join(f"{r['query_roi_id']} -> {len(r['candidates'])} candidates" for r in rankings))
'''

HEADER = [
    "DL2 torch smoke: 합성 fixture(3제품 + 음성 제품 1)를 BASE_DIR/psdl2_smoke 에 만들고 patches -> make_split -> train(1 epoch)",
    "-> index -> search --all-ready --exclude-query-product -> eval_position --search-batch 를 돈다. 외부 데이터가 필요 없다. torch 가 있는 Python 으로 실행한다.",
    "BASE_DIR 은 쓸 수 있는 폴더의 절대 경로로 적는다. DEVICE 를 \"cuda\" 로 바꾸면 train·index·search 가 GPU 로 돈다(CUDA torch 필요).",
    "끝나면 로그(LOG_FILE)를 그대로 돌려준다. 다시 실행하면 같은 폴더를 덮어쓴다.",
]
DOCSTRING = ("psdl2(DL2) torch smoke 단일 파일 실행기 (자동 생성: scripts/build_etx_smoke.py). 소스 해시 @@SHA@@.\n"
             "합성 fixture 로 patches -> make_split -> train -> index -> search -> eval_position 을 한 번에 돌려 torch 환경을 확인한다.")


def _hooks():
    config = json.loads((ROOT / "configs/synthetic.example.json").read_text(encoding="utf-8"))["config"]
    config["views"] = VIEWS
    template = json.loads((ROOT / "tests/fixtures/review_roi_v1_export/review_rois.json").read_text(encoding="utf-8"))
    row = next(r for r in template["rois"] if r["status"] == "ready")
    top = {k: v for k, v in template.items()
           if k not in ("rois", "excluded_records", "assets_root", "producer_commit", "source_manifest_hash", "status")}
    text = HOOKS
    for key, value in (("SMOKE", repr(SMOKE)), ("PRODUCTS", repr(PRODUCTS)), ("NEGATIVE", repr(NEGATIVE)), ("PER_CASE", repr(PER_CASE)),
                       ("TOP", repr(json.dumps(top, sort_keys=True))), ("ROW", repr(json.dumps(row, sort_keys=True))),
                       ("CONFIG", repr(json.dumps(config, sort_keys=True)))):
        text = text.replace(f"@@{key}@@", value)
    return text


def build_smoke(out, *, overwrite=False):
    return build(out, overwrite=overwrite, jobs_source=_jobs_source(_jobs()), hooks=_hooks(), docstring=DOCSTRING, header=HEADER,
                 extra_edit='DEVICE = "cpu"                                 # train·index·search 의 --device. "cuda" 는 CUDA torch 가 있을 때만\n',
                 log_file=f"{SMOKE}/smoke.log")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "bundles/etx/psdl2_job_smoke_torch.py"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build_smoke(a.out, overwrite=a.overwrite), indent=2))


if __name__ == "__main__":
    main()
