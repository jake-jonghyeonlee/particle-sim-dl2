#!/usr/bin/env python3
"""ETX(Linux 서버)용 실데이터 실행기 2종을 만든다.

bundles/etx/psdl2_job_db_search.py (torch 필요): 제품마다 patches --mode grid -> make_split -> train -> index --dataset
    -> search --all-ready --query-dataset ... --exclude-query-product 로 DB 를 만들고 질의 전체를 검색한다.
bundles/etx/psdl2_job_eval.py (torch 불필요): 후보 제품마다 baseline_ncc --all-ready -> eval_position --baseline <여러 파일>
    -> eval_position --search-batch <db_search 봉투> 로 두 방법을 같은 정답으로 평가한다.

수치 기본값은 넣지 않는다. 사용자가 정할 값은 FILL_ME 로 두고, 남아 있으면 어떤 작업도 돌기 전에 멈춘다(edit_value_required).
경로는 예시 이름(productA 등)이다. 입력 파일이 없으면 input_missing 으로 멈춘다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_single_file import build  # noqa: E402

COMMON_HOOKS = r'''def _log(text):
    print(f"[psdl2_job] {time.strftime('%Y-%m-%d %H:%M:%S')} {text}", file=sys.stderr, flush=True)


def _check_values(required):
    """EDIT 블록에 FILL_ME 가 남아 있으면 어떤 작업도 돌기 전에 멈춘다. 그 값을 쓰는 모듈의 작업이 JOBS 에 있을 때만 요구한다."""
    active = {name for name, _ in JOBS}
    left = [name for name, module in required if module in active and globals()[name] == FILL_ME]
    if left:
        _fail("edit_value_required",
              f"{', '.join(left)} still {FILL_ME!r}. Set each value in the EDIT block at the top of this file and run again."
              " No job has run",
              f"EDIT 블록의 {', '.join(left)} 에 {FILL_ME!r} 가 남아 있다. 값을 정해 적고 다시 실행한다. 어떤 작업도 실행하지 않았다.")


def _check_inputs(inputs):
    """입력 파일이 모두 있는지 본다. inputs 는 (EDIT 이름, 경로, 모듈, 인자 표시 또는 None) 목록이다.
    JOBS 에 그 모듈의 작업(인자 표시가 있으면 그 인자를 쓰는 작업)이 있을 때만 요구한다."""
    for label, path, module, flag in inputs:
        if not any(name == module and (flag is None or flag in args) for name, args in JOBS):
            continue
        resolved = Path(path).resolve()
        if not resolved.is_file():
            _fail("input_missing",
                  f"{label} -> {resolved} does not exist. Set {label} in the EDIT block to an existing file"
                  " (relative to BASE_DIR, or absolute). No job has run",
                  f"{label} -> {resolved}: 이 입력 파일이 없다. EDIT 블록의 {label} 를 있는 파일의 경로(BASE_DIR 기준 상대 경로 또는 절대 경로)로 고친다."
                  " 어떤 작업도 실행하지 않았다.")


def _check_names(label, names):
    """제품 이름은 파일 이름에 쓰이므로 영문·숫자·_-. 만, 서로 다르게."""
    import re
    bad = [n for n in names if not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", n)]
    if not names or bad or len(set(names)) != len(names):
        _fail("edit_value_invalid",
              f"{label} must be a non-empty list of distinct names made of letters, digits, '_', '-', '.'. Got {names!r}",
              f"{label} 는 영문·숫자·_ - . 로 된 서로 다른 이름의 비어 있지 않은 목록이어야 한다. 지금 값: {names!r}")
'''

DB_SEARCH_EDIT = '''FILL_ME = "FILL_ME"                           # 이 값이 남아 있으면 어떤 작업도 돌기 전에 edit_value_required 로 멈춘다
DEVICE = "cpu"                                # train·index·search 의 --device. "cuda" 는 CUDA torch 가 있을 때만
# --- 입력 파일(BASE_DIR 기준 상대 경로. 아래 이름은 예시다. 없으면 input_missing 으로 멈춘다) ---
DB_DATASET = "dl2_db/db_dataset.json"          # psdl2_dataset_v1: 제품마다 template export 한 source (patch_sets 없음). 이름은 .json 으로 끝나야 한다
QUERY_DATASET = "dl2_db/query_dataset.json"    # psdl2_dataset_v1: 질의 export(변경점 ROI)와 그 mask_derivation
LINEAGES = "dl2_db/lineages.json"              # {제품: 계열}
ASSIGN = "dl2_db/assign.json"                  # {계열: train/validation/test}
# --- 사용자가 정할 값. 수치 기본값은 없다 ---
CONFIG = FILL_ME                               # psdl2_run_v1 실행 설정 파일 경로 (views·모델·epochs 는 이 파일이 정한다)
COHORT = FILL_ME                               # 검색 대상 제품 목록 파일 경로 (JSON 배열)
GRID_STEP_UM = FILL_ME                         # 격자 간격(um). 평가의 hit 허용 거리 이하로 잡는다
TOPK = FILL_ME                                 # search --topk
PER_PRODUCT_TOPK = FILL_ME                     # search --per-product-topk. 숫자 또는 None(쓰지 않음)
# --- 이름·출력 ---
PRODUCTS = ["productA", "productB"]            # DB_DATASET 의 source 와 같은 순서·같은 개수. 이름은 patch set 파일 이름에만 쓰인다
OUT_DIR = "dl2_db/out"                         # grid_<제품>.json, split.json, model.pt, index.json, search_batch.json
GRID_DATASET = DB_DATASET[:-5] + ".grid.json"  # 자동 생성: DB_DATASET 에 grid patch set 을 붙인 manifest. DB_DATASET 과 같은 폴더에 쓴다
'''

DB_SEARCH_JOBS = '''JOBS = (
    [("patches", ["--dataset", DB_DATASET, "--source", str(index), "--config", CONFIG, "--mode", "grid",
                  "--grid-step-um", str(GRID_STEP_UM), "--out", f"{OUT_DIR}/grid_{name}.json",
                  "--allow-unversioned-producer", "--overwrite"])
     for index, name in enumerate(PRODUCTS)]
    + [("make_split", ["--dataset", GRID_DATASET, "--config", CONFIG, "--lineages", LINEAGES, "--assign", ASSIGN,
                       "--out", f"{OUT_DIR}/split.json", "--allow-unversioned-producer", "--overwrite"]),
       ("train", ["--dataset", GRID_DATASET, "--config", CONFIG, "--out", f"{OUT_DIR}/model.pt", "--lineages", LINEAGES,
                  "--split", f"{OUT_DIR}/split.json", "--allow-unversioned-producer", "--device", DEVICE]),
       ("index", ["--dataset", GRID_DATASET, "--checkpoint", f"{OUT_DIR}/model.pt", "--out", f"{OUT_DIR}/index.json",
                  "--allow-unversioned-producer", "--device", DEVICE]),
       ("search", ["--query-dataset", QUERY_DATASET, "--all-ready", "--exclude-query-product",
                   "--checkpoint", f"{OUT_DIR}/model.pt", "--index", f"{OUT_DIR}/index.json", "--cohort", COHORT,
                   "--topk", str(TOPK), *(["--per-product-topk", str(PER_PRODUCT_TOPK)] if PER_PRODUCT_TOPK is not None else []),
                   "--out", f"{OUT_DIR}/search_batch.json", "--allow-unversioned-producer", "--device", DEVICE])]
)
'''

DB_SEARCH_HOOKS = r'''REQUIRED_VALUES = (("CONFIG", "patches"), ("GRID_STEP_UM", "patches"), ("COHORT", "search"), ("TOPK", "search"),
                   ("PER_PRODUCT_TOPK", "search"))
INPUTS = (("DB_DATASET", DB_DATASET, "patches", None), ("QUERY_DATASET", QUERY_DATASET, "search", None),
          ("LINEAGES", LINEAGES, "make_split", None), ("ASSIGN", ASSIGN, "make_split", None),
          ("CONFIG", CONFIG, "patches", None), ("COHORT", COHORT, "search", None))
@@COMMON@@


def _prepare(base):
    """작업 전: EDIT 값·입력 파일을 검사하고, DB_DATASET 에 grid patch set 을 붙인 manifest(GRID_DATASET)를 쓴다."""
    _check_values(REQUIRED_VALUES)
    _check_names("PRODUCTS", PRODUCTS)
    _check_inputs(INPUTS)
    plain = Path(DB_DATASET).resolve()
    if not DB_DATASET.endswith(".json"):
        _fail("edit_value_invalid", f"DB_DATASET {DB_DATASET!r} must end with .json (the grid manifest name is derived from it)",
              f"DB_DATASET {DB_DATASET!r} 는 .json 으로 끝나야 한다(grid manifest 이름을 여기서 만든다).")
    try:
        document = json.loads(plain.read_text(encoding="utf-8"))
        sources = document["sources"]
        if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
            raise ValueError("sources must be a list of objects")
    except (OSError, ValueError, KeyError) as exc:
        _fail("input_invalid", f"DB_DATASET -> {plain} is not a psdl2_dataset_v1 manifest ({type(exc).__name__}: {exc})",
              f"DB_DATASET -> {plain} 가 psdl2_dataset_v1 manifest 가 아니다 ({type(exc).__name__}: {exc}).")
    if len(sources) != len(PRODUCTS):
        _fail("input_mismatch", f"PRODUCTS has {len(PRODUCTS)} names but DB_DATASET -> {plain} has {len(sources)} sources."
                                " List one name per source, in the same order",
              f"PRODUCTS 는 {len(PRODUCTS)} 개인데 DB_DATASET -> {plain} 의 source 는 {len(sources)} 개다. source 마다 이름 하나를 같은 순서로 적는다.")
    out_root = Path(OUT_DIR).resolve()
    for source, name in zip(sources, PRODUCTS):
        source["patch_sets"] = [os.path.relpath(str(out_root / f"grid_{name}.json"), str(plain.parent)).replace(os.sep, "/")]
    target = Path(GRID_DATASET).resolve()
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"prepare: wrote {_rel(target)} ({len(sources)} sources, patch sets under {_rel(out_root)}); DB products: {', '.join(PRODUCTS)}")


def _check(base):
    """모든 작업이 끝난 뒤: 검색 봉투를 읽어 질의·후보 수를 로그에 남긴다."""
    path = Path(OUT_DIR) / "search_batch.json"
    batch = json.loads(path.read_text(encoding="utf-8"))
    rankings = batch["rankings"]
    empty = [r["query_roi_id"] for r in rankings if not r["candidates"]]
    _log(f"search batch {_rel(path)}: {len(rankings)} queries, {len(empty)} without candidates"
         + (f" ({', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''})" if empty else ""))
'''

DB_SEARCH_HEADER = [
    "DL2 DB 검색 실행기 (torch 필요). 제품마다 patches --mode grid(DB 위치 생성) -> make_split -> train -> index --dataset",
    "-> search --all-ready --query-dataset ... --exclude-query-product 를 차례로 돈다. 결과는 OUT_DIR/search_batch.json(psdl2_search_batch_v1) 이고,",
    "평가는 psdl2_job_eval.py 가 이 봉투를 읽어 한다.",
    "회차 위치(compare): 질의 = 변경점 ROI export (QUERY_DATASET 이 가리킨다), DB = 제품별 template export (DB_DATASET 이 가리킨다).",
    "compare 의 새 \"변경점\" 형식은 미확정이다. 지금은 review_roi_v1 manual 행 + mask_derivation.json 으로 만든 export 를 가정한다(PLAN D2).",
    "simcompare 4·5회차용 psdl2_job_baseline_ncc.py 와는 별개다.",
    "학습과 색인은 같은 grid patch manifest(GRID_DATASET)를 쓴다. 수치 기본값은 없다: FILL_ME 인 값을 정하지 않으면 실행 전에 멈춘다.",
    "오류 줄은 [psdl2_job] ERROR <코드>: <영어 문구> | <한글 문구> 한 줄이다. 코드: edit_value_required, edit_value_invalid, input_missing,",
    "input_invalid, input_mismatch, torch_required_for_<모듈>. 파일 만들기: scripts/build_etx_jobs.py.",
    "--allow-unversioned-producer: ETX 실행기로 만든 export 는 git checkout 이 아니어서 producer_commit 이 없다(사유는 export 에 기록).",
    "BASE_DIR 은 쓸 수 있는 폴더의 절대 경로로 적는다. 종료 코드 0·3 은 다음 작업으로 넘어가고 그 밖이면 멈춘다. train·index·search 는 매번 다시 돈다.",
]

EVAL_EDIT = '''FILL_ME = "FILL_ME"                           # 이 값이 남아 있으면 어떤 작업도 돌기 전에 edit_value_required 로 멈춘다
# --- 입력 파일(BASE_DIR 기준 상대 경로. 아래 이름은 예시다. 없으면 input_missing 으로 멈춘다) ---
QUERY_EXPORT = "dl2_eval/query/review_rois.json"            # 질의: 변경점 ROI export (review_roi_v1)
CANDIDATES = [("productA", "dl2_eval/templates/productA/review_rois.json"),   # 후보 제품마다 (이름, 그 제품의 template export)
              ("productB", "dl2_eval/templates/productB/review_rois.json")]
CORRESPONDENCE = "dl2_eval/correspondence.json"             # 정답: roi_correspondence_v1 (a.roi_id = 질의, b = 정답 위치)
SEARCH_BATCH = "dl2_db/out/search_batch.json"               # psdl2_job_db_search.py 가 만든 봉투(그 실행기의 OUT_DIR)
# --- 사용자가 정할 값. 수치 기본값은 없다 ---
WINDOW_UM = FILL_ME                            # baseline_ncc --window-um
NCC_TOPK = FILL_ME                             # baseline_ncc --topk
MAX_MEMORY_MB = FILL_ME                        # baseline_ncc --max-memory-mb
HIT_TOLERANCE_UM = FILL_ME                     # eval_position --hit-tolerance-um
K = FILL_ME                                    # eval_position --k
OUT_DIR = "dl2_eval/out"                       # ncc_<제품>.json, ncc_<제품>_sheet/, eval_baseline.json, eval_search_batch.json
'''

EVAL_JOBS = '''NCC_JOBS = [("baseline_ncc", ["--all-ready", "--allow-unversioned-producer",
                              "--query-export", QUERY_EXPORT, "--index-export", export,
                              "--window-um", str(WINDOW_UM), "--topk", str(NCC_TOPK), "--max-memory-mb", str(MAX_MEMORY_MB),
                              "--sheet-dir", f"{OUT_DIR}/ncc_{name}_sheet", "--out", f"{OUT_DIR}/ncc_{name}.json"])
            for name, export in CANDIDATES]
# 정답(roi_correspondence_v1)이 없으면 아래 마지막 줄을 `JOBS = NCC_JOBS` 로 바꾼다. 그러면 평가 작업과 CORRESPONDENCE·SEARCH_BATCH 검사가 빠진다.
# 후보 제품이 하나뿐이면 eval_position --baseline 은 단일 모드(psdl2_position_eval_v1)로 돈다. 둘 이상이면 psdl2_eval_position_multi_v1 이다.
EVAL_JOBS = [("eval_position", ["--baseline", *[f"{OUT_DIR}/ncc_{name}.json" for name, _ in CANDIDATES],
                                "--correspondence", CORRESPONDENCE, "--hit-tolerance-um", str(HIT_TOLERANCE_UM), "--k", str(K),
                                "--out", f"{OUT_DIR}/eval_baseline.json"]),
             ("eval_position", ["--search-batch", SEARCH_BATCH,
                                "--correspondence", CORRESPONDENCE, "--hit-tolerance-um", str(HIT_TOLERANCE_UM), "--k", str(K),
                                "--out", f"{OUT_DIR}/eval_search_batch.json"])]
JOBS = NCC_JOBS + EVAL_JOBS
'''

EVAL_HOOKS = r'''REQUIRED_VALUES = (("WINDOW_UM", "baseline_ncc"), ("NCC_TOPK", "baseline_ncc"), ("MAX_MEMORY_MB", "baseline_ncc"),
                   ("HIT_TOLERANCE_UM", "eval_position"), ("K", "eval_position"))
INPUTS = ((("QUERY_EXPORT", QUERY_EXPORT, "baseline_ncc", None),)
          + tuple((f"CANDIDATES[{name}]", export, "baseline_ncc", None) for name, export in CANDIDATES)
          + (("CORRESPONDENCE", CORRESPONDENCE, "eval_position", None), ("SEARCH_BATCH", SEARCH_BATCH, "eval_position", "--search-batch")))
@@COMMON@@


def _prepare(base):
    """작업 전: EDIT 값·입력 파일을 검사한다. 하나라도 틀리면 어떤 작업도 돌기 전에 멈춘다."""
    if not CANDIDATES:
        _fail("edit_value_required", "CANDIDATES is empty. List at least one (name, template export) pair. No job has run",
              "CANDIDATES 가 비어 있다. (이름, template export) 쌍을 하나 이상 적는다. 어떤 작업도 실행하지 않았다.")
    _check_names("CANDIDATES names", [name for name, _ in CANDIDATES])
    _check_values(REQUIRED_VALUES)
    _check_inputs(INPUTS)
    _log(f"prepare: query {QUERY_EXPORT}; candidate products: {', '.join(name for name, _ in CANDIDATES)};"
         f" {'with' if any(n == 'eval_position' for n, _ in JOBS) else 'without'} evaluation")


def _check(base):
    """모든 작업이 끝난 뒤: 평가 출력의 요약을 로그에 남긴다."""
    for name in ("eval_baseline.json", "eval_search_batch.json"):
        path = Path(OUT_DIR) / name
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = result["summary"]
        keys = [key for key in ("n_queries_with_truth", "n_queries_with_prediction", "n_top1_hit", "n_hit_at_k", "top1_hit_rate",
                                "n_queries", "n_evaluable") if key in summary]
        _log(f"{_rel(path)}: status {result['status']}; " + ", ".join(f"{key}={summary[key]}" for key in keys))
'''

EVAL_HEADER = [
    "DL2 평가 실행기 (torch 불필요: numpy·Pillow 만). 후보 제품마다 baseline_ncc --all-ready(질의 export <-> 그 제품의 template export)를 돌린 뒤",
    "eval_position --baseline <제품별 파일 전부> 와 eval_position --search-batch <psdl2_job_db_search.py 의 봉투> 로 같은 정답에 대한 hit 를 낸다.",
    "회차 위치(compare): 질의 = 변경점 ROI export (QUERY_EXPORT), 후보 = 제품별 template export (CANDIDATES), 정답 = roi_correspondence_v1 (CORRESPONDENCE).",
    "compare 의 새 \"변경점\" 형식은 미확정이다. 지금은 review_roi_v1 manual 행 + mask_derivation.json 으로 만든 export 를 가정한다(PLAN D2).",
    "simcompare 4·5회차용 psdl2_job_baseline_ncc.py 와는 별개다. 정답이 아직 없으면 EDIT 블록의 `JOBS = NCC_JOBS + EVAL_JOBS` 주석대로 평가 작업을 뺀다.",
    "수치 기본값은 없다: FILL_ME 인 값을 정하지 않으면 실행 전에 멈춘다. NCC 점수와 코사인 점수는 서로 비교하거나 합쳐 다시 줄 세우지 않는다.",
    "오류 줄은 [psdl2_job] ERROR <코드>: <영어 문구> | <한글 문구> 한 줄이다. 코드: edit_value_required, edit_value_invalid, input_missing,",
    "stale_done_marker:<사유>. baseline_ncc·eval_position 은 완료 표지를 써서 다시 실행하면 skip completed 로 건너뛴다. 파일 만들기: scripts/build_etx_jobs.py.",
    "--allow-unversioned-producer: ETX 실행기로 만든 export 는 git checkout 이 아니어서 producer_commit 이 없다(사유는 export 에 기록).",
    "BASE_DIR 은 쓸 수 있는 폴더의 절대 경로로 적는다. 종료 코드 0·3 은 다음 작업으로 넘어가고 그 밖이면 멈춘다. --sheet-dir 는 없거나 빈 폴더여야 한다.",
]


def _hooks(text):
    return text.replace("@@COMMON@@", "\n\n" + COMMON_HOOKS.rstrip("\n"))


def build_db_search(out, *, overwrite=False):
    return build(out, overwrite=overwrite, jobs_source=DB_SEARCH_JOBS, extra_edit=DB_SEARCH_EDIT, hooks=_hooks(DB_SEARCH_HOOKS),
                 header=DB_SEARCH_HEADER, log_file="runs/psdl2_db_search.log",
                 docstring="psdl2(DL2) DB 검색 실행기 (자동 생성: scripts/build_etx_jobs.py). 소스 해시 @@SHA@@.\n"
                           "제품 grid patch -> make_split -> train -> index -> search 를 한 번에 돈다. torch 가 필요하다.")


def build_eval(out, *, overwrite=False):
    return build(out, overwrite=overwrite, jobs_source=EVAL_JOBS, extra_edit=EVAL_EDIT, hooks=_hooks(EVAL_HOOKS),
                 header=EVAL_HEADER, log_file="runs/psdl2_eval.log",
                 docstring="psdl2(DL2) 평가 실행기 (자동 생성: scripts/build_etx_jobs.py). 소스 해시 @@SHA@@.\n"
                           "baseline_ncc -> eval_position(--baseline, --search-batch) 를 한 번에 돈다. torch 가 필요 없다.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "bundles/etx"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    folder = Path(a.out_dir)
    print(json.dumps({"db_search": build_db_search(folder / "psdl2_job_db_search.py", overwrite=a.overwrite),
                      "eval": build_eval(folder / "psdl2_job_eval.py", overwrite=a.overwrite)}, indent=2))


if __name__ == "__main__":
    main()
