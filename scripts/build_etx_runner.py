#!/usr/bin/env python3
"""ETX(Linux 서버)용 DL2 실행기를 bundles/etx/psdl2_job_baseline_ncc.py로 만든다.

simcompare 4회차(candidates) 결과로 두 방향 baseline_ncc(--all-ready)를 돌려 검토 시트(sheet.html)와 match JSON을 만든다.
경로는 실행기 BASE_DIR(작업 폴더, simcompare 실행기와 같은 폴더) 기준 상대 경로 예시다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_single_file import build  # noqa: E402

DATA = "data"          # BASE_DIR 기준 상대 경로. "/data"로 쓰면 루트 아래 절대 경로가 된다
CANDIDATES = f"{DATA}/candidates_k3"


def _direction(name, query, index):
    return ("baseline_ncc", [
        "--all-ready",
        "--allow-unversioned-producer",
        "--query-export", f"{CANDIDATES}/{query}_export/review_rois.json",
        "--index-export", f"{CANDIDATES}/{index}_index_export/review_rois.json",
        "--window-um", "20",
        "--topk", "4",
        "--max-memory-mb", "8000",
        "--sheet-dir", f"{DATA}/dl2/{name}_sheet",
        "--out", f"{DATA}/dl2/{name}.json",
    ], f"{query} 질의 -> {index} index. --sheet-dir는 없거나 빈 폴더여야 한다")


HEADER = [
    "DL2 baseline_ncc (simcompare 4회차 뒤, 5회차 전). 두 방향을 한 번씩 돌린다.",
    "--window-um, --topk, --max-memory-mb는 기본값이 없는 필수 인자이며 값은 예시다. 메모리 추정이 넘치면 chunk로 나눠 처리하고,",
    "질의 하나도 들어가지 않을 때만 종료 코드 2로 멈춘다.",
    "--allow-unversioned-producer: ETX 실행기로 만든 simcompare export는 git checkout이 아니어서 producer_commit이 없다(사유는 export에 기록).",
    "끝나면 data/dl2/*_sheet/sheet.html을 보고 data/dl2/decisions.csv(헤더 candidate_id,decision,note)를 쓴 뒤 simcompare 5회차로 간다.",
]
JOBS = [_direction("ncc_r2t", "reference", "target"), _direction("ncc_t2r", "target", "reference")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "bundles/etx/psdl2_job_baseline_ncc.py"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, overwrite=a.overwrite, jobs=JOBS, header=HEADER,
                           log_file="runs/psdl2_baseline_ncc.log"), indent=2))


if __name__ == "__main__":
    main()
