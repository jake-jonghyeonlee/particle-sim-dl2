#!/usr/bin/env python3
"""psdl2(DL2) 평가 실행기 (자동 생성: scripts/build_etx_jobs.py). 소스 해시 5964b4a368cc7465c4f194af18c2d0ad8696efd57b19df2ae06778cb584c6bec.
baseline_ncc -> eval_position(--baseline, --search-batch) 를 한 번에 돈다. torch 가 필요 없다."""
# DL2 평가 실행기 (torch 불필요: numpy·Pillow 만). 후보 제품마다 baseline_ncc --all-ready(질의 export <-> 그 제품의 template export)를 돌린 뒤
# eval_position --baseline <제품별 파일 전부> 와 eval_position --search-batch <psdl2_job_db_search.py 의 봉투> 로 같은 정답에 대한 hit 를 낸다.
# 회차 위치(compare): 질의 = 변경점 ROI export (QUERY_EXPORT), 후보 = 제품별 template export (CANDIDATES), 정답 = roi_correspondence_v1 (CORRESPONDENCE).
# compare 의 새 "변경점" 형식은 미확정이다. 지금은 review_roi_v1 manual 행 + mask_derivation.json 으로 만든 export 를 가정한다(PLAN D2).
# simcompare 4·5회차용 psdl2_job_baseline_ncc.py 와는 별개다. 정답이 아직 없으면 EDIT 블록의 `JOBS = NCC_JOBS + EVAL_JOBS` 주석대로 평가 작업을 뺀다.
# 수치 기본값은 없다: FILL_ME 인 값을 정하지 않으면 실행 전에 멈춘다. NCC 점수와 코사인 점수는 서로 비교하거나 합쳐 다시 줄 세우지 않는다.
# 오류 줄은 [psdl2_job] ERROR <코드>: <영어 문구> | <한글 문구> 한 줄이다. 코드: edit_value_required, edit_value_invalid, input_missing,
# input_invalid, input_mismatch, stale_done_marker:<사유>. baseline_ncc·eval_position 은 완료 표지를 써서 다시 실행하면 skip completed 로 건너뛴다. 파일 만들기: scripts/build_etx_jobs.py.
# --allow-unversioned-producer: ETX 실행기로 만든 export 는 git checkout 이 아니어서 producer_commit 이 없다(사유는 export 에 기록).
# CORRESPONDENCE 를 바꾸면 eval 작업 표지 두 개(OUT_DIR 의 eval_baseline.json.psdl2_done.json, eval_search_batch.json.psdl2_done.json)를 지우고 다시 실행한다.
# BASE_DIR 은 쓸 수 있는 폴더의 절대 경로로 적는다. 종료 코드 0·3 은 다음 작업으로 넘어가고 그 밖이면 멈춘다. --sheet-dir 는 없거나 빈 폴더여야 한다.
# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 각 작업은 `python -m psdl2.<모듈> <인자...>`와 같다. 모듈 이름은 baseline_ncc, eval_position(이 둘은 torch 불필요),
# train, search, evaluate 등이다.
# 인자는 명령줄에서 쓰던 그대로 문자열 리스트로 적는다. 상대 경로는 BASE_DIR 기준이다(아래). wrapper가 이 파일을 다른 곳에
# 복사해 실행하면 "이 .py가 있는 폴더"가 달라지므로, 상대 경로를 쓸 때는 BASE_DIR를 절대 경로로 적는 것이 안전하다.
# 작업은 위에서부터 차례로 실행된다. 종료 코드 0(완료)과 3(부분 결과)은 다음 작업으로 넘어가고, 그 밖의 코드는 멈춘다.
FILL_ME = "FILL_ME"                           # 이 값이 남아 있으면 어떤 작업도 돌기 전에 edit_value_required 로 멈춘다
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
NCC_JOBS = [("baseline_ncc", ["--all-ready", "--allow-unversioned-producer",
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
LOG_FILE = 'runs/psdl2_eval.log'   # stdout/stderr를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                               # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
BASE_DIR = None                               # JOBS·LOG_FILE·WORK_DIR의 상대 경로 기준 폴더(절대 경로). None이면 이 .py가 있는 폴더
# ============================== EDIT ABOVE ==============================================

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path

CONTINUE_CODES = (0, 3)
DONE_SUFFIX = ".psdl2_done.json"
# 완료 표지는 이 모듈에만 쓴다. 표지는 인자에 적힌 파일만 비교하므로, 간접 입력(예: 데이터셋이 가리키는 export, 체크포인트가
# 읽는 파일)이 있는 모듈은 표지를 쓰지도 읽지도 않고 매번 실행한다.
MARKER_MODULES = ("baseline_ncc", "eval_position")
TORCH_IMPORT = {"error": None}   # 버전 줄을 만들 때 첫 torch import 예외(형식, 메시지)를 저장한다
# 작업(모듈)별 바깥 의존. 표에 없는 모듈은 numpy·Pillow 만 쓴다(patches, data 포함).
# 이 표는 src/psdl2 의 실제 import 를 확인해 만든다: train·index·make_split·evaluate 는 import 때, search 는 실행 때 torch 를 쓴다.
MODULE_NEEDS = {
    "baseline_ncc": (), "eval_position": (), "patches": (), "data": (),
    "train": ("torch",), "index": ("torch",), "search": ("torch",), "evaluate": ("torch",), "make_split": ("torch",),
}
# 작업별 산출물 옵션. 표에 없으면 --out 하나다. 폴더 값이면 그 안의 모든 파일이 산출물이다.
OUTPUT_OPTIONS = {"baseline_ncc": ("--out", "--sheet-dir")}
PAYLOAD_SHA256 = "5964b4a368cc7465c4f194af18c2d0ad8696efd57b19df2ae06778cb584c6bec"
PAYLOAD = (
    "eNrUfVlzIkuT5V9pu689NsMiVMWYzQMgsYNKILHkSxuQYk2WKkAI2ua/jx93j8jIJFGp7v16bOahrATkEouH7378P//a7f0g8z/+"
    "4z8Wm8XhP/7jv+/Of/3Pf/vPv/zRYUR//FVb1BavlXlukgmWw/5p9rrK12uVenq87uy9QWs2zgxn42x952WCI34fV8q5cbaXmlz2"
    "s/H6+2zcz688fJ+t0edeys/kz6NzceUNigE9g/61d/ScI95Tmu2mg6wXTDbt3TiTmw7OhXytVDs1l4/HVqnW+Ou//dtf+/kok7vH"
    "yHKTcTY9ucuPs7m76Sg7Tb1NvqXv7kbfRiP/2/04fz/OjLPTcXqUHqf9/PQue5dN0x+Z7/lxajrNpP763/Q4mfx4tH8LFpu3/9hM"
    "JkkL0Kv0DpPKx9yvvB7G/XJqmJnvxuvJbLguX7z+B/39OvPW+XOtWnynhUqN+vljrVoPhpl8xhvUc7XS/Dg+F/dev3wer4Ojd14d"
    "abIVLAp9txyVivthvx6MKx+BVyp++P3e+a1b7L6kVjO/Mg9qVZ/e0Xn3z8V3b1G8jPqdoFYpzprddHbUz63G2cnB76dnw1JxPlnn"
    "aXOeZ6NB51KrtIPxpnP2usXd5LRb+oPnmbepv4+7xdSo8jqja/deN78fDoJgsiji/3dc4/dztAF3vLn8nGrxPM500sPs82xI76tV"
    "0rQW7a3Xp8+b1ext3TuPz+kDzeVYWufeJ+v0fFwJLvx7tv4+ybYOtBbnybpH7+jwsxvl3dNzqrWvPXqVXmlFz/XnNJeLX9KxZcoH"
    "Gje99yNoZj7mb7QmtQr+z6fpGppj8D5e0HrMdnz9MFM+epVgRc8PnPF+q1V682FmRuucW00q+d1488zzknUtp3l/Ku35JINn5I9+"
    "tb4bbnopIe5cUJNxEiGXT7UKvaMg78PnyRnrTWPN9ogueqvmovhMa/CLvl94/fb7uE+/ER3QuJZ+pbey+wGCL3e2ND9a784Jaz6q"
    "lGmd+bsL/X3ynnezYbV48Sqnc3Pdywz7H1Ma/2XE65fDXp/pXavJ+URr9V3mQ/QzHNSPWHdz8Hi/6Lkjmo9Lr3QdXb9LexUfNPc+"
    "nu12tBNHvyIHncZ7HmOcj727Yb+dqlXbNNb6ksZ3Gg3mwbhKe1fJvYPOsVd0PnAm9LvOedgPjrSfG1qTIu1TMB4UU0TT75PZ7kJr"
    "THvTSYMevEEb88bve947oht+ZiV/xpo1S8WjN/BAnxs6F5dJhtf5aZylOdHzazRv7AHPD2vXT++ICZ1pnXLNwo7WyN9NsjQepTuc"
    "uck6uBATWtJ1u3H3rsHMZ+0RzaRmg3R+TXuCsREDKu5oXO+TDdP9kd5JdE90t2k1SuvgwGPkM+efaD0vtGdrOlOHWsU7Y83ABybO"
    "Mzqb+pzmjXE0aJwnYpwpGj/R3nw/6tec73aX6DU4b7OGnl16V2dL52tBa2XG8IzvnDFd3gaRMR6JP5xo/ec0nuOksNVnFejMED/r"
    "hmMcV/Jzr5wP3qr8Obxu4M3H5TyNp73jNViE99Ba4Z4ljZVou30i+iTayy3tepRyF+KZwbj/OLNzeqTzsenwOnUG9TPtC2gnPSY+"
    "OFkw/d35NM7XVLvWeU0Xaw+pWT0r52GQqdOa9WjNcsEgk1sO0/lMqztpvLz2XruP+Ue5dnccZtpTb+2lBsSDxpXeFMKJzvtU+dum"
    "tArar48fZbrvqfdYfu2+5p/43uvnL2h9l6My3nN3qZ8LsxqdcaxprUK8nvgCre1yBP5AfISFYignaF4e8Xef+DHJiEqdeSf9nfar"
    "RfAyOpPELzed3TDbIn6VC/w1eN0jrUFwoPVMDek9pdm2MSQ+SLx09uzwus7Ke6XP9O7X+9KCxlUqEDXXH3DuSU6wzKI1n2NceqaJ"
    "XxDNrcGPwPt6IqOUFz4tiK/QmrUWNPZ+eeN1V3s6j2tv83xu0HhoHkFjrf93i2vizyvwGx9yT3n3cJ2/k7EUK52AeFs/lyL5d5kM"
    "6qlGKMsydGZ2Q9BJdyX8cLCaEQ86eSxjIEOZH+o4inPINeHbDr+otrdDUiZw/biKsbbf+T56FnicjoP2Ktj74HPrNuQD0VhhK/IP"
    "PLKd8vvl/UDlDPGclAc6pbNh+XF3pby0HviDduA97JXnMq/ZEv+n895iPiTf54907z29bznIpOd05rA39IyPLMlZooXOzisJ76a9"
    "uPj9j5SRJ+NNj/ad1iZT3tOcZpD3TB/0uUm7WyNawLPoGQHpA9MBKV80BuJXc/D3deNFf+d/Rf6uGeRJptUPEyhV1UNeaYX/gda9"
    "RW5Kcm83rj7PftCcvcHztvG8tdfQupJ+MjzyNRXSCeictArXv48qQYrf8VBo6NoTfQxnXqZHe8/X0Hk4BG+07kqXe3c/B1nZz9gc"
    "WG6+daEQzn/R3uKeNMmcoJF0b3fl3rvzFuZ+oqVFbD0u7nsi67Fl3ae0p7k8RsdC83iabWfR+/S563p61P8guVz49aNb+HDX2V0n"
    "4mG4LtVcl1deaa7jW90ay1TObntMZyM12QQ+1n+QadOZIDoCnfbM/ppzyzxU9dbelM985XUX2Xco5KQv2DFlOyRL05d+pgc9KAcS"
    "cOiMZCn0i3YQ38ek90TXh2h21SEeXxCdqiLnrJ5tWZ5E30O3YN5H+2R0YOIn5RR4otfviH7UT6muREYInWHh6bUD6aEH/Rt6DvHe"
    "2U7PyvXa6zybG5JfxFhHmV4OZ94nOvK68/Ds9/Lm7O/o9yeMvRFdk4tP102Szx3JQugRd7O3s080G1zqly3J196xEaOBBq1hPWPo"
    "ZnIfP2vNUmEj5yr8jT9nUw2yI/h8/R1aMHtk5AOd++OklCPZVyZdvZwz/J7mvvKrwcl7SeP3NXQE/2GYMvxhRHzMZ75yfRaVB9To"
    "GQPQrL7rqGNo6L4ID6ffG4+zWeNc3NZKxF9o3D06R7XqhGydwod5nxfkdV70zkru1Fx7a7+Ui8mt0yxK88U1rff7WynxHsyLZEPv"
    "0ijNtzTWbIPlnvITfe/k8mjWCPKQeK6Ri97UrGFjYd4Tvr9P9gfG/7Sa7c1cngLf13FdWovImHYT/P9Q2443pHfR+JqbqPytlbYO"
    "7zQ8r/CT9IwpZC+NX+k2GI9KhQPxr/uu7BH/3UundN29+cS+Oz1/o+f7/dR2cqnRGGlceGa19VErfZ81KjPMLQtabZXuTnZNQh7C"
    "42jyOlu6wVqp3Me7ViG9Zl1akfNo5mdoJyqHV45+U9u6NqdLd3TPzq8SPyF7fIw9hM71iL0l/Qc66EJ0lPA8kO0+mC89st9I/vJv"
    "Zi9hKzCvYZuGxr8O5tAfXL7iPufq/ETH9cdnqLn57AzZc6z8KFjXaI9+LIo1njPRGf09EJ2Tvn+h80T6D9lRWXoG5PN9oiyQtVL6"
    "YF3A2QOiH3d863Qw7N9ZPcEvkw6WrR8wRui/I7P3pP+OM2ViVvMUaNcvrY6wmRul1c7On2jCuf+L4yNdrEp0QHsKWiFd/X2yDnni"
    "25p198NwUNgm6Rp0D9m67S3tCfN2+gzdIv+yhu7m0lu9T3ob6WvpwzjTsT6BUf9u1j+nPog/fwxe9jO5r5gdEY9X3dXVI2kvW+wU"
    "89ekMxLdDpXGbtDfUfd9AP0S/p4h6/k9sv0LW7LZHB2/R9Ko88BzMXoqXQ+5NqwUVB915APGxfJB/RfWHigvSSe+0Pe0Lh/7yfkO"
    "lK5rLWeU9Y5Pzq6h2djZDWnUPAd2vuzlV+gQ+zaVMy1nBM/Q9WF+jXER71TfidAUeF4PZyo2NsN3hUeeHHlCPDtbPhMdkm1SI9ov"
    "bH8nV4kvgqfvyBaCnin0BH/bWfQFsq/24ltiG+3IenBc/w1ljpFVbEvzOTIyYB0YWWX5K87SlYyodsZPy+0BspP/76UcGfFlmQT7"
    "/X3cD460x+8sI8L1J5nQ/phsOlPQsV7PfkPlf/e0Ru+TdJ7s50PQqLTfJ5uamdf7iPcDjt882aU5ol3iTw93s1b/9fCk+wYHcj+T"
    "/zVg+Y776+Nx5uBDDsFHNc6kyYYsp2iN+Ht9ttW3sCZkK+8mpTnGRetLZ3RZ2JPsO7ZKoQxi+6+n/s8gLz7FTYtsxzn7KEJ9jk7B"
    "Zn/TXqTriA7g80mD7tOwd59CHyXxn/avxiCF9fnxumAfLp/ziE9SHefw9Uwyhwv7JDf1nfGx4gzbefbLK+MfpjFlJ+IXZhk3qdbf"
    "vWpvSbau8eGwbc3+MfFHGz8j8SnMp3iiZ6x8+PjEho7ISbMW0NX5XScr99f0rIDtaeNDIFr36Dl4HnybMk/S36vwv7y+q37+Df4S"
    "uvZOfVPGNlff2cd8uO7tx/CXOv508Kqx2NS5UZ/eU20pT9ExV+bvOGdso2Q78P8G8I3SXHbw/1oeSDYjycAD2fAp2C/eoMB8HLTu"
    "lQzvrpNc67CPH3o2fQaPvUTkPvuQcpcmj7t8gL+Y9vfblOjjLav6VcWDvKWxrPgdZi3prOL5uYiNJNeKbVupw0ci76f7aF12zTWd"
    "4X7uSPMjHS+A/+foZVuw+4kXt7ZR20N5aaW847Nbmi9GLCtym/7ZZ596PZ1y3uOP68SfSJbCx6fxAN+P2P9i08hcmQfX30fdnHy+"
    "vg688DTsB6R7BMeang3ew/WzPSuT9IGvGZjn9PLsE/E29fm437v0s8U56dh+k/jOqO/vxkH+7q1H57SbcM2muHsj+htkSf+rtLcD"
    "0i/CvWmPeS2ClNodvenbYLarlR5pbWZX63Y936TxvR5N4GuQneci7+vmxEdeztP/7Dd0xvwaGcO1H8HIQRov0fOzxCTgNy4TPyAa"
    "mW3Gg97erwQn+DTduMcgY9Y1v6axTSP8bTHZXfkirvYB/Fa+u74WshtrXWOaNPR/a/zgwXReN6N+GzRF454rf7V7sVcaYr8p2V57"
    "GgfsZL8p8iRv3mH2mGwc+F3hw4FesfSI70KforU8N3v1H12jb2XrxDuvfSl0di5ewjkRHaN3ZD7jrsWt8a3LF9blu/PFOJPf67gk"
    "LsQyn2RWdeX4csskC218B3x7A57tZ+7gozwPB6u4PvuN12i5TTgrp1l8XZobb+cNJmwvEi89+g+zHftmEetgHsBrZtY/D18yydG1"
    "8GXHL77GOJ//lvwju/N9jFgI0Th0m161nhb7nMYjz6WxBHe0B3nldQlxuzbpWa/38H3LtSf2pxLfyoj+rfwZ/vbQ/px7xP85ME3r"
    "OyQ9JB7H8yscH5uTjInGE0vzJdaf5MkF77R89SxrNxo872wsqgrf7jPrJiSTIN/YzjRz8ypkZ9DaN/U8YDyiz5etXI2Mi2h1DBpA"
    "/GoRHRdiNBrzgl+Q/r+bvWIfN8Z3b+hM7Mxrud3Dni/pGpLFwcJnHQ4+3Y7RmfkZrl/5SelX6BH+6TLixqxrNPvyfD73g6j9686J"
    "aOYr8hAyJ2D/CMk1kQ89xKdIp4etMLd6i+7tLiYn+TxANxa5eM2vVDZMYf+wT8noZnot6c8rh1eTzRgQn8pN36rB1O9DLur1n8mU"
    "rtoavXzurRTxNzPtOLSEswcdgHTSOcm/HOSClceTM9N22vu/J8Mhm80ekSwiWiReMIRMKc2Zx9g59vLQR0ROVszake1yk4fWiR+R"
    "Pngd29vWM+n0uNrZwWcfs6mMjJ4ihyIaq6ttEsYfpevNzTlIDKNL697vrBrYC9I5zfOGm2CK++AvGm885CusQCtEq/swPuLy0BPp"
    "iL0D6abniezp1TiSdTHW6XAWoN/ek87P4wqfdRWr4OvJPkrJ2aiTbtEO/IfrNR+uyytjI7I/LEu2S0wv6l/avuN7EnmVoNcxvyP6"
    "8gYz0TkzEn8XW3V1f6UPyH2w9/bjjM88ZED3NCr1OXSkmPwnXSmUK0b/7WdkTwYRGbvfKE/d0PfzSbaTI3t1K/KF/fCsG2DuCfNY"
    "kH4IPjfG2Rd9hsbjxqmsPOoYPQR7E70v7tsMZZjjv9Q4Zjq/IJlwJHtE157secnzkDg2zzGfBe+wdEGyaFwZ4r0Vov2lxNPnpFPU"
    "tpHru6tZ81wwNvYSPHOYXSXeN0TMhvSg/qXos+/6McxBaFTJJlvAx5SD/2Lc6qWs322cncFHynzSyU/YImci5Jsx/bu3Pw0kJnI1"
    "jlu6cf9S9m1crir89NN3JvDe8H6jJxW26meEPxW68Rbr04Svrf+xRgxhnAX/go4YSEyoBHu8syV7Khj1T9tGpZ2rlfbw8+69ReGA"
    "tSB97t2MsdE9XZ2763fOYu+c/e6dd7F33kXfuUqiM9o/+OTmKn+IFnTP4TuSPDAf8svKS7qO6Kh2ot9JpvvTyfJxz7YsPavF8pd+"
    "z7Q+xKf3N+jZ8NJLga6R59Da3DUeCs57YM+miV/Or+Vmef8BfdaMw8YU4Kup1pKed/7N8072ednah+tbHiJ2Q8+jsUoO0FI+07ti"
    "/sfbdpedewa5G9BVe9PJoBdMSJboWm2u4yKy/vDHmXWy+jfZh5IvNr8gP072lM7XBv7OdnpShU+rfRlk5tmILr7yz/AXv3WLRzo3"
    "yEm8G/VJvnaLF38Afw7ypOhMLpGjEMDuxnPex1Uv4JwP0vv9c5r01R30MSenx/H3Imcl9Okhx+nCNkEpHY51MN9NLqHvHDT4o+sj"
    "JroYV5y1qARL4qssq/o29tSGz/MiMY/oXF3Z68okb/0xj8rBImh/ieeSrpVBzhvkPOkAqmNyDA++VOv7ZT70vI3atZy/M58PN63t"
    "BOeRxkLn8TvGyDl/pC/jt+GZf1vqb1nVZ084F5NKa18rfxfZqfNN0udYX8YY+6/uPE4kJ395/ecjjZHzXxqlOc8NuhjHikpKI70D"
    "fw/+ez2PD9L/5+ZZu9rDHftPffgjr/MVFtCvR6c4vRZJlh8CkpmN0lrySyZsA8EemEtcvpeHPrs3OUPEC8R2Pt+Qd6DZwOb9KB9z"
    "83mQj5NGQvBpnMH+ReIm7D+N+Dj5zK1mDSeuB3smktPp5KzCVzDKtkwcYxvPK22wL/5O74e/uYOcSKtvI4ZG5/Vs8l4HWZM3Wt77"
    "mXKOxpaWvMTQLlJ+gPPjrtlRv9exXPOGGC++4vmG36hPUPiy4xdL5M/whdMae4h9l/NWVgw5V7hzcXzXkbEa+eL4tyJyRvxn8zPL"
    "EsiWSgvzWbasnG4TLbMdODY6FedJa/wjZiMxfyG63cbm+LnfM6ab/kvtJRm/eyaNbWHWKebTSpNulR5Dppj1eFq3SMbF/KFX50Zy"
    "ijTm5cRE1e9BckLyFpi3gR8vIaM5XpqReEvDPjPus3L0aeIdrYedobf7IfFCjbXuruXWfkP7vx9XW5sn5MqpTOb9hR56edwhLyWk"
    "rck98Zr9iNcKMRf/Mqn0lsh5mWSLwZB4Q111OXz3h3rf/tr+8K3dRtfe4ZmR9aqmbuQaED291NJt4pM011OUR7Vrz/wb2Rjst7Nx"
    "DdKHiu+I68RiGfvkXESWGcy3IG8ROyAesrZxF8nZnzWqxb2PfEz1dRl90fBqb1BOIxcbOQgjfgbH2+33bi0CPfNEduiluWCd/CB8"
    "C7nUPbJ/03QW/DTRzOVpQXvRh1ybwwaDjwmxSuaj3aD942VhfHgmP1L4tM5XdQP2DRFP6mHOsIFxJiX3knMl2iwvJHaIdVjRWYdv"
    "qX2Bn8crrYw/bj7mezTWZXMnEQPbw0YH/xdaL7m6SYdsi97F5zNh86fZr95avmYakj9Ne9Y7TEiPpfMvfvjSZF8TW4j3Pvyezj3Z"
    "RM01rVUGsSvo7/T+hT8ZW7951Bfg5n2wzIeMLO83Quu+H/VZIUbPuUMaE5Y4eCTnXWUZ+6htHgn7UXZetkc2b1v3Ze7G1Yk/5xdv"
    "A8S6ZtHcPonz/sk76ey2U42E/JOonmHWPpee8Lxet6MNjXORWyEfmmSV8kLVxeVv0fNlTHvzXujlJv88Hi/nPDlZtx3pOkfkkyOm"
    "1yDaaZyLi3r641i/9p3AjzznuREd+pqzl6D7kFzwg0n2+Uj6MvuezbvisWSSNxvkYDfX8+DNzp3W2tHhaS+OyPFl2431WMjyiI/L"
    "xJZDXybqijQ2GcaBO8InJM5pciCXOPM4P3reA45BEW+F/xZ6itcnPRTnq5/LcT3Nwxa8BbGaCH9CjgLTrchb+HuNnL6p8zRNPQ7p"
    "ipMMyyo+B6Ir1bGGe+V34j+uiC6EvBONTR80d57PHHKyPT73/vwNvI++U7kvRWC6/tc6BewErG2R5DZyID8udWtL1JGXivWl6wOc"
    "D/iheRxqc5BufoAfLKqHJKyRyGd9T5bl/2YQ9eeBzlz91KFd+0zHf5dIB9f+tHAOob5RGW5iOQpkh0Dne75/E//euFVO55tZiQ3B"
    "tpl2k+J8wkfGm17ET54UO2R/qM2tTBq7jd/F1yTBJ5Xw/GT5Pqu9pEIeKnIwtu42P0jpkPhjFUWEz1tv4cfjrQfiZUJ3D7tvlh8N"
    "Uu923XrpfJR/fGqfYX5TN4dW85a3sXzWq/NmaCOBByG2cuEcgnX5NIGOUZqv6xFdxd8hr4Hk+hQ6EnI0SDcLnjZmv4u03/lvlscO"
    "Ujf95XSm017UXpRzYt892ybItqiNWe1dIjYq6YiiIyAv2dBjCrofdLs0ataMH/9J9V/QKPTG63xkjofjujF8M82u8AD4bejvJeg8"
    "SUcN6QFjcPhAyVdeB73U8AR6L2o3oKvKXHCd2lg8hxh/SecdPr+mNdzRWKfCh7FeH/Mxr1c+RfoY9v59Utnf4v17sUmJxtVnEtU9"
    "oZcdXD0RutIB+T/9Ve7hmflowdq88MuENXC2HnNXC+sIhf5Z1rVTNH74VxCDgV2V1RwX6IuoM5RxZYoz5b/fjA1GdsN9TX0RyGHi"
    "ukG+93nGdXX9YDOqRutKSb9c+vBHYf5ci1g+1ao+jfUD/DVla/wQR4VOXZq/tExdBvLn+u1fiCmrLgid9zBGYfDCxBFDfXyCOiLE"
    "z0hPldrRsuRQlcI4o/iQtODY1n/mT83+Xmt6Qv8T1+6W5sLPacye8Qto/Eh0C+P/ew311dLqK/JrT++7ll+im737lfL+t3mRoHnU"
    "IRL9wuapk248HHhzWtv5cE02LttZ9hzwuYN+AhvY8Bt6L/tA6bvjmPNjO7AXF4i7OTlpG+hs9HycE/c5LFfZli756r+kswP5WkjO"
    "BaknxfvAM3iePRNrWzr5rFPHf3L0BvNU44YsalR6JKfZJuPxej2yW2P5CTTW3YBjFqK7eF2bQ+LOx+h+Oif2lRj/bAq1n+MSn/XT"
    "6Iy8z/T8rTRD7u825FeI13yftRbso7a/Wx4W/h7GX6xOB/qI7W2mTLQdoDYcPBbrHe4h7Grwn8Dlgb4f5XFKb7qHsp/CZ+T+3nyU"
    "ZV6ZtPbY84Tv7X6xT4N4yQJ224hzm0nnSaBdkxsDPwTJVVM3txyS7Pf7LANWemYxP6l5Q25itTWDD5pzITYr0pPTO7/KeirvizmT"
    "Tc2JNLXAkv/M+r/kFNp9nMxobWnfI7Jd6b+3gj2I67heDznAEpffSC7QfgvZODyv7msmn13PbJ9rV9s+x3DUBxunLZMLqD7dqeqZ"
    "vBfTiFw2vLxD+yZ8eZD1SF8mfkz7KzFq9fUQ/4Rvxlt/vNM6bUUOfYx5nHhn6bvWG8XpYMW2V6tU+E7n61x7sDLM5KZBjss6PhQk"
    "bnKGz1vm6/rSjb2mOfOf5NIl59exTR3Kmr2VNXpGlb5Cf98gFc+VsvvnyOoEm0Ns/maMzzh5nHTGi79sTQDs1Wr9fbjuXX68il/O"
    "td1CmVx0ZXayDxu5tzF8g2jeU+g/N1gCT2fDh1CzXVDagw3dhq0XqW0TPUT41Z/YonqOlbfR/W7OvcY21N8ZsXcc3qk6VszfoTEX"
    "plWnBiNiD2Rj+XY98RE4uiePG7yR3s/XRHVRlZeI4cmaSA2J0DHzI63HEN+qub60N9fH6WjqrVFHSzZJuH50bu72lu4jNJJIQ4Ze"
    "Xf9ARO+K+mlPYvOUThHbPrpONo7h1OsNw+vLjswguY46nyud4HKzLpHrF/0B+3g3PvLx12XgJ3D8DrIkol9HeFOoY6hcOcMWIj7N"
    "Ojx/l3H0tHReeCFqIJ2xi6zy/amNVVzbjKaeyNyTdE3oC7DjuI47VEl+rfeWx9A/1R01P7sP/IHeWfxBET4sMhM6z8NjNB+tivoJ"
    "ftfU1HFjbyV2g3yLWjSewTRyzYdsrL0KWdKK+vKqddjk57duaPPS3tG8c79Yjq/LxxHrfXLu8J2RtU3GnbnbJq251J3GY0isy0Gm"
    "5sy+Cv11wJ+vdIKw3uD0W95fz4T5dxhjon8gkqMXtROhe5FOF3gbOkvVeWgvsry7YR/+gT4VOVMxnSrcA7ETlcbwjJDnV9tTq3tF"
    "6Efl7iCMSei7NH9gFaeRWHypddumjH5vazlozD9j8bqIfYp5xuWa8CKRI0ZuXvErHafD8yIxwbj/lO4LSKbtfeKD4+zk1lmY2vUw"
    "sWbEAZU3JsqNc6iLa23VLn6mQtp0an42vaOLH2LnlzH1KMZenE9tvhrxfbovNQLmTWm7t3hMaY258jhMHJKxVhY4924sFnPVPIuM"
    "95J+pecFkdpHySdn+aJ+I+TMMXaG+BV572Y/SoWPKP/5JOcyq/UzpD8Psrj/AB5FtBVgfcP8jNieam5tYm2UzXcxOeJJfNvukeTQ"
    "X2Ov5C4mHzNcd1MjmEzPilvBcWnIUejq5jfmJyT7lMcTnbKOIvZ/iGWj+cSBa78wb0yWt50L6fNzYDh8fb3biCtj3ZLpkHWBcnrE"
    "+qiNK92gWa7LprE/A4ciQP5e4nXV1s7NaUUuOvKu+XxtIIfYR7fTNbrX9RG92+RfX52BnNSYObo2n+FqQiy/Gtoi8iwTS5ibMxOT"
    "l3g/52XyPc21H/glm1ey86K51AvgTbAeefY1HgL+68wJ8i9iP8Togn5XPRD7LbVYV7ohx8xpD/1xXN4Nuqev2DQbm3MuMtKJuQk9"
    "xuNbbu56RObd8KHEYt8ml9PkrH/pncBko7VOzO1oDNIRTJiR1EfubvmjYnTh+oR/Ntbgr6+cA/BH+iXkriMfUM/NZwa0SLQxDuxZ"
    "/mOds+/oB7/1jVc757fk2IPi9Lh62PYn46qhFnixM3an0ur8Zp7ULZl2Q/ePypyXdJJcz4scCqLzJB5Le30iu/NB8tXLS/+xd55w"
    "Hg6wDej3y5/PlXhdaggZj30r+SQfPtet+lmOuwdvpNv+k72L6HYZ6FfpeQgaeYdnvTyn5uWX10eM4eABmynDZ7Ld6XVqLynmEe76"
    "4bevy2weO6837hN78PrMbm7IlN/p0LQGOdEJ18jbhB7J/iRjQ16S8pFjexE5Y05+47nv2HvwaZFs20R9xyoXpF7MeafyyYR9TsjL"
    "sfeJbWLoCHMLmOdAr8M4QPOilxN/CfKKj9lDPo+t+41hvYEfRDD+oF96mQCYgoLNF+TfSV+h39q/6hF8HG9N90r9ftrkTGgdv8X5"
    "+Q2W0aK2aDwiz5NstPLQ+PQ4HhPx51QMXqXExa+wxFAvh9wWA6CqNYOI+yHXVfFfZsAr1PzIuVdhvyfxy1BG2Hs5HsO5FiuPfalS"
    "F9Ysae6OvU7ypZp9rqkPxroH4nstIH99b3yb7eXz+zCTx1oEbw+P56dofOM3WCSCKRDBDjFrbHOCdaxfxPxx8lNlLTjXu/Czxjl5"
    "nRXsrFYvpf51up/sdOAjtF6G+6t7ws8f/PlhBp9ADe8flcJ7G2WMoZiFz6jVXRF/L9xZv06F6Bq4Emy7CqbfIJMOxv08+D70ZaZ5"
    "D76AQfGi9hjyuAKiQ6G/S+qkPgjSd4MUyTj7/XVscM9nSWr2niP0o9ifgl3Xr+eafUN/Mq5mSXOomDbovKyRv020sVnNgB8oMS2m"
    "vzRjblp8uSLXmpJdC7zRb/y31DPdc+1kiDFoMXOS4uAWf2IAWUWyDnintjbvU8y9b0pvs2HFJ/2sd0KMUvNjOF9JfK/hMyfwmw5q"
    "UXwf3oPCsfZo6yvPPmr+pYZyDZ6k/lutxxRcPsQeaD8vghv4anLgULvr4qeYnLg97KwnrWvlddVcEIPnpGu043hFBKtqZ7AWWTdt"
    "LXYPtL405w/k3ZOO1bH2E+NZZiSncFgp7MwZV6ww5AIaXATwnxOtQU5zAYE1+s7YfZANwJp44FpZOeNM94LlOWZ/a4dxHZ16UOJl"
    "hYzlIYP27m0d4uAabEWVB9thpWhsR4nV0F63S5jXHv93+f/Frs9zWBR/jFmWFgw96/M+ctHcHKUBU1dbbZ+GXKvGeZoYt7xz0wIt"
    "BBNaG+OPINplPFrEesfdMJfc1O9yjLo8dOLYXEO7JluNffbjs/2O3gG6U7pdFF/j/vymzQ+XPEZj87q1t4yzGj2P9HzExTXPVGQH"
    "+wCkJqTDmJO0rlEewuehrPdznlZm1O9laX9INwKeSJvWrKj7h1i/7B/LzpCOXHyRE+PpRep6awfUStace3luZA823Di+U9ftZ0Br"
    "r+xvaXTd3E3NpWe/vdYY8vklPSPMe+aaD9ixzHMetjNjs460/t0vsY4SKFYm5yGOs3W6r7YzvmQa172pQzbPSdYDZj817nUxdfPy"
    "DpOziNxb2LYyzuZa8kkaJetLuJA8E39H2eLRyHiFJ2GssFnTtGaxfHWL6XyJrslB/ReHXyZuPZLnMJbg9Brjkn0Z7vMa3cK/A+8t"
    "AYOSzk0ulvsT3oe8Ikf35Th78wV5O0R7gx2wCN4na/iMD5FrWuW0P+jtz4MQx4jPfpPPZZ3zKBk3lNbVYIdKPcAqlAtXfnihl1hN"
    "7QM9s0v/+rJm2Of2mPesHMlLY57M18DGrQ71fd6O7IWUjIv/z6jvgtdFdGa5pvZwytP4OAZKPPjiR/0TlrdxnRrHc/d78NHGoqi8"
    "jvSLco30G+KdpFP0oJtUPnb+OorFCX02zGlG7sDcrhnpDmfc+2zy1Whsb6o7TeCvHnSmRh9onBP0idO17YT6Yzov8Pm4egzzA5sb"
    "AD3MvIvXxNTBiD/BWyNvne0pO1aTc6HnTM8X2zFGljn4Ey4v4pj7epzJp9jXvnmO5Q/aGtyoPk4yb4I6gnXb5P1ovviJcUEmsA2A"
    "L4Da4IFgApm1Ai6E1tuF9T+OXoccJNERSCZkTgarg59bi7xXfIGjfj3NeFZV1GvvXZ4a8LV91m2geyJHK4ueAHaP+zz2EL+E7b8i"
    "8fCPlc2dEh1F5Y/4tCZng58ELNRIvTVowdRP7+LYRbAx2a9GvLtRSe1Ibzko/z4k5K3aWovmxuQizEI+G1lnqecbd+eH5jqnay78"
    "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"
    "mW7Zl34HrDCG6PWNbluxxzwHbuct2zilwKdYWlXUrGyF/tvx3Y5gbAUb7IQZuLE2/PXjZHvx/Fdz8W+eprSfPnqlKJnBjtmN7+yJ"
    "xImchbUqttb/OcBSovwFvxsm6UeEFNP1OaxUc+wYwMbx9AHFHJ+p/CSbwBXcPnyvZlFnv2zW8zQb6fobxl7GGRP8d4gq8Bps0djj"
    "zl6Vn6VKZHyBc4XOclWuM/o8dDqM7e+lI/vA3m0eawtOPTsVCituA6JInukJPvrUGbfan1E0/IbJN4bW0e/MdFrY80O6LNoxJn13"
    "XTtVzAt9NrXd+baKqrx3W9omNJi1yEVb8d7TvuOP5Lgcv/JYuxt555FiZHbWDs+jQgPQWEuRu5bJb36cD5OsZb6925pqz1SYYhmZ"
    "aFkH5o+L2B5YWd97O6pxbGbzD1gXtMd/WabF+mefFe90N7xZLxmwOg+4u7wJO78yOaN2d2oU7qxd4ex7rbJy1XFZHU3YKa8u729u"
    "T+jqlfie4qFhL9zFpdEamX7+6qudGwNxhmf/HFTWQ+Rv8Pz07NvHZnECNQ5+RthgqcIjJrNoT8WMz35ASf4Tdqh1kc6JrRe/uPOt"
    "4hjvnfm2Tq1H6dh31nY/uZ6TKiYmljiEat2xSoOXO0G3P/lSGXkXdajKbZktjdYrcnwRzvDW8wXiB+mW/pJxzQax78rf5f0g/8ix"
    "f5ryg50X7j5394JiSuK5L/Tluaa/pdrqo1f0HHpzO7TPxGxiYIzVPnwSdeRcj+z9LygXHKvmyu9kulZ+CDPVe2uB1Z9cG+H7q07M"
    "xygMtW8lxlTvXmKO54CC8532baaW4QP2Wn/P8zUS4zJ/E1+QVTRU/hRoJ51f08gHKCBJLq1f4Up5Lcwez0i4mA/E6hDfzUvCH6vm"
    "yjmg5UQtT6t6KeVlnG2pLPeRsgn1pY539HkoVXLYZbITN2W2E97vc90l+TUZtiflIuziL//v/Psx/Rf7vefnYb77MXUI43s5eepQ"
    "NwvUq8kucX55oefjZwOKdLH5uDNqs0Waz00/W7TryWemmYG1xfjBMYXevo0Rv3tHf3q8iTVwjPJKdzbdgSmbc/Q3AfYtJxZ016Lq"
    "5vJtcOy9UYxw3gs9p1zTPUf4vEAnUa9U4N/XXF+/GWQyw/kU9LmMb6bQ6eY9HssownmyeiCGZAUXd54Dn1E5+3Lc5wx8183xZLPR"
    "t/MkrDZAYrpzp1h17P40KKdEV6b7jKjOCWOP/l5SYYhz9e790Qnt+DKvaV0h7HPHzyzf/zQxQGw+U2KB2HqIsTDpeNf/XMOyDrD/"
    "fSLTTtw/jcfqibEH1iKjw7guYny/Oavnmjk9SVkjdj7gHTIz+JLZwvMGAaJjJ/YlnPvvjQq2s25hz/RadTv6ho1DqFrcU9XQXBuC"
    "NZTWVXPA7rhrBbEGnxUBVDP7oOz/hddYEIUes6sJPxbv6iXL5xzd2/+9dNJYv/jGfA71SGESdJ6lLeu36P7uWyajij0Hf7acM7MU"
    "64wOvA/1LAtmQimtiuyz35T7sjaiwFy4Pj38s+sZxQLoQF94Nb0U2yFMnrjnKrGu20d0cpsxkL/grkvHNqS+h4Tf5qFXHRX2Cte1"
    "snJWcs5hp1HSpqskr1AfyL9rPylLPudLJuYrGdbbDdap+G7u9fU9yf6g5tl6VowIoqwciJl0zLXms1PYLhylNPZtoAaOzoqJn3/O"
    "kl2ryzOJH2g+V2tcfT7srnYxBjyyTWUZd76KriXyZdZrslcTeo7IURtzrzO9v7nXqJZF8nuvowfy6ek8+2WfuU5n93TSpWuRPzSy"
    "47g2qBxVM9moGGOp/Dry284W0r3EPrauSaq5hW07k9+DjaDN3SjkN39NjR+tUNv3JXrG93Q/OobSj+X7pnOol3cl5+ki19MQyXNn"
    "3w0lF875E+MH0976pLEL08W7dx5TrNyZa1sBdDyNafiWWxelzlPcUoyu9vPliGNR8nW9jvnUeCE7pTU2DNo3Ht8T/X+c8OfR2f+w"
    "k3cRWq/KH0rs2WPHRN8Zk9+cGkfodTpM5GtTlZpcxC+/g3heRs2lxYpAOYLjp1XAB3BzWWBhcdbZzp//anwd7pxzd65ilYb/HeWf"
    "65zTO5SkOH5gRDqdL8k1RjapuaZzmGJs9d6ktoLn5fgYXQk6nzjYJee+io66CjNLTfFsGjVddev/gmLLScwP1q3xkt+Tk6sSBmia"
    "d9hDdD2S70NjByNnhnycT/J1drRWflXzYGxbfNA+XD80/vh/oz4FE1FlsnbwO4yRuS+NtSoVz6+Pb8giJ71EzpzOjo/SO7rVgQVq"
    "bvRZJ9ggyXs8vF9DKQqYl6XGT0l9H9iwq4W991bjhlYu9qumMUh3j4xJomuPI5OzOkPXp65F4/p/Ra36XPBVzR3GpuqfLkbpmZUA"
    "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"
    "aPpvzuqfbtadC+cRmQX6/lCWTK88//xf2lMpay8g879CJRX5ju6emj41buhvHdOP9xh3Ydh5oDNyIhnC9mIlN/wsf7M/fiM/+Qxb"
    "vXdVR6L6cn1hXaPMcBHZ6zzD/jt9GqaAXvGmz7r/qrS42z9+38Mmhk/m7W6f8wa1i9bzeEnnIaa3PCAo+us5KvQWgprMejOeXUar"
    "w+n3YaSXHCoKaipvOfY13VxAjAnpEs1wOb61raLvJJFxq6XxG4vyY1FYIG9wX44rMtJULxyUv1LV1ZuPHBuvrB2Y8jzym+94rNi/"
    "fjIIcav3PldAWNtw456PgM+2QLwFCEjMVkn2+de2m1rcqh9i4sQIqttsuSRa/qDqOOBfG597wFUnrEekmjGXTUNGcbUIfDZBHgVT"
    "4mFMcyXsHeTHScWQQR3iWGcJiL9AXFKdig5YJMPkfO+kY75oZejO+sIn4wPS+Drw/RjJW9nDUbEh6Lo05g3iLiS3doK2gcrfPPlT"
    "iAU7PpXu5cp3vkWx6oHdA5nap1QNqFMco3DsVkbrobmddiPxmf20m0NMWn029q8QB8nPGRmEu10C/qj42BF/gfc+V+MOUiQDltE9"
    "8OTbfDaT7XTQwme/khiKoo6J7/3NmfhXyE3+OfQTKmxTmi4+lwlWu+jSjKyH6pDuPA7Z7/HAHU7k4yOrL7GSJe2ZBcdbKoWS+/3c"
    "+d55llSbcDc0GGREx2pmndF/kpxXWC80cRjobH3QubcM7wYZahATJ4E/bGxcE/fosn0PH/+PoGQnv8gfXiZ77a80nRnytVrdt8RX"
    "OC7BeRmJ7eMcpz+f2jTPyTI6gFVeodL7zcTiEgMfuQqsNHoOFxKr4M4rZPrRBYAzzFUFglpR8JOLPEbaU4hjc/e02FuW2ayXu5nW"
    "dB4G3M0sOQJztpFH8hlc+Z3IV1abmuaGfGxmFaHzDPtHOn2B5KcxP5prVMDPe9wpjrwOx8PM+sKmpb34Z/s0z76CKWrA66HIw6j0"
    "QY5MYp3TbhGVJ8vVuD7dYi3E5mMWxxUSiYaRFXESX7ZhHC7SHKO/mjyNzC+jNpucCuIxtJcLfpytp3Jk1E6z3Ygxkm84E7bF5YLZ"
    "hzu1NOaL1pRszSquYTY7nf+ZSRz3OdYtMUrdd/RMFD0gWYqqu/Kp326ZXNUZdqT6yW6cc49KI+0qFqQ93hsP89m2NAUSHNm6y9Gc"
    "44x8JoGERbIlxc+ie/M+5vPjx4ECyXiDMFkXW7byExlUAFNUIP7KvozvU1tfzD2zP/SVoM+A0vc2rWHemjRfj4KgdNFvM/G3cBI+"
    "N3snWebEamWuuinYtNBVabsvhZXSQyXogOe4lYiXmUCAvmT7q/ywCfgoqp/DcGuq64SxQXKpPhJ+TKKc5QqjDs+Hh7Eg/2Se1tB/"
    "0XiI6iYHNZARRi1KYMsr2xgT7EGSYQ/NRSvbyoHlUq6ptWYPDefvxiJTaCS+nhvecFd6yH5W398SF8YfielIvJcRLBesL+fi85VT"
    "ozn8MOhyg6B5yZYW/9DY6T3dc8hx1IAEegEJ010/WV+eY7ah712fe9HrdFE9fOit/hzsebrkWxm0T1MRyZWPj9t4PesgSOaOdv2H"
    "qfK7+qqhs1I7G1vUnVf2R5KKXmJsptSztS3oeahOBlLiWf0oRmkQ2SI6JOoviIxxGbgHQAxu7Xm/deLQbtreJxDUZDwxsZZCaxea"
    "63DVrdHbP0Do8nWBkftGPrwBLa2pfgd3cl06a1I1z0iB/F7N2PMVK+tCY1bd/J/Ely4igfp5od4erC6Sa4zaVuVTYi3rK1WNMWyN"
    "iJvK7xubQ7XQy5SP1neRTqeW+i7B+PZOu4y3XMWZ2+xs/YgiwCFf7ed02c5ipHvoMmb/I3sKfgyf96KpUEdVbbCT/innoPEXLUsI"
    "bB5mtO4kR7NhzonvFcH2U7aoMPR7qYNpZUivZQyr/OZ57j2WCoomS+uvdqHmtjzLuNFO7aXuhN5Dakpi35HvPWklov6C5uueVlxX"
    "oT5BANGEbVRXdxs7lHO1K86jGlaWgF7vrpdaRcvo9gn4Scgt9OrWTjxxvg7X1YN+o7XfUszsZBlCaB9M0f3joxqOmEEcrJUyfmb9"
    "JD+6hQpsvwLedBS5MRutu9EYvWV08e2rCFvWEmxZ2DuVgiKwWcSmhPN55lR64GK/Hb+Hy2RSd7tUlVHqNPxTatZe37xar/nYnHK9"
    "0kN6OHj/ont6uxLP8zBTWu0/S+83f7jT4+HmD8k56F0wR6P7MvFaB6JDd/+SK/3zWqyRDv5avh6S+fVLsvn+clh9nh4+treN3aGf"
    "mN5+VO7vuv1p6eX0RN9XDg/HT5I9CbLJ/ymfYirQC931K9lpYxRyrveNVn75XOsk/oBxij97P36a9yJ/YzFu3HCR32th+vWbd+g/"
    "PF55/v0/d0+5TOklmf14+Xuzy2XveK5yaeQqZ0Op2D9VAmd9sZ3UH3L7dmJHz1nPX06j1cvh6VB/2E1virt16W06TvLeJ9u3+g4m"
    "LkUmpr2X/ajUl5nbj9zwaX4fFzdb31Vyb3eleff25m+T7j9djg9TerfzZ+5hv+idd/P0ebf6u6YN2/j4O7if3pX7idt0f0FjGMxe"
    "Duf9y+F1lnvYrbcPH+9puu7p/LHpHad3T4XHu+7L6I4E6e1H4Xh7U+mWXg4n+s3qg8Yxpd8sux+7+UfnY37z/jG/fTvEd1ZPPnbl"
    "94/dXXN615u37kov2bvymN65PjTvvHg5lVYvf490z7+7Ab1Lb0prk+vdlZP3/nvy97vl7fljfXvE90N6n9zq5dT9oM/n5SmN7bZ0"
    "Vz79ub05lOi+Ixrb5wHv+nJY0nzQOzT+7m5pDkqV3V3575auf6P70DP+njd0zZ72wccYe+HU3eZuaG/d/1MOMgkxaj6zEBufDSxg"
    "7TQ6KRjho+Kf6R0Y3IOxM8TqE5qPy98pasXt80NtEczFublMQasAqop8Fup+Ss4+R8kbRhwK70XVsS6K5o5RnzlGzqwAJDtnel+X"
    "PYptC8TYIBMZdaPzFkKbDjK6rcrnr83z+91ue/PO6xfZ03YcZe5o77ff5h3a26+5DP3u5g/psLmc/WA3jcT6fXYxek5iyNdP/7mr"
    "1Gt3pQPJvnecry3dK9F/0O9K8yHthUeg95kzEvheP6N9cvwLfTc+DWcvldLi/QlnMnivXmFIzxm6nzn705ct/5xGOEuLu4eP+faN"
    "njum/Zi0z30i2fA6H4bY8RCXdxhscsM/oyLLzwXJOcxPosfPfPuH9B/usXh9/6LP0neTBmTmG19LviSKe88kg3k9S+/B79176Xsc"
    "TM1WJdJlk2W5/7RqkU7LH3v1KcvSp5WgAvfmo7vR+3ZNezbIhsdnYhmL6AP2s1I+sxowYhP0lOwH93P7+5Rru8R3JHE33ikt3bnc"
    "dTQ6gBW5kt+vyv+onKW12D2RnL89Je56fze01t3p+NT5qGfvzLNKa0VWAsoU2f+M2ldY3nD969LLDItT6G6cB7bPxqT7h50W2R/Y"
    "w+kk1x8yuyPXuPp6HkhUdUFTJs/50F1Xjy/ceR46F9x1ODz0VxmgD/nPSCK2BZY6RrdgpC3W67nh3kdrubyHaI25Riy83/pciG/n"
    "2e+yFKSC2QixJDmL63j0/eB5JB/6z/Dhztepd/0SZCri/WF5VJofp6W5rpkjDwxLYmmZ2FZYp9+H5MC1XIz6/N4etjrHtyetjNZ2"
    "p1E/yV3GppbSH7tZ9yr5GPc+gpvN36U/OqZDbf2s9Rv67jijrowpytmiM05nhFH0keveg8Ed90H3vuqPvXS7Wplu0bv5GvidD8+H"
    "50aJdFkyQTKKbPAuUGC4XpVkTffwkuyt7r+VxYluu7zrNW6MLNm8HuyZ2M965/2ydP7YfvzBPtmQbAiNt8rjYTlLv/9OJmMPsJxB"
    "A8Qpex6unjOldexazJE7GT0k5kY2BuTnid5VZFSow1nmzkEAtOfEtcNj9qv47BzrGd1xt1fHmww6e7lXfjcHAmn/VB69vN+fesXE"
    "YVhIr8tR397sMekuc/aTPh8yL2YfXTifbEuyHN8O2kOOX5DcvmN93Un4c+Tvr8j7kk25X900yKZ5m8P2xfx/rzvpOtkPJ9EhN6sB"
    "7Zbhw070yjwOfVv7Hy7v25j43uxzoAiDtBfn2EdAW0KXNMmW8/CcmLM/SusdJ5vNnJZWF2Uy7P5jt33zT1mQ7c38rK/K5Ljcc+T8"
    "TGVtDmqXgqW1gTGQbVMu7uZke45TntWtJBNWpKv2YLO9EKeY9FYex0sUPdL6ieiSDcvIgF6jfTJOdW97KcQpXLRj7xoSJeLom0Hy"
    "axlisrm8H+jMqbwIIANFdIrUTuN6tg9el18HH0lX/HSMT+Ihab5GumDvmUnK2A967y3ZF+iCXcC+sGwdb6HcoMbsmNn2Sp2vyT3Y"
    "Zr4LjEOM7CrIZRPN7W6cuGBM7o/2UCFSH21iittcIFZuGhyXu0E+g7zVCTXuL3j+KSZXVhxxP8wgxSiZnKcedIC6ovfIZee9up+D"
    "8GuFOcaI2lDphWmzbf/J3b+FPO0JRn2Y0jWzwVrqCmNiyZMnU9OYz3BsW+bSIt5eyLn+sBZOapcWbkxN4oiZBBB0FC32wPVZgTh0"
    "b8HnotDa9XL/tRomP3+Xz4C5RlCKFf2I5CuPeRJCLfo295v7aQ46tkaQZGHm3KvPuM4uuEd/nq+l/UiffU0QSxuu7L4Hmx86wNOX"
    "a0Y0Xm6R+kw3tbKySC0k2SR/pmYuNUa3zk3/b6hDOnPr3d2lvERm8CeZGicnd+Pb5GDoeYnR7cD7Mx5k/mRSwwx93R8PbkaDiTcZ"
    "3oxHt5P+TfouNZy4DYK74Wy86se1BtYA+oNSZ5R7J1ufCD/BBZdSLjr+SbSvCJgsl1wVlQyloOTt6Gdtf83GDGKYrTcSXE6PXuI9"
    "wHWkNMG0XGUmvTWn8LmZ3G/3khJ3JXo/58z1nLbIi3oTIBPT0sVkPNpX7LSNedwKpmVzvP1obHPndwYI7g3v0EJZeoEm0ZMWstyy"
    "+lB/bFXfWj6hJUQkgOu4bZJBjZYAiVvnFsvHWjPdrLdqZF30yrXm12OTSXqnYZJYAKAsyDUB8I5LMMTAKD4BztvaEMyCPGjoKQh3"
    "koHDr2x7BY6S3m3zPCU9XhqAFR9IWkDIDQiPv675jAVDUiJCmfdTTLldELAg0DZUb6c5ZOy3LVjQTU6jofVnkATAXDkp7Y105JJc"
    "WrC3bYlIfa9q2x63j0rY2gLKWgIDeQeAsYRAyk3a9tYpZwmSU9nfZueDZGbnEjSFyJw0zQFyd4DFcMifgSLd38C06ufS9AyINSbL"
    "C6VHvgcpQnmFiCeEvcpLEa+x4Pb+WLW1VgD+yTSkvRokMblOCmLv08ooCHx54gCdxrSbKWCgzp9Tckbm0sz/PFD6FkMO5s//mUyl"
    "re3TJhEs85YGMeF2GCaiugp+BnUoJUfY51wO0QYY0zdj0nfxCUJn6d+P4x5y5oz9QeKe5GnT7A8mCtH1EUKxh9KP98SwMEvT80DG"
    "yabo9XHUvGHhC6B5sn/ms6QlihdgLrMnfZLRZH5Fay/mmZS17dxym+/H718bs5ZuCxTIsdad/HEtYVL/dz8npGgd9H0FjE7B6XyA"
    "0Huf/Mwdg0s8ijPqZZwWtam8bz7r0Tx2EDoGSLiShDnvvzj0aD+O1Nxyy5aZVI1cnf/eOKR90hkL/g6Sg/lrL8Rl0gqzkTbI+0+a"
    "i9jro+0x5prWAtebOeukap/muRw+kXI7EMRhD+6EdKdF5tH/4ZDs8+kmXTmGSC2DOuLHe35g2sU8Mp+XGXXh+O/wfMWUx/ugWc7c"
    "8HtLmdNv5Im3peexSQvyaw0t8t8G0C8EbO2XH/vPxlm751KatoSSKyCTSG2Xg3dDKGv1bIEB3u2Z83U2k8MXSza9yXaXCQM3Nipz"
    "WgPVTTsjpzvW7WM3H+4+g5mRS3trrtF3J/cyf6Q9mR4/+PfrSzskQPZHYYwUkGBIedIyQbZcLIaHEkgqzoY/nqArmJn3Uc6aS086"
    "WF+k3YvlSedkME3u18ZGsED3ZMOMycVUXBJy0ZYIURh3ffvb+8fMD8rrLTGKPkdCHxwCYUJqLjkwZCB0zhQYPAYES0C8FKNkRvu5"
    "NRl3pkYfsEwRYh8hBRvNY8nKkqWHI11/HyT7MdgsZv1yvwQNjSUBK0/Me8bpnB/Pqz9WhpMYkbsIgkz+fo2SiSzJmuUNtx3LPrwX"
    "m75KvgPZ7sk3FysmBEbLwF6nbmcYxbch2U2fG2BV0sHpNblyAPzcjui86u/M9zOUqINMi4EW0U5B8zWyALf2vMFWTopdwTayEoRO"
    "L9jmI0OaR2vC/0fqeUJ7YUXn5p3mYNdJthYAFIxrpbomI2kcZ5rLo5Q21CaGPIRD+SgdLAiBaSwWTaG6A+5TtxMhmJFxA2g5fN6c"
    "7+LOCpPWrPIpCVOZfcigv+76mfOTGHAK38Esks9vxiF7mWEIbHoCpJRZnrenNs8b/EzAZcAHRUnKlNxrhp6gMXIoG5+h9adUWEwr"
    "p/vj03tz65bKh1x0sz8/kfIakl9hcYWiY1Yf+M8OdgfO4Xi1PJozHp4/GiONbeaOcxuaG5BZYV7kGhfPqbgkn4z8oZj5idh+uRlg"
    "CqYoY386KT6RwnF0UtktwptCcidEGLi/2GNbsrWNnA7pl7rK2oAtSu8vduRPddsDSpXeFEDanNcBQGWXkJ/bpZmrF7MOrf0KQNa+"
    "z2VbkgEfQtdXt2OyQVTm35o5aCctQYD83pT3KTYYnZNFHyXiLeA2LeFnOS0v1cTQvBv5ocMiiEP8eXuZ6z1Rdqrht9eg/eH7bHm9"
    "//rZf7cUSlAEH47THbnp9t/oVdl7++W4rsCrQtoyYYiWllk3ljlxcxUvb/BbBuQk/6cwUnludJ+UTMWVrrqkdOx/5gI+B6/tvyM/"
    "dPRSypb+6F7xiTLiUpqDZPkvwsFPywyTv3CL+tLBc+O5O+qYI0DvKP/fjEBYWbjRub6hcz6MnvOY/Wux3dajbQ/6M7Ami98SJMv6"
    "GeJLA4W1zDD5UgxIfQDT7MpedwkqEXNaApoINr0BAZf213waMSEhvFaCXQ3zs78rbUuzcZve18skn+vD347PIXfNBHSa4uDtAXPB"
    "gML6HJCWIl45jiX/C4FGFyA7ufxfSL/EVoo7o3RWZlyC69spv16rmPtWZZ7yl9/VzDFj3UnsYKN2c9h/133l730799F9wHEH7P1K"
    "wV4j9oqQcZpx/3aMixHDg1XRquHYhmavlH5vI0buUWbiDvPvS1h/3+4PS8qs96cz0Ldt3e6+Vng23i86V78EGO/ZGHptEjeey/LK"
    "PbOWAMLxL75Pf3wr63IjLeOtfeffxF0XSxJrbZdHTT0nPVrb2Wfv3tFlTJroIa1I6z97HxSf1a6O2DDGv4AuvfutLsMe+n4vjDLx"
    "oPL+urBvJXY/45xWfu1HBXUU4kz0d1SO/VB+8dmV+Zo+1R27IgcsS9joUfskJKMd+TDK/PKce4PV/oDyLynlyU588o/elvZreB/7"
    "JPHkl1Vym7/wB6Kgv6nbdMbr/xkP03f0iOFgdHebTGfGqXQylRqk6JvJLbB+E5O75BgXpG684c34LpEe/Enfel7GTWmN+x/DWVxK"
    "640RuTxGG0KoilG9CtVPDivVcdzyJ3ToKkfSDBnAsfCInoBoNxREqI1JTfWYUwndd4xmCZVFJgJ4e26kG04RAzteZjXiaWiRiPNR"
    "A8k8AnzrjN2y9bOLTAgkAeHbeTMpL3pP5kwwKS1yDzrAu86mnd+R61zl1JbzGePY/weoiekgImMaePhphkEupI82JZdLn1EZNGg/"
    "+veKHodd6TEuPXaUkNC9TRtMuCuzXWPUnWjYPX8azlGtmz4CSQ9did3OwoTT0YmkPEb4Xsz+6vubmyqhtSwdGPmnDs6V+y8JT+O+"
    "aT6mzPs+d1I563uEC1cMAYwOYnpmpY6sL02OpnyGOe7ApM+GSHGdyJXFc7Yanvk3KR+aPxIZq0xgPmLVhu+u4pk2lE/7FaYOuqW3"
    "ZHYkxAwFH4+PNCDd0FyZsESKoFdgRKSNdmwgLBJaj+qSj33O5XV+1BBWkxE+uib09HNuZefZI6gH4LBz2rGDkKt2d/jj1S4idH7R"
    "2vQKGTJpHmntS9un0yx2LSr1bdznc79Lw86hoEpytXaGkXTod6TSYOLb1M7+mebtmSuA7pNOyksqogpsXjPqIzo4h6tlUtLYN/4+"
    "LXLqd8aIqA3lJ3XCvMItzB16zpmq5ZuLZfutVSs3EulJM1Et1Zpe1t/b95s4E9c5GwiTSXqjCA4wm5LkdRtzxRTOcT6ckiYV/0Vq"
    "ynCT3QTT0kmoFZozMgug2ozK4S5d14RLJWKQPUqaqmxxNTKjq7actLNywdQeW/V6q5dtPS5fG8tM4y0xyzeaj78I3TsVbsp31wFi"
    "Z3I/YZRSRnB0Q2TB9CUqTMNz4HMRg9O1ukQqQhBT7G8AC//eQ2oAIbl2eiH8tG8WAYfVujOPVzkQLqbc3OdzJReZV1IRLSaD+71d"
    "6xMq16QUACUcLYSTxAWY/9o0oDH1wNexgHziqrt87P5fhzmOxuCHdsbfNu8FbuuTWQPn/fO7NdaV+XrUlPE5u8R07rdHB662XjWv"
    "d2syt2vs+sjv1i1Uhi3HzLsHlGMOkQnf0Apch8Kl86SIOuj46NV9VEtaozXCkCNGkPbPx/B0EzL37zeRPSDvq+HkmTMmdLtxx6wn"
    "/H8oxbHcPfbdDAf77/nCfntO7n+0j43NROb55lld2F7nXtMV/jzR+gOVz3BRa0fA/YbsHe6uGyjPCv8d3bejirh4l/bUrvSDvVRR"
    "u+B/fQYCbr6VgTFy03ZGXTj/1p2Hi4ySg4Wibej9sYf8LqSzcnkjnKDzZjqX5D7KL2ZDBP8TGWtLK2TOlB9K5i0XpaqIhDeDXFJB"
    "vm2WaXw2/RIqw4fmyO8wcoTIGrN2e7h8o0CpgLhf+r3LseYjbUh3Ac839GuZ9NYL6axmu5Up1Jut51r++RfzyShFux7bg075E8Zd"
    "4JROIt5l1BCRlgDxGdP1QDrGLy+rISWyHRUXdKbSYguA9z52Xmx5jz8f/j2D8686+9tnovsmUBaHjl7YJv4cM5KydtTKvH43/5YT"
    "TfnVFVVGqCZsekuuC6S6fvzOJrXG9/+5jfveZSTRGofJ/HsuY/kYpYzhnuxqVIR6M6dcUUoCJAUTKAuAHKnxuzUP/PtWhrvb3T0r"
    "7w4ZCl4flDW5cx3ijvf1qnD5YQ8WZD/ReDTczIgYW/BfhfbNisfP5/0+sG5GRjj8vDSve3TA7XotWpeUylFeH+E7DvHbXfY7TalT"
    "rN10f4RvZ8sFC/79Seb/+RcyP7im5BfDb+skM3MNndMYHJ4qv4Jf2QSUY9Vf22bDy742HjNFkhdvjcRX+c1bkN9cfW0s0kCVVru/"
    "VX2DH92sNsj2eGgkvLe3VrkMfvpOopxtJpYN1ycwe/qJEQqna3QFAiG/l1oitYTQbRqlM9j/rUd+7mvNW06aj5lnekYPZzIOwUZk"
    "tcv5mgbqG8p1EXYEFyTNbfVs7v1G79dYtB6MDQ/fpbGs0RifpxZ5OogWG3hvQeTOkx6hZy0zc9Kt2yHQJiEXMEetbJb8owbm6G2R"
    "b9Rbz7/SXXp/eucyuM4Ca6ulviZV4pfauTbeQ0B+SLltyvHzLJ8iKIiYm3M5Fn7Hu6hNx3pthlLqMcbpGRuHPw/YSNBv8MkHjAC3"
    "xNx8dsGLvbb6lst/gTLfXS0FlfHEnyX75OvRbz34JBNjm5+YJ/Tc94AwgxK3vJY3cOj5c1jkNZ4NHjZToE4zH6ahaQJ7CNBbkzdA"
    "e4I/v+vVPeWxfbPzM1REKdLnB/Z96zdhGyc4R0yHBP3rzueCP++uWEdwiBboQGPumoraXZNfyGpGJS+0ErwPwusYr3P9NWFarEUQ"
    "zaJzf0t2/HIIGSycxid0ZNgyi0LrGLBT5W/Ehjyk100qxfqcyrOrNh2N937j+y+1CVKziPcE/QbXB+FOYdF5oPIqtBbonBrSO4bG"
    "EYxVgI83CaYIdHZwrIg7oX58zhLVlxZiTf4z5n1F2I6N9fB+4PTUN2Nhm/3085LLWmrgBc+4M3/x6RkuI4rfW9rp78R0Anvt27hQ"
    "yAeKfUZH0tP2OiODgKpI+kwYXCx9G2wd93MpvYmJUynSwkz/b0vkjY7Vzx1WnGb6sdaaxcSfFLWBbTOxW4LxEua4LjcWtXw7b2M4"
    "P46NjtrpD9BpkXzT52SM/o/YTSyvU89ObLv2Cn3WSIx6Hdaxy4c2zecVnbS7oFOcvWDmTHUgEE9Jnw6So7SfZlL9FPv87Ocwam8E"
    "7/k/0KuKhmFK3TQnUt4iV0DnmmznpXvvavMxX6+3qrh37q1VLTdPP29hoGcFci4dbq9p3dC+gO3j2h8u+gfLT7UzBbk2JWe9w+0U"
    "4LAUfkuxP+VeiJ33wGkrtIZhe1RjncgdyDu0oVtaiVh+3UstH4w6c7GMv5vQXMJhXJ+9G1+d7N/188Pj6fn85lXfhzcv7fy8m8Q5"
    "VFSHwjOP1dzr13EHOQvIT6BMUziMY8sLEP9vHYSb2juXjE7NsYyHjRW091VXSzxA54x0lMqXSEey2vWmDIB/a/2LXCTF7cspBzVJ"
    "5eiS5vaXcSsup9qSD+HLtHzGIKCE4o58/0O3XVsE/HnDeS6xR56jp1V+QfpVSid8287NlbD+c2QgX4scxhDIsfOQT3F1DI689uC7"
    "obu0yuc+5CMaOc7I7vAxhO97FkSZ5lQx9rtwwCKf0Ss+b7SVQa6tCzo8uM8Ri6249kV8ewfdMxRnkrQ5fe6fCydV7ZaYREtzr8/H"
    "jazpMtFxW3lSZGPS3kabih930DmI3/c2FT3Wcdpzq93ecr65pAZlsBkbn3LitmyLtjLoQndjAhuy7XxKVVN++V+y9XAk31q1h7qv"
    "L7nLe1isLgYPKIlozYar6gZyW1L+3a/ycdsgHV1sPLaAiPnW8MrNjld9bbXKD7WWIGwYRsUOzzH2ztBpAZJ7VoqC3PGU89GYTA5c"
    "W+wYOQvxb85/r95MB7PPaR9v79jrwmccrSKDIp0Bg2bhrPFrEzEoZms6cjm6Hzvi93htKkJbweoYG1vsPURaAF0byMQbN4I4EzjP"
    "og9YZ84mjoxbBPexE8dUHVfR9wjEzDi+ghg0rXWqNhF0wlBJj7bwmdL/+N/EtgPEzgvns3/Rdui3L5Acj3u2d7ENMU53kv4Hh/nb"
    "riSx552waaKtTEuU644cKeg5/nnLolsu+Nh4yLvj12fXYlsWbD0HIx6qXDCxcG2lMNdMnDVU9j5FjfPXrs9+cEYYTQu76WvuF7I/"
    "UNqK3+/jW/tg27TzJy5VTbWOmD/mli/oZ6sWl35p6dQAPkMbMia/H1mKYMenh5/TN4xhvj0QykPLPnbiVI5Nt9wh7hrwj9/cmL3b"
    "tsvy1fcRbCwghChTQJm8X3cDbnfHTyN7wd+PGuPdMuKJ7LN4e66dnvhz55b3Qf7lwSL5UxQk3DP0Po79gjycyk2R8UvzjvxMlz33"
    "ciweOSyDFnoxzxCLKm3W0kEpunNyJbeB9WgFbB7O8wtC1I3QOBv0rU6NyxGhK4AoT3thUU6xjcSt4jS3Xi9cklmw107Mfey5zMXO"
    "RSSHANR9/W10zWPbY+ze99FUis4Yz5vwM531kZoFRnWd0zzlRhH987vxc+4G8bd5n/QM4vwB3z20DprncHPX0P9ST+PHJbcDyfm4"
    "123sHGnew30O27JxFNbtLewSsLpYlJqxMx8vV/a4lujO4C9iTs1zYRe9JdJNPzY0moT9v+/nTWxOkiWfXFIeE4eMGU9KWTqAurjq"
    "rSzSqz1D8fNrfOz4HCWQWQJzED+/xo/+vt7Blx/mfcP1CaIHk4LE/2OkE44Fjgybaojq3J8PPOsHiCn/5b0RM7fLjC0l1f1wUSep"
    "/3XxeZXG7ipzUHR+OAb1yQy5q5ANl/vfytO4vRyIef7P5iKrKEBfn/CPmUXjAYwmoXeP1zMhOWxhQvwcdD2IlHphX8XJ/JXkMiN6"
    "037+L+8bg9YV1Kkxuc7K9edw7SlqWELzb+cjULN2Yc7ca/7T50nrO9fIXLIXGCInEUay/c08xuWw4KtG81KRNfzVb3/6ftdzFcsJ"
    "bE53//5MhntxuZHv10f+y7iI4nTGGXH1m5jB7gfzFKqDWmxDzGNLsv93I643RK3K4tatC7f1GlprpfkgN87hzEuTa44HqP0QBFaO"
    "vzJbXzCHkAm1aZxH7a/E07p1FPt+Joh7De99wDAJrm0SzEVE1yWUq6gfr88/6pDpHPQ5B5A59hqJtW/fD9lPC9n8ikw4kpayAth/"
    "0mBKTF+HUuJncb2H2+7K9kTDi8jVoL6fIRa77zJjNGyboF+jzAeR90FtudG7Zo4rrk9FdrTkNpBjtfdDDNej3+0Ela6Gz8KxNvzu"
    "wGcoN8MaZwaFPM17bcbfkb6u5DzyR0cWGVjbvD+5zduJj/46phG0Od5Lj9CnX1thLAI7dElZzaTlXtnfIP/mvTaQXpvTp7rHPjfp"
    "SNlngI7i2kq2GWe9wlZYm3M3FxlsaAyp4UlqyCpcNybrgzyCv1fAnjhdR/3rYewcx7fOtpYMD5BqMURUn5kG+axmaO8PYlhKNOae"
    "SSkcDtbqEhIex/G5toT2TDul6xOqpeJ2Yo51p02OX2zVYvz1YM+t1O8/rrKkBu5XG+B9o76bN5LWqSy3l9rvOK6xu4o0KPPk7v9M"
    "KsaGxh49t89bZn/h39wH6qllHYP5nzj2Do6Fdjtce8EoztgHbVtv2UKMMvC+Jh5kaxWLXL8xYCYzrN/8coxD5q/8i/jFdV3IfgPp"
    "wHGhuhtxG5wiLdv7G/bVuGeGbIbQXqgE82Jm7ni+aM8F0KtDOoFtf2GN3SEGPHFtCtL5aFW2587PXQydFl0nlq+5Jl2LaZnrq0SH"
    "h+p13Hv6di0YQYLtp4wWyu9X3J9CMRp3Dd3YoalZsfGkCs/pz+M+w+V/Zl9XOpE85ScjUa6flSnGjusPretFHyI+FqCo4MxQMjoM"
    "vLiYfXpypT7Kr1EO2CUGwX2xZpRyskHAuEOy6widPfYh8Zz63mOwblV+/2P75CmCsE/+KPrsivSOpN/MvxlCiPamIiBP/H1LPp3L"
    "7uPqCPzeyJbW7lZi4YldVJ7F2Bj87JA+8e8haPGd2VHWUvZ62/qQxh+Xd7P+YNxY7fuZ2Df769Y+NfILyK+SX8PZbElsbJX5HECe"
    "SC5nG3d/kwcxtgDPV/7qOYqVhZ04pgGBGkL+ntvC2U9CDAG2ZHu5Q40C18sm0zsaH/onsW5rZgmz/nR1NuK6MfIDUqhp2K3d2D3i"
    "P8J4EBOXCLJxkX1R9QbF2uBCDCNQN2HQl839e4ah5btY6MOlZ/qxRse2OstzSN9ciou2nF6BeP17tSfG2JvQ6XG5JH1+7HdxkGe4"
    "l59LFXmCfBtQ8GHjAXZIx3x2GLK0XrDGkFPP783k8BzqkQNL8ZKRmmkMX2lh6iX7vlimvbtH7BHsm2Cgmo9o79A8O/ZdFjqF2Uf7"
    "5I8zRGOBmTwj59bYo0/KVEA6QOqiHh4Nc9W0WzQMB5pLzYOp7PFUbfTOYAPtgYGV7Y7sUhhOMV7OfymbMq4l3Zfsqg1scrLaRyQy"
    "cNpcZMqATh2gzygFVq8/04rGDZWJ1MbVKnWH2cudE8h4a9/MZkNG9L+3uituDLKXrn4f6vVR+ZXgnP2olNso3JnqhuJzuM5E/EjJ"
    "IwfW3YEicvcJ9zrTnMb5N1x3ycx55nn5neMbcq3gh+YgN85+j8psW/9uc6aHPnK7qedgfsGcaSfP4NxXWRnYpw/lG8L2Ws3EV305"
    "oNdwjCJiwx1/YCuOLp4pvKeBdkG/WLdR8qrzH90zqPvpPu0U5DsYE1hH0X7bse5yYsbqV4ZixUFY5NvSJd/kfjP9YTwm/K4SS2wF"
    "72/7IML9HKwXw6w6/u9i4oa772NX8f+RXgvGMJzn2LoO9sOziFFl/t1zzBnD2hx9/xK2Jfmwgl8wY3+K1uOr0vIyURhZX374stzU"
    "WKi9SraZ1hUGoJ+bkJ0dvabO/ZZJRtFfcb+kYfzU+pKb6RvkIekF8iv2Zm+aZ7G8OjGL+9TY0sPkvbJOLwW6A7KpaBkPwe536CkM"
    "aTml9ej2LLytn+bZrEKKozbB1EZOGeE/BRYHZZEumjFAr1j5zHJI4xRHYU3x5D50P7B2vOBd3b1crE5F9tSESdnP68s19O7QA857"
    "GN9jTfMm6wUma4E5Jf0AZHvzbGBSkHzEmOz7CrMlQ2gz5gOPBXHjIBy3MNUjfwQmbbKzniuBPo8iIMtGmx7Z5ICFhw2h8ZrDoLiA"
    "bcvPl3x2c6rQ8YmhsjGExyfr4/vQjLtRYLh4M5bAXnnKGX9nxMyjauPg/jHzy/rIA1N2xe9HCMD82f1z8utBTZ+J/12gtsXk6s9O"
    "f2489F6+Wm4sRs+1Vibfzmdfm8vapJnItJqLar55jKz3Dj0TzIybDNp6ysT5X6nNJP9oTTol4fao2PFC5oAhhOt1EOPaZyomN5rK"
    "aj0+2FgSW3+OWo7OiGG+Ms+71s+ubCnGPgUrN+1f8g/JxkbMs0A+fLLF9pnWpWw1B+bUjoZ1YpCZGzHfmGeSnaT7PZ9xnz/rFd42"
    "V/oazRocUINo1oznlHRB5eFR47P27DFEl7NvTH7bnBOTO5Y9T3oA59dnHbO2otTyBGLA8c8I1tHo/GmMymEj03VDTIrrH4XlKXCv"
    "S+/G8bFtJ5K3t8+KqUtx16imdceax9G9DZZyjp/nNhvnbMg7FN7sGTD5jJ6wU5ENMlpCllZCELscW4+LywZkxXOADXbUZnyQD/Re"
    "O++w1XWb+N+/cczmOY5pNuDHBetQorYW9L2J98R9rzZSQWMU7plbP69fnPFHbKkifx/Yv8NUYvutj3bFzq+Y+i/EHhCDTS5hi6Mf"
    "YdvDfRSG26/DtPdwPw/YB62gvxPwhXrtPPSA2s9MI3IoPT5OTe1qWA/3rE4Cewt0kNQtkQ6euToeOTrodfRl6n3Zl8L1pk7e6LGK"
    "1ZniW9D9gQ+z7rNv1tqi5jKi98GYniLd3imZfEOij9+3Rxu1d2ju0N+AuKa1gcwz13g30y8nEOULwZTydVFkzi/pJFfu272zzDDL"
    "WiW0PuH8ouLhDMw9OvX7v6aOG2tv6yjh+9DeBHUG+tExNvQ0TuqsM1RXo6+k6UPYFzzS68sV3zclDDgMmw+o4vuLck57kLg25YLM"
    "s2zgera/l2VRGFVhX3OY2HC/GPY1+G6An98H2CTZt+IcSMLaGcjnRvwvL773xLJSx0AOku//rrVFDh1AVIaEfVfVB5/dlPi3nDcB"
    "5cE8DYwssi+nG/feAlFu/2a4fPR6DpJfHjCwKkb26bo+nbJ0n6rWTi0OiKeSbhXsgIKwj8Guf62PlLl6uK18lwNz3ld953gWPPmt"
    "7r2lyUn5vnYr4ewFzjWHZFvXx+NRuz0YJ2EdqWcWMjOzDa6NkcOsp/EekAU219avp4+GNS9oNyEWIb0wdO+DyPlppCZM88TmbLn3"
    "jfjUlUt62akL+2V/USR+YPvwwnafzp1TK2XmJaJ3zLVW7/h6RmkZIr0sRh5ZnDuGi+d5nAWYtACH2gW+gLDSo6flQM+6MXIM9c2m"
    "18FhAEsMkt62e7KsbLJ37fOF7gL959iPTyvvQ9jxVGYVnxXqH9g0M/h3k0FyZ+D/KwbvaNBhShJ5xhqMs7OJ5Kb1uStmmAM0+fpp"
    "XdsPc+g/vV8KoxjZ3kXaL7CDLtJ38DMOwrgGOg7GeZGxvIUYygAbW/iaMQwus6nzb7dK27IRKovFN0xniA3KM3XcF2Ay9b3UhnLX"
    "gPQxGHmBe5gM2whDYbw/4Tx2O+W1/N08vNG/R5KDf2OsxE5pQ/vo3EXMvyg9JmVP4yu5rNZWWlopiaUWef6hV8CY6fcR5LKPDbLl"
    "fbxH1FDCX2j6LIzMZkk+4hw+A+01+z5vm7LfU4MYL3o4Mk1lWaS1PvVSz5ny+c8a9XpmfOzX5DOYA4/zQ+RjPdU9U+O1FxqX8P1+"
    "8C6nxF7jx3uGZe/Ajo6+F1gAhw8J21dkr80DwtmxN3LZ1JDPN/Jtwfi0jdfkZpzzhD0F+rVobe0iwH55ZfxaB9zcc+3kyQvhERyn"
    "Br+D9Mp5VOB+owRg5r+t6WF5pWNO7rdDjbXH1WhGa7mBBTpbdpO7O2PPGT/NidckmE2S41IjxK3O/eQScnHxoz2EdSM989QGnhGw"
    "Wq5fK++zl3oxXl+/f10hev/V3umCWqHD8jT935tviW2SbYV6yDTHgej+xvco5cuvdWB4qY8IOwv2t+g4Q0Moclsx147DZG33vMwo"
    "juts0l2JzzR6f4Q9lSQd+jkERs3P9p2tqXjqaA9cx9ZUACZ5Jv3YJGNSzDCKnh3M8fr7OHV2Q+9ypPlbgP4gUD8M/wU522Qz6Kfm"
    "sljL2cDGNXWeAn71m1sLpnMUrVX68d7juK0d815quny5wTRAPx/rB3L+JT4P3ifspEZdang0J2b9VMiO3ppzc5KjQ/wW+27VQg3l"
    "spRL7G3/G2yxUE9PTM2W0V9H1RPc90v6nuSw2JaV3HBP54xjaFwH5vddWZ3OPV8NT+yM7+W2zb0/ufQPD0KBAduCMWpbYq/86P06"
    "nLu9fr82zv73ssLUGVw/zz/bJ4gr/2jOfnbuYF98jk6eN1gb3Ig0+Xqe2R+Cf1NQXKn3BGo7PgUzB8/86Vp7s3FOdWvOo89pT5J+"
    "HLRLa38uLWWI2cN1qcN4C+dopZe9U5sBa7eH/rp29bPHcot1H+3xEfn+XwupA2hxjL0Pf53sMuRlhsklswjTb97RJ0zyzuP+hLWp"
    "byQbrlieVX48vhqoSSWWWyBbOblEfiYDLIVhsSl4QanWolvnHhcbb6Ux8vfWr4e9xblhuwfkuS1et3MFtnCqa97pnXxQto078EdQ"
    "e6j+SykXkluF8pLrO00NL88V/L3Hqcjt6Z0yL8vnomOX0p9KfhzHc7hmD3M513zBO2hEu8iFo5812VpcYqaGbyvv/nwwOdsA5iZj"
    "ac7YF5c5YLuXa4R6xcXv6SwUP7QD3ORW5rO3NLgENe6Zgo7lesI2+iiv0gLwvD+tf1TXoXVhOs7YGt37P6WHx39PDZaKe4/9FXow"
    "2z8Knfned/WH7TUFbkz1w+KwWly3IO5HIGajmC+DNtltZNv7+OWWyhD2Z5Js+2W41oUp+Vqw93pb+EG63uZvoTrkveboIBOvKyBO"
    "YeUAx1Hi3ott6bqPn8M4DPkM0xVr3Ix/a2OusfMS7cF1sYtsTse/lzC+d0qkswUfxYfrZxzoCAatjt3GHm2/GHC2Ggn/zGi9g+9P"
    "+LGF39KWkS7C36jnQq0I+cw9SzcjNVwmHm3it7Vr+2tVuv6OV3vpmLZIz7v4XaC2zHyOftqHdxXLQntxmC46w3gWZNuJvuO8n8QC"
    "HB8tJn7nQa/s0MfRb7dSwHvtrTmeEF5zs5YB7JynnMqPldEPcdRNntLtvDn3tbFzc18TE3L3MPaN1CmSz8jvd/bMe1Z+W7cAmUj2"
    "IWJNtmfKzcvQGDOObAbFRtBm8PN6pLd+/3yyKW9M/1Qn6QGTF+Pdd989nUOyIfKWflRtCGB5ln6Uh4qfZyNLZvoM8zfJHqEXBS4Z"
    "KFgz8fv0t+PMfj93drz5HeQd8p4WG37l5Kth0/22RqdwcQ7N+PbdzsxSq5I9QPYmZIE3v3TuXV0ap6cRQw7kH92clsQN0ScV6o0p"
    "3dJ/h0hc+QL1meLhfPY7F/s3rlNGtfOLsY9/ws+7LPOiPQqmFq7fudCb4OdQ5Fkag6bxh/t8WO7rWhibJdqr52P0SV7oIn72D6h+"
    "af/5tpCpj2Pf/yo1ziW5dCkv9r+Xp6amKYq/pPcyOG/x+Eu5KKZeIMYTxvsy+MiF3lH6vm1NCI8tgpumeMchzLSN4pyiP5/uPzJU"
    "TTo/i3+BTebHzgaMeyz146GxgNZZfSH0eiVCe9z0iCiOsGAHSx27Offm72CM+6j8OlhTxUJmW9nIlwh+kWMD4e9vcYzMWl7DK/hW"
    "Ll6tD/esLriMp3GM78E7e9f8hUtnyPV1YuTKFZyjOnMqfGq9llBu03sZvy96/sz6VAflZAxmk/TVTDmeENf7qnUtwVqU73J1al+k"
    "sEdbjJOn+WrJwUVkvON3QQdyTxLLzAu1H3r/dYvG87UUbH/GojyYzyR3F9Udyj0lvlEujT4c1NQuYvMpcva0NgIyJoAPb/GszDNR"
    "p4HYwiA3G5jPOvVoX277nB2RD3Yc1ZFHq30Kztt0K/Uei0Pc/o7rz7O12n4u0HBsGYwti9P1lMs6GFGbv3Qm+b7DU7CmJ2bNHJrV"
    "MM3jz+wBp77KvjPmIDLf+UyKzj3mlmRZZt9vWzn+OWJMK5HxdB73Q5LxTk27sbHoXW5cfLQD93c9JE5xveZGHmuOns6tdxyegD1R"
    "TTB+ifYp2fp20+MkPUOBnqVA7X2kFh516JE4MWqHDObh1fr4X9h8V/utzLvFXzMaTUL1XcGc8P1Reyvj5oNtGPK/kl30EKF3rT7U"
    "nPr9OWir2zyBYHLH8RHRfL4t8s23hI9/K/jb13QzcEKsPr6OZ/pNPfklrNMXx05QHR6xD57c/i2/xvX2Wi339fUF50nWG0mt2Z3t"
    "Z4ipr6Dvbd0Lj7nQDMcbtq4d79R+x/aWSFziN/7OKGIbBurwovUHqjvD8Yfge34bZwjqid/0Cjhr5fdrRGu45D3Mek2uz8naj9/X"
    "uG9jfBpJXxf6/wI20f3a5gn83kDHL0XN+4JlTWyfmH+/2O9/6ys6/XAx9g/2MPnFyaO/py73RG9//+yrNiD29nyQzOz8tbjSYzhf"
    "CBb/td7MzncywNo/bq/Q+Zuz6vTcDW//o16808zxIY9iYyGXdDk2FIpnLLYT12b4pW515DRyQRKXAB/Bu8H4PEr/BMnT10YpWGdW"
    "SE/cfv9o7XbJ8OWw3tOe7xPrP+mB1975MG5AwG9TmTtj/k3yD8GbeeZeMacXOayT2HZZZtzeetZd4d452wufQl6mtu22j5zTod/y"
    "3JOvdo7offrdz/Y96F0D73UbnjN9ljdYha+TGIDZQ+QP7MO/Detx2zPxcP/r8T9rLwUwY3udLHLHER+U9t5Tr7PU+MJx2hKeLf9v"
    "5Cib9trXZkJ913p2xnW4hWAMpyfxXOXskGu1r/yEetruaRmhFQ7brVc56cQf5PsG/RC28T16b8hu7jUMzGO8vag5M9jarHMzw9SS"
    "dFo1AZlEn985NtQJtXh43w7bqt6y4/0R28ljmwy4DZPy2ceW5lh3vpoG5sqg2brpdxj/aDuYc31aiOb3j5dJ3STvEn+Sk8RgPLpN"
    "eQnPu7v1Rv3UYDJIZVL9P7d3ydTo5vYuc3ub9v7cZtLp0d3Nn2Fmkknfjlya3+1yvo9j+W3Akmnn1wZB10fsOe4HYCJYfc2GuSwi"
    "XGTxPt+x10S7duhWfzHTHyK/JPXXz6YKXX6Tk65a7b6akCXjKSPIxMnM0LNJcmkHOxh0r7A5xe1YrtbzK/ItmxyjSGHlxYNkFG4d"
    "73EajNz4KNuG6Q3ZabAQOcw1t4yGH8oOkfdHO/z+MAJKRUuIq023o2ZU//Y7NUbsoVOg0SWwFNB9Tmn9ziKvaQVtlb089lSlcjWM"
    "0Bau8gyi7F2odmXLu9Naa7WHIgBkVtKBm9nGRDTj0Brs3JjuXXdeyzHduU6kINwhFHd/sswY3ctlVQ2OCRYF9tsygBC3s92r9fD1"
    "hsUs9nlzstZusGYG/Y7eedBNTfdCoL47dup/Tk/t7XZURJcr/ibtnVqY779C3391fmxp0xrn9rxnOJJwLn30SbOjY5b/5j1gvs+H"
    "vs+POp2EU5mLTve8WEbMUmyRuzkiQ3MfXDcTrV6SF2nWr+WwyhXyYIz0eknxcg1TUs8g1xXKF78PdZmbPcrVNzr/76XH2lZRJYHG"
    "7Vc0KTsbaZP9iDvi0tx5Il2XHipmSUa2Dhq5OgvCClk16KpUdAtFA2WL6sl2MHJXs+4PaAf6DVno7eTejgfIItLRsDNsGjFMiTF7"
    "jz/XPeZ3o908k7U3bqDjdvpFa5l+lufSvJXng9QMZ1k7pmnOTlkP14/ecX0T1yfl+nLs9ZHOpMD4FPGonb+pFKdHuhfdezF9zWXJ"
    "WrzZjB8e+RmVwHWLIz/Tva6B67pfDoOC0QUbv3LT2WcaYVQZGtLIi7DFKL8PsHUKcxEyp+j2MYxLbNEUls6zfsR+s4XnQx7gESjA"
    "Dssoul5jM0Sx73eOZb2kNUGFc2vWs8yJsA6UoZfOVKycDXWZdFflz7FE6a+s8f13DIfoTJ+TfglnQjgiJc+QeawK6v1ssF5sLKtf"
    "sasMnN0gkrv+DtYb/zsPuXf/Dzo2IC/b57LKCX7vwTO6Rt4T5tpz55eVDZYJNQU0CjCv4T6X2CkUEST/B/tNkZaQ8QYape12Pcd0"
    "5otVfG/384Tmddkj+WoRIF3ElAUjfojnI2yq3B1uKlqYRbKAfZRnVHfYQaVCluSqN+2iin6erYBl8Kn9te3TuJW19lzKTb9Y5jw8"
    "HnDmJWrMSLv0ntm4DrYQq4hhVtcsTSOMiiMygeQXyyzsnTHJWekKnn0wS70dd9d2iQLxYCRZT0WaSqMrYj88oZPUZTq5xJjOKCHb"
    "ePl/vyknj1wxKGOcBd7BMItIBZtB4/zBMw0rwuU54IpLeocjomhgyey1W7dgKTLjDo55Eem+0Wf5ulaYZpHRRydLjCy8p9Urw1Y9"
    "GHaB4B5l1utpl7MjX8yOAeQbp6ufq+ulmjw7A+pOqVhFxZ7vgZCtXSmM0F15RDRBPGXyWqc6D49GPtnKtqCunme5s4TeZQa0yB5Q"
    "OZmlG3t+OoX8BMrZ0zz7Cv1LtsVRoouCYAAkk5FWIg5h+9ZRZVj2usk9onopRlrQ81cJVl7vOJoS0qFPypLALIL1rOqxhano5DPG"
    "HZN17j65eepUU6iSRvfNQFAujuRdg01i0+3QfBeykPHM8N2jM2mqLmm+dmQb8Nh5PMXqR5+7jt8O/tqpLC1w12CAzWjoVvrK+rsI"
    "OjNEpyuFGxtFEfQPRK1Jp4VYRUbJGXfemi6pobffKutbXOYY1aQciWwnl9It6f8u8F2nQfZLXStR+R2rg34QodDvLHPfs6hd5QV4"
    "wJvQOfbvxdH03GxbkU5BRM4XFbreRVyga8gefTN2VND2ZHspznYMsR3RvA3XVWUcSURQSuHTaZXGEvoM7Ljotow5j849N/LOjGQV"
    "QlAgPSzvXYPtvb2ARv9XKgcuZhm9wQrna4Ys4F+/67f13meED0E1defqt+Pw7xV6/zzsv9ZyvGpuKrEolkdGVkTnF6KuQbZB/c5n"
    "PN4y8irpItuVX7CyMYDmxmN6uHe7P1GdNGc2da5UVrlQfD4YHRjO0qJaCfYIxiSVCqRvG55lBO03No7NSOcEqORiI9yMOwZdyBs8"
    "50M2Fir2O71loEM1sId43u37xpw7347P/yF9s/hyUUEuv0sUsTzwLsmN+hobsbfMd5qZE6QXOrdLL9Y2ZPvr4WY6fojJvsoa/JW5"
    "5+r/5FU2BF8+sU+o5/tv5Trzgc9KIZUMpLO1M0m6jESeJ9Oo1jedXyKjOfvOWar3PjIjsI0U4Ubts/gqR5fRGPrvnJg+v8Vf2zmp"
    "L8Vr1g2v2d+frpk/R+iCmKXVR9syM3Z7eSC/SX21xVWkXtGnNxuc9f58cXFvOBUAvg4BYnEnCzaGeNkRQLcJIlVZWzhoH4kfx1V8"
    "1eUISOirPGzqjZXtwoS0dd4nloHQ2thif2gm2CIVIZZy4LiTlVk28j0nv/oDVZqSVWc7ZxJaY2bPCMoa1h2oLOG9JPe7+Us6d09r"
    "8Pl0LmeivrFWL+b3Nnskc3pv7E6NV7IscOZG7FDYv506I4ajmsDvwnare9TPrPgoTTsbF11JBtGugSJDc+WqINnvxS4UP22cCqHl"
    "apWOIl0b2ZDx2TOnUV+L/ag3y3gGhG74XCRrOPvhM7NZxusEur3BZKpIEjwelSGBcbjzUvl3LNEuC9kZyOYjqWz15yKeNU1sC1k7"
    "lpt6Ns7OvG8jHdey3+5QGcvZxrbYMcPVcmHRV1dLPT9iU0809vD8Sz/WiaX5CA9LrtgjHT3SahBlp1Vfl+Zk1k8ifhrDwJjkDvdb"
    "h736hO5FRbbfckZR15r1SGCMz2EkYJlrYWHgrMUlVAnpshfk93IK1fnLg7LHmj3PdgPvJfgCtEaTSBbjNjkYJIap5G3GSyVHN4n+"
    "bfp2cPdneJP508/QIRuOvEly3L8Ze3deejC5Sd54yX7yZpC5yYxv7u4GThZj/9Gfr+OyGDXUPhSW77zbgcHRPk5lp6NmdCj4IOvF"
    "VHtk7rg+An3s9J1eR1YKZpt7tE3fiXjgK+kn6XiZ1Ui4tCcd/xp4VsDVnWndZCW38rMaw9VI+r/etoKBir7owpfhJj4HshopwUl0"
    "fs+5UPN8jnKC23PuP5v7ku+39m/Ug/YKmb1zj/O481zx+3voufBiV4HI0XEsubvD0LkXehu7STLcV7/rM/L7aQLXQ3LO0FcUqPu5"
    "3xrObI7UOvMR7qNAZOETdaymzsV/r69P1MMPnTVpkOc5TJgMC2npZJl5zli75gwHLPKrrd1QejF35kSilor2wc1TkO8SOdvlQHCi"
    "Khf7px6dHqJHqZV+M1kiiVp9Doq9JeohuP75zY4fUWbgPZ/dcTRRr8G1C8+seTgHuQQnV/rDjn+tNVdOv1RUKtEcaLTP+Z2R7mbO"
    "jXYXTpiUZMwCHMUiZSqXOcM5M/agGbWd/q19oCwtaB3B09I6icaExke/FZ7do+eMNqjN/Rcc4/Sscr7u/6Zea+bxfSOQFZQ6aF4P"
    "g31iNAWdm3ft04UFo3UIvOfOlTxfa/A9glg9Odcj4nFz7TRJfu3TdXnJsR/RY+rj9DgRShONvlxLx5aP7CuHx/ssn+McPK+lDin7"
    "SedMPzc8t9UteYVS91AAnkCa+ebESmTtD44M8PSFLRSbIZUosd9vKdaHjsu7xmHv8O+uWwfm3D3/K77S00i4NKQvj+4T2/8ZOnuV"
    "3Oav+U37NNK6XuH/amEPW1mR5vOo/X4b5zdazy+/CcqX6d/Kyp+DAb47+Vaxj9OvmjTZItlX3QxPPOdzg6+FDJ+sKfhCIlzs0Nzc"
    "Z12x8w0Oo+Uolk81dA08lcpDSfDck6209OuGn611d1pX4lwn+8J+//j7noa1WAwdM3YvY54Nq0jvbyK9aQ9c4cO52XfI3n6h7xk1"
    "e7TmHBmRaIXl3W2GcINsxtHs3UDkdiSZdCcC4Vtb2r8xGzH+TjbBPXQFsvbWz5lnrt1qARNy0kXWnuQecIz0c+XSrDnfVSeM9aZ1"
    "ua+dfeZSTRF5f1yT4z+vOumBY4sjQa3FK6IP5Ml3wd0iGPccFSav9QwMAX4H4LIUs+hHk/klq0x+x30De6wz5pt+fxyeE0cnq8ee"
    "FJ6N/TpCf3xupufEr1cLVUywTibd61dCoOZ9hX29wXeiX8M49I41PpxHKxRCns4neYEZ6Y8yUVi8y8+zgAbXxMrXThU9Omdw/iEq"
    "5mMcgUfc4Lelz/5+lAjkE3MI4Vxxj8tC7YaL3MEuZ5Fy/3AfXWj/6Rl/ntakasbBhONznQQ3tETZM8h67GRvl5kHinkdyHZT/I+Z"
    "YI3QtVyhMrTvx9H1aZAPS6Ne3H8WirhyhtDsf10HiWpHKl+4H5H3DvDAOFNUDEca7gNzAD67mLMFPfGBHrhoJEvma8hVKZGzp7/L"
    "ci2X/Tx3/w/JsZj691qI48h9ziOta8lkFH3s6JaMz1QQRc5AsUw+QzSaY2vsojYQeWjP/P2lfj/mZLL1cQvfg79g43zDyXkFm8Bg"
    "PaPShsd18t8zbeUWuDgrqWqCZDLq+zhbKP4Mvd/xUs/i99zClyqOSOfYPSJ10WZdd19hvlS3jrDydjFqeJHvlPbxaZhy3rHwCE5R"
    "c9adz0v4HPa/M54fYAoq1iLeuX/cRDmfCmJrvMn5gN4z91NuZV8mDlJDU5lg5nZbCckYh1vQ58HJZ+x8QveOud+JeVYz/rscv9dB"
    "bZbbmjlZXK65juopw6eWkUoOkamVelBv8jMNB9yDZ7JudqxPPtduYH+SjrGcTC7PXvBZwO4ohbnJjEyyWGMdtZd7Wrcc5Hbz7QQ7"
    "3y7/r+r28fsjZIjpH9P5MtwzVg6skaUbv5fca82+22qlnJXfLl/UwFRKnnzMGBlvfo4+HaeH12BhapaZI1F7Ok+nfgf4pXL9k1Tu"
    "GD9mN0ihf5wjuZ/dlbwv2VgbGhfJu83pSa/xqxgDPvT5teO+o9GjvhyP22fgIDL9i4F10/VkuzrZcve37BP/ve3a2GuDa2NsHdin"
    "W9N7/eT7bbCTvFEx68EHrcydatFONJKLOeCYANl/2lujNfDe0sdzVj+/kVh3U1kvgvnGWfERbJJzh7kWdv69rO9fRQ8w7IVkt33M"
    "PDfuuWLE9s88hO0r01stvb9hPJdKIdR77bwD5t6JOWzcd0D1Sf908b3MGoSqW8FpF4gJHDh20MpI1YrjO2oswccIEDtRYzTX8EQu"
    "xmnc/T34oa05cvxxzbxiDPuY2MiMz4HJag+9vY8HEOGVjFYbxz3bYGmbCO2vsws+xqmNfcbgGQimA/v3pdj+3YGJEqM3TjMGAZ4s"
    "sd/lHHHMFBWD96dQ1c8Vu0OiyeLP+dV56sfTWailBp4TPSf7MMznqvar/w7BeALJztaK7rvuF5kTc/sU2ZuO7R3sDbR4ZV2uRkXs"
    "C3LPxMuYu+RTMFPTs0GOObCWw3VvK5XKHEtN0b7j9TF4l6M26wyxm808nbJNxgNl30SraYo1VMIDC559ZcY2K4BDQrKtNBachw1s"
    "Aea1OjH/iJ+JReUHuEQKmaWeGTtO8oPYjtd3Up0jfUdPycxcePKWSXCE8V5Z+/sP4zaciaMk+/qalSP7BdVxkrnd9+rKJVusLgft"
    "POyu94A8MNzLWuGl5wCf7fzPlgcbwzj5FWiy9xBLkZiSjXPZmJVvd3DFyGnzd4h+bXBHtl0/9X7tn5GRw1epXIbmfOVGehaRYQzJ"
    "/wt+fBx3ueH8e/IxY85x5/8H97Qc5qhQ92VaHMd59MwLV5ORqZyFlX+7MT+X2+da380p++hjhiDe1hR/FFUayjfw1LZYX/9/d9ey"
    "nMiSZP9ltrMhE6ESi1kIJECAqJIQkLADEkjEQ9wCCYFZ//v4cY9XPkCo+t5ps1m09S0EmZGRER7+OH5OvLe29sjV4gk4umltgbN1"
    "IO+I9lNMizjq+60l7TXyYYFa0T2j4ifYfxfi1799M/pjqIegwq36BXU+T3JOXg4+/fu4XEBP32Lk54CQRIXN5p5UPk6hez/Aw4q4"
    "d7TuRqOK5K4asRxodxmriq8Ka/Zp1rfvZjwrXW1HX1Jhfe73usZAdrwCbhDUABqX5jfbC9p/N2/m7NfXZL4AbzNZCU+0qvW8y+e5"
    "96feczRq6/wPcxiCW1DnLJDbOfzquTnPZD7ZyX/yHp1B63HvoFkFwQvN0jNrzP42s6b0pv0ty9XgxjBP8bzXqyc2u7e8Fo432Cvh"
    "wsQ7sJy/kuPgz4JkbM9jd/KSOpcbin6zstnfzI+quQD3eOUwYP4skyflDojAVzyY0D0LBvh/6Ha9Dm1V/A3oQ3A9gZuF/1Ypms8U"
    "8onzUt9+B6rWE+dLLjk1R66FxHr00+8wmg4551Y8qI4mVYtxrxnp8cTynEZXSc1J4DNnOvhH1wqdlvJjm6vubshaTLZqzeiUaqco"
    "fBCMAFkzl6fwFhYlj2B84zOxbYvO44rxp38F2ibg8/t3eX/qOl3pC3Suq7Se5LfNNXMDK74F/v360n59N1ep4nLXH1NIlbpZS796"
    "8v9mfVDc4tgM/Znjf/P5ZDhKY3nAqrlW5l5I1C70egRy+6DPZuzLQbtA+xa1+OI7epgTXESsIyqorcKaOXlWnqe7KRppHzTGvxLv"
    "suAc8FL58sKTvNxtfq4Wv505iNdDoEcic7CJPfsd8ul8b6WJCntobaHJN3O3YEoX3OnK0GMi33l++6n8yvjfvO3usZvbqE6BLD0V"
    "latojZov9RD94qEgtUz+n+btY+zoH8R4vNayP2weOva+dHytOywkP7VOvie1L/OuRrzFM5y4thnfKVSmnkddt25a3toUStaZy+sT"
    "nElyBswLZPvrH4GP3n46P5jT4vkD3JKjZA0pvtfS+YSyoKsZnSSdpnnab+QLGI60keNjAn2taoEL1V3DYw7Tccn3uRcTZ80jcjMn"
    "fAPO2zAPja7V2DqNzmdssvnqGK2pckMt+EUKNf88tfOf6PzLmMMe82eQf4Fu1BV4h3cyb+Dhjr8rrlPF1487JoX7gP/h+DMUT5Kf"
    "UlK6z1n1BBNvCHpMnU86n3cuJ881UdTAXY3x07nV9dm8vN43vs6LqVpteVu0zyafZfS7J/fcO8WgHKc1nHm5pCaZrXuk9Jw4hgq1"
    "r3Ft++mffj+2LbZA+E6S+/V2ndA8t7aauxxO5mM2CjEuuIUvOmmRJ9f67oHGQ4iesM4bf8Eh8/bXxAfSHGjsz49+8KQ7i8Dxk4OW"
    "KTAeSrdFnfVL5ChoHlo03kh3Delcq/z76e0inVmsKYU18pRml86/JpF7vGYfhJtpaxB+lSLzxTjaU+tpkEYxW/+rEKFGDLS3wsUh"
    "BkrZZQf1q7QOs7k7bT7wEXwjyzPcaw7GqbAkPxz6YioXO9bI2uWvdgiMB+eGuHNNYoCi4YTgGtBXPszlNlTxpxeT67dxRD3Q1AJU"
    "3cL8+xucOPI+VU5Y64RpW3ii7qCwFqyBh9qT7MeY7Szf+gnuy8S53RrF9oZTn6V7FQK3vsSxNOMe9DVS9ZS4zbrMPjTmm6QvYTEv"
    "Sgfb2F99huRPaLwl7DTNhRPP1afg1Bjj3WfVdw+l+5HKH7NOMcUXg7bSr6lWjqHbecaderFYRzo+q+C7Ac8KavHK/03HeYo/BHWZ"
    "LE7rwgfwGsz/0jUYGPCEyDzdvc3qh12cAycx59ztxlzfZLMCraG12JziRAF3CewCzc+mL1rv+p3Ivn46GavKs8yy3r+JwZz6aHyc"
    "cW3EDOxnOfoIg+e98BUZnykeoztxoNnzd56JSXiPpuL1r/d+Ks4+eplxTFYsreMoZw6SvDLWHpJPEsZQ9yXmREI9yFnPO3Svhqxz"
    "9A/Zu0z8j5e2F37G92QesuxUMWNPnrBnX40PtUPU3J+nqOViPHTuwKdbNlfeB3IT9H42/cNs09Tzb+P4LTBpX9lj1u9gPrKI9SOk"
    "toBxenNaN2Jna/FYXjjNDZc581L2fWHYMDm19s2hMX87oCb3+O34fpHB4uLwVbodE5yvcHLe8Lmy1wIzVCg9k2vWdJ9HRp9BuBe1"
    "rkGCSxH2OMFv7OTTsuZ4rTDBzMsmuXLc3+Zo6HMnN2XG7+SDcG6k9lQWSwt48Ci+L67pXdD3P8me8fWTjDpZ38O5w+9eYQysz845"
    "/8twTVnPj9+MpG6C2voR+NfRmm0txnb3kiv8fO60Ou1O8WfgtSovncrP7v3yrsPYXOF6HpM/CTwDnYHHYaB/5z09det1sAcEuXqp"
    "k1u+ZN3/NN8/dFLMu8M1mfWB4iqnLqE1y09z1NPzGxsM+6T4WDPql48b0V7uHhDnovZDv5vG+NuYV/ZzmvoO/Ezh0y0avHbVg81k"
    "XrvwNVHjNvo6zKZyGOTl382F1fXpKN2kBneatw4Uq2rdpo74nKp+X/a4Y/bh/vnxcc6MDa6uIdmD5Zp1HMtK91hh3kf+YMX4gPbV"
    "RRppRmf2ENd5Au8zc/cdc+tfhzF88S+0ipQuxsFjXYzk9S7UjTJcgxnj0fpNWiPO4f/sfz44OsTc+U9+V5+xiP1Zv11Cb8IaHe32"
    "eS/UzVN772/SEwMP4t+nJdYzTC/HS8bnzCGzfyR155ovV7M441JpBz9KYp4L9eaU3b3gGcHoQPHEbbGe27y85ITjmzWcyrn/BiYS"
    "OcBmvKvrQzq6vj8ujbk4s65+ObVzrdVgsENYY6jJ0XkFTGtWB+AXum+6Dv1n2p/67GatNz/6wH/T3kT/xj6UPonXsPcc1ZFjt12z"
    "RfN3Pddy3tu1Ejxif9x1O88l1PSHrG9WNDmFn8JgcdepdBBnbEdV5Ma2l+ov0m89J74+q/lmxyg81oL9uVdrw/qvskZRG/KvUD8B"
    "h/nh4v3MejTjXfx8eto5ejRndSEvfG5an7Od6J087bTeSbPnzZ31t4KOROhotnZUbkjrpBn9JuQzVsj5YG9afTnZny0wpeQVewj4"
    "clEXBlZkMz7cvsHHH61bG6mVPC+5D6xdmoe9wUpYEGid8r1ov7UXf6oZl9Jm+8c1uUyN3WAN/1gH6/9E/+qbulf/v/SuXJxNot/N"
    "6n7Iv9Nxv3l/6F8z2A7GY8b119LvkzW89zrXrHyk4rd7q9r/Tu3jP6zN9I/oNJ3QKjL79baY7IttJPjgmUvaXatGHyW+3r73vP/J"
    "uS6dnLtfPXn2k1pkWawbah0qdhDGMiiMptEw0pjNpN0F4w7XsiW/82bm3OpiOdhiNbZ89+u5rlp/81cvUwvJ4k4PKT2kFC70xDVU"
    "vJ7+/Vec+9rHc3PaelyqhuicASaXzfZI/1avU/NvFz/d01qjCt/z5XxpnLWX0JeK45UZw6z60dwzABoart+j6jhHsKLQuWOwzw7m"
    "2trCWJ5T8iiN72mbbDiesbzeDatx1EUv01tWP3eSl7vtdwvqLNk+VLgPzPztxWup/15k9x6dGDfn9vl3Jo/E/T8U5//VmBttFZ1b"
    "Ro4Bf38XrA332L2O8zmpndKaRA1SXYOxlYzlCNDf0Doy64jM+49pe1nEemBdiRcPHAbv2v+Mz9UD+jpRI5gyn30P3AzCNR94tN/J"
    "jwu8G1OH1X5FJ4+xdXfPQUTrP9opDMSmPPufBHuFP/R/XI1u8kPvKvdj9KNwHebG15PJMJefhOHYD/O5q9GVdzMaDcOxN/T94lVx"
    "7IXDUf66MJxObooOe8XHfLLfnmCviKCSjEzwCJ16jDSnXbUSpgValdGYo7j7GVmEDyDxxrS6+bRfP/NvsXonNex8rtLvhzVwJQJN"
    "iUjq+dgEo91MsQ1UitPBmmeQGcstk0MFO0FzkxzLM8taMa4tnO+RRxyAR69U0GwCvDN6ra1Y5Uf3u/DGXlFlUR2TUn15erOsFOj4"
    "DUqKO7OwN/ctF15F8d0wFJA3TSuiUpwjWztiNQNBaQe+ZHDlHoIQ6vf2tBNUdvs00wSyOt/lG7/DuBxmB82/ShFnVKDxd1680q+X"
    "+2Kt1y0+veQ+608e1Jc4W8jvGNnKMVdETPbQB38oRb/H+n5zJmt4C/QBusznyKRJZ1sFnQY4eYUbDCfyakmnTnch10pnEnFq1X1k"
    "nUOKYlT3H1lI6SgdNzqLbsPev1556rZ+vSy5eoR5vKbfMEfvRHW0cSaTvJwRd29ByYO9148+IhoaD0XvouKhVT5Wu6U6aXOsgE3z"
    "3geCV2W5YZEZhY2OjmCJStGU/v4KVTvbRaOvt1HKQPA0C6pbaZmjuVDcYpK5ZBUQv8LVFr73ybEAQUHjUKdO9u/UPfO20yn2DLeb"
    "onRXy57I6ALNsOox7tBfoxV49cPliDwYdPz2D5Lpxkk61BnnObyhBXsxkvVgDk7eC4hOpWNWccnO3hpPsf0YqXd5ZZBPff8zwgkY"
    "89x1Z0WVee6vKXqh/y/pjh7dtY8O/em4uuQqetjLXVPEyYrwKnLND1ldprQRNoru7ie6UhCZq3mGxzNegvOJ7FbAnTXrgNkuOnL6"
    "aH5+4dn/8nuaJ11z8dP9hHe2uzMZAUF8aIXOkrxbZsroap79a80v2/ONwr38vqI78qAOiG6VEKo5QCC+Mgo+/3xwmEHedScfXd90"
    "RoyPye9ZnjNU0Mbozp6TJ1LTfKTZlQlWs1o/NdR1hOnAK5J9Zd5yIEnoZER34ecqg1lC2CMC8uJRLaZ9NmjLd5W2wTbxzr/LAMF2"
    "XPi3l/vBF8qk5FVtB/PMKg6jX7r3bFN/PXvLaee++NjutAbfVph3uMdYpY054UQFku5FnuByindK3/lKYV7Pk7E93Jlqu06OjBgD"
    "a82iUHnynutgrenkuq2nToHsa+uFGTvQLVPtv8e475ZACLSQXQL7QaYd/7ueW3yMKDbuL55b2EjUuDni0p2+tjt/46iRsS/DHSny"
    "fk9U49Cle5+lznsmc4JomNaHjy6y0scYHMrBgG171l4581xp3jM/Zdeke0hs0H4oCA5v1M7cE6p79om80CKiVTzngXyuN5mDvbqX"
    "ZWdqoNMLqkm1h8R3gOi2z6a6FRKoXq7IvQsnufD9nbGp7nNk/r1xIgMG5ITy5GXtlxWPXFuz0DDfL6obB+ETZL5GRJk405ENBif1"
    "uvFHWTL3XZtzmTtPuFPmnG1R50FT2XlGL9UWap5TOixAyF5NrB2UThHW+1B+Vlp5mCuswznW8G2M21Xfu9G2/PtY6+iK0ToWuuuN"
    "fP9/g2FHa5Awsw4/Z9YaF3S6rHH1/Gyjmyu13mPPuZ/FvrMON6wqF5+L62zUMLgpo4OKKSgeiV4b7e2hsa5LZydFseCNBo+t47e/"
    "jeaKgxKRYhVsMlc8j5PVct+QOYLKmTzrfK+eJ0tt+QKGmIRPPgooWociZYCs2oCV8/T74zOClXeL+cS8frnX6/bdqDkbO8/w3Wtl"
    "rcfxVq+fspqHp6r4n4gZyXdeuP5BohNbMeFV8gn/QMWAjt/od13mQsQZW7Dipc7FAB1adD1h7/lB60fZJl7vK1nv5F/2Pilui16B"
    "PJ9+vxqgmB+fXSTVNKnKfbIS4DJF1R7fGvPND/ofrVeJV7DeWcE0T/e7z85k4xmhUsrf9XKbKdkFd56tijGflVvpXk7xeRv1Y/rv"
    "zQgVtZj6cWIPIq4n+xr4irGg4pxFL56yWamz63I1zZrscWQrxzVU/q3f/iuQMz99Bl6evT9zLhXVc8rzZdn4b9znzPmm76Pikqz7"
    "xPakzjkAdU0+MneFW9R9Zgyg3sM642/fqXRUT/uW+l2cQSF9Q0E1O/5w7pHhmy6y2VHU2jVoHoqbB73lNqWxYrVBdo2ypzVHDH81"
    "zqrm4WbmIPLfaYz0jvhvG4sWEnsIHQKu2B/A7sFdcGA9tOcCxcEdyxoHxedln85ko29RQ5VtphgABpsJo30jOlv7O4pHlmwnaK00"
    "uQJMZ3C5tIeuG/myNBdXVjeidssK2c65xxpoilv/nfbk1YCRUGZv6grvnNGPwu5wpVhYNLNYrDMPqEwgZmELBIW12YzVffq+dF/p"
    "2JcRnMFmqVXJ9Tg0+5n53iWVGLwzruR0NQOS0uooHIc0h7TfCpNy4ixG7oRZYHG/KCfVFnve9w+LnftvWlcfj/N98ju/U9/J0IoQ"
    "lG1xj3twVrscbQe9Qa4ZuD7HzabRvvkYo5tZr03mVQYDqee+t49Ge4HvZXTxDXJgVoz5LQfRS2LUGFTS56XEtXbvjfXtTOljgHmK"
    "NZMYoXF3+uzjfIzn5n0y/JYVqsbFOdASYTq2iSbxrg50vUfcke533tRYuAuD4/3qsyCIegXyCZfv4V0/10AHfFXnWVHtf+Yukkbi"
    "3ZquhFqE9VkQlitey41ENxmPYVTdC0PHivNn76gCoCrMrC7lMOz5PIar4BAV4KNMyE9QirKfJ7jHe8iTdRz2O8U9vmOUpeQVwdo8"
    "HflAEtV91y+wWlF6/PtZQz+3p7T/VlBzfT5Cg6Fp9mwLGljbB/vdI7ppNfPkmPbNYPUpc6TOdFONj2lOIZda36Cy1mfmJrKz1Y5o"
    "AzH6v3vos6bPp+FKhzYe/GXWfVsv4h3+YA4Obl09qRSKZAJ7U2M2pCn7IDJHUEinWKt7TI73bDxkKlfp9yA5fvr75eyvgu7NOwzB"
    "lZsv2ejGoqlm93vcTpHPEoElyPFdYl3Px75f2cZzAA82Xk7G+jXxjZI2zYmTHbZAaDiRT3cIYzmFRCcPsxlwV85X11ihmwQIcXTV"
    "Ffl3GbkaeU6wItZulRaUMHaobrJUbP7d/N7IZ5TWlP26FcdPvL/G0KaqivZC4A8Qkwrag3xqMH5xBTaZF0gidcz8kq29+66mnNQ1"
    "jP+Pjnw+++HbMiudoK6Wai0kcgZG89XJyyR0X+PvIx6Pxa5l4tmajmlvttpvEf+Bu/+uh8GttodL7rheunE4xuPuw2RM+YcImvj+"
    "0KhnnQuhWCi6YtsbPG5Os3s+OZ3iFBcd9qajUtb6LEtPSfsxf8GXYbRRr7JvOHuX9h2d/fTsB+im7Gejw+JEh7Ksaem24NhniTrO"
    "yY5kzllKF4n1FWZ0r91fzeCR/A+PzuvivpF37Ej5ymuwxsj95kSX4w7zxuc5o3PQ6UROavVxN5x7x+bxYUfjp7/tHL9g9t6C1tLh"
    "5pP8A1WvTKOzgKK49FnIJ1gOyQbRfX/H7+ul7nvRc1gf6bXZe/s9PkDv9mbTzN/G/Lbma+eUZpbke/PPESPMlB8d+JUriieg2cQ+"
    "EfR0AkaCkv9szyHyXZbkE+Q+gz/JGeZVh5hz/sZ8JsR+zJTHPjN3Sjp5NEZsIp6yyNeT3Tp7rn+rGubXTKee0+2ga5ianSlCbPJb"
    "WGw+o0mviBjfnseuv7CseyPUcKBLe7j9/VAbIFbEHr5mhKroDu24s+duO3s5RK9gJxsfdkswi40PghsgW0RxnlVLGB+Kpcn9/bvO"
    "JbXJjlMMwZgDILdDVs1gVOtGYmvumFDXvCKflfzQKmuJqzNHPUdZaq3CrtZdDDifFH2M/K2e3w86O1Q+lM6VVYV9MMRk9JzSPQhm"
    "kOoGuAFmuePYlBl9WAkCrGRkfwwzGs4WMPq+GnYyzNtMMyxRTOsLKwlYmugd0hqtR5O2+w60P0xj8w3boTCaCEOG/i7i5P2kPWMW"
    "Wfl7a/ST7Bb+9/iyCR/KffJNCur7z+ZvP+e0vg3bpWLy1GOi2Az13YZidQHqC0xAtIeRp14opDP8adRZjo9zVimQeMw+xzu0TMmX"
    "NEyfgtZqjaYrXQfPuUxN8i69YtS0/z3PYBPC3+eCFJVx9450vt297cQfVmva217Tvvj8udRxLO2HKjNDvfWF0W3faN+u6KyZc83M"
    "MDg747DxS9QD5qMaMSM/zeMazN79ue1u7y935Jc/v1lGG42XESbunr+IMXLoWEg6vdV86M7/qsPCnLEfgxjbv17n0s2v8QsmNxH3"
    "fYXtROM45Ddmf8S/q85a1vZm5Kea152+R9h03qnFhmTo0NGzDGHXaR71XNO6pDjnQTSPMlg3zBw7DAL9rpnjH3ovB/6D0xVcH/G9"
    "jCafs1awrn2wmqOjssBMLvYebJP2UlflTs13Ybcw93C7QbRqRyymFUSbtqfYbzOgG7kzfaA6QzAGfO/xaK6XH1aX8P2j5ppzRwm0"
    "PTqO63NXe7PnVzDvWn9z1F/GOtgNBkXig3ue3/HqIbMGz3PSKxQaVYMlyaoFxdikFf6C5gp5seWO/LV38blkfE2Js3H+kp2GDanT"
    "3tHX5zVzoL1ixhmYuTfXQ457pNkjyUbwHLjXSMQveg65Gylj/H9BdUix5pt50+MNsv0Hebewy90i6rfIp+jrbI026Wn2ifSclVPr"
    "R18vec6bMWr7FljlBNSwcc5JnqGmddHiOsA0T7SO5Sy0LM31gtE3ZU3jCLHnR99XLArk/4OZeQSWKe6QxPOxbjD5EJX9sFyqP9ce"
    "3bh+CmyKsE1DlUg6byR+ofe57r4Pg4oH3WPOs4LVWD/jkm3ZcmI7yFQutzA113mRv1nG8co7d2VTrKz3s9T9F29Jlu+MffZq5+Rz"
    "JOdufc06fZYxO7RnsmYeizGQp5if3TH14CdInjJ08tCOwkKkc/pmj414rurI7WzBTobaG2Mbk7mhcrSLs+bSM3bB3likcbRY45YZ"
    "LYPnAjPaVz2XpVPFWdxZj+vQu34Wv0Z1cMevpTs+zfcjxCIT5N89xsctx3NmXDRrEvlxxkDkQ7zXj9Dkw0vclU/XAbOQ9SP/PF/N"
    "TILO7/bC0lVYQ0+vuS4hh75160fMxF67vVjB489jeMl9qPHYd9kVVuPRuoVOqF1CrYjRw+NqVKBY+1Uwll1oTb4yWw7YiU1syV0X"
    "GOs0fH1YpzQHY/knNe+rENp9mfMp6kq338S+tD5Ut5jBWyjsZRzfw6zVwo70c17yMNdc++gCIwC8qKs9qPUjZ8htMMtLg88DILjr"
    "9K4Xb/2MM7T1+qS7wiOpR+hn9pA7YdwtGCIkf+nUarzi1aSdUE2wthd2H3706/AQ4YxAHuFdzqIC+zlDMH3NDY7it/hoYEdphdxV"
    "kPEbrWHIe7PWgtrbcWJ87+4K9hM1BNhBrUoGP4Y+23FeTr13nQfW79d29Rm2wdDVrUQnDDphcc2m6S6/2mp2fFvHATu/HQeeo6Hy"
    "65Jvfs6NfMXi9u/Wc9Jn5UFwDy3WmozZAY4futNJkNL9VDYZ9hM6uTPMHfbOZsTPQs8dq78UMa/k6y6ycwhxf8LUc1hvV82Ve/0h"
    "mL4TdRs1txvkcGL3lhxOnCGS35f1a+NnTryulBFv+S17Fh0nrMcJvLfUXJJnFO8lxlm2fjcqO1Wvc9ezx+fhMIimFO8x48+Ez8Ir"
    "8ssVa5jkCtmGB1kdO5LPY59uJLGX0nuls6q743FPKjdv4zzXSK8aXWcfODULxkdXJP9CY9Hnl4l7mqYe0TVKSu5nKu9nlRWg+GXW"
    "fAK7YH2314d7zb7OqkezMcf7yBU9mh4HrrsqTHhzXipBt3XI7E1gKo+icMU2bk7XPqCmPapBAZTiAV+U62T9XelYNIJNRCwaqzGa"
    "M9is19+ptbo286Kew1PMRlZditag+UzNiWZJR20GayU5Zx9J7EVK5ULWKNtj9nWQuyQbPWjvmLll2F78xaxYZclXxtlM0+8yOGHT"
    "s7WOWblRM+sbn5P8FMSui2Y2lnqbiZU2WBjBlf4x7vdi3CB3vYuWbqWYcb9lJl7Idrg+qXVyQr0opgpk55n9d3q3E/H7eN8r9QXX"
    "H3Q/T/qFjTO5+zNM8PtY/rEPNhP0B1GcTbFIAZ34MdWNuI+sfW1+B+iNQLwusU5ryXWIstS8Y3+zqjvAfgF3LGyZdxfX/XKhX7R9"
    "KYLJQz4L/p9lZanENJbJhuJMXgqbTD6hGmH8eqnrOTVv/Rwx38sqwzH7eib+B/U5Gqs7HvYTHRwp+oMyflsZke+WqO8kfCBhnOTe"
    "A3Ne4vySOc3EeXyzbnfpHNt+MvN8D+k6XTk6N0eZ7E3qWTL3PK2pE8yvGfFZ21VZzcZK1ZW2+hl12T+/ZwaLk3q2TNz+d1kPFE4+"
    "01aRT3SEajGdGeadOPV8jSfCGlY1F94frxT/a/sNnHOWvba1nENJ9SMhpmZFIbAg0nWF8d3kPOk+tLZ/DySvkwOz6ETbWu4BeAaL"
    "95Luj3zfTOxP/YhzcNj73JrzX53fyPkPg4dUb6TkNjkfIszzfKbXkXNdjKwaqi/YiJDOyNKbO97xQZ/77meMGbC21Hf/Fmo/nn0u"
    "yR9vlV3ZqvxxHMMieeOt1NhtvKHtrJ3HeC56q69HYwfTyUHynsz2qHLR23epoz0fBRMCzflKAf7CzzmUhSrvg9tTOOjYGB1bKXGU"
    "uu9CFBaebV5b9YdYfJ2qbapnaHg7418gtyAxWNfg+AbtSPINislWKcu+kp+yFsXa9DORXYrwPMkx6GeVegPY3qMD5yHbEbOn470z"
    "IwrU8eK5fzdPbGPKJPbI353CHP23xfXLOwJ779izue0RXUdyEsqfZ0UK8VGDPHfmS56PbL7ObY1WxXfgdEaHyPQgSu5K5jauiMW+"
    "b8Z9Cozvy/SHMnoPmYlU10DVfAa+zkFapqUkpkXmanAAjjsztxNTaRJcRKwXZc64asbc1pMqrk4eNR473/7xGa6ZijQ+x8HT2B5i"
    "jd9I551t/U/5vs14fHd5ni0zp337xjUlxF1Sa5AauHxWcD+b1KRG/gv3rwCDxn0kV0ohAPV0fM7/ffIMk/sWBMOGHFyr0DuWQuk/"
    "U/V1rHF8zv99Mpf+t+DsnOt5YMCWfpzH3Tgf4Vnl/ApkjGcY6Q2GUCnVO7EzsPadE1jB2wv3yZd9zZuvxpaoWcTqvuESNVFhQmKV"
    "OK6XGlv1ZT/LCVwo5l3fZ2rOyorbD52lcmBrzsI94LJ16PrwTmoPbu1TarYKr6nmGrFkwl5nvHeVx+3wXorVUWxtM9Omhkv7+YRz"
    "F92RtbNRGGSwlcfti5PrUXv4xB5dZ+RJ5bxXecWYT8+KM4Ujs1s6CmB07p/aS4JNLpcEd1u50QrEgi+vFLX/9qbudwqv8+HkFpK5"
    "/K9z8ifY6sd5eTfAZAlW5/FNfMtzeDFP7V9RSmFscLXo5nlO7MOWL9glN+fEPuzZHGfvKPVXM4eX3EvilWNI/jyYe0QdVnDZ4zz6"
    "WZkRBKqGe9Wf6zErbHmGXrXiQzKfb/Ki0ehrG6znVvpm+T6vObOeBni+soOl1/Z+9qWigFND/nRqDl/iqo8hsPb5jvZxNidt5T9i"
    "/zXWUnEslHW90uRI9fp18pm731/7Y7cn68jov7QxBNmvWkHhaLsLxq7QWuT/Z8a7LjNgk23eXKS0nXWOfJN74tQcnbXRNelxwtmS"
    "8tfJt6+j1/N8LYTrKJfZbsOCpZg7r75tw8f5gWOzl2qOHbte8cJT92UVBcbjcF2KMWKBj5wdagvAkkDF0lUlvvqb3915/o802+BJ"
    "f4Dj5KHi3IGvo7AkzmeV8Fy+4IwP8BfnDmV9Cf9EMNiEZKMNlsOtlUp+8lXlu36MfVpP4NSoVrYj3S8SH/ta96a6NRLBzidrGvBr"
    "Yr3tOk986vuby/rWYvWQRC3lcXMpnn9qcrwDnoc+1wJS+E6rRil8Nm7ukWwqq4dmKsKfVPbUPECr+sekUryi84PPNGBGVWxLfhzi"
    "HMZR294SqJ1h/Bl8Ffht1ufnxnAyZ4be19P5sq1mqE/2DKrftTr3lXabuYOK5aduq945qc4Qmh5NYZBkXz41lnqM38H2Ojk5/Sn8"
    "BP0s0hM18EbVT3MGaZWc0/PhqJFWVL8Cq6U6n9divXg6lhTuA2cvaDX3ODYgI5cIvIfiwlBrDHZwKSpW6CPu7gVXGoHZbiFY1hQe"
    "2f7NYkGAe5UcTFe4MFK9PKIEZsZobStyvpwnwX7QuULwZW0kZyO8HmDHQ+4G4zVKpod9nEMoGeufy3WC+bCGGgxqKhSvqHdgGP69"
    "LI6HknpP6Zp7oGyD5Ihmh0TdWvE+PLDPTPPwDrw0++/t2c6tsZ3uudB4OTmH7BykOSUSGIePkPNolk+tYewHM8GrPGS6TuBgMuSe"
    "qt6bqD1sGr7hFyDbTnYgaOUSinzCKSVYe/V8en1CkRN40pbtQV8V86yWkejB4BrCIb6ebf2EMTSxOoVa1zKeE2P/6vpJfid1Tf0+"
    "3ef9oRQe3P5Hfu5Gtt13FeVje6axVtexzJ7v6qzl/JvTb2v+nmWL072Gcl11/sta9rafXJO1Y2vE2Qmv815ufDMp3IyL18Wbm+ub"
    "XDi6vr66LuQnhUIh9+O6GI7z/mga+j/Gw9F0fOP51144zl2Pp+FwMgr/61//+l/ipcFM"
)


def _files():
    files = json.loads(zlib.decompress(base64.b64decode("".join(PAYLOAD))))
    for name, entry in files.items():
        data = base64.b64decode(entry["data"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"embedded file is corrupted: {name}")
        files[name] = data
    return files


def _extract():
    """psdl2 패키지를 실제 폴더에 푼다(바이트 그대로). 이미 같은 내용이 있으면 재사용한다. 풀린 위치를 돌려준다."""
    files = _files()
    here = globals().get("__file__")
    bases = [Path(WORK_DIR)] if WORK_DIR else []
    bases += ([Path(here).resolve().parent] if here else [Path.cwd()]) + [Path(tempfile.gettempdir())]
    errors = []
    for base in bases:
        root = base / f".psdl2_pkg_{PAYLOAD_SHA256[:12]}"
        try:
            for name, data in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file() or target.read_bytes() != data:
                    target.write_bytes(data)
            return root
        except OSError as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("cannot extract the psdl2 package anywhere: " + "; ".join(errors))


def _console_write(stream, text):
    """콘솔은 인코딩 오류를 '?'로 바꿔 흡수한다. 콘솔 때문에 작업이 죽지 않는다."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))


class _Tee:
    """로그 파일(UTF-8)에 먼저 쓰고 콘솔에 쓴다. 로그 파일이 없어도(LOG_FILE=None) 콘솔 쓰기는 같은 방식으로 흡수한다."""

    def __init__(self, log, console):
        self.log = log
        self.console = console

    def write(self, text):
        if self.log is not None:
            self.log.write(text)
        _console_write(self.console, text)
        return len(text)

    def flush(self):
        if self.log is not None:
            self.log.flush()
        self.console.flush()

    def isatty(self):
        return False


def _fail(code, english, korean, early=False):
    """오류 코드와 조치 문구를 stderr(와 로그)에 한 줄로 쓰고 종료 코드 2로 멈춘다. traceback 은 없다.
    ASCII 콘솔은 한글을 못 보여 주므로 영어 문구를 앞에 둔다. early=True 는 로그 파일을 열기 전 단계다."""
    line = f"[psdl2_job] ERROR {code}: {english} | {korean}\n"
    if early:
        _console_write(sys.__stderr__, line)
    else:
        sys.stderr.write(line)
        sys.stderr.flush()
    raise SystemExit(2)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path):
    """현재 작업 폴더(BASE_DIR) 아래면 상대 경로(/ 구분), 아니면 절대 경로."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _option_values(argv, options):
    """argv 에서 `--옵션 값`·`--옵션=값` 의 값을 모은다(옵션당 값 하나)."""
    values = []
    index = 0
    while index < len(argv):
        key, equals, value = argv[index].partition("=")
        if key in options:
            if equals:
                values.append(value)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                values.append(argv[index + 1])
                index += 1
        index += 1
    return values


def _marker_path(name, argv):
    """표지 파일 경로. 표지 대상이 아닌 모듈이나 --out 이 없는 작업은 None."""
    if name not in MARKER_MODULES:
        return None
    out = _option_values(argv, ("--out",))
    if not out:
        return None
    if out[0].endswith(("/", "\\")):
        _fail("out_must_be_file_path", f"--out {out[0]!r} of job {name} ends with a path separator; give a file path, not a folder",
              f"{name} 작업의 --out {out[0]!r} 가 구분자로 끝난다. 폴더가 아니라 파일 경로를 적는다.")
    return Path(out[0] + DONE_SUFFIX)


def _output_files(name, argv):
    """작업이 만든(만들) 산출물 파일 목록. 폴더 값이면 안의 모든 파일."""
    files = []
    for value in _option_values(argv, OUTPUT_OPTIONS.get(name, ("--out",))):
        path = Path(value)
        if path.is_dir():
            files += [p for p in path.rglob("*") if p.is_file()]
        elif path.is_file():
            files.append(path)
    return sorted(files, key=lambda p: _rel(p))


def _input_hashes(name, argv):
    """argv 에 적힌 기존 파일 중 산출물이 아닌 것(입력)의 sha256. 키는 적힌 그대로의 경로."""
    outputs = set(_option_values(argv, OUTPUT_OPTIONS.get(name, ("--out",))))
    hashes = {}
    for token in argv:
        for candidate in (token, token.partition("=")[2]):
            if candidate and candidate not in outputs and Path(candidate).is_file():
                hashes[candidate] = _sha256_file(candidate)
    return hashes


def _marker_state(marker, name, argv):
    """None 이면 표지와 현재 상태가 모두 같다(건너뛴다). 아니면 (사유, 한글 설명, 영어 설명)."""
    try:
        done = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(done, dict):
            raise ValueError("not an object")
        if (not isinstance(done.get("module"), str) or not isinstance(done.get("runner_payload_sha256"), str)
                or not isinstance(done.get("argv"), list) or not isinstance(done.get("input_sha256s"), dict)
                or not isinstance(done.get("outputs"), list)
                or not all(isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("sha256"), str)
                           for e in done["outputs"])):
            raise ValueError("a key is missing or has the wrong type")
    except (OSError, ValueError) as exc:
        return "marker_unreadable", f"표지를 읽을 수 없다 ({type(exc).__name__})", "the marker cannot be read"
    if done.get("runner_payload_sha256") != PAYLOAD_SHA256:
        return "runner_changed", "표지를 만든 때와 이 실행기의 psdl2 판이 다르다", "this runner bundles a different psdl2 build than the one that wrote the marker"
    if done.get("module") != name or done.get("argv") != argv:
        return "inputs_changed", "표지를 만든 때와 작업 모듈 또는 인자가 다르다", "the job module or its arguments differ from the marker"
    recorded, current = done["input_sha256s"], _input_hashes(name, argv)
    if recorded != current:
        changed = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        return "inputs_changed", "입력 파일이 표지를 만든 때와 다르다: " + ", ".join(changed), "input files differ from the marker: " + ", ".join(changed)
    for entry in done["outputs"]:
        path = Path(entry["path"])
        if not path.is_file():
            return "output_missing", f"산출물이 없다: {entry['path']}", f"output file is missing: {entry['path']}"
        if _sha256_file(path) != entry["sha256"]:
            return "output_modified", f"산출물이 표지 기록과 다르다: {entry['path']}", f"output file differs from the marker: {entry['path']}"
    return None


def _write_marker(marker, name, argv, inputs):
    outputs = _output_files(name, argv)
    if not any(Path(v).is_file() or Path(v).is_dir() for v in _option_values(argv, ("--out",))):
        _fail("job_output_missing", f"{name} exited 0 but its --out output is missing; no done marker is written",
              f"{name} 이 종료 코드 0 인데 --out 산출물이 없다. 완료 표지를 쓰지 않는다.")
    done = {"module": name, "runner_payload_sha256": PAYLOAD_SHA256, "argv": argv, "input_sha256s": inputs,
            "outputs": [{"path": _rel(p), "sha256": _sha256_file(p)} for p in outputs]}
    temp = marker.with_name(marker.name + f".tmp{os.getpid()}")
    temp.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temp), str(marker))


def _check_requirements(name):
    """작업 실행 전에 바깥 의존(torch)이 있는지 본다. 없으면 오류 코드로 멈춘다."""
    for dependency in MODULE_NEEDS.get(name, ()):
        if importlib.util.find_spec(dependency) is None:
            _fail(f"{dependency}_required_for_{name}",
                  f"job {name} needs {dependency}, which this Python ({sys.version.split()[0]}) does not have."
                  f" Run this file with a Python that has {dependency}, or remove the {name} job from JOBS."
                  " Outputs of earlier jobs are kept",
                  f"{name} 작업에는 {dependency} 가 필요한데 이 Python({sys.version.split()[0]})에 없다."
                  f" {dependency} 가 설치된 Python 으로 이 파일을 실행하거나, JOBS 에서 {name} 작업을 뺀다."
                  " 앞 작업의 산출물은 그대로 남아 있다.")
        # torch 는 버전 줄을 만들 때 이미 한 번 import 를 시도했다. 다시 import 하지 않고 그 예외를 쓴다.
        error = TORCH_IMPORT["error"] if dependency == "torch" else None
        if error is None and dependency != "torch":
            try:
                importlib.import_module(dependency)
            except Exception as exc:
                error = (type(exc).__name__, str(exc))
        if error is not None:
            _fail(f"{dependency}_import_failed",
                  f"{error[0]}: {error[1]}. Importing {dependency} for job {name} failed."
                  f" Check the {dependency} installation (CUDA and shared libraries). Outputs of earlier jobs are kept",
                  f"{name} 작업을 위한 {dependency} import 가 실패했다."
                  f" {dependency} 설치(CUDA·라이브러리 포함)를 확인한다. 앞 작업의 산출물은 그대로 남아 있다.")


REQUIRED_VALUES = (("WINDOW_UM", "baseline_ncc"), ("NCC_TOPK", "baseline_ncc"), ("MAX_MEMORY_MB", "baseline_ncc"),
                   ("HIT_TOLERANCE_UM", "eval_position"), ("K", "eval_position"))
SETTINGS = (("WINDOW_UM", "baseline_ncc", "number"), ("NCC_TOPK", "baseline_ncc", "int"), ("MAX_MEMORY_MB", "baseline_ncc", "number"),
            ("HIT_TOLERANCE_UM", "eval_position", "number"), ("K", "eval_position", "int"))
INPUTS = ((("QUERY_EXPORT", QUERY_EXPORT, "baseline_ncc", None),)
          + tuple((f"CANDIDATES[{name}]", export, "baseline_ncc", None) for name, export in CANDIDATES)
          + (("CORRESPONDENCE", CORRESPONDENCE, "eval_position", None), ("SEARCH_BATCH", SEARCH_BATCH, "eval_position", "--search-batch")))


def _log(text):
    print(f"[psdl2_job] {time.strftime('%Y-%m-%d %H:%M:%S')} {text}", file=sys.stderr, flush=True)


def _check_values(required):
    """EDIT 블록에 FILL_ME 가 남아 있으면 어떤 작업도 돌기 전에 멈춘다. 그 값을 쓰는 모듈의 작업이 JOBS 에 있을 때만 요구한다."""
    active = {name for name, _ in JOBS}
    left = []
    for name, module in required:
        if module in active and globals()[name] == FILL_ME and name not in left:
            left.append(name)
    if left:
        _fail("edit_value_required",
              f"{', '.join(left)} still {FILL_ME!r}. Set each value in the EDIT block at the top of this file and run again."
              " No job has run",
              f"EDIT 블록의 {', '.join(left)} 에 {FILL_ME!r} 가 남아 있다. 값을 정해 적고 다시 실행한다. 어떤 작업도 실행하지 않았다.")


def _has_job(module, flag=None):
    """JOBS 에 그 모듈의 작업(인자 표시가 있으면 그 인자를 쓰는 작업)이 있는가."""
    return any(name == module and (flag is None or flag in args) for name, args in JOBS)


def _invalid(name, value, english, korean):
    _fail("edit_value_invalid",
          f"{name}={value!r} {english}. Fix it in the EDIT block and run again. No job has run",
          f"{name}={value!r} {korean}. EDIT 블록에서 고치고 다시 실행한다. 어떤 작업도 실행하지 않았다.")


def _check_settings(settings):
    """EDIT 값의 형식을 본다. settings 는 (이름, 모듈, 종류) 목록이고 종류는 int(양의 정수), int? (양의 정수 또는 None),
    number(0 보다 큰 유한한 수)다. 그 모듈의 작업이 JOBS 에 있을 때만 본다."""
    import math
    for name, module, kind in settings:
        if not _has_job(module):
            continue
        value = globals()[name]
        if value is None and kind == "int?":
            continue
        if kind in ("int", "int?"):
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                _invalid(name, value, "must be a positive integer", "는 양의 정수여야 한다")
        elif not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0):
            _invalid(name, value, "must be a finite number greater than 0", "는 0 보다 큰 유한한 수여야 한다")


def _input_invalid(label, path, reason):
    _fail("input_invalid",
          f"{label} -> {path}: {reason}. Fix or regenerate this file and run again. No job has run",
          f"{label} -> {path}: {reason}. 이 파일을 고치거나 다시 만들고 다시 실행한다. 어떤 작업도 실행하지 않았다.")


def _reason(exc):
    return str(exc) if type(exc) is ValueError else f"{type(exc).__name__}: {exc}"


def _check_cohort(label, path):
    """COHORT 파일은 비어 있지 않은 고유 문자열(비어 있지 않은)의 JSON 배열이어야 한다."""
    resolved = Path(path).resolve()
    try:
        names = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names:
            raise ValueError("must be a non-empty JSON array of product names")
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("every entry must be a non-empty string")
        if len(set(names)) != len(names):
            raise ValueError("entries must be distinct")
    except (OSError, ValueError) as exc:
        _input_invalid(label, resolved, _reason(exc))


def _check_dataset(label, path):
    """psdl2_dataset_v1 manifest 를 가볍게 본다: schema_version, source 마다 export 파일이 있고 export_sha256 이 같은가.
    export 는 해시만 계산하고 자산은 읽지 않는다. (manifest 경로, 문서)를 돌려준다."""
    resolved = Path(path).resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != "psdl2_dataset_v1":
            raise ValueError("schema_version is not 'psdl2_dataset_v1'")
        sources = document.get("sources")
        if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
            raise ValueError("sources must be a non-empty list of objects")
        for index, source in enumerate(sources):
            export = source.get("export")
            if not isinstance(export, str) or not export:
                raise ValueError(f"source {index} has no export path")
            export_path = Path(export) if Path(export).is_absolute() else resolved.parent / export
            if not export_path.is_file():
                raise ValueError(f"source {index}: export file {export_path} does not exist")
            if _sha256_file(export_path) != source.get("export_sha256"):
                raise ValueError(f"source {index}: export_sha256 does not match the file {export_path}")
    except (OSError, ValueError) as exc:
        _input_invalid(label, resolved, _reason(exc))
    return resolved, document


def _check_inputs(inputs):
    """입력 파일이 모두 있는지 본다. inputs 는 (EDIT 이름, 경로, 모듈, 인자 표시 또는 None) 목록이다.
    JOBS 에 그 모듈의 작업(인자 표시가 있으면 그 인자를 쓰는 작업)이 있을 때만 요구한다."""
    for label, path, module, flag in inputs:
        if not _has_job(module, flag):
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


def _check_envelope():
    """--search-batch 작업이 있고 봉투가 이미 있으면, 봉투가 이 QUERY_EXPORT 의 질의로 만들어졌는지 sha256 으로 대조한다."""
    if not _has_job("eval_position", "--search-batch"):
        return
    envelope = Path(SEARCH_BATCH).resolve()   # 봉투가 없으면 _check_inputs 가 input_missing 으로 이미 멈췄다
    try:
        rankings = json.loads(envelope.read_text(encoding="utf-8"))["rankings"]
        hashes = [r["query_export_sha256"] for r in rankings if r["query_export_sha256"] is not None]
        recorded = set(hashes)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _input_invalid("SEARCH_BATCH", envelope, f"not a psdl2_search_batch_v1 envelope with rankings ({type(exc).__name__}: {exc})")
    if not recorded:
        _log(f"prepare: SEARCH_BATCH {_rel(envelope)} records no query_export_sha256; skipping the QUERY_EXPORT check")
        return
    digest = _sha256_file(Path(QUERY_EXPORT).resolve())
    other = sorted(recorded)
    if digest not in recorded:   # 질의 데이터셋이 export 를 여러 개 가리킬 수 있으므로, 봉투가 기록한 것 중 하나면 통과한다
        _fail("input_mismatch", f"QUERY_EXPORT vs SEARCH_BATCH: QUERY_EXPORT -> {Path(QUERY_EXPORT).resolve()} has sha256 {digest[:12]}"
                                f" but SEARCH_BATCH -> {envelope} ranks queries of export {', '.join(h[:12] for h in other)} only."
                                " Point QUERY_EXPORT at the export that db_search searched, or run db_search again. No job has run",
              f"QUERY_EXPORT vs SEARCH_BATCH: QUERY_EXPORT 의 sha256 은 {digest[:12]} 인데 SEARCH_BATCH({envelope}) 는"
              f" {', '.join(h[:12] for h in other)} export 의 질의만 검색한 결과다. QUERY_EXPORT 를 db_search 가 검색한 export 로 고치거나"
              " db_search 를 다시 실행한다. 어떤 작업도 실행하지 않았다.")
    shared = hashes.count(digest)
    total = len(rankings)
    unhashed = total - len(hashes)
    if shared < len(hashes):   # 질의 데이터셋이 export 를 여러 개 가리키면 일부 질의는 다른 export 의 것이다. 멈추지 않고 알린다
        note_en = f" ({unhashed} without an export hash)" if unhashed else ""
        note_ko = f" (export 해시가 없는 질의 {unhashed} 개 포함)" if unhashed else ""
        _log(f"prepare: SEARCH_BATCH ranks {shared} of {total} queries from QUERY_EXPORT; {len(hashes) - shared} from other exports{note_en}"
             f" | SEARCH_BATCH 는 질의 {total} 개 중 {shared} 개가 QUERY_EXPORT 의 것이고 {len(hashes) - shared} 개는 다른 export 의 것이다{note_ko}")


def _prepare(base):
    """작업 전: EDIT 값·입력 파일을 검사한다. 하나라도 틀리면 어떤 작업도 돌기 전에 멈춘다."""
    if not CANDIDATES:
        _fail("edit_value_required", "CANDIDATES is empty. List at least one (name, template export) pair. No job has run",
              "CANDIDATES 가 비어 있다. (이름, template export) 쌍을 하나 이상 적는다. 어떤 작업도 실행하지 않았다.")
    _check_names("CANDIDATES names", [name for name, _ in CANDIDATES])
    _check_values(REQUIRED_VALUES)
    _check_settings(SETTINGS)
    _check_inputs(INPUTS)
    _check_envelope()
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


def run_jobs(jobs):
    root = _extract()
    sys.path.insert(0, str(root))
    import psdl2
    if not str(Path(psdl2.__file__).resolve()).startswith(str(root.resolve())):
        raise RuntimeError(f"another psdl2 shadows the bundled one: {psdl2.__file__}")
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    import numpy
    versions = [f"python {sys.version.split()[0]}", f"numpy {numpy.__version__}"]
    for optional in ("PIL", "setuptools", "torch"):
        try:
            module = importlib.import_module(optional)
            versions.append(f"{optional} {module.__version__}")
        except Exception as exc:  # 로그용 정보일 뿐이다. torch 예외는 작업 전 검사가 다시 import 하지 않고 쓴다.
            versions.append(f"{optional} unavailable ({type(exc).__name__})")
            if optional == "torch" and not (isinstance(exc, ModuleNotFoundError) and exc.name == "torch"):
                TORCH_IMPORT["error"] = (type(exc).__name__, str(exc))
            continue
        if optional == "torch":
            try:  # CUDA 확인이 실패해도 torch 를 쓸 수 있다는 판정은 그대로다
                versions.append(f"cuda available: {bool(module.cuda.is_available())}")
            except Exception as exc:
                versions.append(f"cuda check failed: {type(exc).__name__}")
    print(f"[psdl2_job] {stamp()} {' | '.join(versions)} | psdl2 from {psdl2.__file__}", file=sys.stderr, flush=True)
    _prepare(Path.cwd())
    for name, args in jobs:
        argv = [str(a) for a in args]
        marker = _marker_path(name, argv)
        if name not in MARKER_MODULES:
            print(f"[psdl2_job] {stamp()} no done marker for {name} (indirect inputs are not hashed)", file=sys.stderr, flush=True)
        if marker is not None and marker.exists():
            stale = _marker_state(marker, name, argv)
            if stale is None:
                print(f"[psdl2_job] {stamp()} skip completed {name} (done marker {_rel(marker)})", file=sys.stderr, flush=True)
                continue
            _fail(f"stale_done_marker:{stale[0]}",
                  f"done marker {_rel(marker)} does not match job {name} ({stale[2]})."
                  " Delete the marker and run again (if the job needs an empty output folder, move the old outputs away too)",
                  f"{stale[1]}. 완료 표지 {_rel(marker)} 가 이 작업과 맞지 않는다."
                  " 표지를 지우고 다시 실행한다(산출물 폴더가 비어 있어야 하는 작업이면 옛 산출물도 옮긴다).")
        _check_requirements(name)
        inputs = _input_hashes(name, argv) if marker is not None else None
        print(f"[psdl2_job] {stamp()} start {name} {' '.join(argv)}", file=sys.stderr, flush=True)
        started = time.time()
        main = importlib.import_module(f"psdl2.{name}").main
        sys.argv = [f"psdl2.{name}", *argv]
        try:
            code = main(argv)
        except SystemExit as exc:  # argparse 오류 등
            code = exc.code if isinstance(exc.code, int) else 2
        code = 0 if code is None else code
        print(f"[psdl2_job] {stamp()} {name} exit {code} in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
        if code not in CONTINUE_CODES:
            raise SystemExit(code)
        if code == 0 and marker is not None:
            _write_marker(marker, name, argv, inputs)
            print(f"[psdl2_job] {stamp()} wrote done marker {_rel(marker)}", file=sys.stderr, flush=True)
    _check(Path.cwd())


def _base_dir():
    """상대 경로 기준 폴더: BASE_DIR, 없으면 이 파일의 폴더, __file__도 없으면 현재 작업 폴더."""
    if BASE_DIR:
        return Path(BASE_DIR).expanduser().resolve(), "BASE_DIR"
    here = globals().get("__file__")
    if here:
        return Path(here).resolve().parent, "runner file folder"
    return Path.cwd(), "current working directory (__file__ unavailable)"


def _main():
    base, source = _base_dir()
    launched_cwd = os.getcwd()
    if not base.is_dir():
        _fail("base_dir_missing", f"the folder {base} ({source}) does not exist."
                                  " Set BASE_DIR to the absolute path of an existing folder",
              f"{source} 가 가리키는 폴더 {base} 가 없다. BASE_DIR 를 존재하는 폴더의 절대 경로로 고친다.", early=True)
    os.chdir(base)
    log = None
    if LOG_FILE:
        try:
            Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
            log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
        except OSError as exc:
            _fail("log_file_unwritable", f"cannot write LOG_FILE {LOG_FILE!r} under {base} ({type(exc).__name__}: {exc})."
                                         " Set LOG_FILE to a writable path, or to None",
                  f"LOG_FILE {LOG_FILE!r} 를 {base} 아래에 쓸 수 없다. LOG_FILE 을 쓸 수 있는 경로로 고치거나 None 으로 둔다.", early=True)
    sys.stdout, sys.stderr = _Tee(log, sys.__stdout__), _Tee(log, sys.__stderr__)
    print(f"[psdl2_job] relative paths resolve from {base} ({source}); launched in {launched_cwd};"
          f" runner {globals().get('__file__')}", file=sys.stderr, flush=True)
    try:
        run_jobs(JOBS)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if log is not None:
            log.flush()


if __name__ == "__main__":
    _main()
