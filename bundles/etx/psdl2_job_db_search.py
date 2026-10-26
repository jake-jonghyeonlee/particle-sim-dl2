#!/usr/bin/env python3
"""psdl2(DL2) DB 검색 실행기 (자동 생성: scripts/build_etx_jobs.py). 소스 해시 3859f9f86aeebc492e3846221e55f7e9dceb9a99dc542037fb7ce8210f7777f1.
제품 grid patch -> make_split -> train -> index -> search 를 한 번에 돈다. torch 가 필요하다."""
# DL2 DB 검색 실행기 (torch 필요). 제품마다 patches --mode grid(DB 위치 생성) -> make_split -> train -> index --dataset
# -> search --all-ready --query-dataset ... --exclude-query-product 를 차례로 돈다. 결과는 OUT_DIR/search_batch.json(psdl2_search_batch_v1) 이고,
# 평가는 psdl2_job_eval.py 가 이 봉투를 읽어 한다.
# 회차 위치(compare): 질의 = 변경점 ROI export (QUERY_DATASET 이 가리킨다), DB = 제품별 template export (DB_DATASET 이 가리킨다).
# compare 의 새 "변경점" 형식은 미확정이다. 지금은 review_roi_v1 manual 행 + mask_derivation.json 으로 만든 export 를 가정한다(PLAN D2).
# simcompare 4·5회차용 psdl2_job_baseline_ncc.py 와는 별개다.
# 학습과 색인은 같은 grid patch manifest(GRID_DATASET)를 쓴다. 수치 기본값은 없다: FILL_ME 인 값을 정하지 않으면 실행 전에 멈춘다.
# 오류 줄은 [psdl2_job] ERROR <코드>: <영어 문구> | <한글 문구> 한 줄이다. 코드: edit_value_required, edit_value_invalid, input_missing,
# input_invalid, input_mismatch, torch_required_for_<모듈>. 파일 만들기: scripts/build_etx_jobs.py.
# --allow-unversioned-producer: ETX 실행기로 만든 export 는 git checkout 이 아니어서 producer_commit 이 없다(사유는 export 에 기록).
# BASE_DIR 은 쓸 수 있는 폴더의 절대 경로로 적는다. 종료 코드 0·3 은 다음 작업으로 넘어가고 그 밖이면 멈춘다. train·index·search 는 매번 다시 돈다.
# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 각 작업은 `python -m psdl2.<모듈> <인자...>`와 같다. 모듈 이름은 baseline_ncc, eval_position(이 둘은 torch 불필요),
# train, search, evaluate 등이다.
# 인자는 명령줄에서 쓰던 그대로 문자열 리스트로 적는다. 상대 경로는 BASE_DIR 기준이다(아래). wrapper가 이 파일을 다른 곳에
# 복사해 실행하면 "이 .py가 있는 폴더"가 달라지므로, 상대 경로를 쓸 때는 BASE_DIR를 절대 경로로 적는 것이 안전하다.
# 작업은 위에서부터 차례로 실행된다. 종료 코드 0(완료)과 3(부분 결과)은 다음 작업으로 넘어가고, 그 밖의 코드는 멈춘다.
FILL_ME = "FILL_ME"                           # 이 값이 남아 있으면 어떤 작업도 돌기 전에 edit_value_required 로 멈춘다
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
JOBS = (
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
LOG_FILE = 'runs/psdl2_db_search.log'   # stdout/stderr를 이 파일에도 남긴다. 필요 없으면 None
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
PAYLOAD_SHA256 = "3859f9f86aeebc492e3846221e55f7e9dceb9a99dc542037fb7ce8210f7777f1"
PAYLOAD = (
    "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"
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
    "6f/kM3tMrH6u6UfX/Qznh2zp6X3N8qwJfA/N03R7mDvPXZ/RF3M1gS+QNSoyPue8L5eLTNsemevdODROP+es1k1VaUm2A34fesLa"
    "Us9p3ez2jr8D+7ga1HZJDnuxiebMuZNepae1wYwJ/6fOPVXXg4rqw6ucZa67eO/bQsPf9iNrr7fAu3PgDAv37GVrYu2+77O/PW1t"
    "GjxDZZLKwdN+uKd3otbcbhxhn1iusw737aqfG4efk+sGtKfyvW/3COr1ak38OHjN/FB48GDnMG/pdXMbbd11keillryW6vWTnAOv"
    "/VWwxzR5rayxhfiGgGVEXb0DWwvdNcSJ0RT9a/9UD7JaP+TbODoR4NCqRF/4r76/mtE7Tq4pnftaYozBvYd9LH27mgPJxVG6fr3O"
    "YTC+k+Kfr7Qd53PSy3UEzlIHCyr83q7NdXEm6ixQnK3llV7/dL6W0+/QyXkfPF/cGg3v92/Ga3hBdczB+7lP4xm8Td4Un0qgTnUV"
    "4vD7MnpEgrXUNSEP397wxrhz730sDhj434z1y7zwmj/AiX14LnjPKf/2j88uH1vj+MyZNb3w3lfjUGdVaMxZzyj+L52L+kwvhvBB"
    "Kc7Q4Dkq+VX4XFlr48h9PcX+Qk+Vi5noUhxC//6i+M3mcLmW6PgyAY3q8HnURN+z4gI+7fsai45zD3VIw+NYfFj93Kf3vuMTaLx3"
    "iOdSfFHF89dw9miCB5gxTvdzWu+a85R8vwad55abuETrpcVYg7TP/ogxOhzioc8GecQTOYcS2ZoHh8fSyU9wLdbk2kx/Bc7sUC7h"
    "59s3GOa21rfkOJX7uM1Ybfz69V3Nv/G2XKpass0vW+yN8ElBt6aY8/Km5vd7iyNpdHNJPVSVU7I4j1fn2YPPyNi9q0WaQ5ExRqm8"
    "j9MTFgdy++cmty5Y+xXqOd/nJ8K5UBqvzaME4h8vZ5KMbar5cqpP0vJuns5p6vCYOp9V+bxDOZhDXAWpeiFqI7Zmxj396p3e5rL5"
    "UqU/hfXJGuBpKL4k9LVUXlbwtZPhPJaf26Vr+rfqvRKexOHbhGKDJtm42U4wL7lFJFyznB/xdasi6dEVe1V4aJtcIudQevNPmp/n"
    "C2CTEL/d3ayennJnjoaavif4uymu7tYXCrtic3UV009bFm2HI767z210nuy+q+tQ18yF8CtS53KN7ODtVbmew30Q489yjv4XrzGl"
    "O8H+TOK5HR9W5/4ebD2U3gHthz37wtBD03NcGS/u23l77lfyTq4rb7RP7rk/7nP3qx23dU0FuSOseVqHE/DxuLEUv4eur1f0zRgk"
    "97bPJ7SJ3PEMhbtBrQGK6z7Eh4ae0DA4f9xjo/hHkT8bLCLEca4ukX5G6XVljkv07nyyppflIR+uEuuWz0g6e2ltSw9mozLc3Hfy"
    "k9dKXulH5pXe3HQz6NzZ992RvmJvToMaceJT0JqheAFzAZwisATARTAWDFy0H3Hncyo5/QifyfXBaVSZwHfYUoyveZG4357i4nxv"
    "jvlVOmzwD8n3ath6Pt6L1tqQ97XQfKV41oclr+s3o+WciF85D5kf3LJmiY9TQb89dAXn+ufdOJE7LuPaw30poZVydI7X9V9DGj76"
    "+t997uj89Qv0mYrxKJv/9s480xG55WOfMxsX1eF3tM3WhvPjMVWX+os5+eet/AAxHvA3xWhNZ/YX2S+6V73UqGRyOayCXL4/HpR2"
    "d0TX+Mw5ftZY6Weo3wd9KuG0al/tHF7n/yR+9aw5gb/B32OfJb694/MX/o70L37OfnWbJXrPXzHO/cVdxrzcAYOHfT3ieexEo+4e"
    "HFZXi25e9Dy7+cuF5emQGOm5CN6zaNPqTgTzgXMV/Q3j//7f//V//Ld/rSf9Qun8X//3f/vXsHBZ7p+dX1z2C8VBbjg6K5Xzr3H/"
    "Nb48GxUuL/MXpWKhkCv2B6XLUbnfz/UvRue5Url4kTsrj15z/X/9v3S51TqeFf6v14/+7H+sluu3zdty8X+u9nT5/+dfcX/Tx32A"
    "uHuukTcqJ9OGq7cdVNB5xlUVm9kFYMHg8RdwUr104gmjE6Dm3MHp9gmrhe4uWI4d7QpG37qdDcP9WaMya+1jivpE8Qxs14Kme2lf"
    "j1+kC9S7Bmc34AGAyaPG3ZDjmE7MGF792/Xbq1SG6fpN8prytHPphNasUmTR6N5giBq/1KroNt2iI0ErTg67ETOfq65Sff9Vr3Y9"
    "buiOV/P9Or7P3xvUqm+0steclbRzhwizoVi7EcnrLlttGUSh8raaRzeqMNbdjMnKk3cQL8FS9dLWLAh8SrjqCabDFFkcZEuE0WG2"
    "rcyvx3Tq0gn+uIlr5TWz06Gq3CWvsHbFDO90Xe8dqa4Qy3p+20THhWXJUh2suhPjrsrVpS/MXWWen8H7ftkzMgHRAHlwOGFbH0AK"
    "CqM1WGgiOiXIo8cYoDp+G2n2aXT40lwLupMsCjLZa5yMtJPeXwp4z1DJvc4zOmu3UnOO6/G8cFfBQI1R5rtEazRP76O6HzDa0F7z"
    "Duugzd7BTimVshpJD55vTViz0d0y6D5saW2yJ0Trkk6uaI95dt+vdC8KW77yMhAdq6re5+QFiPWvNVRabRUF1T6orwjjIe7XoGh5"
    "3xNGeCB7xvBSaN2u1fxsaX5czwbPCzYlvqZmm8Haa4jXOxZm/um2AmzkeCWM3tXyqLdgazUjS+Sy1GwxLhXNfMGKkWVkVnw+4dhb"
    "eG7YzzPKw/kZ6M2x83P5qzKXPcxVfq7SX5trsmd9Ze8xJAtY0WOkk0mxW9vxhU6GGpTQ61y15UzY1bLxmCu3n6cRBTbln62b6Odj"
    "LkqoMxywJe1hI6jk4Ck1w859CoOTYWgRRChON5x8tDaQwTDorH7FjYoyVUv9Lu9/AmVqFU0TzNYnKJ3eNH89z+pVmpJfT9NW9Wna"
    "rFqGqxNUPmt00i1QET+dASOJukvPc2SUVG2kJ+eJZJd0B4zprl6pqranJEqn7j6l5KM6IPqWRcOosmgVQxXRhzpvw1UPdFDV7gS9"
    "RPeke0+4M0oxsNULdz4qS58NP3AeptXAUsg3dX4dQrZJlE+eUOeZq2s0x+5Z8Bl3IiDguZvsJysVl/g9v+B8QGZ7rm07UOhi0++0"
    "Z3nldW7Luf4jJx0N9LycvQdzkKxtPQf2b3l4XVdGfRIsm6J0KWixF1Trvb+xkua4XpDOwsT131BJ426+yHw2yxNPd//SdToFbWcm"
    "BV0NVkxHb8HOsCjndcXT/L2ZfaCrF/tZOdGlnKcz6+ueIjF4mL15FVHWUj2Deh9TRmM1HGUf+d44xdinr+ewSO1p75ANEPveLcA2"
    "50f63Zmo4PbBsshBcRzo35pRWk0gjHUkoRiiOTJsHYkw1t/1Ecb1oqz1+t52ietogm0XM2POOGoJfebELkqK1pr5oWSQBAlHvoJi"
    "JODrJVizHd8rWN1Slajqti8qVUXNbotMEboXWYVQ5om76uh8o+h7ytnehxC6zPUZu37mWe7DrIr0mXVyzXmKcfr3vE800gpKtOaz"
    "qsvP+X33pI43d4/xmKQKLb4i7Qn978h0SdYLSaY3ZoPEXL1RZL2MDYL4xVRpsvbbCd2A6FqAcnxyjr44Om0PV6N2MoqOuENDYhUv"
    "0y370u+AFcYQvb7RbSv2mOfA7bxlG6cU+BRLq4qala3Qfzu+2xGMrWCDnTIDN9aGv36cbC+e/2oh/k1zRvvpvV+J0xnshN34zp5I"
    "nMhZWKtia/2fAywlyl/wu2HSfkRIMV2fw0o1x44BbBzNdyjm+EzlJ9kEruAO4HtFZZ39slnP02yk628Ye5lkTPDfIarAG7BFY487"
    "e1V+liqR8QXOFTrLVbnO6fPQ6TC2v5eO7AN7NzrWFpx6dioUVtIGxLE8UxM++swZt9qfcTz6hsk3gdbR78x0WtjzQ7osugkmfXdd"
    "O1XMC302dd35toqqvHc72ia0mbXIRVvx3tO+48/0uBy/8li7G3vnkWJkdtYOz6NCA9BYK7G7lslvfliM0qxlvr3bmWrPTJhiGZlo"
    "WQcWD8vEHlhb33s3bnBsZvMPWBe0x39bpsXWx4AV73Q3vFkvObA6D7m7PIKdX5ucUbc3Mwp31q5w9r1RW7vquKyOJuyUV5d3P25O"
    "6OqV+J7ioVE/3MWl0Rq5QfHqs1uYAHGGZ/8Y1jYj5G/w/PTsu4eoPIUaBz8jbLBU4RGTWbSnYsZnP6Ai/wk71KZM58TOi1/c+VZx"
    "jPfOfFun1qN07Dtre5Bez2kVExNLHEK17lmlwcudoNuffKmcvIsWVOV2zJZG6xU5vhhneOfxAvGDdEt/yrjmw8R35e/yfpB/5Ng/"
    "S/nBzgt3n7t7QTEl8dyXBvJcsz9SbfXRK3oOvbkd2WdiNjEwxmofPo06cq5H9v43lAuOVXPldzLbKD+Emeq9tcDqT66N8P1VJ+Zj"
    "FIbatxJjqncvMcdjQMH5Vvs2M8vwAXutv+f5Gqlxmb+JL8gqGip/CrSTzq9p5AMUkCSXNqhxpbwRZo9nJFzCB2J1iO/mJeWP1QvV"
    "AtByopanVb2U8jLOtkyW+1jZhNZKxzv6PJQqOewy2YkfVbYT3u8LvRX5NTm2J9Uy7OJv/+/8+wn9l/i95+dhvgcJdQjjezl56lA3"
    "C9SryS5xfnmp5+NXG4p0ifm4NWqzZZrP7SBftuvJZ6aZg7XF+MEJhd6BjRG/e0f/9HhTa+AY5ZXefLYHUzbn6H8E2LecWNBdi6qb"
    "y7fBifdGMcJ5P/Scck33HOHzAp1E/UqJf99wff0oyGSG8ynocxnfTKHTzXs8llGE82StQAzJCi7uPAc+o3L21aTPGfium+PJ5+Nv"
    "50lYbYDEdOdOserY/WlQTqmuTPcZUZ0Txh79vbTCEOfq3fujE9rxZZ6zukLY506eWb7/aWKAxHxmxAKJ9ZBgYdLxrv+5tmUdYP/7"
    "RKadpH+ajNVTYw+sRUaHcV3E+H4LVs81c3qSskbifMA7ZGbwFbOFFw0CRMdO7Es49/8yKtjOuoU902vV7egbtQ+hanFPVUNzbQjW"
    "UFZXzQG7464VxBp8VgRQzeyDsv8XXmNBFHrCrqb8WLyrpzyfc3Rv//fSSWP94h/mc6hHCpOg8yxdWb9l93ffMhnV7Dn4q+OcmZVE"
    "Z3TgfahnWTITSmVdZp/9R3UgayMOzIXr08M/u55TLIAO9KVX08uwHcLkiXuuU+u6e0QntxkD+QvuunRsQ+Z7SPltHnrVUWGvcV0r"
    "L2cl5xz2GiVtukqKCvWB/Lv2k/Lkcz7lEr6SYb3dYp2K7+ZeX9+T7A9qnp1HxYggysqBmEnHXBs+O4XtwlFKY98GauDorJj6+ec8"
    "2bWWPJP4geZzjfbVx/3+ap9gwCPbVJVxF+voWiJfZrMhezWl54gdtTH3OrO7H3ca1bJMf+95fE8+PZ1nv+0zt+jsnk17dC3yh8Z2"
    "HNcGlaNqJlsVY6yUX0d+29lSupfYx9Y1STW3sG1n8nuwEXS5G4X85s+Z8aMVavuuQs/4lu1HJ1D6iXzfbAH18p7kPF3kehYieeHs"
    "u5Hkwjl/Yvxg2lsfNHZhunjzzmOKlV8W2lYAHU9jGr0WNmWp85R3FKOr/Xw55liUfF2vYz4zXsjPaI2NgvaNx9ek/09S/jw6++/3"
    "8i5C61X5Q6k9e+yY6DsT8psz4wi9TkepfG2mUpOL+OV3kMzLqLm0WBEoR3D8tA74AG4uCywszjrb+/NfT67DvXPuLlSs0va/o/xz"
    "nXN6g5IUxw+MSKfzJb3GyCZFGzqHKcZW701qK3hejo/RlaDzicN9eu7r6KirMbPUDM+mUdN1t/4vKLaCxPxg3Zqs+D05uSphgKZ5"
    "hz1E1yP5PjR2MHLmyMf5IF9nT2vld70IxrblO+3DzX37H/9vPKBgIq5NNw5+hzEyd5WJVqXi+fXxDXnkpFfImdPZ8V55Q7c6sEDR"
    "Vp91gg2SvMf92zWUooB5WWn8lNT3gQ27Wtp77zRuaO1ivxoag3T7wJgkuvYkNjmrM3R96lo0rv9X3GktBF8V7TE2Vf90MUqPrASo"
    "GG/deBK/e+mCsV9Q8XfVq3PDnNYGMxiz486A4BzQngb7Wb+m8lByPZ3bBTJ3iXcutbcN8lZrfa1hgZUFJoMrerce5gg4NdiAmXPN"
    "KyfPcrXy48MzPSczg3Er0LnD2CAnrrzVfqm8n+fKmXpnBzBlnVIpY/4YvyW+6tn4adr8QXuJYwHNokPz98XP4/weuLHBHM+NmNeq"
    "N0o99HnDtUpWZFXdTBbLlGP83zQfPVVb9Qz1tIPrtL5bVbSKjLJxY7cepGvc3UJddZrhOyav7edOoYwxL2/0Zz0MDbMnzrYK0ToS"
    "tjLyBYByp+d9fTKoWVdZ4xerafg52x2j6OW9n6u1VezD9greUa2p+KtTmFEM6OYIWB2D1Z15H6tahmb3A44DypBAv8p8l8G6t48L"
    "M1Zmc5Sc/o21uOvU+8hSyrE1QoNrMc8uCnPCzPbHLHK6tsl1POxvi/T367rs88Zenp3XLJQ107U7xFrMIhvCC9GzvbYn78hxGjwB"
    "as4J9vtkLYpibtQdyadgtQfGBei8zIuqZen7hvKX/xj2QY3VZ6fZMEIbOScay7Fjkq5usDuhkykqC46rJh0Vat1SfNEc1O17Ql7C"
    "VZ22c5TPHcZiFL/FRLyncTjTYK30JZ/GGHFewslP4XpJbNERyq6hnMhgsIg2L/No37h90errBaWEeVSNlP3Jbu5wXkTFc3hfVmnT"
    "w6b8RicNPednI8r9ExgIvZeDtaZGMm8dlVPv4ZCf/A/UQ4PnA/llKTxVosZpmHb934M5uP4xiBzMQPfB2jgHW8WMtsgXJs57+DV3"
    "Bu+1yzrn0c0qWHV7bgOLP9F+AmOCmen12rtG18u9tugscZQsK/Sv6Y1TfyvP3bHYe6EeWGJsIM6dXls6JLC+uAumqH3EJt49zjRW"
    "PRY89LXTQ0C+MvuoM2aNxfVQ44LCH+NjOxQDFBmbDb+joetgqnbImFbudppr5T1WnVzzmfW09jHsFvOZiwuXRg0ae9Hxy3i+kgrV"
    "ofX5Urtyz9LE/tjReRdvVfcj13RfizmXMU1jnNYOFlfUfjqofzObvYPfbX01jlb7+c/E0fwBdte5ZgbGzsyxUqIQjGhi/Nlr+tHi"
    "oisn1Mwdv0/7647vt8pgfJoMFtNl6KyQWrP1y4Wh5UHq6N/UqQ8xLGYzR2XZ0aSiytWHwwaSrLelbHnX3/cBZhQwBMFXP/lawTyp"
    "2jfoE8kBZ/YC+6AVyQ02RNiP9Zn5t2v4GXOXmq8MNqRD77MbqIc56+LwtRZurtY7J44+n7qFFcUq8ayRxAE6tXTO1dv6FP1t7Z1T"
    "Ekteo2PYj6elnqFiclGU5b6iWhZWSNjBhwXyl9puPUfOB7BMiupN62NYyf8me56PK17MojE3Xv5d21i3Div7zeu3wp73/IPO1+p3"
    "Qo3gf7L6quCaApg9zrkn6j+pzzDWhsfj1kyD3/V8WA8P59Y6kRPnjn8/D5pVL0urUv3T9VO1Hv59NdJ0XHB4zr+plR5fc/2vqp3+"
    "59ZHRakDe1hwYak6Hq+JwNmb3P/hd53KjXPdY+3d58qqreqcekOpzFuWaltbUozKnFtwPpfswwrXoizG0q1dmRoROupRTzD1owpq"
    "DJfjh3FSfT1UQ+K5TH2/3r7a6/fTCHwPNaCHKAd8mhlzp6BqZlFuhXoK3T+EAd3Ad+u3Jynf1/SyVK7+8s8WheOsrdduXtQ5I79C"
    "yh90nri4VZs3LFyPfyp183Cc1PrSOd9+rQrVjkk6H0vzota1nBk3Y/JRWXnCqg4/r+65h1KrLNsaCdtEE7s7WFalXG9xpM/liJks"
    "XGUdZhlZ9hUjr8qPbWNhxP8NBbSHE9Ry1DlbAIsg6hRaZcP41zWwCMCH2TmxlWKDnpVNjYTrjkV6Tz+u1qneudOwp4rd8GqZ9JFS"
    "uEHpf8vGRiVzAyE2aMMeqJ4z4EcfjRPV14iElfIoLNqpDF96DrzxsMLSHL6psKFd/T4Gt9sIsbgXSh9ss4Bx+HFzqAfQfWdvr8xC"
    "ORtgPrsp3LDLjimsZ7Y+3Rx8/57zyOuNeS8Y+8z1uLFhLWVm9U/d+2DZxw3TDHwuG1uBIXDk9VyYusfWYXvMcZ3KzcEIa63xgyn2"
    "K2Vd14ndc7reNdD/vm0ewBNfu5hIzK3t+0/2+ondDPVhcZ+Ci/eQMUxDancZfQqqXoc6vNzHsJiASUiYugR/4eJwX2R96phtnaz7"
    "Unw/oTNhhe+QLRX1jXk0AbsGY2/x/g7jAdI1d9vDYjFE3NuC81bq0gH8kGHv4ro1fl+o0n/8e61ERf6N947xt4uRj5Pl/CjWo+fj"
    "OOz4aSZpp/7GOHnlH2Svrax8Qvic2x/yhQJxI9QljK0X37Jh3r9cpxHCoUOh8s3Jv1i24vC4shgTEzlNH6uTZruh94+a0EJUzSOj"
    "kqrvE7BtjA35B+bC3iM4H2S/F1O1p5rgRsjzmq4EsMUBbLhgc5K47Kvd/d9Z+4frC39jX3yHH3eUX9wcnrUhSRycij/t3tD+TZbv"
    "l/U+5T3MBi5mhM8lqDW7OUrn/GCV0KSdC2H7ahjHS8qv+W6/HRoTPad/pr9l31cUlb7BhGflAy3jqV17ee4BoWuUPpJrr3viPuov"
    "ommvwPmhZLw1kr89L7/zpe4PnH8d++7jUA4qw14oNtEg5j9xFl0n919Z8G/fzDcr8/rfM3NxLB4+jf1C7GOuA8ztEbhZlVtrgV3S"
    "4FP998FYum3/K+++o6Mx2449KKcwyU5eJmCDVf+mf552o4310xlzKWs0jalle6qY6Dx/Lpn7tf4SmC0LJYsDy5eLfWEK5r4l2F30"
    "zYU/451JYodPUUv9pvfVYvWUvUbsxPbf79d5uM2tsuai42E9myPmEqqWla+HuCTsx3M+xtgjfU6Z+rm3JzLPTN5DqDV/P3fAsQZ7"
    "1FRMAP9wmF8n5kSN39os729HXIvXOnCf/X3qWt7fMq4F30ZsB3DYgXPyaCy25yfaWrh5B6Eernxudcy4wn4Bq9h7+/Eg07SXnw2M"
    "rxi0TUG27KweKLdnxfr76r3QGVjvPpzTf4u7yvO4YX0A24NluB30z+xnmnlwr+nno12/i+PDmYrLZq76gR+3kM1c6FzdWPX+RN/u"
    "lW5KEcW5d1BxrVy8X0Q7yelOdJxUTvURun1oUjMwGOhfHcYOzZL7w3tGbw6+788J9VpwHl/wGV8NyR2q9ZzEXbt9FvWPQWG35POm"
    "+PC92nTGGXhyDaE9PZTzd1Tc6bwvWr4hF+uerNOrnoMBzamrcsTXltxc/ej1QbZpp3PDCf/CKiR4OPedj1nHGViMNg2+p9MXmlAi"
    "sr6WMEWb77hnzKHvPC7d/pdUrQt9Hdn2x+0FcZ9P1o5drwmsOjBgXlwQ+Rg7PndUXTzhB0iuOfpqJLjCOKY78sz0eqXlXtu4ez0V"
    "fIWda+kreC6756DtV+C67Yf0VSz/elXnDWOmIsz7JXgBxvX29W9mD808sy7Hg4XBFh3o80g+B+oxJRXPjFVvR+qzXh6WewTUuszK"
    "v7g5X4Vl/4i7rUTs4t17fN/GHsP7z/Ix422yFz7pT4D30fXZeM9UN59qjX2qvB369B3MP6+zTycnoXNv8aFe8BN6WEy+9MB8+T5e"
    "Yk/jfon4mj5//VtzQxzuFVe+Ao1T97TYHrVLvEuTdwT2x9g5v59E94fvBddA49gj/w/swvo80cf24b5XjkXtPbwz4Nhx0/02NK8L"
    "Z55nvUqox6cka8w8C68n1T+lchUZ7yDZA1RndbQ8jTMeubbu0DP/6b3B/3moJ0fscbrnp8G2g3vGgn1Etj/Qt8lHjolrlt0Mn+6n"
    "V9cMjS1r77j8q3kaL2P5Rjhb9NpL5FFt/jiVx9dcd9cmT6HyqXrdyfmp65Qqh6JxgugZcnvgdK9Qcg+onqGkTdNr3Pq3tg6bqZTw"
    "FEUP0c3McIea3FkxWeOT9W05bLAHee//DvcoZfJLBOfiiN4pM6f3B3qgAv1bO2ALrI+Mz292P7k3wfl9dbMzSpAVrSQu7zlxXq9G"
    "jK3wehXFVwvMQ1+wDXjWnFXfuFo4vB8jy6s7G2le48T8U6wGtfXZBvND5wT6q+j9V9d0jVK3gLVaXw3pfYysasWIeaLFfnzQWn1X"
    "nKtrxbECHJPlQGT+1hJzBNPvt4POhvbo3VL6Wyie+sqj/2nG6kqiSv3RV7+z/la0GTK/iijSqvOaOafVvbcD4G/QH1oZbuNaflen"
    "MwmxE9RmwA81rM2mDYO/w9nn+0Ryj1IRc0DzIvhr+zxQBab9XmbubXBjxLWXzc8EdyjuBewv+bfvHL/c3i3lulOl6D1hZSbLO9sj"
    "m1BdD279sYg94/HInNw+LEOYRvWZ/IBVO9ZLR1lkxHjpaplZ2jUn7s8UDsJXmha1Dj3GIxWmNZ9dLT9hlWqoH7yLmq7lqNLKH8hn"
    "gYFffr6fWnWIZ6Wm0AA7fKG573evRemwHT8a/JvCugjWnvHaN08/7ix/KnPzci/Qz8fcg5Nnfz6KV70n54ZVO6nFtP+inaf2IOoc"
    "G7Kbeagb0H4RVaRuk7muekWKl7pX4DOjfTTD2ZjALcg1M1Q+VA/JmHOvzBv9lVs09klFld24D/v2I7dI8oI7HN+N51z5p8ZmJ5/p"
    "m/tvFIYjW4XFcDsn+oOqoipjewCBXUiMsZLbYO7uu6JK5ahrAD+WpzUdVoix7/O7+fmm38Eoi0h/3FspyetmudWFC/zCw+kzhok+"
    "i/7FWv6jBxXA21ZxsGfcCdQmBNdSsBgZ7kvI4K8/Sl2Hnt/0AdF7SvYB4Vx7UbELPRf5J+SXR+yP4sxrfJ/HlvUUWAPCg/60hJrH"
    "Fz0j7Rmsl+tAL5KX8x83bG3+XBSF+CzVvOrHKt+k96jKAWl8juKy+yPVIebK6Lr+Rx4+ycHr96CY1n08UvFmjbVq+e9+mPNOqd0A"
    "r3TcWLEm68yLm7QtwIzNdr2nvHCgq3kVZRU5N9V1ZawRz/GXUp7KPBO5VinqLLTnRPmtm+6xU9hDUVZJ9gId5GY5iMc/0CcaWT5B"
    "qPrpvjtts3zODpfHPJtnXj1nYK2l8Jai0rNI9kUmeVPYFjCeUF3b2hj+2xX8gPGvt6vPrN6Io+7zdAyGQOKF7F4rXHfi+wwZ77PB"
    "2LfsfkStsHPoM1kKiVl18W/wD9ZeyGcv6gW2yeIPJ3vTpIct2dN6rvyzjDk/nsPp4Lpl3LLkgI/YV6vRqbgN2ycnmP4C9z6NRCdh"
    "YpS4VRyi1KacOPXxCDyIcPR+2/sz3Nueo+E+i9MpjGMQ/OT1d/hFladVquLm2QPKtUesE9VHN/J9jCsXHyG4lKL48d/OZQD7YXjY"
    "6fnSPXzTgzywSa4Rs8+CMTXt0a+8GtMBrHAn/PtArdqdO5v/rQEbCH7ma7LvLVbX+9V19tE8/Xe2H6ExP+nxcj4shHnWSmG/Q7V0"
    "R1UtMyfhcAwH9DoE+2/PXFH5dH/+Xil0dQHMJ8XtzCfUU2sEv6PzXv/O41SmOHs26FJ8JzkZxvuiNyv4DMV8em1XyzpmpbVUJh/x"
    "QdsWikMQt0qMR7Z0M6QYz3CkoydUFNfIbzkrtygOHTJuZMZnYvwjt29wPB0Prbqff74gB9FIXP91H+Jobnm1oHSexeF6N7+PY+RD"
    "tIIcxcnbX8+C4WoElXC1gtiVp6zWccYjfIgprh3J1f54OHzOB9TKj7aLhh9+Mkho1uyO1KnxbJ16v9rv8bHLkvc6D72v4/0B7ado"
    "lTfVQ6t8AscmDh6quS3rt1R1ziScJ0n0lLr79Y94sf+GPfQVeO1e2eqc1P08/5vuu0fehHujbx+Mn03PiNzzaFBY+3G9tZN0D4rB"
    "U7mkf2QfHpWHonkDb/dSrTU6M/M7Wms+BtPbc4aLTvDJqq6WtJnm9/l8OWPcq6TqtOSJr3aqJhy6Z4D/bqi54L7+f4XDUxUOS4XC"
    "5Xk8zJ0NX0vnudfSxXA0vLgYXl6WRvGwVB6d5/KXF+X44nUYF+Pi5XD02h+dXQwLcXGQ65+XLxIKh9v+5jUoblgEic7jhhcIhOuk"
    "mP0xvGUDMxlU8pJoq9gNLiRSQvLE8tS0uZkUC7Lo3bEmlpr0Cs/StKoE0Ghi5jE36mKyrAiaiCq1JqZpdz7bKGlYK5n8aEXSeENo"
    "kTROik7I+bmzImnVa1+E7ZZe6G7ZYMPYvkKiFJKhADjo+0iStKBE6iQ4XysRuBGa9mgxnennoMNEDvW2fYZUc3WoWcyK3uimMBCQ"
    "QAYWRVcOrPjeZMBoLnieDXCUjJcubso8T0wzyfArkaj99zZV0Vw1YTSyBAEYPM+ik0oY0ACeK2M0XSDBzE55bOStucDOYG+aHyTS"
    "MomSXhgkH2+VE+sLujHh0DidYOf5f/6jcciBLcZHgY5Q1FsM5sqZdABVjePnE9K8aICCkwyHckUHr7e/yNnAvsoFG9hucbBHXOCV"
    "gosEn0xeL41CpwtUQTK+IsmVbwR7kg4WOZAuqN2fnxPJmBJzoBp4CtW3IZrMq8YR47lBwNeAPHglkNxx91R+Q2NsCXiJk2gtcU4C"
    "Ig4IpHoCjHIJYtX8JQjbg0G8+z4s6JDshwEegRhuABuqxeDsXmGBAOdnrEV7vWrZI3DEwcXJmQwQCoNYzfO0HBCJEQOAA6gk0aea"
    "DHunSIEM4VS3wKRqC53I4AKtNEd9ZSUGHSLOvxpziNnmaW9x0ubIBhIphIZEDvW96eD9oDWhElMz2KCsBA+AlO8IQhxQimkGoXvk"
    "uYEcMs04M2pMygKbz+uRx2obUnyi2ch/Jw+ZRQvHhnabXpOGJF3YUTxLkVjVyL5z0tM08F0MZxtyVOi9Vn2Qm30nDCCSz/Cev1rW"
    "CzeSCHlzhL1uUUCraiLBCc7/18pEnzdrubdJahvRGW5SnEeLFxGGsY2yySZZc4aeeePz7U4LZ/wslsKwcw8lC08BnmmitNczQaAp"
    "er9pZ/N62+vEE5lnI8+tz3JuVNHPbJ1Wl0Bp6AFO0gVdCi4KUe57IPI1kqRotEoC+b1nepFmB2nsbE+5+NBw3gs57G9B0LSZtxLk"
    "0ckG9CZw8APfZaGdmLxVCCb2aizSZOeD1oULCAHplLz7FshfV/JvC3wRMEJ9yn4KGqbfnDOnrcV65H6JtaDECwXUps+vf/IMOiTO"
    "czwJjtrbXQe0t5gmihEsXKTfVfL8ccGxCIxXSJq9VjUBUEuAtRIYnpnm1EDjmFqHSYGs1PUSxFF8/rljOCSAemheUn6m6+9VlZiJ"
    "IxDl+P0J0sQEWJ3Xx8QdYyrIEtsrYOLXIgDBtsk7Qchu1woCZpV0YlIbv6FVgfqSdkaJ/ND57OzDRqgxk4m4LADUHX9SyIXmTj+r"
    "AmdVUISzCavE8/LZ5JDEk28Yr/A5fM+dj0a7zMUoblItzHjuFJDffieTeEaCd97jRxSg9Ljc5gn9O//9huyDR47oCXL2mUQ2tg2v"
    "Rfs+nP0QBtPtZ2X/3sZOJURBHSEmfrfWB1AJEf73Ec1Piz78DIBoHbI8+OU4L3s0ZgF3mvMZZFTwBwoAVNbfvPNI9jDbDLafOQT/"
    "SD4kiaxe98u/tJ94fwRgzr4rlQgN+FYB8Jvja2niIWeN3Grw5RUDEZU9wN7/TT8XAdJz4kJNdKb9h0UQ+Jf2w+hcyC3t+KcfqmFB"
    "j2EVHlMmINSQtX/nqyXvc7Qv6JyNIUDhsX6e8zfE1CCe+To8Zyi+oDhQQpFgL2thswVZnLuHyE4jgf3F65Sub+4ZInsvXtOzkj2d"
    "51US1vPP/XisUJ7rAifNoc2X5Mso8I3cvIUTM+g8iCS7BKygchmuIDyDL3Be2pxPu3RtClZVDX6Y0BpsvtP72EkMl1u0TBGOYvxC"
    "b66K85YU+tbs1QnZvbbKX1z1hbgXon02HqczAGKEccWLw7ZHgXuKRjw6G3CVYyCSQw5/bYA6jsDzW38uhBRHgmt+fwMoWTGYc/8d"
    "yEXnv4Z/BMIZSi7xMFBG5uM40E3XidXfMsA8teaSzhYA/iBmhnzCSuVwJK+ifX0GDNC+Lj46/u63QC3kMw+8y7KsQdojSECLiCU3"
    "DfN+U34K2TO6ry0WohFtjdgEAFNDrJ3l00mxhotSeq7UWhhJ0YOLtAUDsr2tl5JkW5K74CZk7dvOGgnQqCqg2PykV9BC7OAXo48B"
    "UiRydeVE0UwX5jiXE7qfa090YcjLjR7TuKzIR3SOVDUw8rUCOQzrc3v5YF0I1z8fByTJytO6Y9A5c+Ojt9X5Xd3sfeFcr0EvWNBT"
    "1/ILeeRfCDGCnC8gHmokhH5DBeBvClsrEAv/Y8VaFG8K0Y7W6DIdh+zG7UJUyiwEVWGnIvPZp3xT/XtK6++Bc2T+fsgseM/kewZE"
    "NHutPSBXslbgz/KwOMN6RSy/p+tfOIWxPQDUmINuXgDt3fyl5BbyZdQPII4yMqTBGjBXbZbIr5sNnqMzsptL/tzbGMW3RDHo8jw3"
    "iC9z/fyg8JofxcXLi7PCRa5U6o9y+fPCsFi4KPcvzkv0u9HofHQxypXPcnGxWCrTxwrng5FTDHpbxK+foUrQU63MkrU9UDKTNeaI"
    "R1uyW7ZeNEM9lqYljwm75K3PVGCg/8dunZTIA5FrCOXhl6KxlPaFeXXRawvNtlRiUDrr5TGD5JmNnGoMeRPkcSvLjcqNqeQ4sHNd"
    "jUHG8tWpxNB4Z0NAWm6nzvf4bev7MjSerM6bc0+GXDqVp6/XrleJytF13vtXtooE6MlLYbMbFLiqYu8/B6SY5XNW0nIg8C3v81eq"
    "elUpAYIzccbB1pE8PTppqohGzfe50kbPYqpQc8CeorXz3arADiKUT5uQuBncMBX4qifeAc3BJzx1shK9FShE9bNoGmznWqnsvK1+"
    "0fW6sVs965A3WHxkqd/Zgox3/WnaqnYyZEi0VYzf7xa27aAJCNNIKgewHHU6ofLwsnl+6HnIk2v+hgdMXl1BZQGTXiR5xy2ZX4GG"
    "bSiamsADeAHVAd2XxrtiSJDJID2MH/G+anIKDx36R18+hyFrdIrjuTFWeB6c/VugNXfIEiMyP7SLv/oi63JG628V3063utUVpyF7"
    "1/NnBV1jq8gVUfbUGFponxuUtIjgMB/dIntkSia4SlbmE/vtC6fUK0U1Sg4r98KVLMj4BT4TpGaN3XljyqPXwprGEe06hY3OtvzW"
    "lTNVMQDMTKh0pRXrQ9OeoaSP1iNzT4rYddu+gtk592sGJaxAyTeAByxUgB8vC7Y9E9uK680RqAK3PUO9yZLzie/EGfdBhFSH9wW7"
    "hmf31ow3jgXL0i7Y9kVlb8667VBkRXssKiuYvam4fHv9eD6bkCeMNZVjGnRbCeD2nGFhM8Iep0hv4tA0ZWVy3u9ujC2h9UF7vw0Y"
    "3yckq64xR0xXjZbOxQNkcySyf0N24tK0R9x34N3qSrl4Enc1tOKgunsJmzcb1B63viS33cv0PkL7V6ii0F7QxRlxXWoEny/Vruju"
    "I0PxreSWaA3Q+xSJMfdzS8+2mJYmc4IvE+MN7bt0ZZkzx7i2orqrTGbDKmefWP4R1Vn8nNoT+dwqsSecsYLSOCLP5Pqjz4gA0GaH"
    "PM7x4jFXeub9l+/VH9tDlSnhyudEzlW9nkQuguG2Sp6xnmv+jCrDNbKddg5dynaWeNv3KpM3+i6oeRHlgOJ1KvZyJ7Yuj/Okvorn"
    "cl4r6tq1IDOayySVrbVROBeu9bW3zu+dagTuryTWnYyeZ+eczDBnyjwJgpStUHLrkK4RmQ2yz1JdSMrchDIxhTK9nybbPUSpaK/k"
    "Sqz1OZb+2JTMCsYHGBjWJdOgh2T3hJ78yOsJVXnbyscMC0aOx7MxwTHn8wP3O0H7laLVlsyXs1ft+azkKAaoxgM63/kcpWjheb24"
    "51OIHj7YomvemYX5Zr/XJBTYPL+R+BBqwuw5dj4HutGD60Db6ey5cceGiJjeTeZ11dni2i2hO+9yZs7u67ZzjrTD+zC4xtAeXyuT"
    "jxmDruwDVa/XtM8Bugb6W3kBmjryGVcvxWnWGqG9XZoZiosC2QessU7zK+2PHPhs2C+hcz2PrMcK0a60JczC10WmaD5D+zdD//vd"
    "zDMZ8pNTtodzQEtj3m+gHUEWLrkuXhZkh+ncvse4u3fbxDpJXyv8XrPbTIrxSloXrmlvRjuJZYy9RYs0ndclbz8rtJJXbeu2Z+Xg"
    "eiLbOKyWyWcr4xoKpg1agWjCka63Zut58mno3CrxXFB8YdYE25hTx5oxJje+5FaMQg8xTy4t+eWcNXkNmdfosm9bqDL9Hhp/sQ/E"
    "YsVchz+jZCJmtJbWQp/eWnJFFnGxQjHEBUBN4S+LNBHFthzH3ScybnxuSUUSa2MrLYlMn6QQP6pdyZ1/dV4af8/6O8rnejTntGrx"
    "mul4r1FD/Nb8LfHNo1Tv57P33lOO1sB1vm4p/HXcCQTje78oPkYaCeldz2QtYkaikA2ZN5eSfXpc+nEw26IdtywVqrqN9X14y62+"
    "xV5ntujftr5UdSDRUq1jVUYATESiXWJKE6+Z1j/3nhPvWSDpBurwfscgElnudQA/4kZg173u3fIJa6Lz+UN9dtl4W6nvgc5RYmvs"
    "Z50tU3Pu3Ss97/K97DGUcoM9o0A/Bre9Gfxhfk/tiY7JgdLgqkujkvJ71TV24wjVqJyer5KqLDEScqnnjOx4AXG5+I+7ZF5i6drb"
    "V4mhMF6u0L50Y6bvG8yRFZOMmYKVm3coVdRqEb49j6sKmhNU0+sKeo3WD8e3XETkl7dU/D+GdPdK6M7xHnbkL3JeYaTHjzl057qb"
    "GI/rw/N+A1KF/s2Vk5ogB3yqavMOEvMrSGZU1F4KoKAoo/1/0quMExTpgjoYFKaCNMH4A9TPqQy8mxFX5wkjCfAO82U971v9N1C1"
    "9dtZrXau/43YSdZbo7pRdjiKG5zrcGws+Xpo74AdoGfeouWCZXwq0yw6Qz2ftqLuycPoauWjPR+Re+mgzaOlJIDxDKVQzKf/pmxg"
    "SEpLZ4PTOagw1N8fu37/CTkchwKe7y+0rcnYQdEtCo1Ri+wq3ctBHLitf6/B9jMgBWc/WzfRi5XwZf/csROtUWIvso93aK2HfQrZ"
    "38nPsg+T2Ov+WotH3I5TmajvazsyzfBdwv4g2uP875dCn8tAph72G1N7w88HfSFXyRKJxYfs6wd8wsCYgz5mxjzYKgrWh5G/cewP"
    "pNGe0DFB+4/i2+E7fW7mIddWap39pVDGirpoJSjcWyOPqDsNVMuZzS9btHqkbamyY+KPCopN8vYp5LLrU0g10tjnjDVizjFt01QO"
    "nf3Cl8InkNaM1jTdHN0mt84lES369w1zztuzQSofUslM5gt05Rt7jquU/PPz9tG2dD6qyuiSnvnrBVRHt5IfrtOp1e98gtYAckHz"
    "3lxoZGycuts8T8t1366qaiz5K32m1WJfBrLmM5qXJVpnj0IDFHTOLE37Mphzxa5c/7pccCXtcHWf55juu6F19JG+3nHVbPbJu0De"
    "fUdDUx+ZzwoNDPbmqgfbh3WIChj5iCJjeuNQqJjnPZJSw/OdjkINfI9gQMveP0crIm2rw00MCSTVvkt7dqNzRTQOjHuiKKHLQLeD"
    "NoYp2TrPx6I36JwcbyQ38Lih34N6pnRPNqS+NwiOOfzP2HlXz8qGqBoFECzSHSWV2BzQNkzpR+cK/J6eqqHQXi0OiowM53Y8fCau"
    "TcgXu4L0CtnV5krkYFsz9pU5R9eb4/m5/sb3au1fjqd+0b6t8unKK1Du/WqPVWzC52Oe7OWC/VSNlP5hYpd/DglRKK+Z3qkW0TnQ"
    "nOnKufm5TZ/bX3Pu965G6xC0mG0lMUwxTby388zxwFUCXVnRSAGTLzcSCg2phGukBdvPoUaS3yTkvo7tIqrRz7VP7tAgv+yjNyur"
    "8SnpObuPA51aKXoYf0yH0e22tXqucmB5oFfKuWMpug8hj3WeDGhQUNsBFd0FSq9WLqKTRVHHmfHWA9QUibiC5TKZfk7XU4UKbebQ"
    "HCRiucTaCNB46LUAybpE3Tb1nu/l83LOk13l2Owrr2K0Y6W9DaKa56KL2pL2PcnWGfoH2osaMaNb730b1ho5NuxoWhSNiH7pTnQu"
    "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"
    "v/71v8iovNc="
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


REQUIRED_VALUES = (("CONFIG", "patches"), ("GRID_STEP_UM", "patches"), ("COHORT", "search"), ("TOPK", "search"),
                   ("PER_PRODUCT_TOPK", "search"))
INPUTS = (("DB_DATASET", DB_DATASET, "patches", None), ("QUERY_DATASET", QUERY_DATASET, "search", None),
          ("LINEAGES", LINEAGES, "make_split", None), ("ASSIGN", ASSIGN, "make_split", None),
          ("CONFIG", CONFIG, "patches", None), ("COHORT", COHORT, "search", None))


def _log(text):
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
