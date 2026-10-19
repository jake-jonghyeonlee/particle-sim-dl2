# psdl2 사용 안내 — 절차, 입력 형식, 출력 형식

DL2는 사용자가 고른 **검토 위치(ROI) 하나**와 국부 구조가 비슷한 다른 위치를 찾는다. 같은 위치를 가까운 범위와 넓은 범위 두 창으로 잘라 원본 형상, 전극 역할·layer, CNT/VIA를 함께 본다. 결과는 참고할 유사 구조 목록이며 원인 판정이나 위험 예측이 아니다.

- 입력은 원본 layout 형상뿐이다. hotspot, CA, fab 값은 받지 않는다(있으면 오류).
- 정답 없이 학습한다. 정답(ROI 대응)은 선택이다.
- 창 크기, 격자 간격, 흔들림 폭 등에는 기본값이 없다. 실행할 때 사용자가 정한다.
- 실제 자료 검증은 폐쇄망에서 사용자가 한다. 여기서 검증한 것은 합성 자료뿐이다.
- 세부 규칙의 근거는 [data_contract.md](data_contract.md), [training.md](training.md), [validation.md](validation.md)에 있다.

## 1. 전체 절차

| 순서 | 하는 일 | 명령 | 만들어지는 것 |
|---|---|---|---|
| 0 | 설치 | `uv pip install` | `.venv` |
| 1 | ROI 목록 내보내기 | `simcompare export-review-rois` (수치 저장소) | `review_rois.json` |
| 2 | 설정 파일 준비 | (사용자) | 실행 설정, lineage, split, cohort |
| 3 | ROI 목록 검증 | `python -m psdl2.data` | 요약 JSON |
| 4 | 학습 | `python -m psdl2.train` | checkpoint, 학습 기록 |
| 5 | 색인 만들기 | `python -m psdl2.index` | index JSON |
| 6 | 검색 | `python -m psdl2.search` | 검색 결과 JSON |
| 7 | 평가(정답이 있을 때) | `python -m psdl2.evaluate` | 평가 JSON |

ROI는 `particle-sim-compare`가 만든다. 전극 수 지도에서 고른 영역이나 사용자가 직접 지정한 좌표가 출처다. DL2 검색 결과가 자동으로 새 ROI를 만들지는 않는다.

## 2. 설치

Python 3.12 기준이다. 네트워크 내려받기는 쓰지 않는다.

```bash
uv venv --python 3.12 .venv
uv pip install --no-index --find-links /approved/wheels -r requirements.lock
uv pip install --no-index --find-links /approved/wheels -e .
.venv/bin/python -m pytest -p no:warnings          # 설치 확인
```

`requirements.lock`은 개발 기기(macOS)에 설치된 버전 기록이다. 폐쇄망 OS에 맞는 wheel로 같은 버전을 설치한다.

### 2.1 GPU 환경 (Windows, pip mirror)

DL1과 같은 torch 빌드(`2.7.1+cu118`)를 쓴다. psdl2는 `torch>=2.7.1`이면 되고, numpy·Pillow 버전은 DL1과 같다.
기존 venv는 건드리지 않고 새 venv를 만든다(Git Bash).

```bash
export PIP_INDEX_URL=http://<mirror>/simple PIP_TRUSTED_HOST=<mirror-host>
bash scripts/setup_gpu_venv.sh .venv-gpu        # +cu118 wheel이 다른 index에 있으면 TORCH_INDEX_URL 지정
```

스크립트는 CUDA 빌드 여부, GPU 이름·메모리, psdl2 CNN forward 한 번을 확인한다. CPU 빌드가 깔리면 실패로 끝난다.
학습·색인·검색에 `--device cuda`를 붙인다. CUDA를 쓸 수 없으면 CPU로 넘어가지 않고 종료 코드 2로 멈춘다.

기준 PC는 GTX 1660 Super 6 GB, RAM 16 GB다. 모델이 작아 GPU 메모리보다 RAM(원본 mask)이 먼저 한계가 된다.
- 1660 Super는 tensor core가 없어 혼합정밀 이득이 작다. fp32로 돈다.
- 원본 mask는 같은 파일(경로·hash)을 여러 ROI가 공유하므로 제품당 layer 수만큼만 메모리에 올라간다. 검증 요약의 `asset_memory.decoded_bytes`로 확인한다.
- `--max-asset-memory-mb N`을 주면 디코딩한 mask가 N MB를 넘기 전에 종료 코드 2로 멈춘다(행 단위 blocked가 아니라 export 전체 거부). 이 값은 남아 있는 mask만 센다. 파일 바이트·디코딩 중간 배열은 따로 잠깐 더 쓰므로 RAM 전체보다 넉넉히 작게 준다. 16 GB에서는 8000 정도에서 시작해 `decoded_bytes`를 보고 조정한다(실측 전 제안값).
- GPU 결과는 CPU와 소수점 아래에서 다를 수 있다. 같은 checkpoint는 어느 쪽에서도 읽힌다.

## 3. 입력

### 3.1 ROI 목록 `review_rois.json` — schema `review_roi_v1`

수치 저장소가 만든 파일을 **손으로 고치지 않고** 그대로 쓴다. 학습용, 색인용(후보), 검색용(query) 모두 같은 형식이다.

파일 머리:

| 필드 | 뜻 |
|---|---|
| `schema_version` | `review_roi_v1`이어야 한다 |
| `producer_commit` | 만든 수치 저장소의 commit(40자리). null이면 `--allow-unversioned-producer`와 `producer_commit_reason`이 필요하다 |
| `source_manifest_hash` | 원본 입력 manifest의 SHA-256 |
| `assets_root` | 원본 mask 폴더. 옮겼으면 명령에서 `--assets-root`로 다시 지정한다 |
| `rois[]` | ROI 행 목록 |
| `excluded_records[]` | 내보내기에서 제외된 기록과 사유 |
| `tensor_input_policy` | `original_layout_CNT_VIA_valid_only` |

ROI 한 행:

```json
{
  "roi_id": "roi_8392d9e6…",
  "status": "ready",
  "reasons": [],
  "product_id": "PRODUCT", "design_version": "DESIGN", "case_id": "A",
  "source_identity": {
    "product_id": "PRODUCT", "design_version": "DESIGN", "source_id": "SOURCE",
    "process_step_id": "STEP", "exposure_id": "EXPOSURE", "case_id": "A",
    "frame_id": "FRAME", "source_contract_version": "synthetic_v2"
  },
  "source_kind": "manual",
  "frame": {
    "frame_id": "FRAME", "length_unit": "um", "axis_order": "yx",
    "origin_convention": "top_left_cell_edge",
    "origin_xy_um": [100.0, 200.0], "pixel_pitch_um": 0.5, "shape_yx": [8, 10]
  },
  "center_xy_um": [102.5, 201.5],
  "endpoints": [
    {"electrode_id": "A", "role": "gate", "actual_layer": "M2"},
    {"electrode_id": "B", "role": "data", "actual_layer": "M2"}
  ],
  "failure_type": "lateral_short",
  "analysis_layer": "M2",
  "source_assets": {
    "layouts": [
      {"endpoint": {"electrode_id": "A", "role": "gate", "actual_layer": "M2"},
       "asset": {"path": "assets/layout_a.npy", "sha256": "6219…",
                 "value_semantics": "binary", "observed": true}},
      {"endpoint": {"electrode_id": "B", "role": "data", "actual_layer": "M2"},
       "asset": {"path": "assets/layout_b.npy", "sha256": "6219…",
                 "value_semantics": "binary", "observed": true}}
    ],
    "valid": {"path": "assets/valid.npy", "sha256": "1628…", "value_semantics": "binary", "observed": true},
    "cnt":   {"path": "assets/cnt.npy",   "sha256": "4952…", "value_semantics": "binary", "observed": true},
    "via": []
  }
}
```

| 필드 | 뜻 |
|---|---|
| `status` | `ready`만 자르기·학습·검색에 쓰인다. `blocked` 행은 사유와 함께 보존되고 실행되지 않는다 |
| `source_identity` | 제품·설계·출처·공정 단계·case·frame을 묶은 식별자. 결과를 수치와 연결할 때 이 전체로 맞춘다 |
| `source_kind` | `manual`(직접 지정) 또는 `electrode_count_region`(전극 수 지도 영역) |
| `frame` | 원본 이미지의 좌표계. `origin_xy_um`은 왼쪽 위 모서리, `shape_yx`는 [세로, 가로] |
| `center_xy_um` | ROI 중심의 실제 좌표(µm). 픽셀 번호가 아니다 |
| `endpoints` | 관여 전극 2개의 이름, 역할, 실제 layer |
| `failure_type` | `lateral_short` 또는 `vertical_short` |
| `source_assets.layouts` | 관여 전극 2개의 원본 형상 |
| `source_assets.valid` / `cnt` / `via` | 관측 범위, CNT, VIA mask. 모두 같은 frame |

### 3.2 mask 파일

| 항목 | 규칙 |
|---|---|
| 형식 | 2차원 `.npy`, 또는 단일 채널 `.png` / `.bmp` / `.tif` / `.tiff`. 색상·palette 이미지는 거부 |
| 값 | 정확히 0/1 또는 정수 0/255. threshold를 적용하지 않는다 |
| 극성 | 참조마다 `value_semantics`로 적는다. `binary`는 0이 아닌 값이 형상, `binary_zero_is_marked`는 0(검정)이 형상이다. 자동 판별하지 않는다 |
| 크기 | `frame.shape_yx`와 같아야 한다 |
| 무결성 | 파일 내용의 SHA-256이 `sha256`과 같아야 한다 |
| 경로 | `assets_root` 기준 상대 경로. 절대 경로, `..`, 폴더 밖 symlink는 거부 |
| 관측 | `observed: true`여야 한다. 빈 형상(관측된 0)과 누락을 구별한다 |

### 3.3 실행 설정 `config.json` — schema `psdl2_run_v1`

```json
{
  "schema_version": "psdl2_run_v1",
  "model": {"roles": ["gate", "data"], "layers": ["M1", "M3"],
            "width": 4, "feature_dim": 8, "projection_dim": 4},
  "views": {
    "crop": {"near_size_xy_um": [3, 3], "wide_size_xy_um": [4, 4],
             "intermediate_layer_policy": "preserve_exported_layers_v1"},
    "near_pitch_um": 0.5, "wide_pitch_um": 0.5,
    "near_resampling_mode": "nearest", "wide_resampling_mode": "nearest",
    "jitter_xy_um": [0.3, 0.3], "core_size_xy_um": [2, 2],
    "topology_policy": "common_core_connectivity_v3",
    "connection_semantics": "global_contact_proxy_unverified",
    "comparison_basis": "between_views_only",
    "max_pair_attempts": 3
  },
  "loss": {"alignment_weight": 1, "variance_weight": 1, "covariance_weight": 1,
           "variance_floor": 2, "variance_epsilon": 0.25,
           "supervised_weight": 1, "supervised_temperature": 1},
  "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "seed": 7
}
```

위 숫자는 **합성 시험용 값**이며 실제 권장값이 아니다. `configs/synthetic.example.json`은 이 내용을 `config` key 아래에 감싸 두었으므로, `--config`에는 안쪽 객체만 따로 저장한 파일을 준다.

| key | 뜻 | 규칙 |
|---|---|---|
| `model.roles`, `model.layers` | 자료에 나오는 전극 역할과 layer 이름 전체 | 목록에 없는 이름이 나오면 오류 |
| `model.width`, `feature_dim`, `projection_dim` | 모델 크기 | |
| `crop.near_size_xy_um` | 가까운 범위 창 크기 [가로, 세로] µm | 필수, 기본값 없음. 격자 간격의 정수배 |
| `crop.wide_size_xy_um` | 넓은 범위 창 크기 | 필수. 가까운 창을 포함해야 한다 |
| `crop.intermediate_layer_policy` | 사이 layer 처리 | 필수. `preserve_exported_layers_v1`(내보낸 layer만 유지) 또는 `require_listed_layers_v1`(+ `required_layers` 목록) |
| `near_pitch_um`, `wide_pitch_um` | 각 창의 출력 격자 간격 | 필수. 원본 간격과 같거나 더 거칠게 |
| `near_resampling_mode`, `wide_resampling_mode` | `nearest` 또는 `area_coverage` | 원본보다 거친 간격에는 `area_coverage`만 허용 |
| `jitter_xy_um` | 같은 위치의 두 view를 만들 때 중심을 흔드는 최대 폭 | 필수 |
| `core_size_xy_um` | 두 view가 반드시 함께 봐야 하는 중심 영역 | 필수. `core + 2×jitter ≤ 창 크기` |
| `topology_policy`, `connection_semantics`, `comparison_basis` | 연결 판정 규칙의 이름 | 위 값 그대로 적는다 |
| `max_pair_attempts` | view 쌍이 거부됐을 때 다시 시도하는 횟수(첫 시도 포함) | 필수, 기본값 없음 |
| `loss.*` | 손실 가중치 | 모두 명시 |
| `epochs`, `batch_size`, `learning_rate`, `seed` | 학습 값 | |

학습 전에 설정과 각 ROI의 frame을 맞춰 보는 사전 검사가 돈다. 창이 frame을 벗어나거나 core가 흔들림 범위에서 잘리면 첫 학습 단계 전에 오류로 멈춘다.

### 3.4 `lineages.json` — 제품 → 계열

```json
{"PRODUCT_A": "lineage_0", "PRODUCT_B": "lineage_0", "PRODUCT_C": "lineage_1"}
```

같은 계열의 제품은 학습·검증·시험 중 한 구획에만 들어가야 한다.

### 3.5 `split.json` — ROI → 구획

```json
{"roi_aaa…": "train", "roi_bbb…": "train", "roi_ccc…": "validation", "roi_ddd…": "test"}
```

`ready`인 모든 ROI를 `train` / `validation` / `test` 중 하나에 넣는다. 같은 계열의 ROI와 서로 겹치는 ROI는 한 구획에만 있어야 한다. `train`에는 서로 다른 위치가 2개 이상 필요하다.

### 3.6 `cohort.json` — 검색 대상 제품 (검색에 필수)

DL1이 고른 비교군 제품 id 목록이다. 검색의 주 결과는 이 제품들 안에서만 나온다.

```json
["PRODUCT_B", "PRODUCT_C"]
```

### 3.7 `labels.json` — ROI 대응 정답 (선택) — schema `roi_relations_v1`

```json
{
  "schema_version": "roi_relations_v1",
  "relations": [
    {"a": "roi_aaa…", "b": "roi_bbb…", "relation": "similar"},
    {"a": "roi_aaa…", "b": "roi_ccc…", "relation": "different"},
    {"a": "roi_aaa…", "b": "roi_ddd…", "relation": "unknown"}
  ]
}
```

없어도 학습된다. `unknown`은 손실에서 빠진다. 같은 쌍에 similar와 different가 함께 있거나 ROI 목록에 없는 id가 있으면 오류다.

## 4. 명령

```bash
# 3) ROI 목록 검증
.venv/bin/python -m psdl2.data review_rois.json --assets-root original_assets

# 4) 학습 (정답 없음)
.venv/bin/python -m psdl2.train review_rois.json --assets-root original_assets \
  --config config.json --lineages lineages.json --split split.json --out runs/dl2.pt \
  --device cuda --max-asset-memory-mb 8000
#   진행 상황은 stderr에 JSON 줄로 나온다(DL1과 같은 형식): started → running(10초마다) → epoch_completed.
#   epoch_completed에는 걸린 시간과 cuda_max_allocated_mb(GPU 최대 사용량)가 있다. --no-progress로 끈다.
#   실패하면 오류 JSON은 stderr의 마지막 줄이다(앞에 진행 줄이 있을 수 있다). 결과 JSON은 stdout 한 줄 그대로다.
#   CPU checkpoint에는 CUDA 난수 상태가 없어 CPU 전용 PC에서도 이어서 학습할 수 있다.

# 이어서 학습: config의 epochs만 늘리고 --resume
.venv/bin/python -m psdl2.train review_rois.json --assets-root original_assets \
  --config config.json --lineages lineages.json --split split.json \
  --out runs/dl2.pt --resume runs/dl2.pt

# 5) 색인: 후보 ROI 목록 전체를 벡터로
.venv/bin/python -m psdl2.index candidates/review_rois.json --assets-root original_assets \
  --checkpoint runs/dl2.pt --out runs/dl2_index.json --device cuda

# 6) 검색: query ROI 하나
.venv/bin/python -m psdl2.search query/review_rois.json --assets-root original_assets \
  --roi-id roi_aaa… --checkpoint runs/dl2.pt --index runs/dl2_index.json \
  --cohort cohort.json --topk 10 --out runs/dl2_search.json --device cuda
#   --expand 를 붙이면 cohort 밖 결과를 supplementary로 함께 낸다.

# 7) 평가: 검색 결과 여러 개를 JSON 배열 하나로 묶어 준다
.venv/bin/python -m psdl2.evaluate results.json --k 10 --split split.json \
  --partition test --labels labels.json

# 8) 위치 평가 (수치 저장소의 roi_correspondence_v1 라벨; a.roi_id = query, b = 정답 위치)
#    hit = 반환 후보 중심이 같은 제품·frame의 정답 중심에서 --hit-tolerance-um 안에 있을 때
.venv/bin/python -m psdl2.eval_position --search runs/dl2_search_1.json runs/dl2_search_2.json \
  --correspondence pairs_ids.json --hit-tolerance-um 5 --k 10 --out runs/eval_dl2.json

# 9) 비DL 기준선: 같은 role/layer layout 합집합에 대한 ZNCC template matching (학습 없음)
.venv/bin/python -m psdl2.baseline_ncc --query-export query/review_rois.json --index-export candidates/review_rois.json \
  --correspondence pairs_ids.json --window-um 20 --topk 10 --out runs/baseline.json
.venv/bin/python -m psdl2.eval_position --baseline runs/baseline.json \
  --correspondence pairs_ids.json --hit-tolerance-um 5 --k 10 --out runs/eval_baseline.json
# 9b) 배치 모드: 라벨 없이 query export의 ready ROI 전체를 질의로 사용 + 검토 시트
.venv/bin/python -m psdl2.baseline_ncc --query-export query/review_rois.json --index-export candidates/review_rois.json \
  --all-ready --max-memory-mb 8000 --window-um 30 --topk 4 --ambiguity-margin 0.02 --out runs/baseline_batch.json --sheet-dir runs/sheet
#   runs/sheet/sheet.html 을 열면 질의별 템플릿과 상위 매칭 이미지가 점수 내림차순으로 나온다(점수는 정렬용, 자동 채택 없음).
#   --max-memory-mb 는 --all-ready 에서 필수다. 층(role/layer)별로 FFT를 만들고 해제하며, 시작 전에 최대 메모리를 추정해 N MB를 넘으면 종료 코드 2로 멈춘다(자동 축소·절단 없음). 추정치는 출력 JSON의 memory_estimate_bytes 에 기록된다.
#   전체 추정치가 N MB를 넘으면 질의(같은 템플릿 해시는 한 묶음)를 N MB에 맞는 가장 큰 chunk로 나눠 순서대로 처리한다. chunk마다 FFT를 다시 계산하므로 chunk가 많을수록 느려지고(fft_recomputations), 결과 순위·점수·행 순서는 N과 무관하다. JSON에 chunk_count, chunk_sizes 가 추가된다. 질의 하나의 누적 맵 + 층 FFT 하나가 N을 넘을 때만 종료 코드 2.
#   --self-check --query-export … (--index-export 없이) 는 자기 자신에 매칭해 좌표·코드 오류를 점검한다(실패 시 종료 코드 2).
```

8)·9)는 같은 지표(top-1 hit rate, hit rate@k, top-1 거리)를 내므로 DL2와 기준선을 같은 조건에서 비교한다. 허용오차와 창 크기는 사용자가 정하며 기본값이 없다. 수십 개 라벨로는 통계 주장을 하지 않는다.

종료 코드:

| 코드 | 뜻 |
|---|---|
| 0 | 정상 |
| 2 | 입력·설정 오류. 표준 오류에 `{"error": …}`가 나온다 |
| 3 | 실행은 됐지만 결과가 온전하지 않음. 검증: blocked 또는 제외된 행이 하나라도 있음. 색인: 색인된 ROI가 0건. 검색: `status`가 `unavailable`. 평가: 정답이 없거나 계산할 query가 없음 |

## 5. 출력

### 5.1 검증 요약 (`psdl2.data`, 표준 출력)

ready 행과 blocked 행의 수, blocked 사유, producer 정보가 JSON으로 나온다.

### 5.2 학습 출력

`--out`에 지정한 파일이 checkpoint다. 매 epoch 끝에 원자적으로 교체된다. 모델, optimizer, 난수 상태, 설정·자료·분할 fingerprint, epoch 기록이 들어 있다.

표준 출력 JSON:

| 필드 | 뜻 |
|---|---|
| `status` | `completed` |
| `epochs`, `config_fingerprint`, `split` | 실행한 횟수, 설정 fingerprint, 실제 분할 |
| `history[]` | batch별 손실 항목: `total`, `alignment`, `variance`, `covariance`, `supervised` |
| `history[].attempted_pairs`, `retry_attempts`, `rejected_attempts_by_reason`, `accepted_pairs_generated`, `accepted_pairs_used`, `skipped_batches`, `optimizer_steps` | view 쌍이 얼마나 만들어지고 거부됐는지 |
| `epoch_history[]` | epoch별 같은 통계 |
| `performance_status` | `real_performance_pending` |

거부 사유 예: `core_joint_connectivity_changed`(두 view의 연결 관계가 다름), `core_marked_content_changed`, `common_core_not_observed`, `no_core_layout_anchor`, `resampled_core_joint_relation_changed`. 한 epoch에서 학습 단계가 한 번도 없으면 `no_usable_batch_in_epoch` 오류로 끝나고 마지막 완료 checkpoint는 그대로 남는다.

### 5.3 색인 (`psdl2.index --out`) — schema `psdl2_index_v2`

| 필드 | 뜻 |
|---|---|
| `checkpoint_fingerprint`, `view_fingerprint` | 어느 모델과 어느 창 설정으로 만들었는지. 검색 때 대조한다 |
| `topology_policy`, `connection_semantics`, `comparison_basis` | 연결 판정 규칙 |
| `records[]` | ROI마다: `roi_id`, `product_id`, `source_identity`, `frame`, `case_id`, `center_xy_um`, `source_assets`, `case_signature`, `valid_coverage`, `crop_geometry`, `embedding` |
| `rejected[]` | 색인에 넣지 못한 ROI와 사유 |
| `blocked_export_records[]` | 내보내기 단계에서 blocked였던 행 |

표준 출력에는 `n_indexed`, `n_rejected`와 두 fingerprint가 나온다. 이 출력은 승인 기록으로 따로 보관한다.

### 5.4 검색 결과 (`psdl2.search --out`)

```json
{
  "query": {"roi_id": "roi_aaa…", "product_id": "PRODUCT_A", "source_identity": {},
            "frame": {}, "case_id": "A", "center_xy_um": [102.5, 201.5],
            "case_signature": {}, "valid_coverage": {}, "crop_geometry": {}},
  "status": "available",
  "no_match_reasons": [],
  "primary": [
    {"roi_id": "roi_bbb…", "product_id": "PRODUCT_B", "score": 0.999994,
     "search_scope": "primary", "same_physical_case": false,
     "source_identity": {}, "frame": {}, "case_id": "A", "center_xy_um": [],
     "case_signature": {}, "valid_coverage": {}, "crop_geometry": {}}
  ],
  "supplementary": [],
  "primary_cohort": ["PRODUCT_B", "PRODUCT_C"],
  "expanded": false,
  "rejected": [],
  "index_roi_ids": []
}
```

| 필드 | 뜻 |
|---|---|
| `status` | `available` 또는 `unavailable` |
| `no_match_reasons` | 결과가 없을 때의 사유 |
| `primary[]` | cohort 안의 유사 위치. 유사도 높은 순 |
| `supplementary[]` | `--expand`일 때만 채워지는 cohort 밖 결과. primary와 섞이지 않고 순서를 바꾸지도 않는다 |
| `score` | cosine(−1~1)을 소수 6자리로 반올림. 같은 점수는 `roi_id` 순 |
| `same_physical_case` | 출처·frame·중심·관여 전극·분석 layer가 모두 같은 실제 같은 사례인지 |
| `case_signature` | `failure_type`, `analysis_layer`, 관여 전극의 역할·layer, CNT/VIA 유무 |
| `valid_coverage.near` / `.wide` | 각 창에서 관측된 칸 수와 비율, frame 밖 칸 수 |
| `crop_geometry.near` / `.wide` | 실제로 자른 범위(µm), 칸 수, 격자 간격, 변환 규칙 |
| `rejected[]` | 후보에서 뺀 ROI와 사유(예: `case_or_CNT_binding_incompatible`) |
| `index_roi_ids` | 검색에 쓴 색인의 ROI 전체 목록. 평가에서 정답이 색인에 있었는지 확인하는 데 쓴다 |
| `interpretation` | 결과를 어떻게 읽어야 하는지에 대한 고정 문구(유사도는 원인·위험 근거가 아님) |

후보로 삼는 조건은 불량 유형, 관여 전극의 역할·layer 조합, CNT/VIA 유무가 같은 것이다. 전극 이름과 순서는 보지 않는다. 결과가 없다는 것은 위험이 0이라는 뜻이 아니다.

### 5.5 평가 결과 (`psdl2.evaluate`, 표준 출력)

| 필드 | 뜻 |
|---|---|
| `status` | 정답이 없으면 `unavailable`, `reason: correspondence_labels_not_supplied` |
| `recall_at_k`, `k`, `n_evaluable_queries` | 주 결과 기준 Recall@k |
| `partition`, `n_selected_queries`, `excluded_queries` | 평가한 구획과 제외된 query |
| `counterexample_retrievals[]` | query별 recall과, 다르다고 알려진 ROI가 결과에 들어온 목록 |
| `positives_missing_from_index[]` | 정답이지만 색인에 없는 ROI. recall 분모에서 빠진다 |
| `performance_status` | `evaluation_of_supplied_labels_and_partition_only` |

## 6. 알아 둘 점

- **CNT/VIA 연결은 근사다.** 같은 위치에 CNT/VIA와 배선이 겹치면 이어진 것으로 본다. 내보낸 자료에 어느 전극·layer를 잇는지가 없기 때문이며 실제 도통을 확인한 것이 아니다.
- **거친 격자의 창은 거부가 많을 수 있다.** 거친 격자에서 떨어진 형상이 한 칸에 합쳐지면 그 쌍을 버린다. 거부율은 학습 기록에 나온다. 높으면 격자 간격을 줄인다.
- **색인과 검색은 같은 기기에서 돌린다.** 수치 저장소가 결과를 읽을 때 점수를 색인으로 다시 계산해 대조한다.
- **유사도는 위치 정합이 아니다.** 점수가 높아도 같은 원인이라는 근거가 아니다. CA 값은 항상 원본 결과에서 읽는다.

## 7. 폐쇄망에서 정할 것

- 가까운·넓은 창 크기, 격자 간격, 흔들림 폭, core 크기
- 건너뛴 layer 조합(예: 1–3)에 사이 layer를 넣을지
- 정답 형식과 수락 기준

## 8. 수치 저장소와의 연결

검색 결과와 색인은 `particle-sim-compare`의 `simcompare.dl_results`가 파일로 읽어 원본 CA·line/pixel 값 옆에 붙인다. `source_identity` 전체가 맞아야 연결되고 `case_id`만으로는 연결하지 않는다. 절차는 그 저장소의 `docs/handoff/commands.md`에 있다.
