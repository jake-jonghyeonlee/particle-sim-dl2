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
| 0 | 설치 | `uv pip install` / GPU: `scripts/setup_gpu_venv.sh` (2.1) | `.venv` / `.venv-gpu` |
| 1 | ROI 목록 내보내기 | `simcompare export-review-rois` (수치 저장소) | `review_rois.json` |
| 2 | 설정 파일 준비 | (사용자) | 실행 설정, lineage, split, cohort |
| 3 | ROI 목록 검증 | `python -m psdl2.data` | 요약 JSON |
| 4 | 학습 | `python -m psdl2.train` | checkpoint, 학습 기록 |
| 5 | 색인 만들기 | `python -m psdl2.index` | index JSON |
| 6 | 검색 | `python -m psdl2.search` | 검색 결과 JSON |
| 7 | 평가(정답이 있을 때) | `python -m psdl2.evaluate` | 평가 JSON |

여러 제품을 한 학습·색인에 넣고 무작위 patch로 학습하는 절차는 [4.1](#41-여러-제품무작위-patch-절차-psdl2_dataset_v1)에 있다. 위 표의 1단계 대신 `export-template-rois`(hotspot를 열지 않음)를 쓰고, 4–6단계는 `--dataset` / `--query-dataset`으로 돌린다.

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

### 2.1 GPU 환경 (Windows 폐쇄망, GTX 1660 Super)

기준 PC는 Windows, GTX 1660 Super 6 GB, RAM 16 GB다. DL1과 같은 torch 빌드(`2.7.1+cu118`)를 쓴다.
psdl2는 `torch>=2.7.1`이면 되고 numpy·Pillow 버전은 DL1과 같다. 명령은 **Git Bash**에서 실행한다(PowerShell 불필요).
기존 venv(DL1 포함)는 건드리지 않고 DL2 전용 새 venv `.venv-gpu`를 만든다.

#### 처음 한 번: 순서

```bash
# 1) 코드 받기
cd /d/work/particle-sim-dl2          # 실제 checkout 경로로 바꾼다
git pull

# 2) 드라이버 확인: GPU 이름과 "CUDA Version"(11.8 이상)이 보여야 한다
nvidia-smi

# 3) pip mirror 지정. DL1 설치 때 쓴 값과 같다. 모르면 DL1 venv에서 찾는다:
#    <DL1 venv>/Scripts/python.exe -m pip config list -v
export PIP_INDEX_URL=http://<mirror>/simple PIP_TRUSTED_HOST=<mirror-host>
#    +cu118 wheel이 다른 index에 있으면 추가로:
# export TORCH_INDEX_URL=http://<cu118-mirror>/simple

# 4) GPU venv 만들기 (이미 같은 이름의 폴더가 있으면 지우지 않고 멈춘다)
bash scripts/setup_gpu_venv.sh .venv-gpu

# 5) 설치 확인: CUDA 시험까지 실행된다(개발 Mac에는 GPU가 없어 이 시험은 여기서 처음 돈다)
PY=.venv-gpu/Scripts/python.exe
"$PY" -m pytest -p no:warnings -rs tests/test_device.py
"$PY" -m pytest -p no:warnings                     # 전체 시험
```

4)가 정상이면 마지막에 다음과 비슷한 줄이 나온다.

```text
python 3.12.x | torch 2.7.1+cu118 | cuda build 11.8 | numpy 2.5.3 | Pillow 12.3.0
cuda available: True
device: NVIDIA GeForce GTX 1660 SUPER | memory 6.0 GiB
psdl2 encoder CNN forward on cuda (2, 32)
Done. Run with --device cuda, ...
```

5)의 `tests/test_device.py`에서 `test_cuda_training_checkpoint_loads_on_cpu_and_cuda`가 **passed**여야 한다.
`SKIPPED ... CUDA device not available`이면 GPU가 잡히지 않은 것이다(아래 문제 해결).

#### 이후: 코드 갱신

`.venv-gpu`에는 이 checkout이 editable로 설치돼 있어 `git pull`만 하면 새 코드가 바로 쓰인다.
`pyproject.toml`의 의존성이 바뀐 경우에만 새 이름으로 venv를 다시 만든다(`bash scripts/setup_gpu_venv.sh .venv-gpu2`).

#### 실행 (Git Bash)

4장 명령의 `.venv/bin/python`을 `$PY`로 바꾸고 `--device cuda`를 붙인다. 경로는 실제 위치로 바꾼다.

```bash
PY=.venv-gpu/Scripts/python.exe

# 검증: 메모리 사용량 확인(asset_memory.decoded_bytes)
"$PY" -m psdl2.data /d/dl2/review_rois.json --assets-root /d/dl2/original_assets \
  --max-asset-memory-mb 8000

# 학습
"$PY" -m psdl2.train /d/dl2/review_rois.json --assets-root /d/dl2/original_assets \
  --config /d/dl2/config.json --lineages /d/dl2/lineages.json --split /d/dl2/split.json \
  --out /d/dl2/runs/dl2.pt --device cuda --max-asset-memory-mb 8000

# 색인, 검색
"$PY" -m psdl2.index /d/dl2/candidates/review_rois.json --assets-root /d/dl2/original_assets \
  --checkpoint /d/dl2/runs/dl2.pt --out /d/dl2/runs/dl2_index.json --device cuda --max-asset-memory-mb 8000
"$PY" -m psdl2.search /d/dl2/query/review_rois.json --assets-root /d/dl2/original_assets \
  --roi-id <roi_id> --checkpoint /d/dl2/runs/dl2.pt --index /d/dl2/runs/dl2_index.json \
  --cohort /d/dl2/cohort.json --topk 10 --out /d/dl2/runs/dl2_search.json --device cuda
```

- `--device`는 `train`, `index`, `search`에만 있다. 생략하면 `cpu`다. `cuda`를 줬는데 쓸 수 없으면 CPU로 넘어가지 않고
  종료 코드 2(`cuda_device_unavailable`)로 즉시 멈춘다.
- GPU에서 만든 checkpoint는 CPU로, CPU에서 만든 것은 GPU로 읽고 이어서 학습할 수 있다. 다만 GPU에서 저장한
  checkpoint를 GPU 없는 PC에서 `--resume`하면 CUDA 난수 상태를 복원할 수 없어 오류가 난다(색인·검색은 된다).
- GPU 결과는 CPU와 소수점 아래 자리에서 다를 수 있다. 같은 결과를 비교할 때는 같은 장치로 실행한다.
- `psdl2.baseline_ncc`(NCC 기준선)는 GPU를 쓰지 않는다. 그 메모리 상한은 별도 인수 `--max-memory-mb`다(4장 9b).

#### 메모리(6 GB GPU, 16 GB RAM)

모델이 작아 GPU 메모리보다 **RAM**(원본 mask)이 먼저 한계가 된다.

- 1660 Super는 tensor core가 없어 혼합정밀 이득이 작다. fp32로 돈다.
- 원본 mask는 같은 파일(경로·hash)을 여러 ROI가 공유하므로 제품당 layer 수만큼만 메모리에 올라간다.
  검증 요약의 `asset_memory`로 확인한다: `decoded_assets`(디코딩한 파일 수), `asset_references`(ROI가 참조한 횟수),
  `decoded_bytes`(남아 있는 mask 바이트), `max_bytes`(상한).
- `--max-asset-memory-mb N`(`data`, `train`, `index`, `search`)을 주면 디코딩한 mask가 N MB를 넘기 전에 종료 코드 2
  (`asset_memory_limit_exceeded`)로 멈춘다. 행 단위 blocked가 아니라 export 전체 거부다. 이 값은 남아 있는 mask만 센다.
  파일 바이트·디코딩 중간 배열은 잠깐 더 쓰므로 RAM 전체보다 넉넉히 작게 준다.
  16 GB에서는 8000으로 시작해 `decoded_bytes`를 보고 조정한다(실측 전 제안값).
- GPU 사용량은 학습 진행 줄의 `cuda_max_allocated_mb`로 본다. 6 GB(약 6000 MB)에 가까우면 `batch_size`를 줄인다.

#### 학습 진행 표시

학습 중 표준 오류(stderr)에 DL1과 같은 형식의 JSON 줄이 나온다. 결과 JSON은 표준 출력(stdout)에 한 줄로 따로 나온다.

```text
{"cuda_device": "NVIDIA GeForce GTX 1660 SUPER", "device": "cuda", "epochs": 20, "event": "progress", "locations": 120, "phase": "train", "start_epoch": 0, "state": "started"}
{"batch": 7, "batches": 15, "epoch": 3, "epochs": 20, "event": "progress", "loss": 0.412345, "phase": "train", "state": "running"}
{"cuda_max_allocated_mb": 812.4, "elapsed_s": 41.2, "epoch": 3, "epochs": 20, "event": "progress", "optimizer_steps": 15, "phase": "train", "skipped_batches": 0, "state": "epoch_completed"}
```

| state | 언제 | 주요 필드 |
|---|---|---|
| `started` | 학습 시작 | `device`, `cuda_device`, `locations`(학습 ROI 수), `start_epoch`(이어서 학습이면 저장된 epoch) |
| `running` | 10초에 한 번 | `epoch`, `batch`/`batches`, `loss`(건너뛴 batch면 null) |
| `epoch_completed` | epoch마다, checkpoint 저장 직후 | `optimizer_steps`, `skipped_batches`, `elapsed_s`, `cuda_max_allocated_mb`(CPU면 null) |

`--no-progress`로 끈다. 실패하면 오류 JSON `{"error": …}`이 stderr의 **마지막 줄**이다(앞에 진행 줄이 있을 수 있다).
로그로 남기려면 `2> runs/train_progress.jsonl`처럼 stderr를 파일로 돌린다.

#### 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| 설치 스크립트가 `CPU-only torch was installed`로 끝남 | mirror에 `+cu118` wheel이 없어 CPU 빌드가 깔림 | `TORCH_INDEX_URL`에 cu118 wheel이 있는 index를 지정하고 **새 이름**으로 다시 실행 |
| `Venv already exists and was preserved` | 같은 폴더가 이미 있음 | 다른 이름을 준다(`.venv-gpu2`). 기존 폴더는 지우지 않는다 |
| `CPython 3.12 required` | Git Bash의 `python`이 3.12가 아님 | `PYTHON=/c/.../Python312/python.exe bash scripts/setup_gpu_venv.sh .venv-gpu` |
| `CUDA build installed but no usable GPU` 또는 시험에서 CUDA SKIPPED | 드라이버 문제 | `nvidia-smi` 확인. 드라이버가 CUDA 11.8 이상을 지원해야 한다. 그동안은 `--device cpu` |
| pip가 패키지를 못 찾음 | mirror 주소·인증서 설정 누락 | DL1 venv에서 `pip config list -v`로 값 확인 후 `PIP_INDEX_URL`, `PIP_TRUSTED_HOST`(필요하면 `PIP_CERT`) 지정 |
| 실행 중 종료 코드 2, `cuda_device_unavailable` | `--device cuda`인데 GPU를 못 씀 | 위 드라이버 확인. CPU로 돌리려면 `--device cpu` |
| 종료 코드 2, `asset_memory_limit_exceeded` | 원본 mask가 상한 초과 | `psdl2.data`로 `decoded_bytes`를 확인하고 상한을 올리거나 export를 나눈다 |
| 종료 코드 2, `resume_cuda_rng_device_unavailable` | GPU에서 저장한 checkpoint를 GPU 없는 PC에서 이어서 학습 | GPU PC에서 이어서 학습한다 |
| `CUDA out of memory` | GPU 6 GB 초과 | config의 `batch_size`를 줄이거나 창 크기·격자를 확인 |

### 2.2 ETX(Linux 서버) 단일 파일 실행

서버에 `.py` 파일만 올릴 수 있고 명령줄 인자를 넘길 수 없을 때 쓴다. `scripts/build_single_file.py`가 `src/psdl2/*.py` 전체를 `.py` 하나로 묶는다. `baseline_ncc`·`eval_position`은 numpy와 Pillow만 있으면 되고(torch 불필요), `train`·`search`·`evaluate` 등은 torch가 필요하다. 묶음 실행 경로는 Python 3.8 이상에서 테스트했다(패키지 설치용 `requires-python`은 그대로).

```bash
.venv/bin/python scripts/build_etx_runner.py --overwrite   # bundles/etx/psdl2_job_baseline_ncc.py
```

- `bundles/etx/psdl2_job_baseline_ncc.py`는 simcompare 4회차(`candidates`) 결과로 두 방향 `baseline_ncc --all-ready`를 돌린다(정방향: `reference_export` 질의 → `target_index_export`, 역방향은 반대). 출력은 `data/dl2/ncc_r2t.json`, `ncc_t2r.json`과 검토 시트 `data/dl2/ncc_r2t_sheet/sheet.html`, `ncc_t2r_sheet/sheet.html`이다.
- 맨 위 EDIT 블록(`JOBS`, `LOG_FILE`, `WORK_DIR`, `BASE_DIR`)만 고친다. 상대 경로는 `BASE_DIR`(없으면 실행기 `.py`가 있는 폴더) 기준이며 simcompare 실행기와 같은 작업 폴더를 쓴다. `/data`처럼 `/`로 시작하면 절대 경로다. 로그 첫 줄에 기준 폴더, 둘째 줄에 Python·numpy·Pillow·torch 버전과 쓰인 `psdl2` 위치가 나온다.
- 종료 코드 0·3이면 다음 작업으로 넘어가고, 그 밖이면 멈춘다.
- `--allow-unversioned-producer`: ETX 실행기로 만든 simcompare export는 git checkout이 아니어서 `producer_commit`이 없고 `producer_commit_reason`만 있다. 이 플래그 없이는 `producer_commit_required`로 거부된다.
- `--window-um`, `--topk`, `--max-memory-mb`(20, 4, 8000)는 예시 값이다. `--sheet-dir`는 없거나 빈 폴더여야 한다.
- 다음 단계: 시트를 보고 `decisions.csv`를 쓴 뒤 simcompare 5회차(`confirm-pairs`)로 간다(simcompare `docs/usage.md` ETX 절).

#### 실행기 오류 코드

실행기가 스스로 멈출 때는 로그와 콘솔에 `[psdl2_job] ERROR <코드>: <영어 문구> | <한글 문구>` 한 줄을 남기고 종료 코드 2로 끝난다. traceback은 없다. 콘솔이 ASCII뿐이어도 영어 문구는 읽힌다. 로그 파일은 항상 UTF-8이고 로그를 먼저 쓴다.

| 코드 | 원인 | 조치 |
|---|---|---|
| `base_dir_missing` | `BASE_DIR`가 가리키는 폴더가 없다 | 존재하는 폴더의 절대 경로로 고친다. 이 오류는 로그 파일을 열기 전이라 콘솔에만 나온다 |
| `log_file_unwritable` | `LOG_FILE`을 `BASE_DIR` 아래에 쓸 수 없다(폴더 자리에 파일이 있거나 권한이 없음) | 쓸 수 있는 경로로 고치거나 `None`으로 둔다 |
| `torch_required_for_<모듈>` | 그 작업(`train`, `index`, `search`, `evaluate`, `make_split`)에 torch가 필요한데 이 Python에 없다 | torch가 있는 Python으로 실행하거나 `JOBS`에서 그 작업을 뺀다. 앞 작업의 산출물은 남아 있다 |
| `torch_import_failed` | torch 파일은 있으나 import가 실패한다(설치 손상, CUDA 라이브러리 등). 메시지는 `torch_import_failed: <예외 형식>: <메시지>`이며 버전 줄을 만들 때 낸 첫 예외를 그대로 쓴다(다시 import하지 않는다) | 메시지의 원인을 보고 torch 설치를 고친다. 앞 작업의 산출물은 남아 있다 |
| `out_must_be_file_path` | 표지를 쓰는 작업의 `--out`이 `/`(또는 `\`)로 끝난다 | `--out`에 파일 경로를 적는다. 폴더 경로는 받지 않는다 |
| `stale_done_marker:output_missing` | 완료 표지에 적힌 산출물이 지금 없다 | 표지(`<out>.psdl2_done.json`)를 지우고 다시 실행한다 |
| `stale_done_marker:output_modified` | 산출물 내용이 표지에 적힌 sha256과 다르다 | 같음. 산출물을 일부러 고친 것이면 다른 이름으로 보관한다 |
| `stale_done_marker:inputs_changed` | 입력 파일 내용 또는 작업 인자가 표지 때와 다르다 | 같음 |
| `stale_done_marker:runner_changed` | 이 실행기의 `psdl2`가 표지를 만든 실행기와 다른 판이다 | 같음 |
| `stale_done_marker:marker_unreadable` | 표지 파일이 깨졌거나(JSON 아님) 필요한 키가 없거나 형식이 다르다 | 같음 |
| `job_output_missing` | 작업이 종료 코드 0인데 `--out` 산출물이 없다 | 작업 인자의 `--out`을 확인한다. 표지는 쓰지 않는다 |

`baseline_ncc`·`eval_position`·`patches`·`data`는 numpy와 Pillow만 쓴다. `train`·`index`·`evaluate`·`make_split`은 import할 때, `search`는 실행할 때 torch를 쓴다. 이 검사는 각 작업을 시작하기 직전에 한다.

#### 재시작 표지

표지는 `baseline_ncc`와 `eval_position` 작업에만 쓴다(`MARKER_MODULES`). 그 밖의 모듈은 표지를 쓰지도 읽지도 않고 매번 실행하며, 시작할 때 로그에 `no done marker for <모듈> (indirect inputs are not hashed)`를 남긴다. 표지는 **인자에 적힌 파일만** 비교한다. 데이터셋이 가리키는 export, 체크포인트가 읽는 파일 같은 간접 입력은 해시하지 않으므로, 그런 모듈에 표지를 쓰면 입력이 바뀌어도 건너뛰는 오류가 난다.

`--out`이 있는 위 두 작업은 성공(종료 코드 0)하면 `<out>.psdl2_done.json`을 쓴다. 내용은 `module`, `runner_payload_sha256`, `argv`, `input_sha256s`(인자에 적힌 기존 입력 파일), `outputs`(`--out`과 `--sheet-dir` 안의 파일 경로·sha256)다. 같은 폴더의 임시 파일을 `os.replace`로 바꾸므로 반쪽 표지가 남지 않는다.

- 다시 실행하면 표지와 현재 상태(실행기 판·인자·입력·산출물 sha256)가 모두 같을 때 `skip completed <모듈>`을 로그에 남기고 건너뛴다. 1번 작업이 성공하고 2번이 실패했다면 고친 뒤 다시 실행했을 때 2번부터 돈다.
- 하나라도 다르면 위 표의 `stale_done_marker:<사유>`로 멈춘다. 자동으로 덮어쓰거나 지우지 않는다.
- 종료 코드 3(부분 결과)은 표지를 쓰지 않는다. `--out`이 없는 작업도 표지가 없다.
- `--out`이 구분자로 끝나면 `out_must_be_file_path`로 거부한다(표지가 폴더 안에 놓이는 것을 막는다).
- `baseline_ncc`의 `--sheet-dir`는 비어 있어야 하므로, 산출물을 다시 만들려면 표지와 함께 옛 `--out`·시트 폴더를 옮겨 둔다.

#### torch smoke 실행기

`bundles/etx/psdl2_job_smoke_torch.py`(생성기 `scripts/build_etx_smoke.py`)는 ETX에서 torch 경로가 도는지 확인하는 번들 하나다. 외부 데이터가 필요 없다. 올려서 실행하고 로그(`BASE_DIR/psdl2_smoke/smoke.log`)를 돌려준다.

```bash
.venv/bin/python scripts/build_etx_smoke.py --overwrite   # bundles/etx/psdl2_job_smoke_torch.py
```

- EDIT 블록: `DEVICE`(기본 `"cpu"`, `"cuda"`로 바꾸면 `train`·`index`·`search`가 GPU로 돈다), `BASE_DIR`(쓸 수 있는 폴더의 절대 경로), `WORK_DIR`, `LOG_FILE`.
- 실행하면 합성 fixture(호환 제품 3개 `P0`~`P2` + 음성 제품 `N`)를 `BASE_DIR/psdl2_smoke/` 아래에 만든다. 음성 제품은 `mask_derivation`이 달라 어떤 질의와도 호환되지 않으며 `rejected`로만 나온다. 이어서 `patches`(random·grid) → `make_split` → `train`(1 epoch) → `index` → `search --all-ready --exclude-query-product --per-product-topk 2`를 돌리고, 끝에서 결과를 확인한다(질의마다 후보가 있고, 질의 제품과 음성 제품 후보가 없다). 실패하면 `smoke_check_failed`로 멈춘다.
- 다시 실행하면 같은 폴더를 같은 바이트로 덮어쓴다(`patches`·`make_split`은 `--overwrite`).
- 각 단계의 시작·종료 코드와 python·numpy·Pillow·torch·`cuda available:` 판이 로그에 남는다. torch가 없으면 numpy만 쓰는 `patches` 8개가 돈 뒤 첫 torch 작업(`make_split`)에서 `torch_required_for_make_split`로 종료 코드 2다.
- 이 번들은 `train`·`index`·`search`·`make_split` 작업이라 완료 표지를 쓰지 않는다.

#### D10 기준 commit 갱신 절차

ETX `baseline_ncc` 번들의 `baseline_ncc` import 폐포(`baseline_ncc.py`, `data.py`, `eval_position.py`, `schema.py` 등)는 기준 commit의 소스와 바이트까지 같아야 한다. 기준 commit은 `tests/test_etx_runtime.py`의 상수 `D10_BASELINE_COMMIT`(현재 `efc6dac`) 하나다. 폐포 모듈을 **일부러** 바꾸는 Task는 이 상수를 그 Task의 기준 commit으로 옮기고, 이유를 commit 메시지에 쓴다. `search`·`index`·`train` 등 폐포 밖 모듈만 바꾸는 Task는 이 상수를 그대로 둔다(폐포에 `search`가 없음을 시험이 확인한다).

#### 로컬에서 ETX 환경 재현

ETX는 Python 3.9.14, numpy 1.23.5, Pillow 10.1.0, torch 2.1.1이다. 개발 venv(3.12, numpy 2.x)와 다르므로 실행기는 3.9 venv에서도 시험한다.

```bash
bash scripts/setup_etx_venvs.sh     # ~/.cache/psdl2-etx/ 에 venv39-notorch, venv39-torch (네트워크 1회 사용)
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_etx_runtime.py
```

- `venv39-notorch`: 3.9.14 + numpy 1.23.5 + Pillow 10.1.0. ETX 시험이 이 venv를 쓴다. `PSDL2_ETX_PY`로 다른 인터프리터를 지정한다.
- `venv39-torch`: 위 + torch 2.1.1 + `setuptools<81`(torch 2.1.1이 컴파일 중 `pkg_resources`를 import한다). torch smoke 시험이 이 venv를 쓴다. `PSDL2_ETX_TORCH_PY`로 다른 인터프리터를 지정한다. 여기서 `import torch`가 실패하면 스크립트는 종료 코드 1로 멈춘다. macOS는 PyPI wheel 그대로, Linux는 CPU wheel index(`https://download.pytorch.org/whl/cpu`)를 쓴다. ETX 실제 빌드는 cu121이므로 CUDA는 재현하지 못한다.
- 스크립트는 설치된 판을 출력한다. 정확히 3.9.14를 못 받으면 가까운 3.9.x로 만들고 그 사실을 출력한다.
- venv가 없으면 시험은 skip이 아니라 실패한다. 일부러 건너뛸 때만 `PSDL2_ALLOW_MISSING_ETX_PY=1`을 쓴다.
- 이 시험은 합성 데이터로 `baseline_ncc` 두 방향 → `eval_position` 흐름, 비ASCII·공백 `BASE_DIR`, ASCII 콘솔(`LOG_FILE=None` 포함), 위 오류 코드, 재시작, torch smoke 번들 전체를 확인한다. 실제 ETX(Linux 파일시스템 권한, CUDA, 실데이터 규모)는 확인하지 못한다.

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
#   --max-memory-mb 는 --all-ready 와 --self-check 에서 필수다. 층(role/layer)별로 FFT를 만들고 해제하며, 시작 전에 최대 메모리를 추정해 N MB를 넘으면 종료 코드 2로 멈춘다(자동 축소·절단 없음). 추정치는 출력 JSON의 memory_estimate_bytes 에 기록되며, 기여 ROI 집합별 유효 영역 맵(validity_map_count, validity_map_bytes)도 추정과 한도 검사에 포함된다.
#   전체 추정치가 N MB를 넘으면 질의(같은 템플릿 해시는 한 묶음)를 N MB에 맞는 가장 큰 chunk로 나눠 순서대로 처리한다. chunk마다 FFT를 다시 계산하므로 chunk가 많을수록 느려지고(fft_recomputations), 결과 순위·점수·행 순서는 N과 무관하다. JSON에 chunk_count, chunk_sizes 가 추가된다. 질의 하나의 누적 맵 + 층 FFT 하나가 N을 넘을 때만 종료 코드 2.
#   --self-check --query-export … (--index-export 없이) 는 자기 자신에 매칭해 좌표·코드 오류를 점검한다(실패 시 종료 코드 2). 실제 계산된 누적 점수 맵의 자기 위치 값(~1)과 순위 항목의 좌표 환산과, 각 항목의 점수가 그 항목의 좌표에서 실제 누적 맵이 가진 값(map_score)과 같은지를 검사하므로 틀린 템플릿·맵·좌표는 실패한다. 반복 형상에서 다른 반복이 1위여도 통과하며 반복을 대신 고르지 않는다.
#   --sheet-dir 은 없거나 빈 디렉터리여야 한다(비어 있지 않거나 심볼릭 링크면 종료 코드 2, 어떤 파일도 쓰지 않음; 디렉터리 fd 기준으로 쓰므로 실행 중 디렉터리를 바꿔치기해도 바깥에 쓰거나 지우지 않고, 바뀌면 만든 파일을 지우고 종료 코드 2). 유효 영역 구멍은 해당 role/layer에 기여하는 ROI만 대상으로 query_valid_hole / candidate_valid_hole 플래그와 무효 픽셀 수로 표시되며 점수는 바뀌지 않는다. 출력에 implementation_revision 이 기록된다.
```

8)·9)는 같은 지표(top-1 hit rate, hit rate@k, top-1 거리)를 내므로 DL2와 기준선을 같은 조건에서 비교한다. 허용오차와 창 크기는 사용자가 정하며 기본값이 없다. 수십 개 라벨로는 통계 주장을 하지 않는다.

종료 코드:

| 코드 | 뜻 |
|---|---|
| 0 | 정상 |
| 2 | 입력·설정 오류. 표준 오류에 `{"error": …}`가 나온다 |
| 3 | 실행은 됐지만 결과가 온전하지 않음. 검증: blocked 또는 제외된 행이 하나라도 있음. 색인: 색인된 ROI가 0건. 검색: `status`가 `unavailable`. 평가: 정답이 없거나 계산할 query가 없음 |

### 4.1 여러 제품·무작위 patch 절차 (`psdl2_dataset_v1`)

학습 위치를 사람이 고른 ROI 몇 개가 아니라 **형상 기준으로 DL2가 만든 patch**로 하고, 여러 제품을 한 학습·한 색인에 넣는 절차다. 수치(N, 간격, 격자 step, 반복 주기)에는 기본값이 없다. 실행 때마다 명시한다. 아래 순서의 실제 예는 `tests/test_e2e_dataset.py`다(합성 3제품·3계열).

| 순서 | 하는 일 | 명령 | 만들어지는 것 |
|---|---|---|---|
| A | template export (제품마다) | `simcompare export-template-rois` (수치 저장소, `docs/template_export.md`) | `roi/review_rois.json`, `roi/mask_derivation.json` |
| B | dataset manifest 쓰기 | (사용자) | `psdl2_dataset_v1` JSON (아래 스키마) |
| C | 학습용 random patch | `python -m psdl2.patches --mode random` | `psdl2_patch_set_v1` (제품마다) |
| D | 색인용 grid patch | `python -m psdl2.patches --mode grid` | `psdl2_patch_set_v1` (제품마다) |
| E | 분할 | `python -m psdl2.make_split` | `split.json` |
| F | 학습 | `python -m psdl2.train --dataset` | checkpoint |
| G | 색인 | `python -m psdl2.index --dataset` | index JSON(`sources[]`) |
| H | 검색 | `python -m psdl2.search --query-dataset` | 검색 결과 JSON |
| I | 위치 평가 | `python -m psdl2.eval_position` | 평가 JSON |

```bash
# A) 제품마다 (수치 저장소). 1.png/2.png(처리 마스크)만 읽고 hotspot는 열지 않는다
simcompare export-template-rois --config template_A.yaml --out exports/prodA

# B) dataset.json — 상대 경로는 이 파일의 위치 기준. mask_derivation은 인라인 객체 또는 파일 경로
#    {"schema_version": "psdl2_dataset_v1", "sources": [
#       {"export": "exports/prodA/roi/review_rois.json", "export_sha256": "<sha256>",
#        "assets_root": "exports/prodA/data", "mask_derivation": "exports/prodA/roi/mask_derivation.json",
#        "patch_sets": []}, ...]}
#    patch_sets가 비어 있지 않은 source는 template 행이 빠지고 patch 행만 들어간다.

# C) 학습용 random patch: case(= ready manual template 행)마다 N개. --source는 manifest의 source 번호
python -m psdl2.patches --dataset dataset.json --source 0 --config config.json --mode random \
  --seed 3 --per-case 200 --min-spacing-um 5 [--repeat-period-um 40 [40]] [--allow-shortage] --out patches/random_A.json
#    N을 채우지 못하면 종료 코드 2(파일 없음, stderr에 case별 깔때기). --allow-shortage는 부족을 헤더에 기록하고 쓴다(전부 0개면 여전히 2).

# D) 색인용 grid patch: frame origin + (i·step, j·step) 중 위치 조건을 통과한 점 전부
python -m psdl2.patches --dataset dataset.json --source 0 --config config.json --mode grid \
  --grid-step-um 2 [--coverage-labels pairs_ids.json] --out patches/grid_A.json
#    격자 간격은 평가의 hit 허용 거리 이하로 잡는다. --coverage-labels를 주면 정답 위치마다 가장 가까운 격자점 거리를 보고한다.

# 제품마다 C·D를 반복한 뒤 patch set 파일을 manifest의 patch_sets에 적어 학습용(random)·색인용(grid) manifest를 따로 둔다.

# E) 제품 → 계열(lineages.json)과 계열 → 구획(assign.json)에서 split.json을 만든다. 누수가 있으면 오류
python -m psdl2.make_split --dataset train_dataset.json --config config.json \
  --lineages lineages.json --assign assign.json --out split.json
#    assign.json 예: {"lineage_0": "train", "lineage_1": "validation", "lineage_2": "test"}

# F) 학습. export 인자 대신 --dataset (둘 다/둘 다 없음은 오류). --assets-root는 쓰지 않는다
python -m psdl2.train --dataset train_dataset.json --config config.json --lineages lineages.json \
  --split split.json --out runs/dl2.pt --max-asset-memory-mb 8000

# G) 색인: grid patch dataset 전체
python -m psdl2.index --dataset index_dataset.json --checkpoint runs/dl2.pt --out runs/dl2_index.json

# H) 검색: query는 기존 query export를 감싼 manifest(query 쪽 mask_derivation 포함). 학습에 없는 계열의 ROI를 쓴다
python -m psdl2.search --query-dataset query_dataset.json --roi-id roi_aaa… --checkpoint runs/dl2.pt \
  --index runs/dl2_index.json --cohort cohort.json --topk 50 --out runs/dl2_search.json
#    [--ignore-mask-derivation] 을 주면 derivation 비교를 생략하고 결과에 "mask_derivation_comparison": "ignored"를 남긴다

# H2) 검색 옵션과 배치. 새 옵션을 주지 않으면 출력은 위와 바이트까지 같다
#    --exclude-query-product : 질의와 product_id가 같은 후보를 primary·supplementary에서 뺀다(rejected 사유 query_product_excluded)
#    --per-product-topk M    : 정렬 뒤 제품마다 앞 M개만 남기고 --topk로 자른다(primary·supplementary 각각, 점수 보정 없음)
#    --all-ready             : --roi-id 대신. 질의 export/dataset의 ready ROI 전부를 roi_id 사전순으로 돌려 봉투 하나로 쓴다
python -m psdl2.search --query-dataset query_dataset.json --all-ready --exclude-query-product --per-product-topk 2 \
  --checkpoint runs/dl2.pt --index runs/dl2_index.json --cohort cohort.json --topk 4 --out runs/dl2_search_batch.json
#    --roi-id 와 --all-ready 는 정확히 하나만 준다(아니면 exactly_one_of_roi_id_or_all_ready_required, 종료 코드 2)
#    종료 코드: 모든 질의 available → 0, 하나라도 unavailable → 3(ETX 실행기는 계속 진행), 계약 오류 → 2

# I) 위치 평가 (4절 8번과 같다)
python -m psdl2.eval_position --search runs/dl2_search.json --correspondence pairs_ids.json \
  --hit-tolerance-um 5 --k 50 --out runs/eval_dl2.json

# 기준선: baseline_ncc는 dataset을 받지 않는다. 같은 cohort의 template export(patch가 아닌 review_rois.json)를 기존 인터페이스로 준다
python -m psdl2.baseline_ncc --query-export query/review_rois.json --index-export exports/prodA/roi/review_rois.json \
  --correspondence pairs_ids.json --window-um 20 --topk 50 --out runs/baseline.json
python -m psdl2.eval_position --baseline runs/baseline.json --correspondence pairs_ids.json \
  --hit-tolerance-um 5 --k 50 --out runs/eval_baseline.json
```

알아 둘 점:

- **처리 마스크.** 1.png/2.png는 case별 처리 마스크다(위쪽 전극은 중간 layer를 뺀 것, lateral은 그대로). 의미와 `mask_derivation`은 [data_contract.md](data_contract.md) “Processed masks”에 있다.
- **검색 호환.** query와 후보의 `mask_derivation`(규칙 버전, 처리 endpoint, 제거 layer 목록)이 같아야 한다. 다르면 `mask_derivation_mismatch`, 한쪽만 있으면 `mask_derivation_unknown`으로 제외하고 `rejected`에 이유가 남는다. 표는 [training.md](training.md)에 있다.
- **patch 좌표 조건과 preflight는 다르다.** 생성기는 창 전체가 frame 안인지, core가 전부 valid인지, anchor가 있는지 본다. 학습의 사전 검사(`views.preflight`)는 그것을 보지 않는다.
- **종료 코드.** patches: 0 정상, 2 오류·부족(종료 코드 3은 쓰지 않는다). make_split: 0 / 2. 그 외는 위 표와 같다.
- **ETX 실행기.** 2.2의 `psdl2_job_baseline_ncc.py`가 도는 작업은 `baseline_ncc`·`eval_position`이다. 이 절차 전체는 2.2의 torch smoke 실행기가 합성 데이터로 한 번 돈다. 실데이터는 `python -m psdl2.<모듈>`로 돌린다.

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
| `excluded_products` | `--exclude-query-product`를 줄 때만 있다. `[<질의 product_id>]` |
| `per_product_topk` | `--per-product-topk`를 줄 때만 있다. 준 값 M |

점수는 표시용이다. 수치 계산에 쓰지 않는다. 새 옵션을 주지 않은 단일 질의 출력은 efc6dac의 출력과 바이트까지 같고, 행(`primary`·`supplementary`)의 키 집합은 어떤 옵션에서도 바뀌지 않는다.

#### 5.4.1 배치 봉투 `psdl2_search_batch_v1` (`--all-ready`)

```json
{
  "schema_version": "psdl2_search_batch_v1",
  "checkpoint_fingerprint": "<64 hex>", "view_fingerprint": "<64 hex>",
  "index_sources": [{"export_sha256": "<64 hex>", "patch_set_sha256s": ["<64 hex>"], "product_ids": ["P0"]}],
  "inputs": {"query_export_sha256": null, "query_dataset_sha256": "<64 hex>", "index_sha256": "<64 hex>"},
  "arguments": {"topk": 4, "expand": false, "per_product_topk": 2, "exclude_query_product": true,
                "ignore_mask_derivation": false, "cohort": ["P0", "P1", "P2"]},
  "results": [],
  "rankings": [{"query_roi_id": "P0_case0", "query_product_id": "P0",
                "candidates": [{"rank": 1, "roi_id": "patch_…", "product_id": "P1", "frame_id": "frame_P1_case0",
                                "center_xy_um": [106.0, 204.0], "search_scope": "primary"}]}]
}
```

| 필드 | 뜻 |
|---|---|
| `index_sources` | 색인이 단일 export로 만들어졌으면(색인에 `sources`가 없다) 문자열 `"single_export"`. dataset으로 만들었으면 source마다 `export_sha256`, `patch_set_sha256s`, `product_ids`. `product_ids`는 그 source의 patch set에서 나온 색인 레코드의 제품이다. 색인 레코드는 source가 아니라 patch set 해시로만 이어지므로, patch set이 없는 source는 `[]`다 |
| `inputs` | `query_export_sha256`: `export` 인자 파일의 sha256(`--query-dataset`이면 `null`). `query_dataset_sha256`: manifest 파일의 sha256(`export` 인자면 `null`). `index_sha256`: `--index` 파일의 sha256 |
| `arguments` | 이 실행의 `topk`, `expand`, `per_product_topk`(없으면 `null`), `exclude_query_product`, `ignore_mask_derivation`, 정렬된 `cohort` |
| `results[i]` | 같은 인수로 `--roi-id <그 질의>`를 돌린 단일 출력과 같다(canonical JSON: 키 정렬, 같은 float 표기). 순서는 `roi_id` 사전순이고 아무것도 덧붙이지 않는다 |
| `rankings[i]` | `results[i]`에서 하나의 유도 함수(`derive_rankings`)로 만든다. `rank`는 1부터, primary 순서 다음 supplementary 순서. `frame_id`는 행의 `frame.frame_id` 그대로 |

한 번에 `--out` 하나를 같은 폴더 임시 파일 + `os.replace`로 쓴다. 중단되면 옛 파일이 남는다. 질의 하나가 계약 오류면 종료 코드 2이고 파일을 쓰지 않는다.

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
