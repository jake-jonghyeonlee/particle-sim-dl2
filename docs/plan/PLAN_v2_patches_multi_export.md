# 무작위 patch 학습 · 다중 제품 입력 계획 v2

버전: **PLAN_v2.2 / 2026-10-21 / Orca run `run_c2a33ea63250` / coordinator 수락**. v2.2는 T0 결과와 사용자 결정 D6–D8을 반영했다. v1.2(`PLAN_v1_patches_multi_export.md`)를 대체한다. v1.2는 기록으로 남긴다.

- 계획 검토: Codex `gpt-6-astra` / `xhigh`(사용자 지정, 이번 한 번), dispatch `ctx_42905d86bc80`, 판정 ACCEPT-WITH-CHANGES(`/tmp/astra_plan_review_v1_2.md`). v2 재검토 dispatch `ctx_6a3019e0e23c`: B1·B3·B5·B6·N1–N6 해결, B2·B4 부분 해결(`/tmp/astra_plan_recheck_v2.md`). v2.1에서 R1(B2 id 범위)·R2(B4 종료 코드)와 연결 세부를 고쳤다.
- 기준선: DL2 `feat/etx-single-file` @ `f85e8c0`(미추적 `.DS_Store`, `docs/plan/`), sim-compare @ `f30111c`.
- 사용자 결정(D1–D5, Q1–Q8)은 확정 사항이다. 이 문서에서 다시 열지 않는다.

## 0. 확정 결정

| 번호 | 결정 |
|---|---|
| D1 | DL2는 학습 / 검색 / 평가로 나눈다. hotspot·CA는 어느 단계에서도 모델 입력이 아니다 |
| D2 | 학습·색인 위치는 DL2 안의 patch 생성기가 형상 기준으로 만든다 |
| D3 | 여러 제품을 한 학습·한 색인에 넣는다 |
| D4 | layout 입력은 case(전극 조합쌍)별 처리 마스크(1.png/2.png). vertical은 위쪽 전극만 중간층 차감, lateral은 차감 없음 |
| D5 | CNT는 제품 전체 1장을 공유. VIA는 쓰지 않는다(빈 목록). VIA로 이어진 layer는 이미 같은 전극이다 |
| Q1 | 두 endpoint layout이 near 창 안에 모두 marked 셀을 가질 것 **그리고** core에 layout anchor가 있을 것 |
| Q2 | 수치(N, 간격)는 실행 시 명시. 기본값 없음 |
| Q3 | 색인 격자 간격 ≤ hit 허용 거리 |
| Q4 | 반복 주기 제한은 선택 사항이며 **random 학습 patch에만** 적용 |
| Q5 | patch 행은 `source_kind=manual` + `provenance.patch`. 새 source_kind 값은 만들지 않는다 |
| Q6 | sim-compare 수정 허용. 단 기존 `export-review-rois`·`candidates`의 동작·출력·오류·테스트는 그대로, 새 경로는 추가만 |
| Q7 | `mask_derivation`은 dataset 파일에 두고, 검색 호환 조건에 기본 포함. 명시 옵션으로만 해제 |
| Q8 | 처리 마스크임을 provenance와 문서에 명시. 기존 export는 고쳐 쓰지 않는다 |
| D6 | (T0 후) `removed_layers` = `layer_stack`의 stack_order가 두 endpoint **사이**인 layer 전부. 규칙 이름 `between_stack_order_v1`. `layer_stack`에 모든 layer가 들어 있다는 것은 사용자 전제다. lateral은 빈 목록 |
| D7 | (T0 후) T0b를 만든다: sim-compare에 hotspot을 열지 않는 단일 제품 `export-template-rois` 명령을 추가한다 |
| D8 | (T0 후) CNT는 DL1 `cno/*.png` 합집합이 맞다 |

## 1. 현재 상태 (근거, v1.2의 틀린 줄 번호 수정)

- 입력: `train`은 export 하나(`src/psdl2/train.py:167`), `index`도 하나(`src/psdl2/index.py:96`), `search`도 export 하나에서 query를 만든다(`src/psdl2/search.py:143-173`).
- ROI마다 `versions`가 있다(`src/psdl2/data.py:22-27`). 학습 fingerprint는 split 기록·`source_versions`·`source_assets`만 담는다(`src/psdl2/train.py:94-97`). 행의 다른 provenance는 담지 않는다.
- 색인의 `producer_versions`(`src/psdl2/index.py:109`)는 search가 읽지 않는다. 검색 호환은 case signature만 비교한다(`src/psdl2/search.py:111-112`, 키 검증 `:51-53`). 색인 레코드는 crop bundle 메타데이터에서 만든다(`src/psdl2/index.py:31-52`, `src/psdl2/crop.py:150-164`).
- `source_kind` 허용값은 `src/psdl2/data.py:166-169`. 채널 검사는 `:131-215`.
- **preflight는 창 전체가 frame 안에 있는지 보지 않는다.** core 관측, pitch, jitter 끝점에서 core 셀 표현만 본다(`src/psdl2/views.py:103-130`). frame 밖 창 셀은 계약상 padding이다(`docs/data_contract.md:91-98`). layout anchor 검사는 view 생성 때 한다(`views.py:234-237`). Astra 실험: 100×100 frame, 중심 (5,5), wide 20, jitter 1, layout 전부 0인 ROI도 preflight를 통과했다. 학습은 `train.py:89-90`에서 preflight를 호출한다.
- 분할: 쌍 비교 O(n²)(`src/psdl2/split.py:43-45`), 모든 id 명시 필요(`:53-56`). 학습은 train 구획만 쓴다(`train.py:86`).
- mask cache는 `load_export` 호출마다 새로 만든다(`data.py:228-231`). 키는 실제 경로·hash·의미·shape(`:93-99`).
- `tests/test_single_file.py:38-47`은 `src/psdl2/*.py` 전체를 묶은 `bundles/etx/psdl2_job_baseline_ncc.py`가 최신인지 바이트로 검사한다. 소스를 바꾸면 이 번들을 다시 만들어야 한다(`scripts/build_etx_runner.py --overwrite`).
- `eval_position`은 `roi_correspondence_v1`(물리 좌표 정답)만 받는다(`src/psdl2/eval_position.py:26-40`). 정답·후보가 없는 query는 지표에서 빠진다(`:90-93`).
- sim-compare 기존 export는 manual 선택이어도 데이터 로딩과 electrode-map 계산을 한다(`particle-sim-compare/src/simcompare/pipeline.py:564-583`, `:602-605`). real 로더는 hotspot mask를 읽는다(`adapters/real_hotspot.py:660-670`). `candidates`는 두 제품이 필요하다(`candidates.py:415-420`).

## 2. 단계별 준비물 · 입력 · 결과

공통 준비물(제품마다): import된 제품 폴더(case별 처리 마스크 2장, CNT 1장, valid), template export(T0/T0b), `lineages.json`, `mask_derivation`(T0이 출처를 확정).

| 단계 | 넣는 것 | 얻는 것 |
|---|---|---|
| 학습 | random patch(T2) → dataset(T1) → `make_split`(T3) → `train --dataset` | checkpoint |
| 색인 | grid patch → dataset → `index --dataset` | patch마다 embedding·위치·case·`mask_derivation`·patch provenance |
| 검색 | 색인 + query dataset(기존 query export를 감싼 manifest) → `search --query-dataset` | 호환 후보 순위·점수, 거부 사유 |
| 평가 | 학습에 없는 계열 query + `roi_correspondence_v1` → `eval_position`, 같은 cohort로 `baseline_ncc` | hit 비율, 기준선 대비 |

## 3. Task

모든 편집 Task는 한 명씩 순서대로 한다. 각 Task는 `src/psdl2`를 바꾸면 **같은 Task 안에서** `bundles/etx/psdl2_job_baseline_ncc.py`를 `scripts/build_etx_runner.py --overwrite`로 다시 만들고, 전체 `pytest`를 통과해야 한다. 번들 생성 스크립트와 실행기 동작은 바꾸지 않는다. 기존 테스트를 약하게 만들거나 건너뛰지 않는다.

공통 게시 규칙(새 산출물 patch set, split, dataset 관련 파일): 같은 폴더 임시 파일 → flush/fsync → `os.replace`. 출력이 이미 있으면 `--overwrite` 없이는 오류. 실패 시 자기가 만든 임시 파일만 지운다.

### T0 — template export 경로와 derivation 출처 확인 (읽기 전용, sim-compare)

- **확인할 것.** 각각 `파일:줄` 근거와 결론을 낸다.
  1. 한 제품, manual 행만으로 template export를 만들 수 있는가. **hotspot·CA 배열이 없고 두 번째 제품이 없어도** 되는가. 실행 중 읽는 파일 목록(의존성 목록)을 적는다. 이미 import된 identity/frame 메타데이터와 실행 중 필요한 numeric 배열을 구분한다.
  2. layout 참조가 case별 처리 마스크(1.png/2.png)를 가리키는가. CNT가 제품 1장인가. VIA 목록이 비는가.
  3. case마다 "위쪽 전극에서 뺀 layer 목록"을 sim-compare 데이터에서 얻을 수 있는가. 어느 파일·필드인가. 위쪽 전극이 어느 endpoint인지 알 수 있는가.
  4. case identity 정의(composite key)와 같은 case_id가 여러 문맥에 나오는 경우.
- **결과.** (a) 기존 명령으로 충족 → 설정 예시와 의존성 목록. (b) 불충족 → T0b 계약 초안(파일, 출력, 실패 동작, 회귀 게이트). 3번이 불가능하면 꾸며내지 말고 coordinator에 계약 공백으로 보고한다.
- **실제 결과(`/tmp/T0_report.md`, dispatch `ctx_9b680ee1cd45`).** 기존 3개 명령(`import-hotspots` → `import-electrode-assets` → `export-review-rois`)으로 export는 가능하다. 하지만 입력 폴더에 hotspot이 있으면 읽고, 이를 막는 코드가 없다. layout은 case별 1.png/2.png, CNT는 제품 공유 1장(`cno/*` 합집합), VIA는 빈 목록이다. 위쪽 endpoint는 `analysis_layer`로 알 수 있지만, 제거 layer 목록은 어디에도 없다 → D6·D7로 해결. valid는 항상 전부 1이다(`real_hotspot.py:634`).
- **수락.** 위 4개 각각의 결론과 근거. 소스·데이터 수정 없음. 보고서 `/tmp/T0_report.md`.
- **위험.** 읽기 전용. 코드 리뷰 PASS 불필요.

### T0b — sim-compare `export-template-rois` (High, D7)

- **대상.** particle-sim-compare 저장소, 새 브랜치 `feat/template-export`(기준 `f30111c`).
- **새 명령** `simcompare export-template-rois --config <yaml> --out <없는 폴더>`. 제품 하나, 데이터셋 하나.
- **입력 `template_export_config_v1`.** 1.png/2.png만 있는 트리, `cnt_from_dl1_root`, id 필드, `layer_stack`(actual_layer, stack_order만), role 규칙, `source_version`, `max_asset_memory_mb`, case별 `center_xy_um`, `mask_derivation_rule: between_stack_order_v1`(이 값만 허용, 생략하면 exit 2). `density_process_layer`는 받지 않고 내부 sentinel로 채운다.
- **hotspot 미접촉.** 트리에서 hotspot 파일명 패턴이나 `result/`가 보이면 열지 않고 `template_hotspot_input_present`로 exit 2.
- **출력**(임시 폴더 → `os.replace`, 기존 출력 거부): `<out>/data/`(import 결과와 adapter를 합친 폴더), `<out>/roi/review_rois.json`(`review_roi_v1`, 스키마 변경 없음, 기존 export 함수로 생성, asset은 복사하지 않고 참조), `<out>/roi/mask_derivation.json`(`psdl2_mask_derivation_v1`, ready ROI의 case마다 한 항목), `<out>/template_export_execution.json`(입력 hash, 읽은 파일 목록).
  - vertical: `processed_endpoint` = `analysis_layer`와 같은 layer의 endpoint, `removed_layers` = 사이 stack_order layer 전부(정렬, 중복 없음). endpoint layer가 `layer_stack`에 없으면 exit 2.
  - lateral: `processed_endpoint: null`, `removed_layers: []`.
- **종료 코드.** 기존 규약: 0 완료, 3 일부 blocked(이때 mask_derivation은 ready ROI만), 2 입력·설정 오류.
- **변경 허용 범위.** 새 파일(`src/simcompare/template_export.py`, 테스트, `docs/template_export.md`)과 `cli.py`의 서브파서 등록 한 블록. `roi_export.py`, `pipeline.py`, `candidates.py`, `adapters/*`는 0줄 변경. 공유 로더·helper 수정 금지.
- **수락.**
  - 변경 전 소스로 먼저 만든 골든과 비교: 기존 `export-review-rois`(numeric/real/manual/blocked/error)와 `candidates`의 출력 바이트(시간·절대 경로 필드는 명시 후 제외), 종료 코드, stderr가 같다.
  - 기존 테스트 전부 통과, 수정·skip 없음. 저장소에 번들 최신성 테스트가 있으면 번들을 다시 만든다.
  - 새 테스트: hotspot 없는 fixture 동작, `open`/`np.load` 계측으로 hotspot류 경로 미접촉, hotspot 있는 트리 거부, 출력이 DL2 `load_export` 검증 통과, layout = case별 1.png/2.png, CNT = 제품 공유 asset, `via: []`, mask_derivation의 vertical(사이 layer 여러 개·없음)·lateral·layer_stack 누락.

### T1 — 다중 export dataset과 mask_derivation 검색 경로 (High)

- **스키마 `psdl2_dataset_v1`.**

  ```json
  {"schema_version": "psdl2_dataset_v1",
   "sources": [
     {"export": "prodA/review_rois.json", "export_sha256": "…", "assets_root": "prodA/assets",
      "patch_sets": [],
      "mask_derivation": {"schema_version": "psdl2_mask_derivation_v1", "rule_version": "…",
        "cases": [{"case": {"failure_type": "vertical_short",
                            "endpoints": [{"electrode_id": "A", "role": "…", "actual_layer": "M1"},
                                          {"electrode_id": "C", "role": "…", "actual_layer": "M3"}]},
                   "processed_endpoint": "C", "removed_layers": ["M2"]}]}}]}
  ```

  - 상대 경로는 manifest 파일 위치 기준. 작업 폴더와 무관.
  - `mask_derivation`은 선택이며, 인라인 객체이거나 `psdl2_mask_derivation_v1` 파일의 상대 경로다(T0b 출력 `mask_derivation.json`을 그대로 가리킬 수 있음). 있으면 그 source의 모든 ready ROI가 정확히 한 항목에 대응해야 한다(`mask_derivation_case_missing`). 어떤 ROI에도 대응하지 않는 항목, 같은 case 중복 항목은 오류. case 대응은 failure_type과 endpoint 집합(순서 무관)으로 한다.
  - 정규화: `removed_layers`는 중복 없는 정렬 목록. lateral은 `processed_endpoint: null`, `removed_layers: []`로 **알려진 빈 값**이다. 필드가 없는 것(모름)과 구분한다. `processed_endpoint`는 null(lateral 예외)이 아니면 그 case의 endpoint 중 하나여야 한다.
- **`psdl2.data.load_dataset`.**
  - source마다 기존 `load_export` 검증을 그대로 쓴다. export sha256 불일치는 오류.
  - mask cache 하나를 전 source가 공유하고, `--max-asset-memory-mb`는 전체 합에 건다. 키는 지금처럼 실제 경로·hash 기반(같은 상대 파일명이 제품마다 달라도 섞이지 않음).
  - `roi_id`는 전체에서 유일(`duplicate_roi_id_across_sources`).
  - **T1 단계에서는 비어 있지 않은 `patch_sets`를 `patch_sets_not_supported` 오류로 거부한다.** 조용히 무시하지 않는다(T2가 해제).
- **CLI.** `train`, `index`: `export` 또는 `--dataset` 중 정확히 하나(둘 다/둘 다 없음은 오류). `load_dataset`은 `expected_views` 인자를 받고, train/index는 자기 views 설정을 넘긴다(검사는 T2). `search`: 기존 `export` 또는 `--query-dataset` 중 정확히 하나. 기존 query export는 바꾸지 않고, 그것을 가리키는 manifest에 query 쪽 `mask_derivation`을 붙여 넘긴다. 후보에서 query derivation을 추측하지 않는다.
- **전달.** ROI의 `mask_derivation`과 (T2 이후) `provenance.patch`를 색인 레코드까지 전달한다. 전달 경로(crop 메타데이터 또는 index에서 직접)는 구현자가 고르되, 색인 파일에 실제로 남아야 한다. search signature 키 검증(`search.py:51-53`)도 함께 맞춘다.
- **호환 행렬(검색).**

  | query | 후보 | 결과 |
  |---|---|---|
  | 없음 | 없음 | 기존 동작 |
  | 있음 | 있음, 같음 | 통과 |
  | 있음 | 있음, 다름(layer 목록·rule_version·processed endpoint) | 제외 `mask_derivation_mismatch` |
  | 한쪽만 있음 | | 제외 `mask_derivation_unknown` |
  | `--ignore-mask-derivation` | | derivation 비교 생략, 결과에 생략 사실 기록 |

  processed endpoint 비교는 electrode_id가 아니라 그 endpoint의 (role, actual_layer)로 한다(제품 간 전극 이름이 다르므로).
- **fingerprint.** dataset 모드에서만 accepted에 `dataset`(정규화한 manifest 내용, export sha256 목록, mask_derivation)을 넣는다. 단일 export 모드는 accepted·색인 출력에 새 필드를 하나도 넣지 않는다.
- **index 기록.** 단일 export: 기존 그대로. dataset: `producer_versions` 대신 `sources[]`(export 경로·sha256·versions).
- **수락.**
  - **기존 경로 동일성:** 변경 전 소스(`f85e8c0`)와 변경 후 소스로 같은 fixture·config·환경에서 단일 export 학습·색인·검색을 돌려, checkpoint fingerprint, 색인 JSON, 검색 JSON이 같음(시간 측정 필드는 비교에서 제외하고 그 사실을 테스트에 적는다). 기존 checkpoint resume이 그대로 됨.
  - CLI 상호 배타 오류, 입력 없음 오류.
  - sha256 불일치, roi_id 중복, 전체 메모리 한도 초과(source 각각은 한도 이하, 합은 초과), 같은 상대 경로·다른 바이트가 섞이지 않음, 같은 실제 파일 별칭은 한 번만 계산.
  - mask_derivation: 누락 case, 대응 없는 항목, 중복 항목, 잘못된 processed_endpoint 오류. 호환 행렬 5가지, endpoint 순서 바뀐 경우, rule_version 다른 경우. **같은 (role, layer) 조합인데 removed_layers가 다른 쌍이 제외되는 음성 oracle.**
  - 실제 `search` CLI로 query dataset을 dataset 색인에 검색.
  - 알 수 없는 role/layer는 checkpoint 어휘를 넓히지 않고 거부.
  - dataset 내용이 바뀌면 resume 거부, 이전 checkpoint는 덮어쓰지 않음.
  - 번들 재생성, 전체 pytest 통과.
- **쓰는 파일.** `src/psdl2/data.py`, `train.py`, `index.py`, `search.py`, `crop.py`(전달 경로로 쓸 때), `checkpoint.py`(필요 시), 새 테스트, `bundles/etx/psdl2_job_baseline_ncc.py`.

### T2 — patch 생성기 `psdl2.patches` (High)

- **입력.** dataset manifest의 source 하나(template export + assets_root), run config(`views` 포함), 모드, 표본 수치, seed.
- **출력 `psdl2_patch_set_v1`.** export를 복제하지 않는다.
  - 헤더: template export 경로·sha256, 생성 인수 전체(seed, N, 간격, 주기, step), views 설정 fingerprint, generator_version, case별 깔때기, case별 상태, `content_sha256`(이 필드를 뺀 나머지 전체의 정규화 JSON hash).
  - patch: `template_roi_id`, `center_xy_um`, `mode`, `roi_id`.
  - `roi_id` = `"patch_" + sha256(canonical JSON[template_roi_id, center_xy_um(정확한 float 표현), mode, generator_version])[:24]`.
- **dataset에 넣을 때(load_dataset).**
  - source의 `patch_sets`가 비어 있지 않으면 **그 source의 template 행은 들어가지 않고 patch만 들어간다**. 비어 있거나 없으면 template 행이 들어간다(T1 동작).
  - patch set 파일 경로는 manifest 위치 기준. patch set의 template sha256 ≠ source `export_sha256`이면 `stale_patch_set`.
  - template_roi_id는 그 export의 ready **manual** 행이어야 한다. 중심은 유한한 수. `roi_id`를 다시 계산해 다르면 `patch_id_mismatch`. `content_sha256`을 다시 계산해 다르면 `patch_set_hash_mismatch`(생성 인수 변조는 이것으로 잡는다. id에는 생성 인수가 들어가지 않는다).
  - patch set의 views fingerprint ≠ 지금 실행(train config 또는 index checkpoint)의 views fingerprint면 `patch_set_view_config_mismatch`.
  - 펼친 행은 template의 검증된 채널 배열을 공유한다(복사·재디코딩 없음). 행은 `center_xy_um`, `roi_id`, `provenance.patch = {set_sha256, template_roi_id, mode, generator_version}`만 다르다. `_ready`의 위치·valid 검사를 다시 거친다.
  - 최종 dataset 전체에서 `roi_id` 유일. 같은 patch set을 두 번 넣거나 겹치면 오류(조용한 중복 제거 없음).
  - patch set 파일 sha256을 dataset fingerprint에 넣는다.
- **위치 조건(생성기 전용, 기존 preflight는 바꾸지 않음).** 좌표는 frame origin·pitch 기준 물리 좌표.
  1. 창 전체: 중심 ± (wide/2 + jitter)가 frame 범위 안. 경계와 같은 값은 허용.
  2. core 전체 valid.
  3. core 안에 involved layout의 marked 셀이 하나 이상(anchor).
  4. Q1: 두 endpoint layout이 각각 near 창 안에 marked 셀 하나 이상(원래 격자 기준, threshold 없음).
  5. 위를 통과한 위치는 기존 `preflight`도 실행 config로 통과해야 한다.
  - 이 조건이 모든 jitter 쌍의 topology 통과를 보장하지는 않는다. 학습의 재시도와 거부 집계는 그대로 둔다.
- **random 모드.**
  - 후보 영역은 유한하다: frame의 원래 격자 셀 중심 전부 중 위치 조건 통과한 것. `--repeat-period-um`이 있으면 frame origin에서 시작하는 첫 주기 칸 안의 후보만.
  - seed로 후보 순서를 섞고 앞에서부터 고른다. 같은 case 안에서 이미 고른 중심과 거리 < `--min-spacing-um`이면 건너뛴다. case당 N개가 차면 멈춘다.
  - 후보를 다 써도 N개가 안 되면 그 case는 `shortage`. 하나라도 shortage면 **종료 코드 2**(오류, 기존 파일 그대로)이고 파일을 쓰지 않는다. 종료 코드 3은 단일 파일 실행기가 "계속"으로 처리하므로(`scripts/build_single_file.py:52,122-135`) 쓰지 않는다. `--allow-shortage`를 주면 shortage를 헤더에 기록하고 쓰며 종료 코드 0. N을 조용히 줄이지 않는다.
- **grid 모드.** 점 = frame origin + (i·step, j·step). 위치 조건을 통과한 점 전부. 반복 주기 제한·최소 간격은 적용하지 않는다. `--coverage-labels`(`roi_correspondence_v1`)를 주면 정답 위치마다 가장 가까운 격자점 거리를 보고에 남긴다(Q3만으로는 가장자리·구멍 근처 정답이 잡히지 않을 수 있음).
- **결정성.** 같은 입력·인수·seed면 같은 바이트.
- **깔때기.** case별: 후보 → 창/frame 탈락 → valid 탈락 → anchor 탈락 → Q1 탈락 → preflight 탈락 → 간격 탈락 → 출력.
- **수락.**
  - Astra 반례(중심 (5,5), wide 20, jitter 1)가 창 조건으로 걸러짐. layout 전부 0인 위치, near에는 두 전극이 있지만 core가 빈 위치가 걸러짐.
  - frame origin이 0이 아님, pitch가 정수가 아님, 경계와 같은 값, valid 구멍, native/area 모드.
  - 기존 단일 export에서 창이 frame 밖으로 나가는 ROI의 학습이 **그대로 통과**함(preflight 미변경 확인).
  - 생성 patch 전부가 실제 학습·색인 config의 preflight를 통과.
  - N 불가능, 전부 invalid/빈 형상, 간격 때문에 부족, 반복 주기 제한 때문에 부족 → 정확한 깔때기 수와 종료 코드 2, 새 파일 없음, 기존 파일 바이트 그대로.
  - 원자적 교체 **전** 중단 시 기존 파일 바이트 그대로(교체 후에는 완전한 새 파일이 정상 결과). 재시도는 같은 바이트, id 중복 없음.
  - 대안이 여러 개인 fixture에서 seed가 다르면 결과가 다름.
  - dataset 쪽: 정확한 admitted id·개수, template 행 제외, stale patch set, patch 필드 변조(`patch_id_mismatch`), 헤더 생성 인수 변조(`patch_set_hash_mismatch`, 그리고 fingerprint 변화로 resume 거부), 중복 patch set, views 불일치, 다른 작업 폴더에서 manifest 로드, 채널 배열 공유(같은 객체) 확인.
  - 번들 재생성, 전체 pytest 통과.
- **views 연결.** T1은 `load_dataset(…, expected_views=None)` 인자를 두고 train/index가 자기 views 설정을 넘기게 연결한다(T1에서는 받기만). T2가 이 인자로 patch set의 views fingerprint를 검사한다. `expected_views=None`이면 patch set이 있는 dataset은 `patch_set_views_unchecked` 오류.
- **쓰는 파일.** 새 `src/psdl2/patches.py`, `src/psdl2/data.py`(펼치기, views 검사), 테스트, 번들.

### T3 — `make_split`과 split 성능 (High)

- **`psdl2.make_split`.** 입력: dataset manifest, run config(views; split bbox 계산과 patch set views 검사에 씀), `lineages.json`, `{lineage_id: train|validation|test}`. 출력: 모든 roi_id의 `split.json`(공통 게시 규칙). 배정 안 된 계열, 계열 없는 제품은 오류. 결과는 기존 `validate_split`을 통과해야 한다.
- **`split.groups` 최적화.** 의미 불변: 계열 union과 증강 union은 전역(제품·frame을 넘어서도), 겹침은 product/design_version/frame **전체 dict가 같을 때만**, 양의 면적 겹침만, 출력 순서 동일. frame_id만으로 bucket하지 않는다.
- **수락.**
  - 기존 구현(테스트 안에 복사본 보관)과 새 구현의 차분 테스트. 무작위 입력에 더해 적대적 입력 포함: 대표 bbox가 아닌 것끼리 겹치는 사슬, 제품을 넘는 증강 연결, 같은 계열 다른 제품, 같은 frame_id·다른 origin/pitch, 맞닿기만 한 bbox.
  - 같은 template의 멀리 떨어진 patch 둘을 train/test로 나누면 누수 오류.
  - 성능: 2만 patch(3제품, 2계열, 제품당 frame 1개), 개발 Mac에서 10초 이내. 실측 시간과 비교 횟수를 보고.
  - 번들 재생성, 전체 pytest 통과.
- **쓰는 파일.** `src/psdl2/split.py`, 새 `src/psdl2/make_split.py`, 테스트, 번들.

### T4 — 문서와 끝에서 끝까지 확인 (Normal)

- `docs/USAGE.md`: §2 절차. `docs/data_contract.md`: 처리 마스크 의미(Q8), 새 스키마 3개(dataset, patch set, mask_derivation), 메모리 한도가 디코딩한 mask만 센다는 점. `docs/training.md`: 생성기 위치 조건과 preflight 관계.
- e2e 테스트: 합성 3제품·**3계열**(train/validation/test 모두 비어 있지 않게) → random/grid patch → dataset → make_split → train(옵티마이저 step ≥ 1) → index(grid) → search(query dataset) → eval_position. 단언: 알려진 patch 수, held-out 계열 query, 같은 derivation의 알려진 hit, derivation 불일치 제외, `n_evaluable > 0`, 예상 거리·hit. `baseline_ncc`는 기존 export 인터페이스로 같은 cohort에 실행하는 방법만 문서화(dataset 지원은 범위 밖).
- 의미 수정이 필요하면 해당 High Task로 돌려보낸다.
- **쓰는 파일.** `docs/*`, `tests/test_e2e_dataset.py`, 필요 시 번들.

## 4. 순서, 소유, 리뷰

`T0 → (T0b) → T1 → T2 → T3 → T4`. 편집자는 한 번에 한 명. T0b는 sim-compare 저장소만 편집한다.

| Task | 등급 | 구현 | 독립 리뷰 |
|---|---|---|---|
| T0 | 읽기 전용 | Sonnet medium | 없음 (coordinator 확인) |
| T0b | High | Sonnet medium | Opus high |
| T1 | High | Sonnet medium | Opus high |
| T2 | High | Sonnet medium | Opus high |
| T3 | High | Sonnet medium | Opus high |
| T4 | Normal | Sonnet medium | Sonnet high |

commit은 사용자 지시가 있을 때만 한다. 그 전까지 각 Task 수락 시 coordinator가 `git diff` fingerprint와 검증 결과를 checkpoint에 남긴다.

## 5. 위험과 확인되지 않은 것

- 색인 크기(추정): 5 µm 간격, 100×100 µm, case 20개면 제품당 약 8천 행. T4에서 측정하고 크면 npz 저장을 별도 Task로.
- 메모리: 한도는 디코딩한 mask만 센다. 계열 10개 이상 실제 적재는 미확인.
- "형상 기준 patch 학습이 취약 위치 query에 잘 맞는다"는 가정. `baseline_ncc`와 비교해 판단.
- validation 구획은 학습 중 쓰이지 않는다(범위 밖).
- mask_derivation 출처가 sim-compare에 없으면(T0 3번) 사용자 결정이 다시 필요하다.

## 6. Checkpoint (coordinator 기록)

| 시각(UTC) | 내용 |
|---|---|
| 2026-10-21 13:11 | Run `run_c2a33ea63250` 생성 |
| 13:20 | Astra xhigh 계획 검토(`task_78578cc45901`/`ctx_42905d86bc80`, Codex 명시 터미널 경로, 화면에서 `GPT-6-Astra xhigh` 확인): ACCEPT-WITH-CHANGES, B1–B6 |
| 13:28 | Astra 재검토(`task_7bc957e4b0da`/`ctx_6a3019e0e23c`): R1·R2 부분 해결 → v2.1 반영, coordinator 수락. Fable 초안은 쓰지 않음(사용자가 Astra 검토 경로 지정). Astra 터미널 종료 |
| 13:30 | T0 dispatch: Sonnet 5.5 / medium(launch.effective 일치) |
| 13:34 | T0 완료(`ctx_9b680ee1cd45`, released). 사용자 결정 D6(사이 layer 전부)·D7(T0b 진행)·D8(CNT 합집합) → v2.2 |
| 13:40 | sim-compare 브랜치 `feat/template-export` 생성. T0b dispatch(Sonnet medium, effective 일치). 기존 `test_retired_paths.py` APPROVED 집합에 새 명령 1개 추가를 coordinator가 승인 |
| 13:56 | T0b 완료(`ctx_9d7a125434c6`): 골든 동일, 보호 파일 0줄 변경, 전체 테스트 1082 통과 / 1 실패(`test_handoff` — DL2 작업 폴더의 미추적 `docs/plan/` 때문, 환경 요인) |
| 14:0x | T0b 독립 리뷰 Opus high(`ctx_ee96868dfaf1`, effective 일치): PASS-WITH-NITS, BLOCKING 0. 비차단 1·2·3·4·5(b) 수정 Task(T0b-fix) dispatch. DL2 브랜치 `feat/dl2-patch-dataset` 생성, T1 dispatch(Sonnet medium) |
| 14:1x | T0b-fix 완료(`ctx_8865dc77566c`), delta 재리뷰 Opus high(`ctx_86dc90d103c7`): PASS-WITH-NITS(문서만: endpoint 정렬 순서·PSDL2_ROOT 미기재) → **T0b 수락**. 문서 nit은 T4로 넘김 |
| 14:1x | T1 완료(`ctx_89fca10b9c92`): 기존 경로 골든 동일, DL2 전체 테스트 exit 0(270 통과/2 skip, coordinator 재실행). T1 독립 리뷰 Opus high dispatch |
| 14:2x | T1 독립 리뷰 Opus high(`ctx_32588ba82938`): PASS-WITH-NITS, BLOCKING 0. 결정: N1은 DL2가 막은 행도 case 참조로 센다. N6·N7은 수용 위험 |
| 14:3x | T1-fix 완료(`ctx_162f60d9672e`): 전체 테스트 exit 0(coordinator 재실행). 후보 diff sha256 앞자리 `d4c1f7b6693cfa96`, 스냅샷 `/tmp/T1_final_snapshot`. T1-fix delta는 T2 리뷰에서 함께 본다 |
| 14:5x | T2 완료(`ctx_4a3e9369e5d2`): T1 테스트 1개(`patch_sets_not_supported`) 수정은 coordinator 승인(T1→T2 이음새). 전체 테스트 exit 0, 315 통과(coordinator 재실행). T2 + T1-fix delta 독립 리뷰 Opus high dispatch |
| 15:0x | T2 + T1-fix 리뷰 Opus high(`ctx_dfc065453027`): PASS-WITH-NITS, BLOCKING 0. T1-fix N1–N5 확인 → **T1 수락**. T2 결정: N1(patch 0개면 `--allow-shortage`여도 exit 2) 수용, N8은 T4 문서. 나머지 T2-fix로. 리뷰 시점 스냅샷 `/tmp/T2_reviewed_snapshot` |
| 15:1x | T2-fix 완료(`ctx_48bd5fdff020`): 전체 테스트 exit 0, 325 통과(coordinator 재실행). 스냅샷 `/tmp/T2_final_snapshot`. T2-fix delta는 T3 리뷰에서 함께 본다. T3 dispatch |
| 15:2x | T3 완료(`ctx_284033720543`): 2만 patch 0.35–0.46 s, 전체 테스트 exit 0, 647 통과(coordinator 재실행, 36 s). 스냅샷 `/tmp/T3_final_snapshot`. T3 + T2-fix delta 독립 리뷰 Opus high dispatch |
| 15:3x | T3 + T2-fix 리뷰 Opus high(`ctx_40d3958fbe95`): PASS-WITH-NITS, BLOCKING 0 → **T2·T3 수락**. A1(잘못된 레코드 오류)·A2(lineage마다 다른 경우 O(n²), 실제 경로로는 도달 불가)·A4(umask 순간 변경)는 수용. A3 테스트 공백과 문서 nit은 T4로. T4 dispatch |
| 15:5x | T4 완료(`ctx_921f3659e3cb`), 리뷰 Sonnet high(`ctx_3cdad9ddca26`): PASS-WITH-NITS → **T4 수락**. 문서 인용 nit 3건은 coordinator가 직접 수정. 남은 nit: e2e의 hit는 순위가 아닌 포함 여부만 확인(검색 품질 근거 아님), 문서의 줄 번호는 commit 전 재확인 |
| 완료 | 모든 Task 수락. DL2 전체 테스트 exit 0, sim-compare template 테스트 통과. **commit 안 함** (사용자 지시 대기) |
