# 무작위 patch 학습 · 다중 제품 입력 계획

버전: **PLAN_v1.2 — PLAN_v2_patches_multi_export.md로 대체됨 (기록용)**

이 문서는 2026-10-21 대화에서 사용자가 정한 방향을 구현 Task로 바꾼 계획이다. 구현, 학습, commit은 하지 않는다. 아래 명령·스키마 이름 중 현재 없는 것은 제안이다.

## 0. 결정 (2026-10-21, 사용자)

| 번호 | 결정 |
|---|---|
| D1 | DL2를 **학습 / 검색 / 평가**로 나눈다. 학습은 비교 모델(hotspot, CA)에 묶지 않는다. 비교 모델이 바뀌면 취약 위치도 바뀌기 때문이다 |
| D2 | 학습 위치는 DL2 안의 **manual ROI 생성기**가 형상 기준으로 만든다 |
| D3 | 여러 제품을 한 학습·한 색인에 넣는 방법을 마련한다 |
| D4 | layout 입력은 DL1과 같은 **case(전극 조합쌍)별 처리 마스크**(1.png/2.png)다. vertical은 위쪽 전극만 중간층이 차감되고, lateral은 차감이 없다. 같은 전극이라도 조합쌍마다 마스크가 다를 수 있으며, ROI가 case 단위이므로 그대로 쓴다 |
| D5 | CNT는 제품 전체 1장을 모든 case가 공유한다. **VIA는 쓰지 않는다**(빈 목록). VIA로 이어진 layer 패턴은 이미 같은 전극으로 묶여 있기 때문이다 |

## 1. 현재 상태 (근거)

- 학습·색인은 export 파일 하나만 받는다: `src/psdl2/train.py:167`, `src/psdl2/index.py:99`.
- 다만 출처 버전은 이미 ROI마다 따로 붙어 있다. `ROI.versions`(`src/psdl2/data.py:22-27`)와 학습 fingerprint의 `source_versions`(`src/psdl2/train.py:91`)가 그렇다. 그래서 export 여러 개를 합쳐도 출처 추적이 깨지지 않는다.
- 색인은 `producer_versions`를 export 하나 기준으로 기록한다(`src/psdl2/index.py:111`). 다중 입력이 되면 이것을 목록으로 바꿔야 한다.
- ROI 행은 case 문맥 전체를 요구한다: endpoints 2개, failure_type, analysis_layer, layout 2개, CNT, VIA, valid (`src/psdl2/data.py:131-206`). `source_kind`는 `manual`과 `electrode_count_region` 두 가지만 허용한다(`:155`).
- 학습 전 preflight는 창과 흔들림이 frame을 벗어나면 **학습 전체를 오류로 멈춘다**(`src/psdl2/views.py:103-125`, `src/psdl2/train.py:83-84`). core 안에 layout anchor가 없는 위치는 `no_core_layout_anchor`로 거부되고 재시도를 소모한다(`src/psdl2/views.py:16-20`). 따라서 생성기는 이 조건을 미리 만족하는 중심만 내야 한다.
- 분할 그룹 계산은 ROI 쌍을 모두 비교하는 O(n²)이다(`src/psdl2/split.py:46-49`). patch가 수천~수만 개가 되면 느려진다.
- `split.json`은 모든 ROI id를 명시해야 한다(`src/psdl2/split.py:55-57`). patch가 많으면 사람이 손으로 쓸 수 없다.
- validation 구획은 학습 중에 쓰이지 않는다. `train.py`는 train만 사용한다(`:81`).
- 손실은 VICReg 계열이고, 다른 위치를 밀어내는 negative 항이 없다(`docs/training.md:140-157`). 반복 단위 때문에 비슷한 patch가 많아도 false negative 문제는 제한적이다.
- 제품 frame은 2×2 단위가 반복된다(`docs/validation.md:226`).

## 2. 단계별 준비물 · 입력 · 결과 (D1)

공통 준비물 (제품마다 한 번):

- sim-compare로 import된 제품 폴더: case별 처리 마스크 2장, CNT 1장, valid. hotspot/CA는 필요 없다.
- **template export**: case마다 manual ROI 1개(위 파일 참조, 복사 없음). 확보 경로는 T0.
- `lineages.json`: product_id → 계열. 같은 계열은 같은 구획에 들어간다.
- `mask_derivation` 표: case마다 위쪽 전극에서 뺀 layer 목록 (lateral은 빈 값). Q7.

| 단계 | 준비할 것 | 넣는 것 | 얻는 것 |
|---|---|---|---|
| 학습 | 공통 준비물 + 학습 계열 구획 배정 + 실행 설정(창 크기, epoch 등) | 생성기 random patch(T2) → dataset(T1) → `make_split`(T3) → `train` | checkpoint(가중치 + 데이터·설정 fingerprint), 학습 기록 |
| 검색 색인 | checkpoint + 검색 대상 제품의 공통 준비물 | 생성기 grid patch → dataset → `index` | 색인 파일: patch마다 embedding, 위치, case 문맥, 출처 버전 |
| 검색 | 색인 + query 위치 (sim-compare `candidates`의 `*_export` 또는 사용자 manual 좌표) + cohort 제품 목록 | `search` | query마다 순위 목록: 같은 failure_type, (role, layer) 조합, 차감 방식인 후보와 점수 |
| 평가 | test 계열 query + 정답 라벨 (`roi_correspondence_v1` / `roi_relations_v1`, 사람이 지정하거나 hotspot 위치에서 만든 것) | `eval_position`, 비교용 `baseline_ncc` | 허용 거리 안 hit 비율 같은 지표, 비DL 기준선 대비 차이 |

hotspot/CA는 **어느 단계에서도 모델 입력이 아니다**. query 위치를 고르거나 평가 정답을 만들 때만 쓴다. 평가 계열은 학습 계열과 겹치면 안 된다.

색인도 patch가 필요하다. 지금처럼 case마다 ROI 하나만 있으면, 검색이 돌려줄 수 있는 후보 중심이 case당 한 점뿐이다. `eval_position`의 hit 판정은 "후보 중심이 정답 중심에서 허용 거리 안"이므로(`docs/validation.md:164-166`), 후보 간격이 허용 거리보다 넓으면 원리적으로 hit가 불가능하다.

## 3. Task

위험 등급: T1–T3는 **High**다. 입력 스키마, 데이터 계약 검증, 분할 누수 판정을 바꾸기 때문이다. T4는 Normal이다.

### T0 — template export 확보 경로 확인 (읽기 전용, sim-compare)

- **문제.** 생성기에는 제품마다 "case마다 manual ROI 1개, 모든 전극 layout 참조" export가 필요하다. 지금 그것을 만드는 곳은 `candidates`의 `*_index_export`뿐이다(`particle-sim-compare/docs/usage.md:160`). 그런데 `candidates`는 reference/target 두 제품과 electrode-map 영역, 즉 hotspot을 전제로 한다. D1과 맞지 않는다.
- **확인할 것.**
  1. `export-review-rois`를 한 제품에 대해 manual 행만으로, hotspot 없이 실행할 수 있는가.
  2. real 모드의 `map.input` 요구 사항 중 hotspot에만 필요한 것은 무엇인가.
- **결과.** 다음 둘 중 하나를 정한다.
  - (a) 기존 명령으로 가능 → 설정 예시만 문서화한다.
  - (b) 불가능 → sim-compare에 단일 제품 `export-index-rois` 같은 작은 명령을 **추가**한다 (Q6 허용). 기존 `export-review-rois`·`candidates`의 동작, 출력, 테스트는 바꾸지 않는다. 이 경우 T0b(sim-compare 편집, High)를 T1 앞에 둔다.
  3. template export의 layout 참조가 case별 처리 마스크(1.png/2.png)를 가리키는가, CNT가 제품 전체 1장을 가리키는가, VIA 목록이 비는가 (D4, D5).
  4. case마다 위쪽 전극에서 뺀 layer 목록을 sim-compare에서 얻을 수 있는가 (Q7).
- **수락.** 실제 명령 또는 코드 근거(`파일:줄`)로 (a)/(b) 중 하나를 결론낸다.

### T1 — 다중 export 데이터셋 (D3)

- **변경.** 새 파일 `psdl2_dataset_v1`(JSON)을 둔다. 이 파일이 입력 목록이 된다.

  ```json
  {"schema_version": "psdl2_dataset_v1",
   "sources": [
     {"export": "prodA/review_rois.json", "export_sha256": "…", "assets_root": "prodA/assets",
      "patch_sets": ["prodA/patches_train.json"]},
     {"export": "prodB/review_rois.json", "export_sha256": "…", "assets_root": "prodB/assets",
      "patch_sets": []}]}
  ```

- **새 함수 `psdl2.data.load_dataset`.**
  - 각 export를 기존 `load_export`로 읽는다. 검증 규칙은 바꾸지 않는다.
  - export 파일 sha256이 manifest와 다르면 오류다.
  - 디코딩한 mask cache는 export 전체에서 공유한다. 그래서 `--max-asset-memory-mb`가 전체 합에 걸린다.
  - `roi_id`는 전체에서 유일해야 한다. 아니면 `duplicate_roi_id_across_sources` 오류다.
  - 결과는 `Export`와 같은 모양에 출처별 `versions` 목록을 더한 것이다.
- **CLI.** `train`과 `index`가 `export`와 `--dataset` 중 하나를 받는다. 둘을 함께 주면 오류다. export 하나를 주는 기존 동작은 바이트 단위로 바뀌지 않는다.
- **index 기록.** `producer_versions`는 단일 export일 때 기존 그대로 둔다. `--dataset`일 때는 `sources[]` 목록으로 기록한다. `search`가 이 필드를 읽는지 확인하고 함께 맞춘다.
- **checkpoint fingerprint.** dataset manifest의 내용을 포함한다. 그래서 출처가 하나 바뀌면 resume이 거부된다.
- **수락.**
  - 합성 제품 3개(계열 2개)를 dataset으로 학습하면 단일 export 경로와 같은 검증을 통과한다.
  - 다음 경우는 오류다: sha256 불일치, `roi_id` 중복, 전체 메모리 한도 초과.
  - 기존 테스트 전부 통과.

- **추가 (Q7).** source마다 선택 필드 `mask_derivation: {case_id: {removed_layers: [...], rule_version}}`를 둔다. 학습 fingerprint와 색인 행에 기록하고, `search`의 호환 조건에 기본으로 넣는다(`--ignore-mask-derivation`으로 해제). 필드가 없는 source끼리는 기존 동작과 같다.

### T2 — patch 생성기 `psdl2.patches` (D2)

- **입력.** template export 하나(+ `assets_root`), run config의 `views`, 표본 규칙, seed.
- **출력.** `psdl2_patch_set_v1` 파일 하나. 이 파일은 producer export를 **복제하거나 위조하지 않는다**.
  - template export의 경로·sha256과 생성 인수를 기록한다.
  - patch마다 `template_roi_id`, `center_xy_um`, 결정적 `roi_id`(template id·중심·생성 버전의 hash)를 둔다.
  - 원본 export는 그대로 둔다(`docs/data_contract.md`의 "원본 export는 별도로 유지" 원칙).
- **불러올 때 펼치기.** `load_dataset`가 patch를 펼친다.
  - template ROI의 이미 검증된 채널 배열을 공유한다. 복사하지 않는다.
  - 행은 `center_xy_um`, `roi_id`, `provenance.patch`만 바꾼다.
  - `source_kind`는 `manual`을 유지한다. 대신 `provenance.patch = {set_sha256, template_roi_id, generator_version}`를 추가한다. 새 `source_kind` 값을 허용 목록에 넣을지는 §6 Q5에서 정한다.
  - 펼친 행도 `_ready`와 같은 위치·valid 검사를 다시 거친다.
- **표본 규칙.** 모든 수치는 명시값이고 기본값은 없다.
  - 모드: `random`(학습용) 또는 `grid`(색인용, 간격 `--grid-step-um`).
  - **frame 안**: 중심 ± (wide/2 + jitter)가 frame 안에 있고, core 전체가 valid여야 한다. preflight 조건(`src/psdl2/views.py:103-125`)을 같은 코드로 미리 검사한다.
  - **형상 있음**: core 안에 layout anchor가 있어야 한다. 추가로 §6 Q1에서 정할 규칙을 적용한다. 예: 두 전극이 모두 near 창 안에 있을 것, 최소 marked 비율 x.
  - **중복 억제**: patch 중심 간 최소 거리 `--min-spacing-um`. 선택적으로 `--repeat-period-um`을 주면 반복 단위 하나 안에서만 뽑는다.
  - **case 고르기**: template의 case마다 같은 수 `--per-case N`(random). grid 모드는 case마다 전 격자.
  - **결정성**: 같은 입력, 인수, seed면 같은 바이트를 낸다.
- **깔때기 기록.** 후보 수 → frame/valid 탈락 → 형상 탈락 → 간격 탈락 → 출력을 case별로 남긴다. 실패한 case는 이유와 함께 남기고 조용히 버리지 않는다.
- **수락.**
  - 손으로 만든 합성 frame에서 경계 바로 안쪽과 바깥쪽, valid 구멍, 빈 영역이 정확히 걸러진다 (oracle 테스트).
  - 생성된 patch 전부가 `preflight`를 통과한다.
  - 같은 seed면 같은 바이트, 다른 seed면 다른 결과가 나온다.

### T3 — 분할 도구와 분할 성능

- **`psdl2.make_split`.** `lineages.json`과 `{lineage_id: train|validation|test}`를 받는다. dataset 안의 모든 `roi_id`에 대해 `split.json`을 만든다. 계열이 배정되지 않았으면 오류다.
- **`split.groups` 최적화.** 의미는 바꾸지 않는다.
  - 같은 계열과 같은 증강은 조건 없이 한 그룹이다.
  - 위치 겹침은 같은 product/design/frame 안에서만 판정한다.
  - 그래서 계열 단위로 먼저 묶고, 겹침 검사는 같은 frame bucket 안에서만 한다. 같은 제품은 같은 계열이므로 실제로는 거의 생략된다.
- **수락.**
  - 기존 `groups` 구현과 새 구현의 결과가 무작위 합성 기록 다수에서 완전히 같다 (차분 테스트).
  - patch 2만 개에서 수 초 안에 끝난다.
  - 누수 오류 사례가 기존과 같다.

### T4 — 문서와 끝에서 끝까지 확인 (Normal)

- `docs/USAGE.md`에 학습/검색/평가 절차를 §2 표 기준으로 다시 쓴다.
- 합성 제품 3개로 다음 흐름을 한 번 통과시키는 테스트를 추가한다: template export → patch(random, grid) → dataset → make_split → train → index(grid) → search → eval_position.
- `docs/training.md`와 `docs/data_contract.md`에 새 스키마 두 개를 기록한다.

## 4. 순서와 소유

T0(읽기 전용) → T1 → T2 → T3 → T4. 순서대로 진행하고 편집자는 한 명이다.

| Task | 쓰는 파일 |
|---|---|
| T1 | `src/psdl2/data.py`, `train.py`, `index.py`, (필요 시) `search.py`, `checkpoint.py`, 테스트 |
| T2 | 새 `src/psdl2/patches.py`, `data.py`의 펼치기 부분, 테스트 |
| T3 | `src/psdl2/split.py`, 새 `make_split` 진입점, 테스트 |
| T4 | `docs/*`, e2e 테스트 |

T2는 T1의 `load_dataset`에 펼치기를 붙이므로 T1 다음이어야 한다.

## 5. 위험과 확인되지 않은 것

- **색인 크기 (추정).** grid 색인은 `index` JSON에 patch마다 embedding을 저장한다. 예를 들어 5 µm 간격, frame 100×100 µm, case 20개면 제품당 약 8천 행이다. JSON 크기와 검색 시간을 T4에서 측정한다. 너무 크면 npz 저장을 별도 Task로 올린다.
- **메모리.** 제품이 늘면 원본 mask 합이 16 GB RAM 기준 PC를 넘을 수 있다. T1의 전체 메모리 한도로 실행 전에 막지만, 계열 10개 이상을 실제로 올릴 수 있는지는 확인되지 않았다.
- **검증되지 않은 가정 (추정).** "형상 기준 무작위 patch로 학습해도 취약 위치 query에 잘 맞는다"는 것은 가정이다. 평가에서 비DL 기준선(`psdl2.baseline_ncc`)과 비교해 판단한다.
- **validation 구획.** 학습 중 쓰이지 않는다. 조기 종료나 설정 선택에 쓰려면 별도 Task가 필요하다 (이 계획의 범위 밖).

## 6. 사용자 결정 (2026-10-21 확정)

Q1–Q5, Q7, Q8은 아래 제안대로 확정했다. Q6은 허용하되, sim-compare의 기존 export 생성 로직(`export-review-rois`, `candidates`)은 동작과 출력이 바뀌지 않아야 한다. 새 경로는 추가로만 넣는다. Q2·Q3의 숫자는 실제 frame 크기와 hit 허용 거리를 보고 실행 시점에 정한다.

| 번호 | 질문 | 제안 |
|---|---|---|
| Q1 | patch의 "형상 있음" 기준 | 두 endpoint 전극의 layout이 모두 near 창 안에 marked 셀을 가질 것. short는 두 전극 사이에서 일어나기 때문이다 |
| Q2 | 학습 patch 수 (case당 N), 최소 간격 | 첫 실험은 N을 작게 두고 깔때기를 본 뒤 늘린다. 값은 실제 frame 크기를 보고 정한다 |
| Q3 | 색인 격자 간격 | `eval_position`의 `--hit-tolerance-um` 이하 |
| Q4 | 반복 단위 처리 | 반복 주기를 알면 `--repeat-period-um`을 쓴다. 모르면 최소 간격만 쓴다 |
| Q5 | patch 행의 `source_kind` | `manual` 유지 + `provenance.patch` (스키마 변경 최소). 또는 새 값 `generated_patch` 추가 (구분은 명확하지만 sim-compare 계약 변경이 필요할 수 있음) |
| Q6 | T0 결과 (b)일 때 sim-compare 변경을 승인하는가 | **허용.** 기존 생성 로직과 출력은 그대로 두고 새 명령만 추가한다. 기존 테스트가 그대로 통과해야 한다 |
| Q7 | `mask_derivation`을 어디에 두고 검색 호환 조건에 넣는가 | dataset 파일(T1)에 case별 표로 둔다 (DL2만 바꾸면 됨). 검색 기본은 차감 방식이 같은 후보만, 옵션으로 해제. 같은 (role, layer) 조합이라도 제품마다 중간층이 다르면 마스크 의미가 다르기 때문이다. 검색 호환 조건 변경이므로 T1에 포함하고 High |
| Q8 | 처리 마스크를 계약에서 어떻게 부르는가 | 지금 계약은 layout을 "원본"으로 부른다(`docs/data_contract.md:1`, `:134-137`). 처리 마스크임을 provenance에 명시하도록 T4에서 문서를 고친다 |
