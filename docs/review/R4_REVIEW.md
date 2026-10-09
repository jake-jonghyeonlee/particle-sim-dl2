# R4 독립 코드 검토 — DL2 (T10 + T11)

- 검토자: fresh Claude Opus 5.5 세션 (작성자·coordinator 아님), 읽기 전용
- 대상: `/Users/jonghyeonlee/git/particle-sim-dl2`, branch `main`, commit `c19a33d` (T10 = `836ee0f`, baseline `886aa74`)
- 계약: PLAN_v2 §2.4, §2.5, T10, T11, §8 R4 행 / `review_roi_contract.md` / `implementation_memory.json`의 `gate_answers_2026_10_09_dl2_and_hotspot_format`
- 검토일: 2026-10-09

## 판정: **FAIL** (commit `c19a33d`에만 유효)

차단 사항 2건(B1, B2)이 있다. 둘 다 실제로 재현했다. 그 외 중요 영역(importer, loss 산술, 검색 scope, split, checkpoint/resume, 네트워크·금지 입력)은 확인한 범위에서 계약과 일치한다. 수정 후에는 B1/B2에 영향받는 crop·view 부분만 재검토하면 된다.

---

## 차단 사항

### B1 — native pitch nearest crop이 half-cell 중심에서 원본 열/행을 건너뛰거나 중복한다

- 위치: `src/psdl2/crop.py:45-48` (`sx`, `sy` 계산)
- 원인: 출력 cell 중심 `origin + (i+.5)*pitch`를 부동소수로 계산한 뒤 `floor`한다. ROI 중심이 원본 cell의 **중심**(half-cell)에 있고 창 크기가 짝수 cell이면 출력 cell 중심이 원본 cell 경계에 정확히 놓이고, `floor` 결과가 반올림 오차에 따라 `k`와 `k+1` 사이에서 불규칙하게 갈린다. pitch 0.5처럼 이진 정확값일 때는 드러나지 않고, 사용자가 답한 실제 규모(약 0.05 um/px)에서 드러난다.
- 재현 입력: frame origin `(100, 200)`, pitch `0.05`, shape `400×400`, `layout:0`에 4 px 주기 1 px 세로선(`[:, ::4] = 1`), 중심 `(103.075, 210.025)`, `CropConfig((2,2),(4,4), preserve_exported_layers_v1)` (native pitch, nearest).
- 관측: near 첫 행 `0010001000100000001100100010001000000011`, 표시 열 간격 `[4, 4, 8, 1, 3, 4, 4, 8, 1]` — 선 하나가 사라지고(간격 8) 다른 선은 2 px 폭으로 복제(간격 1)된다.
- 기대: 모든 간격이 4 (전체가 일관되게 한 cell 쪽으로만 선택).
- 범위: origin 3종 × 중심 280개 조사에서 half-cell 중심 840건 중 **564건**이 비단위 step을 보였고, cell-edge 정렬 중심 840건은 0건.
- 영향: T10의 핵심인 물리 crop 정확성(half-cell·반올림). 지도 영역 ROI의 중심은 bbox 중앙이므로 홀수 cell 폭 영역이면 half-cell 중심이 되고, 수동 중심은 임의다. index/검색 경로(`view_config.bundle`)에는 이를 걸러낼 검사가 없어 선이 빠지거나 붙은 patch가 그대로 embedding된다. 메타데이터는 `physical_cell_centres_nearest_v1`로만 기록되어 구별되지 않는다.
- 테스트가 놓친 이유: 모든 crop 시험이 pitch 0.5 / 0.25 / 1.0(이진 정확값)만 쓴다.

### B2 — view 쌍 수락 규칙 때문에 창보다 긴 전극이 있는 layout은 전부 거부되고, 반 cell 이상의 jitter도 대부분 거부된다

- 위치: `src/psdl2/views.py:132-134` (same-scale `represented` 집합 동일성), `src/psdl2/views.py:139-146` (core에 닿는 component 전체가 네 patch 안에 완전히 들어와야 함), 결과 처리 `src/psdl2/train.py:42-43`
- 재현 (a): frame `200×200`, pitch 0.5, origin `(100,200)`; `layout:0`은 20 px 주기 2 px 가로선(frame 전체 폭), `layout:1`은 같은 형태 세로선; near 20 um / wide 40 um / core 10 um; 중심 9곳 × 20회.
  - jitter 0, 0.1: **180/180 `core_component_clipped`**
  - jitter 1.0: 153 `involved_bridge_or_component_lost_between_views` + 27 `core_component_clipped`, 수락 0
  - 같은 layout으로 `paired_forward` 호출 → `effective_batch_at_least_two_required_after_view_rejection` (학습 CLI는 exit 2)
- 재현 (b): 창을 넘는 component가 없는 layout(2% 고립 cell), near/wide 모두 native nearest.
  - jitter 0, 0.1: 180/180 수락
  - jitter 0.3: 84/180 거부, jitter 1.0: **173/180 거부** (`involved_bridge_or_component_lost_between_views`)
  - 원인: 두 view의 창 경계가 원본 cell 중심을 서로 다르게 넘으면 가장자리 component의 `represented` 집합이 달라진다. 즉 jitter가 native half-cell(여기서는 0.25 um)을 넘으면 창 가장자리에 무엇이든 있을 때 거부된다.
- 기대: 같은 위치의 두 view에서 **연결이 사라진** 쌍만 거부하고, 창 밖으로 이어지는 전극이 양쪽 view에 모두 보이는 정상 쌍은 positive로 수락.
- 영향: 실제 패널 layout의 gate/data 선은 어떤 near 창보다 길다. 이 규칙은 설정으로 바꿀 수 없으므로(`topology_policy` 고정) 폐쇄망에서 무라벨 기본 학습이 첫 batch에서 실패한다. 조용한 실패는 아니고 명시 오류지만, T11의 "labels 없이 기본 학습이 성립" 요구가 합성 fixture에서만 성립한다.
- 문서 상태: `docs/training.md`가 "long wires extending outside near windows를 거부할 수 있다"고 적고 있으나, 실제로는 "할 수 있다"가 아니라 항상 거부이며 jitter 제약은 문서에 없다.
- 테스트가 놓친 이유: `tests/test_views.py:13`의 jitter 0.1은 native half-cell 0.25보다 작게 맞춰져 있다. 제공된 합성 설정으로 200쌍을 생성하면 near 두 view가 **200/200 bit 동일**했다(차이는 wide area-coverage에서만 생김). `test_component_clipped_and_different_location_rejected`(`tests/test_views.py:72-78`)는 frame을 가로지르는 선의 거부를 기대 동작으로 고정한다.

---

## 비차단 메모

- **N1 — producer_commit null 거부** (`src/psdl2/data.py:184-186`): producer 계약은 git checkout 밖에서 `producer_commit=null` + `producer_commit_reason`을 허용한다. 그 export는 DL2에서 envelope 전체가 `producer_commit_required`로 거부된다(재현). 폐쇄망에 사본으로 설치한 simcompare면 해당된다. fail-closed 자체는 타당하나 T12 전에 운영 절차(명시 `producer_commit` 전달)를 문서화해야 한다.
- **N2 — producer version 검증은 기본적으로 형식 검사뿐**: `--expected-producer-commit`을 주지 않으면 40자리 hex면 통과한다.
- **N3 — `negative_pair_count`는 상수** (`src/psdl2/losses.py:105`): `tests/test_losses.py:24,46`, `tests/test_end_to_end.py:112`의 단언은 상수를 확인할 뿐이다. 또한 labels로 `different` 쌍이 supervised 항에 들어가도 0으로 보고된다(재현: known 2쌍 중 1쌍이 different인데 0). 실제 무반발 근거는 `alignment_per_location` 시험과 아래 독립 검산이다.
- **N4 — 검색의 case 호환성은 순서 포함 완전 일치** (`src/psdl2/search.py:90`, `src/psdl2/index.py:19-26`): `electrode_id` 문자열과 endpoint 순서까지 같아야 한다. 재현: endpoint 순서만 뒤집힌 후보, electrode 이름만 다른 후보 모두 `case_or_CNT_binding_incompatible`. 제품마다 electrode 이름이 다르면 제품 간 검색 결과가 항상 비게 된다. 실자료 명명 규칙에 달려 있어 확인 불가.
- **N5 — `expand=False`에서도 `rejected`에 cohort 밖 후보가 노출됨** (`src/psdl2/search.py:90-94`): 호환성 검사가 cohort 검사보다 먼저다. primary/supplementary 목록 자체는 분리되어 있다.
- **N6 — 점수 6자리 반올림 후 정렬** (`src/psdl2/search.py:25,106`): 1e-7 수준의 실제 순서는 roi_id 순으로 대체된다(재현). 문서화되어 있고 결정적이다.
- **N7 — 기본값**: 창 크기·jitter·core·중간 layer 정책·view pitch·resampling mode는 설정 경로에서 모두 필수다. 다만 저수준 API에는 기본값이 있다: `CropConfig.output_pitch_um=None`(원본 pitch), `build_bundle(..., near_resampling_mode='nearest', wide_resampling_mode='nearest')`. 원본 pitch는 물리 크기를 바꾸지 않으므로 허용 범위로 보지만 "기본값 없음" 문구와는 엄밀히 다르다.
- **N8 — area-coverage는 component 병합을 검사하지 않음**: 거친 pitch에서 떨어진 두 도형이 한 cell에 합쳐져도 통과한다. 계약은 bridge 소실만 요구하며 문서에 명시되어 있다.
- **N9 — jitter 0 허용**: 두 view가 동일해져 alignment가 항등적으로 0이 된다. 문서에 "명시적 선택"으로 기록됨.
- **N10 — evaluate는 split을 모른다** (`src/psdl2/evaluate.py`): train split query도 Recall@k에 들어갈 수 있고, 분모는 index에 없는 positive도 포함한다. 결과에 `evaluation_of_supplied_labels_only`로 표기된다.
- **N11 — labels의 roi_id 오타는 조용히 무시**: batch에 없는 ID는 건너뛰고 `supervised_known_pair_count`만 줄어든다.
- **N12 — split의 겹침 검사는 `train()` 경로에서 사실상 중복**: lineage가 `product_id` 단위라 같은 제품 ROI는 항상 같은 그룹이다. 누수 방지는 사용자가 주는 lineage 매핑의 정확성에 의존한다.
- **N13 — checkpoint 디렉터리 fsync 없음** (`src/psdl2/checkpoint.py:31-40`): 파일은 fsync 후 `os.replace`하나 상위 디렉터리는 fsync하지 않는다.

---

## 확인한 것과 방법

실행: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:warnings` → **105 passed, 1 skipped**; `--run-training` → **106 passed** (coordinator 증거와 일치). probe는 세션 scratch 디렉터리(`/private/tmp/claude-501/...`)에만 작성했고, 검토 후 `git status`는 이 보고서 외 변경 없음.

1. **물리 crop** — 손계산 oracle: 비정방 frame(6×10), origin `(100,200)`, 비정방 창 `(4,2)/(8,4)`에서 near/wide 채널 위치·valid·padding 일치, near/wide 중심 동일, 축 순서(yx) 정상. area-coverage: 반 cell 어긋난 창에서 `[0, 0, .125, .125]`(합 × cell 면적 = 원본 면적 0.25), frame 가장자리 valid `[1, 1, .5, 0]` 손계산과 일치. 모든 채널이 같은 index 벡터/가중 행렬을 쓴다(코드 확인). 결함은 B1.
2. **importer** — 실제 producer fixture(producer commit `f5e86006…`이 compare repo에 존재함을 확인) 사본에 변형 적용: blocked→ready 변조, identity 불일치, origin 이동, `axis_order=xy`, 대문자 sha, CNT 미관측, `assets/../` 경로, `tensor_input_policy` 누락 모두 승격되지 않음. 극성은 선언값만 사용하고 자동 판별 없음(valid를 `binary_zero_is_marked`로 선언하면 그대로 반전되어 차단). hotspot/CA/fab 키는 행·asset 수준에서 거부. 파일명만 `hotspot.npy`인 layout 참조는 내용으로 구별할 수 없어 통과한다(계약상 producer 책임).
3. **loss** — 무작위 5×3 입력에서 alignment/variance/covariance를 numpy로 독립 계산해 일치, `total`이 세 항의 가중합과 정확히 같음(숨은 항 없음). view 2의 행 대응을 섞으면 variance/covariance는 불변이고 alignment만 변함. labels 없으면 supervised 0, 전부 unknown이면 0·known 0, known 2쌍 BCE 손계산 일치. encoder에 BatchNorm 등 batch 결합 계층 없음(코드 확인).
4. **view** — B2 참조. 위치·중심·binding 위조 거부는 기존 시험과 코드로 확인.
5. **검색** — primary는 cohort 내 호환 후보만, supplementary는 `expand`일 때만, 더 높은 점수의 cohort 밖 후보가 primary를 바꾸지 않음, 입력 순서 무관, no-match는 `unavailable` + 사유이며 위험 값 없음.
6. **split** — lineage/겹침/증강 그룹의 추이적 결합과 누수 거부를 코드와 시험으로 확인.
7. **checkpoint/resume** — 4 epoch 연속 실행과 2 epoch + resume 2 epoch의 **가중치 fingerprint·view RNG·torch RNG가 동일**. fingerprint 키: batch_size, labels, learning_rate, loss, model, schema_version, seed, source_assets, source_versions, split, training_records, views (`epochs` 제외). seed/batch/split/labels/lineage/ROI 집합 변경은 모두 `incompatible_config_fingerprint`. `torch.load(weights_only=True)` 사용.
8. **금지 사항** — `src/`에 네트워크·다운로드·pretrained 코드 없음(grep), 전 시험 socket 차단. 문서는 실성능을 주장하지 않고 `real_performance_pending`으로 표기.

## 확인하지 못한 것

- 실자료(실제 layout, 실제 창 크기·pitch·jitter, 실제 electrode 명명, 실제 이미지 container/bit depth). G4의 창 크기와 중간 layer 정책은 미정이다.
- 실제 학습 수렴·검색 품질·collapse 방지 효과, variance/covariance 규제의 간접 false-negative 영향.
- CUDA 경로(RNG 복원 포함)와 대형 frame(2000 px)에서의 성능·메모리. area-coverage는 출력×원본 크기의 조밀 가중 행렬을 만든다.
- T12의 C→DL2→C 왕복과 최신 producer commit과의 호환(fixture는 `f5e86006…` 고정).
- 전원 차단 수준의 원자성(N13).
