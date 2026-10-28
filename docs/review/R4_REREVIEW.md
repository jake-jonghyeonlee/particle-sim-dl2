# R4b 독립 재검토 — DL2 R4 수정본

- 검토자: fresh Claude Opus 5.5 세션 (작성자·coordinator·1차 검토자 아님), 읽기 전용
- 대상: particle-sim-dl2 저장소, branch `main`, commit `e03cb5c9cce7a36dfb73c0c53b2a9f99eb1448ba` (R4 보고 commit `9849300` 위의 수정 commit, 1차 검토 대상 `c19a33d`)
- 계약: PLAN_v2 §2.4, §2.5, T10, T11 / `docs/review/R4_REVIEW.md` / `docs/review/R4_FIXES.md`
- 검토일: 2026-10-09

## 판정: **FAIL** (commit `e03cb5c`에만 유효)

B1은 해결됐다. B2는 "정상 쌍 거부" 쪽은 해결됐지만, 새 규칙이 **두 관여 전극을 잇는 CNT bridge가 한 view에서만 사라진 쌍을 수락**한다(B3, 재현함). 이 경우는 `c19a33d`에서는 거부되던 것이고, T11 수락 문장("bridge가 사라진 쌍은 거부")과 `docs/training.md`의 서술과 어긋난다.

판정 근거가 좁다는 점을 분명히 적는다. B3은 코드가 문서화된 "채널별" 규칙과 다르게 동작한다는 결함이 아니라, **규칙의 범위가 계약의 '관여 전극·연결'을 덮지 못한다**는 판단이다. coordinator/사용자가 "core 밖 채널 간(CNT–전극) bridge는 정책 범위 밖"이라고 명시적으로 결정한다면 B3은 문서 정정과 그 동작을 고정하는 시험 추가만으로 메모로 내려갈 수 있다. 그 결정 없이 현재 상태를 PASS로 볼 수는 없다고 판단했다.

그 외 재검토 범위(importer, loss, 검색, split, checkpoint/resume, 금지 입력·네트워크, 메모 수정)는 확인한 범위에서 회귀가 없다. 차단은 아니지만 실규모 실행 전에 고쳐야 할 성능 문제(P1)가 있다.

---

## B1 — 해결됨

재현과 확인:

- **원래 실패 입력** (origin `(100,200)`, pitch `0.05`, `400×400`, 4 px 주기 세로선, 중심 `(103.075, 210.025)`, near 2 um / wide 4 um, native nearest)
  - `c19a33d` 코드(추출본): 첫 행 `0010001000100000001100100010001000000011`, 간격 `[4,4,8,1,3,4,4,8,1]` — R4 보고와 동일하게 재현.
  - `e03cb5c`: 간격 전부 4 (x, y, wide 모두), source index x `42..81`, y `181..220`.
- **독립 oracle sweep**: 저장소 코드와 무관하게 `decimal.Decimal`(정밀도 80)로 `floor((centre − size/2 + (i+.5)·pitch − origin)/source_pitch)`을 계산해 비교했다.
  - native nearest 1800건: source pitch `0.05, 0.1, 0.3, 0.07, 0.025, 0.013` × origin 5종(`(0,0)`, `(100,200)`, `(−13.275, 27.125)`, `(−0.35, −0.001)`, `(1234567.85, −98765.45)`) × 중심 4종(float 연산으로 만든 cell-edge, float 연산으로 만든 half-cell, 소수 6자리로 반올림한 half-cell, 임의 다자리 소수) × 창 4/10/11/20 cell. index 벡터·단위 step·patch 내용·valid 모두 oracle과 일치, **불일치 0**.
  - 거친 pitch 600건: (source, output) = `(0.05,0.15) (0.05,0.2) (0.03,0.1) (0.07,0.1) (0.3,0.5) (0.05,0.05) (0.1,0.35) (0.013,0.1)` — 비율이 유한소수가 아닌 조합 포함, frame 밖으로 나가는 창과 valid 구멍 포함. nearest index·내용, area-coverage 값(독립 overlap 행렬과 `2e-7` 이내), `outside_frame_cells`, 행별 가중합 ≤ 1 모두 일치, **불일치 0**.
- **tie 규칙**: 출력 cell 중심이 원본 cell 경계에 정확히 놓이는 입력(pitch 0.05/0.1/0.3)에서 x index `[5,6]` — 문서의 `higher_source_index_half_open`과 코드(`src/psdl2/crop.py:43-46`의 `//1`)가 일치한다.
- **area-coverage 가중**: 출력 cell별 가중합 = frame 안쪽 비율, R4의 손계산 `[0,0,.125,.125]` / valid `[1,1,.5,0]`은 기존 시험이 고정하고 내 독립 overlap 행렬과도 일치.
- **오류 경로**: 창이 output pitch의 정수배가 아니면 오류, area-coverage에 원본보다 고운 pitch를 주면 `area_coverage_requires_coarser_or_equal_pitch`.

성능 (2000×2000 원본, pitch 0.05, 5 채널, M-series CPU, `build_bundle` 1회):

| near / wide | 시간 | tracemalloc peak | 프로세스 maxrss |
|---|---|---|---|
| 400 nearest / 800 nearest | 0.02 s | 4 MiB | 88 MiB |
| 400 nearest / 2000 nearest | 0.07 s | 24 MiB | 113 MiB |
| 400 nearest / 500 area (pitch 0.2) | 0.12 s | 57 MiB | 147 MiB |
| 400 area / 2000 area (native) | 0.44 s | 190 MiB | 316 MiB |
| 다자리 소수 중심(jitter 후) 400 nearest / 500 area | 0.12 s | 57 MiB | — |

crop 자체의 유리수 연산과 조밀 가중 행렬은 이 규모에서 문제없다. 느린 것은 crop이 아니라 쌍 검증이다(P1).

남는 한계(차단 아님): `Fraction(str(value))`는 호출자가 넘긴 float의 최단 표기를 그대로 믿는다. 지도 영역 중심 `(x0+x1)/2`가 float 오차로 half-cell에서 1 ulp 벗어나면 patch 전체가 한 cell 쪽으로 일관되게 선택된다(단위 step은 유지, 확인함). R4_FIXES.md에 같은 취지로 적혀 있다.

## B2 — 정상 쌍 거부는 해결됨, 연결 소실 거부는 불완전 (→ B3)

정상 쌍(전부 `generate_pair` 경로, seed 고정):

| 입력 | 결과 |
|---|---|
| R4 (a) 줄무늬 `200×200`, pitch 0.5, near 20 / wide 40 / core 10, 중심 9곳 × 20회, jitter 0 / 0.1 / 0.3 / 1.0 / 4.9 | 각 180/180 수락 |
| R4 (b) 2% 고립 cell, jitter 0 / 0.1 / 0.3 / 1.0 | 각 180/180 수락 |
| pitch 0.05, `800×800` gate/data 격자 + CNT 3%, half-cell 중심 포함 6곳 × 5회, jitter 0.02 / 0.1 / 0.5 (native) | 각 30/30 수락 |
| 위와 같고 wide만 area-coverage pitch 0.2 | 30/30 수락 |

`c19a33d`에서는 (a)가 전부 `core_component_clipped`였다. 창을 가로지르는 긴 전극과 반 cell을 넘는 jitter는 이제 수락된다.

연결 소실:

| 입력 | 결과 |
|---|---|
| 같은 채널의 U자 우회로가 core 밖에 있고 한 near 창에만 남음 (자연 crop, y jitter +3) | 거부 `core_connection_lost_between_views` |
| 같은 우회로를 거친 area-coverage patch에서 지움 | 거부 `resampled_core_connection_lost` |
| 같은 우회로를 wide native patch에서 지움 | 거부 `core_connection_lost_between_views` |
| core 안의 CNT를 한 crop에서 지움 | 거부 `core_marked_content_changed` |
| 다른 중심에서 자르고 `original_center`만 위조 | 거부 `common_core_not_observed` |
| **두 전극을 잇는 CNT bridge가 core 밖에 있고 한 near 창에만 남음** | **수락** (B3) |

보고와 오류 경로: `paired_forward`가 batch마다 `accepted_pair_count` / `rejected_pair_count` / `rejection_reasons`를 돌려주고, `train()`이 batch·epoch history에 기록한다(`src/psdl2/train.py:43-50, 92-98`). 유효 batch가 2 미만이면 수락·거부 수와 사유를 담은 `effective_batch_at_least_two_required_after_view_rejection` 오류가 나고 CLI는 exit 2다. 조용한 실패는 없다.

---

## 새 차단 사항

### B3 — 두 관여 전극을 잇는 CNT(또는 VIA) bridge가 한 view에서만 사라져도 positive로 수락된다

- 위치: `src/psdl2/views.py:139-153` (채널마다 따로 core 연결 관계를 계산), `src/psdl2/views.py:176` (채널별 관계만 비교). 채널 사이의 연결은 어디서도 보지 않는다.
- 재현 입력 (자연 crop, 변조 없음): frame `40×40`, origin `(100,200)`, pitch 1, 중심 `(120,220)`. `layout:0`은 세로선 `[:, 17]`, `layout:1`은 세로선 `[:, 22]`, `cnt`는 두 선을 잇는 가로 막대 `[14, 17:23]`(y 214–215 um, core y 217–223 밖). `ViewConfig(CropConfig((14,14),(30,30)), 1, 1, nearest, nearest, jitter (0,3), core (6,6), common_core_connectivity_v2)`, 두 view의 offset `(0,0)`과 `(0,+3)`.
  - view 1의 near 창(y 213–227)은 CNT 막대 6 cell을 포함하고, view 2의 near 창(y 216–230)은 포함하지 않는다.
  - 관측: `generate_pair` → `accepted`. `c19a33d`에서는 같은 입력이 `involved_bridge_or_component_lost_between_views`로 거부됐다.
- 재현 입력 (계약 반례 형태): 같은 ROI, jitter 0, 같은 중심에서 두 view를 만든 뒤 두 번째 near patch의 `cnt`를 전부 0으로 지움 → `validate_pair` **수락**. PLAN_v2 T11 반례 "유일한 연결 bridge를 한 crop에서 제거하면 같은 중심이어도 positive가 될 수 없다"의 직접적인 위반이다. bridge가 core 안에 있으면 거부된다(확인).
- 원인: 규칙이 "한 채널 안에서 core cell끼리의 4-이웃 연결"만 비교한다. 두 관여 전극은 서로 다른 채널(`layout:0`, `layout:1`)이고 그 사이를 잇는 것은 `cnt`/`via` 채널이므로, 이 제품에서 short를 만드는 연결 자체가 규칙의 대상이 아니다. layout 채널 하나가 전극 하나라면 채널 내부 연결은 대부분 창 clipping으로만 바뀌는 관계다.
- 기대: 두 관여 전극 사이의 CNT/VIA 매개 연결이 한 view에만 있는 쌍은 거부. R4가 요구한 "창 가장자리에서 드나드는 무관한 고립 cell은 수락"과 양립한다(무관한 cell은 core 전극들 사이의 관계를 바꾸지 않는다).
- 영향: §2.5 "관여 전극·연결을 유지", T11 수락 "common core/bridge가 사라진 쌍은 거부"가 core 밖 CNT bridge에 대해 성립하지 않는다. 모델은 "두 전극이 CNT로 이어진 patch"와 "이어지지 않은 patch"를 같은 것으로 학습하게 된다. `docs/training.md`의 "A bridge outside the core that only one window retains changes the core relation and is rejected"는 같은 채널 안의 bridge에만 참이다.
- 시험이 놓친 이유: bridge 시험이 모두 `layout:0` 한 채널 안에서 만들어진다(`tests/test_r4_regressions.py:124-136, 346-358`, `tests/test_views.py:57-69`). `cnt` 채널을 매개로 한 연결 시험이 없다.
- 설계상 주의: 채널을 단순히 XY로 합쳐 연결을 보면 서로 다른 layer의 교차를 연결로 오인한다. 어떤 채널 조합을 연결로 볼지는 planner가 정해야 하고, CNT 밀도가 높을 때 거부율이 다시 올라가지 않는지 R4 (a)(b) 재현과 함께 확인해야 한다.

---

## 비차단 메모

- **P1 — 쌍 검증이 실규모 창에서 너무 느리다** (`src/psdl2/views.py:61-73`, `:65`의 `min(remaining)`; `:158-160`의 anchor별 `Fraction` 연산). 2000×2000 원본, pitch 0.05, core 10 um, `generate_pair` 1회 실측:

  | near / wide | CNT 밀도 | 시간 |
  |---|---|---|
  | 200 / 400 cell native, core 5 um | 1% | 0.22 s |
  | 400 / 800 cell native | 1% | 1.8 s |
  | 400 / 800→200 cell area (pitch 0.2) | 1% | 3.9 s |
  | 400 / 1000 cell native | 1% | 3.7 s |
  | 400 / 2000 cell native | 0% / 0.1% / 1% | 5.3 s / 5.7 s / 47–52 s |
  | 400 / 2000→500 cell area (pitch 0.2) | 1% | 77 s |

  메모리는 최대 약 280 MiB로 문제없다. 시간은 component 수에 대해 이차로 늘어난다: `components()`만 따로 재면 1% 고립 cell에서 250² → 0.01 s, 500² → 0.08 s, 1000² → 1.34 s (component마다 남은 집합 전체에 `min`). crop 생성(0.1 s 내외)에 비해 검증이 수십~수백 배 느리다. ROI 500개 × 50 epoch이면 쌍당 1.8 s에서 약 12시간, 77 s에서 약 22일이 검증에만 든다. G4의 실제 창 크기가 정해지지 않아 차단으로 올리지 않았지만, wide 창이 원본 1000 cell 이상이면 실행이 사실상 불가능하다. 결과를 바꾸지 않고 고칠 수 있는 부분이다(순서 무관한 시작점 선택 또는 배열 기반 labelling, anchor 좌표의 정수 연산). 수정하면 `views.py`의 수락 논리가 바뀌므로 B3 수정과 같은 주기에 넣는 편이 재검토가 한 번으로 끝난다.
- **M1 — core가 cell 격자에 맞지 않고 `core + 2·jitter`가 near 크기에 닿으면 정상 쌍이 거부된다** (`src/psdl2/views.py:38-40`의 설정 검사는 통과, `:143-145`에서 거부). 재현: R4 줄무늬 ROI, pitch 0.5, near 20, core 10.2, jitter 4.9(합이 정확히 20), offset `(+4.8, 0)`, `(+4.9, 0)`, `(−4.9, 0)` → `common_core_not_observed`; offset `(+4.6, 0)`은 수락. nearest 창은 cell 중심 기준으로 원본 cell을 고르는데 core는 "겹치는 모든 cell"을 요구하기 때문이다. 명시 오류이고 거부 수에 잡히지만, 설정 검사가 허용한 한계값에서 jitter 끝쪽 표본이 버려진다. 한 cell의 여유를 요구하거나 문서에 적어야 한다.
- **M2 — 거친 pitch + `nearest`는 격자에 맞지 않는 전극에서 거의 항상 거부된다**. 재현: 5 cell 폭 선, pitch 0.5, wide pitch 1.0 nearest, jitter 0.3 → 40쌍 중 39쌍 `resampled_core_marked_content_lost`; 같은 입력에서 `area_coverage`는 40/40 수락. 가장자리 cell이 표본에서 빠지는 것을 core 내용 소실로 본다. 동작 자체는 보수적이고 명시 오류지만, 거친 view에는 `area_coverage`를 써야 한다는 안내가 문서에 없다.
- **M3 — `negative_pair_count`는 여전히 구조상 0** (`src/psdl2/losses.py:92-93`: `eye & ~eye`). N3 수정으로 alignment가 실제 `pair_indices`로 계산되고 `supervised_different_pair_count`가 따로 보고되는 것은 확인했다(different 2쌍 → 2). 그러나 `tests/test_r4_regressions.py:189`의 단언은 항등식이다. 무반발의 실제 근거는 같은 시험의 view 2 치환(variance/covariance 불변, alignment만 변함)과 numpy 독립 검산이며, 이는 유효하다. 다른 위치의 projection을 옮겨도 나머지 위치의 `alignment_per_location`이 변하지 않음을 따로 확인했다.
- **M4 — N4 이후 `analysis_layer`만 다른 후보가 `same_physical_case=true`가 된다** (`src/psdl2/index.py:29-32`, `src/psdl2/search.py:101-107`). 재현: source identity·frame·중심·asset·endpoint가 모두 같고 `analysis_layer`만 `M3`→`M1`인 후보 → primary에 포함, `same_physical_case: True`. 전에는 비호환으로 걸러졌다. 같은 전극 쌍·같은 위치이므로 틀렸다고 단정할 수는 없지만, 플래그가 "분석 문맥까지 같다"는 뜻은 아니라는 점을 문서에 적거나 동일성 조건에 넣어야 한다.
- **M5 — N4의 연결 종류는 VIA 유무만 본다**: `via_count` 1과 3은 호환이다(확인). 문서의 "connection kind (CNT, with VIA when exported)"와 일치한다.
- **M6 — `producer_commit` 키가 아예 없어도 null로 취급된다** (`src/psdl2/data.py:184-187`). 플래그 없이는 거부되고, 플래그와 비어 있지 않은 reason이 있으면 수락된다. fail-closed는 유지된다.
- **M7 — 두 view가 같은 방향으로 밀려 둘 다 bridge를 잃으면 수락된다**(같은 채널 U자 우회로, offset `(0,+3)`과 `(0,+2.5)` → 수락, `identical_views` near/wide 모두 True). 두 view가 서로 일치하므로 alignment 목적에는 해가 없다고 보지만, 규칙이 "원래 중심의 창"이 아니라 "두 view끼리"만 비교한다는 점은 문서에 드러나지 않는다.
- **M8 — 학습 도중의 확률적 거부가 실행 전체를 중단시킨다**: 어느 batch든 수락 쌍이 2 미만이면 `train()`이 예외로 끝난다(`src/psdl2/train.py:44-46`). 명시 오류이고 계약이 요구한 동작이지만, M1·M2처럼 확률적으로 거부되는 설정에서는 여러 epoch 뒤에 중단될 수 있다. 마지막 완료 epoch의 checkpoint는 남는다.

---

## 메모 수정 확인 (N1, N3, N4, N5, N10, N11, N13)

| 메모 | 확인 내용 | 결과 |
|---|---|---|
| N1 | null commit + reason: 플래그 없으면 `producer_commit_required` (API, 그리고 `data`/`train`/`index`/`search` CLI 모두 exit 2). 플래그 `True`일 때만 수락, `'yes'`·`1`은 거부. reason 없음·공백뿐이면 플래그가 있어도 거부. 플래그 + commit pin → `producer_commit_mismatch`. 수락 시 `versions`에 `unversioned_producer_accepted=True`와 reason 기록, checkpoint fingerprint의 `source_versions`에 포함. | 수정됨 (M6 참고) |
| N3 | 위 M3. different 쌍 수 별도 보고, alignment는 실제 pair index로 계산. | 수정됨, 단언 하나는 항등식 |
| N4 | endpoint 순서 뒤집기·electrode 이름 변경 → 호환. role↔layer 결합이 바뀌면(`(gate,M1),(data,M3)` → `(data,M1),(gate,M3)`) 비호환, `failure_type`이 다르면 비호환, VIA 유무가 다르면 비호환. electrode 이름이 다르면 호환이어도 `same_physical_case=False`. encoder는 전극 집합의 평균이므로 endpoint 순서를 뒤집은 bundle의 embedding이 원래와 **정확히 같다**(최대 차 0.0) — 순서를 무시한 호환 판정이 서로 비교할 수 없는 embedding을 맞대지 않는다. | 수정됨 (M4, M5 참고) |
| N5 | `expand=False`에서 cohort 밖 비호환 후보가 `rejected`에 나오지 않고, `expand=True`에서만 나온다. cohort에 후보가 없으면 `no_candidates_in_primary_cohort`. | 수정됨 |
| N10 | split에 없는 query → `evaluation_query_absent_from_split`. 다른 partition의 query는 `excluded_queries`로 빠진다. index에 없는 positive는 `positives_missing_from_index`에 따로 나오고 분모에서 제외. positive가 모두 index에 없으면 `unavailable`. `index_roi_ids`가 없으면 오류. | 수정됨 |
| N11 | export에 없는 label ID → `label_roi_ids_absent_from_export`, checkpoint 미생성 (기존 시험 + 코드 `src/psdl2/train.py:56-57`). | 수정됨 |
| N13 | 파일 fsync → replace → 상위 디렉터리 fsync 순서, descriptor는 finally에서 닫힘 (`src/psdl2/checkpoint.py:31-45`). | 수정됨 |

## 이전에 통과한 영역의 회귀 확인

- **시험**: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:warnings` → **142 passed, 1 skipped**; `--run-training` → **143 passed**. coordinator 증거와 일치.
- **importer**: `data.py`의 변경은 N1 분기뿐이다(diff 확인). 변조 9종(blocked→ready, identity, origin, axis, 대문자 hash, CNT 미관측, 상위 경로, tensor policy 누락, valid 극성)은 시험으로 고정되어 통과.
- **loss 산술**: 무작위 5×3 numpy 독립 검산 시험 통과. 상수 embedding에서 alignment 0, variance 1.5 > 0 확인.
- **검색 primary/supplementary**: cohort 필터가 앞으로 옮겨졌지만 primary 목록·정렬·supplementary 분리는 그대로다(코드와 시험).
- **split**: 변경 없음.
- **checkpoint/resume bit-exactness**: `--run-training` 시험이 4 epoch 연속 실행과 2 + resume 2의 model fingerprint, view RNG, torch RNG 일치를 단언하고 통과. `topology_policy`는 `views` 설정으로 fingerprint에 들어간다.
- **기본값**: 설정 경로(`ViewConfig.from_dict`, `read_config`)는 창·pitch·mode·jitter·core·정책을 모두 요구한다. 저수준 기본값은 N7에 문서화된 `output_pitch_um=None`, `'nearest'`뿐이다(grep).
- **금지 입력·네트워크**: `src/`에 네트워크·다운로드·pretrained 코드 없음, hotspot/CA/fab/density를 읽는 코드 없음(grep; `crop.py` docstring의 "No quantitative CA"만 일치). 모든 시험이 socket을 막는다.

## 잘못된 이유로 통과하는 시험

- `tests/test_r4_regressions.py:189` — `negative_pair_count == sum(i != j ...)`은 대각 index에서 항상 0 == 0 (M3).
- `tests/test_r4_regressions.py:139-148` (`..._removed_bridge_and_merge_limitation`) — 지우는 거친 cell이 core 안이라 실제로는 bridge가 아니라 core 내용 소실(`resampled_core_marked_content_lost`)을 본다. 이름과 달리 merge도 검사하지 않는다. core 밖 bridge는 `:346-358`이 따로 제대로 검사한다.
- `tests/test_views.py:57-69`, `:97-104` — 정규식이 `core_marked_content|core_connection`으로 넓어 내용 소실과 연결 소실을 구별하지 않는다. 두 시험 모두 실제로는 core 내용 소실이다.
- R4_FIXES.md가 바꿨다고 밝힌 두 단언(frame을 가로지르는 선 수락, core 밖 고립 component 소실 수락)은 R4가 요구한 동작과 일치하며, 독립 재현(위 표)으로 같은 결론을 얻었다.
- B1 sweep 시험의 기대 시작 index(`n−20`, half-cell `n−19`)는 구현과 독립으로 적혀 있고 내 Decimal oracle과 일치한다.
- bridge 시험 전체가 한 채널 안에서만 구성되어 B3을 놓친다.

## 확인한 방법

probe는 세션 scratch 디렉터리(`/private/tmp/claude-501/.../scratchpad`)에만 작성했다. `c19a33d` 비교는 `git archive c19a33d src`로 scratch에 풀어 `PYTHONPATH`로 실행했다. 저장소에서 바꾼 것은 이 보고서 한 파일뿐이고, 작성 전 `git status`는 clean, HEAD는 `e03cb5c9…`였다.

## 확인하지 못한 것

- 실자료: 실제 layout, 실제 창 크기·pitch·jitter(G4), layout asset 하나가 전극 하나인지 layer 전체인지, 실제 CNT 밀도. B3의 실제 빈도와 P1의 실제 소요 시간은 여기에 달려 있다.
- 실제 학습 수렴·검색 품질, variance/covariance 규제의 간접 영향.
- CUDA 경로(RNG 복원 포함), 전원 차단 수준의 내구성(N13은 호출 순서만 확인).
- 2000 cell을 넘는 창, 5개보다 많은 채널에서의 성능. encoder forward 시간은 재지 않았다.
- T12의 C→DL2→C 왕복과 fixture 외 producer commit과의 호환.
- resume bit-exactness는 저장소의 opt-in 시험 통과로 확인했고 별도 독립 재실행은 하지 않았다.
