# R4c 독립 재검토 — DL2 view 연결성 v3 수정본

- 검토자: fresh Claude Opus 5.5 세션 (작성자·coordinator·R4/R4b 검토자 아님), 읽기 전용
- 대상: particle-sim-dl2 저장소, branch `main`, commit `b0b11673fffdaaa88af890670c88439e63605ad8`
- 계약: `PLAN_v2_ADDENDUM_DL2_VIEWS.md` §1–§8 (구속), PLAN_v2 §2.4, §2.5, T10, T11
- 이력: R4 (`c19a33d` FAIL) → `e03cb5c` → R4b (`e03cb5c` FAIL: B3, P1, M1–M8) → 보완 계획 채택 → `b0b1167`
- 검토일: 2026-10-09

## 판정: **PASS-WITH-NOTES** (commit `b0b1167`에만 유효)

- **B3: 해결됨.** R4b의 두 입력 모두 `core_joint_connectivity_changed`로 거부된다. 새로 만든 입력과 무작위 대조 시험 7,500쌍에서 잘못 수락되거나 잘못 거부된 쌍을 찾지 못했다.
- **P1: 해결됨.** 2000×2000 원본에서 쌍 검증 중앙값 0.13–0.55 s, 한도 `max(2 s, crop×10)` 안이다. 한 축 2배당 증가는 최대 3.6배(한도 6배).
- 새 차단 사항: **없음.**
- 메모 8건(아래). 실행을 틀리게 하거나 막는 것은 없고, 거부율과 운영에 관한 것이다.

이 PASS는 보완 계획이 정한 기하 proxy(`global_contact_proxy_unverified`) 기준의 구현 일치를 뜻한다. 실제 net 도통이나 검색 품질의 확인이 아니다.

---

## 1. B3 — 채널 간 연결

### R4b 입력 재현

frame 40², origin (100,200), pitch 1, 중심 (120,220), `layout:0[:,17]`, `layout:1[:,22]`, bridge `[14,17:23]`, near 14 / wide 30 / core 6.

| 입력 | CNT | VIA |
|---|---|---|
| 자연 crop, offset (0,0)/(0,3) | 거부 `core_joint_connectivity_changed` | 같음 |
| 같은 입력, view 순서 반대 | 거부 (같은 사유) | 같음 |
| jitter 0, 두 번째 near patch의 bridge 채널 삭제 | 거부 (같은 사유) | 같음 |
| 같은 삭제를 wide patch에 / 첫 번째 view에 | 거부 (같은 사유) | 같음 |
| `generate_pair` 경로 (offset 주입) | 거부 (같은 사유) | — |
| 양쪽 모두 bridge 소실, offset (0,3)/(0,2.5) (M7) | 수락 | 수락 |

### 새로 만든 입력

거부되어야 하는 것 — 전부 `core_joint_connectivity_changed`로 거부됨:

- 같은 채널의 U자 우회로가 한 near 창에만 남음
- x 방향 jitter로 bridge를 잃음
- near는 둘 다 보존하고 wide 창에서만 한쪽이 잃음
- 한 layout 안의 두 선분을 CNT가 이음
- CNT → core 밖 layout 조각 → VIA로 이어지는 사슬이 한 view에서 끊김
- bridge 위 cell의 valid를 한 patch에서만 0으로 바꿈
- 원본에 없는 CNT bridge를 한 near patch에 추가
- 연결된 anchor 쌍이 view마다 다름(A–B 대 B–C, component 수는 같음)

수락되어야 하는 것 — 전부 수락됨:

- 서로 다른 layout의 가로/세로선이 교차하고 CNT/VIA 없음, 반 cell을 넘는 jitter 3종
- 교차 cell에 CNT가 있고 양쪽 view가 모두 포함
- layout에 닿지 않는 60% 밀도 CNT, 실수 jitter ±3 cell: 300/300
- CNT와 VIA가 layout 매개 없이 인접(직접 edge 없음)
- R4 (a) 줄무늬 jitter 0/.1/.3/1/4.9 각 180쌍, (b) 2% 고립 cell — 저장소 시험으로 확인

### 무작위 대조 시험

보완 계획 §1·§2만 보고 판정기를 따로 작성했다(Fraction 좌표, dict 기반 BFS, 저장소 코드 미사용). 36² 장면에 layout 선분, CNT/VIA 막대와 점, valid 구멍을 무작위로 놓고 `validate_pair`와 판정·사유를 비교했다.

| 구성 | 쌍 | 불일치 | 수락 / 연결 거부 / 거친 관계 거부 |
|---|---:|---:|---|
| native pitch 1, near 12 / wide 24 / core 6, jitter 3 | 1500 | 0 | 1218 / 38 / — |
| native pitch 0.5, near 7 / wide 14 / core 3, jitter 2 | 1500 | 0 | 1203 / 39 / — |
| native, wide가 frame 밖으로 나감 | 750 | 0 | 618 / 16 / — |
| area 1→2, near·wide 모두 | 1500 | 0 | 777 / 0 / 586 |
| near native + wide area 0.5→1.5 | 1500 | 0 | 755 / 26 / 584 |
| area, 출력 pitch = 원본 pitch | 750 | 0 | 433 / 2 / 247 |

나머지는 `no_core_layout_anchor`, `common_core_not_observed`이고 역시 일치한다.

### 거친 area-coverage

| 입력 (bridge ROI, pitch 1→2) | 결과 |
|---|---|
| 변조 없음 | 수락 |
| 한 near / 한 wide patch에서 core 밖 bridge 행 삭제 | 거부 `resampled_core_joint_relation_changed` |
| 양쪽 near에서 모두 삭제 | 거부 (각 view가 자기 native 기준과 다름) |
| 원본에 없는 CNT bridge를 거친 patch에 추가 | 거부 (같은 사유) |
| 한 view의 coverage 값만 0.3배 | 수락 |
| 반 cell jitter로 fractional 값만 다름 | 수락 |
| 창 이동으로 한 view만 bridge 소실, offset (0,0)/(0,5) | 거부 `core_joint_connectivity_changed` |
| 두 anchor가 거친 cell에서 인접(pitch 4) | 거부 `resampled_core_joint_relation_changed` |
| 같은 거친 cell의 CNT가 떨어진 두 anchor를 이음 | 거부 (같은 사유) |

## 2. 보완 계획 §1–§4, §8 일치

| 항목 | 확인 | 결과 |
|---|---|---|
| 그래프 edge | 같은 채널 4-이웃, layout↔CNT/VIA 같은 XY만. layout–layout, CNT–VIA, VIA–VIA 직접 edge 없음 (`views.py:131-164`) | 일치 |
| anchor, 분할 비교 | core 안 marked layout cell 전체, 최초 등장 순 재번호화, near끼리·wide끼리만 (`views.py:233-270`) | 일치 |
| centre 창 비교 없음 (M7) | 코드에 centre 창 그래프가 없고 양쪽 소실은 수락 | 일치 |
| 거부 사유 | 6종이 `REJECTION_REASONS`와 계획 문구에 일치, 대조 시험에서 사유까지 일치 | 일치 |
| preflight (M1) | core 10.2 / jitter 4.9 실패, core 10 / jitter 4.9·5 통과. 무작위 644 구성을 offset 격자로 전수 비교: 통과 632건은 어느 offset에서도 core 누락 없음, 실패 12건은 모두 실제 누락 존재 | 일치 |
| 거친 nearest (M2) | `coarse_nearest_not_supported_for_views`, area는 통과, 원본보다 고운 pitch는 오류, 창이 pitch의 정수배가 아니면 오류 | 일치 |
| `analysis_layer` (M4) | 그 값만 다른 후보는 primary에 남고 `same_physical_case=False`. view 쌍 동일성에도 포함 | 일치 |
| 재시도 / skip / counter (M8) | 실제 jitter 거부로 6 epoch × 4 seed: batch별 `attempted = rejected + generated`, `retry = attempted − 위치 수`, ROI당 거부 ≤ 한도, 수락 2 미만이면 skip·used 0 | 일치 |
| `no_usable_batch_in_epoch` | 통계 포함 오류, 이전 checkpoint 파일 불변, CLI exit 2 | 일치 |
| `max_pair_attempts` | dataclass 기본값 없음. 누락, 0, `'3'`, `3.0`, `True` 모두 거부. CLI도 exit 2 | 일치 |
| 의미 기록 | `connection_semantics`, `comparison_basis`가 설정·fingerprint·학습 결과·pair evidence·index 최상위·index record·검색 결과에 있음 | 일치 |
| v2 거부 | checkpoint v1 container, v2 container + 옛 views, index v1, 최상위/record 정책 v2 또는 누락 모두 거부. index CLI는 exit 2, 출력 없음 | 일치 |
| preflight 시점 | optimizer 생성 전 (`train.py:82-83`), CLI exit 2, checkpoint 미생성 | 일치 |

## 3. P1 — 성능과 배열 labelling

### labelling 정확성

`joint_labels`를 따로 작성한 BFS와 분할 단위로 비교했다.

- 무작위 5,969건: 채널 1–6, 크기 1–23, 밀도 0.05–0.95, layout index 집합 무작위 — 불일치 0
- 구조 1,944건: 나선, 가로/세로 뱀 모양, 빗, 뒤집은 빗, 계단, 역계단, 체커, 전면 × 접촉 4종 × 크기 1–31 — 불일치 0
- layout index가 연속이 아니거나 접촉 채널이 없는 경우, 빈 mask — 정상
- marked cell에 label 0이 붙거나 unmarked cell에 label이 붙는 경우 없음

2000² × 4 채널의 같은 구조들에서 0.09–0.29 s다. 뱀·나선 모양에서도 반복 횟수가 늘지 않는다.

### 시간 (2000×2000 원본, pitch 0.05, near 400 / core 200, 1 thread, warm-up 2 + 5회 중앙값)

`tools/benchmark_views_v3.py` 재실행(출력은 scratch): 12행과 증가율 전부 통과, 최대 증가율 3.53, 보관된 JSON과 같은 수준이다.

실제 jitter ±1 um, lane 보호 없음, 채널 추가로 따로 잰 값:

| 장면 | wide | crop | 검증 | 한도 | tracemalloc peak |
|---|---|---:|---:|---:|---:|
| 직선 2개, CNT 1% | 2000 native | 0.118 s | 0.134 s | 2 s | 273 MiB |
| 직선 2개, CNT 30%, 보호 없음 | 2000 native | 0.122 s | 0.216 s | 2 s | 288 MiB |
| 직선 2개, CNT 59% | 2000 native | 0.122 s | 0.356 s | 2 s | 314 MiB |
| 직선 2개, CNT 59% | 2000→500 area | 0.195 s | 0.452 s | 2 s | 231 MiB |
| 직선 2개, CNT 90% | 2000 native | 0.123 s | 0.268 s | 2 s | 353 MiB |
| 교차 격자, CNT 30%, VIA 4개 | 2000 native | 0.184 s | 0.375 s | 2 s | 507 MiB |
| 교차 격자, CNT 30%, VIA 4개 | 2000→500 area | 0.291 s | 0.550 s | 2.91 s | 365 MiB |
| 교차 격자, CNT 59%, VIA 4개 | 2000 native | 0.181 s | 0.485 s | 2 s | 538 MiB |

R4b에서 47–77 s이던 경우가 0.2–0.5 s다.

## 4. Resume

실제 encoder·Adam·`generate_pair`로, 거부가 jitter 난수에서 직접 나오는 구성(bridge ROI 7개, jitter (0,3), 시도 2회, batch 3)을 썼다. seed 4종 각각에서 6 epoch 연속 실행을 2+2+2, 3+3, 1×6 재개와 비교했다.

- 12개 조합 모두 batch별 history, epoch history, checkpoint 안의 epoch history, model fingerprint, view RNG, torch/python/numpy RNG, Adam 상태가 일치
- skip batch가 낀 seed(generated 31, used 30, skip 1) 포함
- wide만 area-coverage(1→2)인 구성으로 4 epoch 대 2+2: 일치, loss 유한, index 6건 생성

## 5. 이전 통과 영역의 회귀

- 시험: `pytest -p no:warnings` → **184 passed, 1 skipped**, `--run-training` → **185 passed**. coordinator 증거와 일치.
- `crop.py`, `data.py`, `losses.py`, `split.py`, `evaluate.py`, `model.py`, `schema.py`와 해당 시험 8개 파일은 `e03cb5c`와 byte 단위로 같다. B1 crop 산술, importer, loss 산술, split은 코드가 바뀌지 않았다.
- B1은 1절 대조 시험이 독립 index 식으로 nearest 창과 area 창 포함 관계를 7,500쌍에서 다시 맞춰 본 셈이기도 하다.
- `search.py` 변경은 정책 검증과 `same_physical_case`의 `analysis_layer` 조건뿐이다. primary/supplementary, N4/N5 경로는 그대로다.
- `src/`에 네트워크·다운로드·pretrained 코드와 hotspot/CA/fab/density 입력 없음(grep). 설정 경로의 기본값 없음.

## 6. 시험의 근거

R4B_FIXES.md가 바꿨다고 밝힌 단언은 모두 보완 계획과 맞는다: core 안 삭제를 내용 소실로 이름·사유 정정, 거친 nearest를 설정 오류로, 거친 병합을 거부로, 즉시 중단을 재시도/skip counter로. 정규식은 `^…$` 정확 일치로 좁혀졌다.

약한 시험(차단 아님, 위 독립 probe가 같은 결론을 줌):

- `tests/test_views_v3.py:338-362` — 거부가 난수가 아니라 `roi_id`로 정해지고 순열은 항등이며, 주석이 말한 evidence의 draw 값은 `paired_forward`가 버려서 비교되지 않는다. counter와 RNG 위치는 확인하지만 "재시도 순서가 복원된 RNG를 따른다"는 것은 보이지 않는다.
- `tests/test_views_v3.py:101-109` — "dense" 경계 band가 4×3 블록 2개, offset 고정이다.
- `tests/test_views_v3.py:88-98` — 층 교차는 `joint_labels` 수준만 본다. 쌍 수준 수락 시험이 없다.
- `tests/test_views_v3.py:275-294` — labelling 대조가 4×8×9 mask 20개뿐이다.
- `tests/test_r4_regressions.py:108-121` — 계획 §6이 요구한 고립성 선행 확인이 없다. seed 2026 fixture에는 인접 쌍이 26개 있다. 고립성은 별도 격자 시험(`test_views_v3.py:112-126`)이 다른 fixture로 확인한다.

## 메모 (비차단)

- **N1 — 거친 view의 창 가장자리 bridge는 양쪽 view가 같아도 거부될 수 있다** (`views.py:189-198`, `:262-268`). native 기준은 중심이 창 안에 있는 cell만 쓰고, 표본 그래프는 coverage > 0인 cell을 쓴다. 재현: bridge ROI, pitch 1→2, near 20, offset (0,5)/(0,4.6) → `resampled_core_joint_relation_changed`. 두 번째 창은 bridge 행을 0.4 cell만 덮는다. 계획 §2 문구 그대로의 동작이고 거부 쪽으로만 치우친다.
- **N2 — 거친 area view의 거부율이 높을 수 있다.** 무작위 장면에서 pitch 2–3배일 때 약 39%가 `resampled_core_joint_relation_changed`였다(대부분 가짜 병합). 계획 §2가 보고하라고 한 값이고, 실제 비율은 G4 pitch와 실자료에 달려 있다.
- **N3 — `no_usable_batch_in_epoch`는 재개해도 같은 자리에서 다시 난다** (`train.py:131-135`). RNG가 복원되므로 결정적이다(재현: 같은 통계, 파일 불변). seed나 설정을 바꾸면 fingerprint가 달라져 그 checkpoint로 재개할 수 없다. epoch당 batch가 1–2개인 작은 학습 집합에서만 현실적이다. 문서에 적어 둘 만하다.
- **N4 — proxy 기준 거부율(참고).** 1000² 교차 격자, native, jitter ±0.5 um, 40쌍씩: CNT 밀도 0–0.3%와 10–30%에서 0건, 1–3%에서 1–5건(`core_joint_connectivity_changed`). 한 창에만 든 CNT가 교차점에서 두 층을 잇는 경우다.
- **N5 — 예제 설정과 opt-in e2e 시험이 area view를 더는 지나지 않는다** (R4B_FIXES.md에 명시). area view 학습·재개·index는 4절에서 손으로 확인했다.
- **N6 — preflight는 jitter가 정확히 ±j일 때만 core를 놓치는 구성도 거부한다.** 실패 12건 중 7건이 그랬다. 계획 §3의 "양 끝값" 그대로다.
- **N7 — `involved` 필터는 현재 죽은 분기다** (`views.py:228-229`). importer가 endpoint layout 두 개만 만들기 때문이다. 중간 layer를 거치는 연결은 그래프에 없고, 이는 계획 범위와 같다.
- **N8 — 이름 혼동.** container는 `psdl2_checkpoint_v2` / `psdl2_index_v2`, 정책은 `common_core_connectivity_v3`다. checkpoint는 매 epoch 전체 batch history를 다시 쓴다.

## 확인한 방법

probe는 세션 scratch 디렉터리(`/private/tmp/claude-501/…/scratchpad`)에만 두었다. 저장소 시험은 `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`로 돌렸고 benchmark 출력도 scratch로 보냈다. 저장소에서 바꾼 것은 이 보고서 한 파일이다. 작성 전 `git status`는 clean, HEAD는 `b0b1167…`였다.

## 확인하지 못한 것

- 실자료: 실제 layout, G4의 창·pitch·jitter·core, 실제 CNT 밀도와 접촉 빈도. N1·N2·N4의 실제 거부율이 여기에 달려 있다.
- proxy가 실제 도통과 얼마나 맞는지. producer에 endpoint binding이 없어 확인할 방법이 없다.
- 학습 수렴과 검색 품질, G6 수락 기준.
- CUDA 경로(RNG 복원 포함), 전원 차단 수준의 내구성.
- 2000 cell을 넘는 창, 8개(valid 포함)보다 많은 채널. encoder forward 시간은 재지 않았다.
- T12의 C→DL2→C 왕복, fixture 밖 producer commit과의 호환.
