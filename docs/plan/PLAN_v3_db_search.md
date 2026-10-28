# PLAN v3 — 여러 제품 DB 검색 (요약)

이 문서는 공개용 요약이다. 절차는 [USAGE.md](../USAGE.md), 출력 계약은 [data_contract.md](../data_contract.md)를 따른다.

## 1. 목적

- 여러 제품의 DB에서, 질의 ROI와 같거나 비슷한 영역을 찾는다.
- 점수(cosine, NCC)는 표시용이다. 점수로 채택·기각하지 않는다.
- 제품 사이에서 점수를 비교하거나 다시 정렬하지 않는다.
- 결과가 없다는 것은 위험이 0이라는 뜻이 아니다.

## 2. DB 구성

- 색인: `psdl2_index_v2` 한 개.
- 모델: checkpoint 한 개. 색인은 그 checkpoint의 지문을 기록한다.
- 입력 데이터: 제품마다 `psdl2_dataset_v1`의 source 하나와 grid patch set(`psdl2_patch_set_v1`).
- 재학습하면 checkpoint가 바뀐다. 그러므로 DB를 다시 만든다(색인을 다시 쓴다).

## 3. 새 출력 schema

| schema | 만드는 곳 | 요약 | 자세한 내용 |
|---|---|---|---|
| `psdl2_search_batch_v1` | `psdl2.search --all-ready` | 질의마다 검색 결과(`results`)와 후보 순위(`rankings`)를 봉투 하나에 담는다 | [USAGE 5.4.1](../USAGE.md#541-배치-봉투-psdl2_search_batch_v1---all-ready), [data_contract](../data_contract.md#psdl2_search_batch_v1--batch-search-envelope) |
| `psdl2_eval_position_multi_v1` | `psdl2.eval_position --search-batch` 또는 `--baseline` 2개 이상 | 질의 단위 hit 율과 제품별 hit 율 | [USAGE 5.4.2](../USAGE.md#542-위치-평가-psdl2_eval_position_multi_v1-psdl2eval_position---search-batch-또는---baseline-2개-이상), [data_contract](../data_contract.md#psdl2_eval_position_multi_v1--multi-product-position-evaluation) |

## 4. ETX 실행기 4종

| 파일 (`bundles/etx/`) | 용도 | torch | 입력 | 산출물 |
|---|---|---|---|---|
| `psdl2_job_baseline_ncc.py` | simcompare 4회차 결과로 두 방향 NCC 기준선 | 불필요 | simcompare 4회차 `candidates` 결과의 export | `ncc_r2t.json`, `ncc_t2r.json`, 검토 시트 |
| `psdl2_job_smoke_torch.py` | ETX에서 torch 경로가 도는지 확인 | 필요 | 없음(합성 fixture를 만든다) | `smoke.log`, `eval_position.json` |
| `psdl2_job_db_search.py` | 제품마다 grid → split → 학습 → 색인 → 배치 검색 | 필요 | `DB_DATASET`, `QUERY_DATASET`, `LINEAGES`, `ASSIGN`, `CONFIG`, `COHORT` | `grid_<제품>.json`, `split.json`, `model.pt`, `index.json`, `search_batch.json` |
| `psdl2_job_eval.py` | 제품별 NCC 기준선과 DL2 배치 검색을 같은 정답으로 평가 | 불필요 | `QUERY_EXPORT`, `CANDIDATES`, `CORRESPONDENCE`, `SEARCH_BATCH` | `ncc_<제품>.json`, `eval_baseline.json`, `eval_search_batch.json` |

- 오류 코드 표: [USAGE 2.2 실행기 오류 코드](../USAGE.md#실행기-오류-코드). 실데이터 실행기의 추가 코드는 [USAGE 실데이터 실행기 2종](../USAGE.md#실데이터-실행기-2종-db-검색-평가)에 있다.
- 사용자가 정할 값은 EDIT 블록에 `FILL_ME`로 있다. 하나라도 남으면 어떤 작업도 돌지 않는다.

## 5. compare 쪽 요청 R1–R5

회차 위치: 질의 = 변경점 ROI export, DB·후보 = 제품별 template export.

- **R1.** 변경점 export에 `mask_derivation.json`을 함께 둔다.
- **R2.** 변경점을 `review_roi_v1`의 manual 행으로 만든다.
- **R3.** confirm-pairs가 `psdl2_search_batch_v1`를 읽는다.
  - R3a. `schema_version`으로 분기한다. 기존 NCC 경로는 그대로 둔다.
  - R3b. 후보 필드는 `rankings[].candidates[]`의 `rank`, `frame_id`, `center_xy_um`, `product_id`다.
  - R3c. 질의 export 해시는 `rankings[i].query_export_sha256`을 먼저 쓴다. 질의가 dataset이면 `inputs.query_export_sha256`은 `null`이다. 후보 제품의 export는 `index_sources[]`와 대조한다. `--all-ready` CLI는 `rankings[i].query_export_sha256`에 늘 문자열을 쓴다. reader는 `null`을 "알 수 없음 → 검증 불가(unusable)"로 다룬다. `product_ids`가 `null`이면 그 후보는 검증할 수 없으므로 unusable이다. `index_sources`가 `"single_export"`이면 따로 분기한다.
  - R3d. 설정에 없는 후보 제품은 `candidate_product_not_configured`로 unusable이다.
  - R3e. `roi_correspondence_v1`은 쌍 단위로 유지된다고 가정한다. **compare의 확인이 필요하다.**
  - R3f. DL2 합성 교환 fixture 위치: 없음 — 필요 시 별도 Task. (smoke 번들 `psdl2_job_smoke_torch.py`가 실행할 때 합성 `roi_correspondence_v1`과 `psdl2_search_batch_v1`를 만들지만, 저장소에 고정 파일은 없다.)
- **R4.** seal/read의 배치 봉투 지원은 보류한다.
- **R5.** compare의 `docs/handoff/README.md`에 있는 DL2 pin을 이 브랜치의 병합 commit으로 갱신한다. `efc6dac`(PLAN_v2 patch generator)부터의 변경분이 이때 처음 pin에 들어간다. 새 pin 후보는 이 브랜치의 병합 commit이다.

## 6. 확인하지 못한 것

| 항목 | 상태 |
|---|---|
| Linux에서 실제 실행 | 확인 못 함. 로컬 Python 3.9 venv에서만 시험했다 |
| ETX 실제 패키지 판 | 사용자 보고값(Python 3.9.14, numpy 1.23.5, Pillow 10.1.0, torch 2.1.1+cu121)으로 로컬 재현; ETX 서버에서 실행 미확인 |
| CUDA (cu121) | 확인 못 함. 시험은 CPU뿐이다 |
| 실데이터 규모·메모리·시간 | 확인 못 함. 봉투 크기 추정식 `Q × (8.5 KB + 0.15 KB × M + 9.8 KB × T)`는 합성 실측 1점에서 맞춘 값이며, `M` 계수는 추정이다 |
| compare의 변경점 형식 | 미확정. 지금은 R1·R2의 형식을 가정한다 |
| R3e | compare 확인 전이다 |
| 순위 품질 | 알 수 없다. 1 epoch 합성 모델은 순위 품질을 보이지 않는다 |
