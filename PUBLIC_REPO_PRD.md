# Public Repo PRD

## 목적

Paper 0 제출용 `Data Availability Statement`를 뒷받침할 **공개 재현 패키지(reproducibility package)** 를 준비합니다.

## 목표 산출물

- 별도 public GitHub repository 1개
- 최소 재현 패키지 1세트
- Zenodo DOI 연결 가능한 release 구조

## 추천 repository 이름

- Repository name: `ib-lbm-comparison` (확정)
- URL: `https://github.com/mulgae-life/ib-lbm-comparison`

## 포함 범위

### 공개 포함

- 솔버 소스 코드: `iblbm/`, `scenarios/`
- 선택 분석 스크립트:
  - `scripts/analyze_sedimentation_canonical.py`
- 의존성 정의: `requirements.txt`
- processed benchmark outputs:
  - `data/df_benchmarks/**/status.json`
  - `data/mdf_benchmarks/**/status.json`
  - `data/dfc_benchmarks/**/status.json`
  - `data/grid_sensitivity/summary.csv`
  - `data/grid_sensitivity/**/status.json`
  - `data/sedimentation_canonical/**/status.json`
  - `data/sedimentation_canonical/**/sedimentation_history.json`
  - oscillating reference CSVs
- 논문 그림:
  - `figures/*.png`

### 공개 제외

- `velocity_field.npz`
- raw mp4 / large intermediate images
- local logs
- 개인 작업 메모와 내부 scratch 자산
- `tests/`
- `run_steady.py`
- `derived_reports/*.md`
- auxiliary runner scripts

## 공개 제외 이유

- 일부 `velocity_field.npz` 파일은 100 MB를 초과하여 일반 GitHub 업로드에 부적합합니다.
- raw outputs 전체를 GitHub에 두기보다, processed outputs + scripts를 기본 공개하고 raw outputs는 Zenodo/LFS/요청형으로 두는 편이 현실적입니다.

## 권장 공개 전략

1. GitHub public repo 생성
2. 이 staging 디렉토리 내용을 repo root에 복사
3. 라이선스 추가
4. public push
5. GitHub release 생성
6. Zenodo archive DOI 발급
7. 논문 `Data Availability Statement` 최종 URL/DOI 반영

## 최종 논문 문구 전략

- GitHub URL이 준비되기 전:
  - 공개 저장소 준비 중 + raw outputs 요청형
- GitHub URL/Zenodo DOI 준비 후:
  - 공개 URL + DOI 명시

## 릴리스 전 체크리스트

- [x] repo 이름 확정 → `ib-lbm-comparison`
- [ ] 라이선스 확정 (MIT 권장)
- [x] GitHub public push (2026-03-28)
- [ ] Zenodo DOI 발급
- [x] manuscript data statement URL 반영
- [ ] figure / report / script 경로 깨짐 확인
