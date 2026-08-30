# 다이캐스팅 공정 불량 예측 및 의사결정 지원

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)

다이캐스팅 제조 공정의 불량 가능성을 공정·센서 데이터로 예측하고, 주요 기여 변수를 분석해 작업자용 4단계 판정 리포트로 연결한 머신러닝 프로젝트입니다. Random Forest, XGBoost, LightGBM을 비교하고 클래스 불균형과 임계값을 함께 고려해 최종 모델을 선정했습니다.

**Team 일오나!! — 내일배움캠프 데이터 분석 부트캠프 심화 프로젝트, 15조.** 팀 프로젝트이며, 본인이 직접 수행한 범위는 아래 **개인 기여** 항목에 명확히 구분했습니다.

---

## 한눈에 보기

| | |
|---|---|
| **목표** | 생산 이후의 수동 육안검사에 의존하는 방식을 보완하기 위해, 공정·센서 데이터 기반으로 불량 가능성을 사전 예측하고 주요 원인 후보 변수를 제시하는 모델 구축 |
| **방법** | 클래스 불균형(SMOTE) 하에서 Random Forest / XGBoost / LightGBM 비교, PR-AUC 기반 임계값 선정, SHAP 기반 기여 변수 분석, 작업자용 4단계 판정 리포트 |
| **검증** | 기본 Accuracy가 아닌 PR-AUC와 임계값 스윕(0.6~0.8)을 활용해 모델을 비교하고, FN/FP에 서로 다른 단위 비용(₩3,000 vs ₩20,000/건)을 적용해 오류 유형별 비용 차이를 반영 |
| **결과** | LightGBM 선정 — 충전불량·기포/내부·표면손상 3개 그룹에서 F1 0.78~0.80; 가정 기반 시뮬레이션에서 수동검사 대비 처리 속도 약 100배, 10일당 약 350만원 비용 절감 가능성 추정 |
| **기술** | Python, scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn (SMOTE) |

---

## 문제와 해결책

| | 기존 검사 방식 | ML 기반 검사 방식 |
|---|---|---|
| **방법** | 주조 후 수동 육안검사 | 공정 중 센서·공정 데이터 기반 예측 |
| **시점** | 생산 후(사후 대응) | 생산 중(사전 예방) |
| **일관성** | 작업자 피로도에 따라 판정 편차 가능 | 동일 기준으로 반복 판정 가능 |
| **불량 커버리지** | 육안으로 확인 가능한 표면 불량 중심 | 내부 불량을 포함한 복수 불량 유형 예측 |
| **속도** | 수동 검사 기준 분당 약 1개 | 분석상 추정 처리 속도 약 100배 |

---

## 데이터셋

| 항목 | 값 |
|---|---|
| **출처** | 주조 품질보증 AI 데이터셋 — KAMP (Korea AI Manufacturing Platform, 2022) |
| **파일** | `DieCasting_Quality_Raw_Data.csv` |
| **규모** | 7,535행 × 57열 |
| **구조** | MultiIndex: `Process` / `Sensor` / `Defects` |

### 피처 그룹

**공정 변수 (독립변수)**
| 분류 | 피처 |
|---|---|
| 속도 | `Velocity_1`, `Velocity_2`, `Velocity_3`, `High_Velocity` |
| 시간 | `Spray_Time`, `Rapid_Rise_Time`, `Pressure_Rise_Time`, `Cycle_Time` |
| 압력 | `Cylinder_Pressure`, `Casting_Pressure` |
| 힘 | `Clamping_Force` |
| 기타 | `Biscuit_Thickness` |

**센서 변수 (독립변수)**
| 분류 | 피처 |
|---|---|
| 온도 | `Factory_Temp`, `Coolant_Temp`, `Melting_Furnace_Temp` |
| 습도 | `Factory_Humidity` |
| 압력 | `Coolant_Pressure`, `Air_Pressure` |

**불량 그룹 (목표변수 / 종속변수)**
| 그룹 | 불량 유형 |
|---|---|
| 충전불량 (Fill Defect) | Short Shot |
| 기포/내부 (Void/Internal) | Bubble, Blow Hole |
| 표면손상 (Surface Damage) | Dent, Scratch, Stain, Burning Mark, Exfoliation |
| 기타 (Other) | Crack, Deformation, Impurity, Contamination, Inclusions |

---

## 방법론

```
원본 데이터 (7,535행 × 57열)
        │
        ▼
┌─────────────────────────────┐
│           전처리              │
│  • 센서 결측치 → 중앙값 대체      │   
│  • 불량 → 이진값(0/1)          │
│  • 불량 유형 그룹화 (×4)        │
│  • id / Shot / Type 제거     │
│  • SMOTE (기타 그룹 1:1)      │ 
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│             EDA             │
│  • 변수별 히스토그램            │
│  • 상관관계 히트맵 ×3           │
│  • Z-score 이상치 분석         │
│  • 불량 유형별                 │
│    독립표본 2-sample t검정     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│           모델링              │
│  Random Forest              │
│  XGBoost                    │
│  LightGBM ✓ (선정)           │
│                             │
│  선정 근거:                   │
│  PR-AUC + ROC-AUC 커브       │
│  임계값 스윕 (0.6–0.8)         │
│  → 임계값 = 0.68              │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│           해석가능성           │
│  SHAP TreeExplainer         │
│  • Global Bar / Beeswarm    │
│  • Local Waterfall plot     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│           판정 로직           │                
│  ① 진단 (불량 확률 %)          │
│  ② 추적   (SHAP Top-N)       │
│  ③ 비교 (정상군 중앙값 대비).    │
│  ④ 판정   (위험 / 안전)        │
└─────────────────────────────┘
```

---

## 모델 결과

### 모델 비교 (임계값 = 0.68)

|  | RandomForest | XGBoost | **LightGBM** |
|---|---|---|---|
| 충전불량 F1 | 0.588 | 0.664 | **0.789** |
| 기포/내부 F1 | 0.545 | 0.695 | **0.800** |
| 표면손상 F1 | 0.591 | 0.719 | **0.782** |
| 기타 F1 | 0.200 | 0.269 | **0.264** |
| **평균 F1** | ~0.48 | ~0.59 | **~0.66** |

### LightGBM 최종 성능

| 불량 그룹 | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| 충전불량 | 0.9688 | 0.9888 | 0.6567 | 0.7892 | 1 | 46 |
| 기포/내부 | 0.9794 | 0.9254 | 0.7045 | 0.8000 | 5 | 26 |
| 표면손상 | 0.9748 | 0.9189 | 0.6800 | 0.7816 | 6 | 32 |
| 기타 | 0.9741 | 0.4667 | 0.1842 | 0.2642 | 8 | 31 |

**LightGBM 선정 이유.** 클래스 불균형 환경에서 모든 불량 그룹의 PR-AUC가 비교 모델보다 높았고, XGBoost 대비 FP 건수도 낮았습니다. 본 프로젝트에서는 정확도만이 아니라 불균형 데이터에서의 탐지 성능과 오탐 비용을 함께 고려해 LightGBM을 최종 모델로 선정했습니다.

### 주요 SHAP 기여 변수 (충전불량 기준)

| 순위 | 변수 | 방향성 |
|---|---|---|
| 1 | `Process\|High_Velocity` | ↑ 높을수록 → 불량 위험 |
| 2 | `Process\|Clamping_Force` | ↑ 높을수록 → 불량 위험 |
| 3 | `Sensor\|Melting_Furnace_Temp` | ↑ 높을수록 → 불량 위험 |
| 4 | `Process\|Spray_2_Time` | ↑ 높을수록 → 불량 위험 |
| 5 | `Process\|Casting_Pressure` | 상황에 따라 다름 |

---

## 운영 비용 관점의 시뮬레이션

| 지표 | 값 |
|---|---|
| FN 비용 가정 (놓친 불량) | ₩3,000 / 개 |
| FP 비용 가정 (오탐) | ₩20,000 / 개 |
| 추정 절감액 (10일) | **₩3,503,430** |
| 처리 속도 개선 추정 | **약 100배** |
| 불량 검출률 개선 추정 | 수동 대비 **약 +15pp** |
| 작업 시간 단축 | **50%** |

위 수치는 명시한 FN/FP 단위 비용과 수동검사 성능 가정을 바탕으로 계산한 **시뮬레이션 결과**이며, 실제 생산 현장에서 측정된 절감액이나 검사 성능이 아닙니다. 실제 수동검사 데이터를 확보해 베이스라인을 재산정하는 작업은 향후 과제로 남겨두었습니다.

---

## 개인 기여

본 프로젝트는 팀 프로젝트(Team 일오나!!, 심화 프로젝트 15조)입니다. 아래 항목은 **본인이 직접 수행한 범위**입니다:

- **데이터 전처리** — 결측치 처리(센서 중앙값 대체), 이상치 처리, 불량 이진화 및 4그룹 매핑, SMOTE 적용
- **모델링** — Random Forest / XGBoost / LightGBM 비교, PR-AUC 기반 모델 선정, 임계값 스윕 및 선정(0.68)
- **판정 로직** — 작업자용 4단계 리포트 설계 및 구현(진단 → 주요 변수 확인 → 정상군 비교 → 위험 판정)
- **비용 시뮬레이션** — FN/FP 비용 가정 설정 및 10일당 약 350만원 절감 가능성 추정

**SHAP 해석 파트(글로벌/로컬 플롯 및 기여 변수 분석)는 팀원이 담당했습니다.** 본인은 해당 결과를 작업자용 판정 로직과 연결하는 부분을 담당했습니다.

---

## 프로젝트 구조

```
Real-Time-Defect-Prediction-for-Die-Casting-Manufacturing/
│
├── 다이캐스팅_불량예측.ipynb        # 메인 노트북
├── DieCasting_Quality_Raw_Data.csv # 데이터셋
├── 다이캐스팅 프로젝트.pdf          # 발표 자료
└── README.md
```

---

**예시 출력:**
```
==========================================================
  [① 진단]  그룹: 충전불량  |  샘플 #0
  불량 발생 확률: 72.4%  |  판정: 불량  |  실제: 불량

  [② 추적 → ③ 비교 → ④ 판정]  상위 기여 변수 Top 5
  순위  변수                                기준값    현재값      편차   판정
  ---------------------------------------------------------------------------
  1    High Velocity                       0.191     0.215    +12.6% ⚠ 위험
  2    Clamping Force                    379.000   410.000     +8.2% ✔ 안전
  3    Melting Furnace Temp              650.000   698.000     +7.4% ✔ 안전
  4    Spray 2 Time                       12.200    15.800    +29.5% ⚠ 위험
  5    Casting Pressure                  596.000   641.000     +7.6% ✔ 안전

  [기준] 정상군 중앙값 | [편차] (현재-기준)/기준×100 | |편차|>10% → 위험
```

---

## 노트북 구성

| 섹션 | 설명 |
|---|---|
| **0. Setup** | 라이브러리, OS별 한글 폰트 자동 감지, 전역 상수 |
| **1. 데이터 로드** | MultiIndex CSV 로딩, 결측 확인, 컬럼 그룹화 |
| **2. EDA** | 변수별 히스토그램, 상관관계 히트맵 3종, Z-score 이상치 분석, 불량 유형별 t검정 |
| **3. 전처리** | 센서 중앙값 대체, 불량 이진화, 4그룹 매핑, 불균형 클래스용 SMOTE |
| **4. 모델링** | PR/ROC 커브 비교, 임계값 스윕, RF/XGB/LGBM 학습, 혼동행렬 시각화, 5-fold 교차검증 |
| **5. SHAP** | 글로벌 bar+beeswarm 플롯, 단일 샘플용 로컬 waterfall, 피처 중요도 테이블 *(팀원 담당)* |
| **6. 판정 로직** | 4단계 진단 리포트: 불량 확률 → 상위 N개 기여 변수 → 정상 대비 편차 → 위험 판정 |
| **7. 비용 시뮬레이션** | FN/FP 비용 계산, 누적 비용 추이, 수동검사 대비 불량 검출률 비교 |

---

## 한계 및 향후 과제

| 항목 | 세부내용 |
|---|---|
| **ROI 정량화** | 실제 육안검사 성능과 비교할 베이스라인(육안검사) 데이터 수집 |
| **다중공선성 점검** | VIF + 상관관계 기반 피처 정제로 모델 안정성 개선 |
| **하이퍼파라미터 튜닝** | 기본값 대신 Optuna 기반 베이지안 최적화 적용 |
| **피처 엔지니어링** | 상호작용항(예: Velocity × Pressure), 롤링 통계량 |
| **파이프라인 자동화** | 운영 배포를 위한 MLflow 추적 + 모델 레지스트리 |
| **실시간 대시보드** | 현장 실시간 모니터링용 Streamlit / Grafana 대시보드 |

---

## 참고문헌

- Ministry of SMEs and Startups & KAIST. (2022). *Casting quality assurance AI dataset*. [KAMP](https://www.kamp-ai.kr/)
- Uyan, T.Ç., et al. *Industry 4.0 Foundry Data Management and Supervised Machine Learning in Low-Pressure Die Casting Quality Improvement*. InterMetalcast 17, 414–429 (2023).
- Pechmann, A. & Kanli, S. *Towards Sustainable Manufacturing: Deployable Deep Learning for Automated Defect Detection in Aluminum Die-Cast X-Ray Inspection*. Appl. Sci. 2025, 15, 13134.
- Okuniewska, A., Perzyk, M., & Kozłowski, J. *Machine Learning Methods for diagnosing the causes of die-casting defects*. Computer Methods in Materials Science, 23(2), 45–56. (2023).
