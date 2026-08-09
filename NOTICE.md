# 데이터·소프트웨어 출처 및 라이선스

이 프로젝트는 아래의 공개 데이터셋과 오픈소스를 사용합니다.

---

## 1. 데이터셋 — MixedWM38

**출처**: <https://github.com/Junliangwangdhu/WaferMap>
(Kaggle 미러: <https://www.kaggle.com/co1d7era/mixedtype-wafer-defect-datasets>)

실제 웨이퍼 공장에서 수집한 38,000여 장의 혼합형 웨이퍼 맵 결함 데이터셋입니다.
정상 1종 + 단일 결함 8종 + 혼합 결함 29종, 총 38개 패턴으로 구성됩니다.
데이터 배열 규격: `arr_0` = 52×52 (0 빈칸 / 1 정상 다이 / 2 불량 다이), `arr_1` = 8차원 멀티핫 라벨.

**인용 (필수)**

> J. Wang, C. Xu, Z. Yang, J. Zhang and X. Li,
> "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition,"
> *IEEE Transactions on Semiconductor Manufacturing*, 2020.
> DOI: [10.1109/TSM.2020.3020985](https://doi.org/10.1109/TSM.2020.3020985)

```bibtex
@article{wang2020mixedwm38,
  title   = {Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition},
  author  = {Wang, Junliang and Xu, Chuqiao and Yang, Zhengliang and Zhang, Jie and Li, Xiaoou},
  journal = {IEEE Transactions on Semiconductor Manufacturing},
  year    = {2020},
  doi     = {10.1109/TSM.2020.3020985}
}
```

**라이선스 상태**: 원 저장소에 별도의 LICENSE 파일이 없습니다. README에는
"연구·학습 목적으로 공개한다"는 취지가 밝혀져 있으므로 **연구/교육 목적 사용에 한정**하고,
2차 배포나 상업적 이용 전에는 원저자에게 확인이 필요합니다.

원 데이터셋의 C7·C9 라벨 오류는 University of Technology Malaysia의 Uzma Batool 씨가 수정했습니다.

---

## 2. 학습된 모델 가중치

이 저장소가 배포하는 `swin_multilabel.pt` / `best.pt` 는 위 MixedWM38 데이터로
직접 학습한 결과물이며, **데이터셋과 동일한 조건(연구·교육 목적)** 을 따릅니다.

- 분류: Swin Transformer Tiny — `timm` 의 ImageNet 사전학습 가중치에서 파인튜닝
- 탐지: YOLOv8n — Ultralytics 사전학습 가중치에서 파인튜닝 (아래 AGPL 항목 참고)

---

## 3. 오픈소스 라이선스

| 구성요소 | 용도 | 라이선스 |
|---|---|---|
| **Ultralytics YOLOv8** | 결함 탐지 (`models/detector.py`) | **AGPL-3.0** ⚠️ |
| PyTorch | 추론 엔진 | BSD-3-Clause |
| timm | Swin Transformer 구현 | Apache-2.0 |
| Streamlit | 웹 UI | Apache-2.0 |
| Plotly | 웨이퍼 맵·3D FFT 시각화 | MIT |
| Matplotlib | PDF용 정적 그림 | PSF (BSD 계열) |
| ReportLab | PDF 리포트 생성 | BSD |
| SciPy · NumPy · NetworkX · Pillow | 수치연산·이미지·그래프 | BSD 계열 |
| Superb AI Python SDK | 라벨링 플랫폼 연동 | 공급자 약관 |

### ⚠️ Ultralytics AGPL-3.0 주의

AGPL-3.0은 **네트워크로 서비스를 제공하는 것만으로도** 소스 공개 의무가 발생합니다
(GPL과 다른 점). 즉 이 앱을 공개 URL로 배포하면서 Ultralytics를 사용한다면:

1. 전체 결합 저작물의 소스를 **AGPL-3.0으로 공개**하고, 이용자에게 소스 위치를 안내해야 합니다.
   (이 저장소는 public이므로 링크만 명시하면 충족)
2. 또는 배포본에서 Ultralytics를 제외합니다. 이 프로젝트는 탐지를 Superb 배포모델로도
   수행할 수 있어, `requirements.txt` 에서 `ultralytics` 한 줄만 빼면 의무가 발생하지 않습니다.

연구실 내부 사용·논문 실험 등 **외부에 서비스하지 않는 용도**라면 어느 쪽이든 무방합니다.
