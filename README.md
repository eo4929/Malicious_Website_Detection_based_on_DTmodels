# Malicious_Website_Detection_based_on_DTmodels
- Summary: Malicious Website Detection (based on 결정트리 기반 모델들)
- Implementer: Dae-Young Park (email: mainthread@gmail.com)

---
## 데이터셋
1. Size: 1781 * 21
2. Label: 정상 웹사이트 or 악의적인 웹사이트

---
## 라이브러리
- Pycaret
- Sklearn
- Optuna
- Pandas
- Numpy

---
## 데이터 전처리
- 이산형 변수 인코딩: One-hot encoding 으로 처리
- 결측치 처리: 단순하게 imputation 처리

---
## 4가지 결정트리 기반 모델들 성능 비교


---
## 1등 모델 성능 높이기

---
## 1등 모델 성능 높이기
- 성능 높일 수 있는 포인트
    - 데이터 관점
        - feature engineering
        - 데이터 증식 적용
        - 단순 imputer 적용했는데, iterative imputer로 multi-variate 사이 관계 기반해서 missing value 처리하기
        - 시간 관계상 interaction feature 고려하지 않았음
        - dimenstionality 조절
    - 모델 관점
        - 다른 방식의 하이퍼파라미터 튜닝 적용
        - 다양한 모델 앙상블 기법 사용
            - stacking, blending에도 weight 을 줄 수 있음
        - 학습된 모델에 대한 probability threshold 최적화
