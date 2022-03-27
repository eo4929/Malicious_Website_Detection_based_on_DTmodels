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
1. Decision Tree
- 학습된 모델 
```
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=1541, splitter='best')
```
- 성능
```
F1:  0.7478
Prec:  0.7049
Recall:  0.7963
```
2. Random Forest
- 학습된 모델
```
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=2,
                       oob_score=False, random_state=1541, verbose=0,
                       warm_start=False)
```
- 성능
```
F1:  0.8454
Prec:  0.9535
Recall:  0.7593
```


3. Gradient Boosting Classifier
- 학습된 모델
```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=1541, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
```


- 성능
```
F1:  0.7928
Prec:  0.7719
Recall:  0.8148
```


4. XGBoost
- 학습된 모델
```
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=2,
              num_parallel_tree=1, objective='binary:logistic',
              predictor='auto', random_state=1541, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='auto',
              use_label_encoder=True, validate_parameters=1, verbosity=0)
```


- 성능
```
F1:  0.8713
Prec:  0.9362
Recall:  0.8148
```



---
## 1등 모델 성능 높이기



---
## Future work: 어떻게 성능 더 높일 것인가?
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
