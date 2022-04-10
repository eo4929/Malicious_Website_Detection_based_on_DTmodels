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
- 이산형 변수 인코딩: One-hot encoding 으로 처리 (희소 컬럼 집합에 PCA 적용)
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
F1:  0.8036
Prec:  0.8535
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
F1:  0.849
Prec:  0.8862
Recall:  0.8148
```

---
## 1위 모델 XGBoost에 앙상블 적용
1. 튜닝된 xgboost
- 학습된 모델
```
CustomProbabilityThresholdClassifier(base_score=0.5, booster='gbtree',
                                     classifier=XGBClassifier(base_score=0.5,
                                                              booster='gbtree',
                                                              colsample_bylevel=1,
                                                              colsample_bynode=1,
                                                              colsample_bytree=0.5713050530657279,
                                                              enable_categorical=False,
                                                              gamma=0,
                                                              gpu_id=-1,
                                                              importance_type=None,
                                                              interaction_constraints='',
                                                              learning_rate=0.39501454875096653,
                                                              max_delta_step=0,
                                                              max_dep...
                                     monotone_constraints='()',
                                     n_estimators=147, n_jobs=2,
                                     num_parallel_tree=1,
                                     objective='binary:logistic',
                                     predictor='auto',
                                     probability_threshold=0.7000000000000001,
                                     random_state=1382,
                                     reg_alpha=7.927683821890306e-06,
                                     reg_lambda=0.10467700422591077,
                                     scale_pos_weight=12.575488528701033,
                                     subsample=0.6787513633215522,
                                     tree_method='auto', use_label_encoder=True, ...)
```


- 성능
```
F1:  0.8807
Prec:  0.8727
Recall:  0.8889
```

2. 배깅 버전 xgboost 
- 학습된 모델
```
CustomProbabilityThresholdClassifier(base_estimator=XGBClassifier(base_score=0.5,
                                                                  booster='gbtree',
                                                                  colsample_bylevel=1,
                                                                  colsample_bynode=1,
                                                                  colsample_bytree=0.5713050530657279,
                                                                  enable_categorical=False,
                                                                  gamma=0,
                                                                  gpu_id=-1,
                                                                  importance_type=None,
                                                                  interaction_constraints='',
                                                                  learning_rate=0.39501454875096653,
                                                                  max_delta_step=0,
                                                                  max_depth=4,
                                                                  min_child_weight=1,
                                                                  miss...
                                                                                               validate_parameters=1,
                                                                                               verbosity=0),
                                                                  bootstrap=True,
                                                                  bootstrap_features=False,
                                                                  max_features=1.0,
                                                                  max_samples=1.0,
                                                                  n_estimators=10,
                                                                  n_jobs=None,
                                                                  oob_score=False,
                                                                  random_state=1382,
                                                                  verbose=0,
                                                                  warm_start=False),
                                     max_features=1.0, max_samples=1.0,
                                     n_estimators=10, n_jobs=None,
                                     oob_score=False,
                                     probability_threshold=0.7000000000000001,
                                     random_state=1382, verbose=0,
                                     warm_start=False)
```
- 성능
```
F1:  0.9174
Prec:  0.9091
Recall:  0.9259
```

3. 블랜딩 버전 xgboost
- 학습된 모델
```
CustomProbabilityThresholdClassifier(classifier=VotingClassifier(estimators=[('xgboost',
                                                                              XGBClassifier(base_score=0.5,
                                                                                            booster='gbtree',
                                                                                            colsample_bylevel=1,
                                                                                            colsample_bynode=1,
                                                                                            colsample_bytree=0.5713050530657279,
                                                                                            enable_categorical=False,
                                                                                            gamma=0,
                                                                                            gpu_id=-1,
                                                                                            importance_type=None,
                                                                                            interaction_constraints='',
                                                                                            learning_rate=0.39501454875096653,
                                                                                            max_delta_step=0...
                                                                                                 subsample=0.6787513633215522,
                                                                                                 tree_method='auto',
                                                                                                 use_label_encoder=True,
                                                                                                 validate_parameters=1,
                                                                                                 verbosity=0),
                                                                    bootstrap=True,
                                                                    bootstrap_features=False,
                                                                    max_features=1.0,
                                                                    max_samples=1.0,
                                                                    n_estimators=10,
                                                                    n_jobs=None,
                                                                    oob_score=False,
                                                                    random_state=1382,
                                                                    verbose=0,
                                                                    warm_start=False))],
                                     flatten_transform=True, n_jobs=2,
                                     probability_threshold=0.5, verbose=False,
                                     voting='soft', weights=None)
```
- 성능
```
F1:  0.887
Prec:  0.8361
Recall:  0.9444
```


4. 스태킹 버전 xgboost
- 학습된 모델
```
CustomProbabilityThresholdClassifier(base_score=0.5, booster='gbtree',
                                     classifier=XGBClassifier(base_score=0.5,
                                                              booster='gbtree',
                                                              colsample_bylevel=1,
                                                              colsample_bynode=1,
                                                              colsample_bytree=0.5713050530657279,
                                                              enable_categorical=False,
                                                              gamma=0,
                                                              gpu_id=-1,
                                                              importance_type=None,
                                                              interaction_constraints='',
                                                              learning_rate=0.39501454875096653,
                                                              max_delta_step=0,
                                                              max_dep...
                                     monotone_constraints='()',
                                     n_estimators=147, n_jobs=2,
                                     num_parallel_tree=1,
                                     objective='binary:logistic',
                                     predictor='auto',
                                     probability_threshold=0.7000000000000001,
                                     random_state=1382,
                                     reg_alpha=7.927683821890306e-06,
                                     reg_lambda=0.10467700422591077,
                                     scale_pos_weight=12.575488528701033,
                                     subsample=0.6787513633215522,
                                     tree_method='auto', use_label_encoder=True, ...)
```
- 성능
```
F1:  0.8807
Prec:  0.8727
Recall:  0.8889
```

---
## 결과
- 성능 1등: 배깅 버전 XGBoost
```
F1:  0.9174
Prec:  0.9091
Recall:  0.9259
```
- 사실, 성능 잘나온 배깅 버전 XGBoost 여러 개 만들고, 이들로 stacker 만드는 게 정통적인 최종 앙상블 구축 순서 

---
## Future work: 어떻게 성능 더 높일 것인가?
- 성능 높일 수 있는 포인트
    - 데이터 관점
        - feature engineering
        - 데이터 증식 적용
        - 단순 imputer 적용했는데, iterative imputer로 multi-variate 사이 관계 기반해서 missing value 처리하기
        - 시간 관계상 interaction feature 고려하지 않았음
        - dimenstionality 조절 (현재는 PCA의 주성분 개수 20으로 세팅)
    - 모델 관점
        - 여러 방식의 하이퍼파라미터 튜닝 적용
        - 다른 모델 활용 (e.g., deep forest, TabNet)
        - 다양한 모델 앙상블 기법 사용
            - stacking, blending에도 weight 을 줄 수 있음
            - 가장 성능 높은 서로 다른 타입의 모델 집합들 가지고 stacker 구현
        - 학습된 모델에 대한 probability threshold 최적화
