
# 마지막에, sns.heatmap(rf_confusion_matrix) 로 결과 확인하기

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from pycaret.classification import *
from pycaret.utils import check_metric


class ModelMaker: # 빠른 테스트를 위함 (scaled data 도 일반 표준화된 데이터만 사용)
    def __init__(self):
        self.pd_data = None

        self.data = None # labeled data

        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None

        self.tree_based_models = None
        self.final_model_stacker =None
        self.final_model_blender = None

    def load_data(self):
        self.pd_data = pd.read_csv('C:/Users/Dae-Young Park/PycharmProjects/MaliciousWebsiteDetection/final_raw_data_SimpleImputer.csv')

        #print(self.pd_data.head())
        
    def apply_PCA_to_pd_data(self):
    pca = PCA(n_components=20) # 대충 15개로 넣어봄 (10일때, f1 맥스 0.8491 (심지어, 그냥 튜닝 버전이 젤 잘나옴) )

    pd_data_categoricals =  self.pd_data[ self.pd_data.columns[14:] ]

    reducted_pd_data_categoricals = pca.fit_transform(pd_data_categoricals)

    #print('reducted_pd_data_categoricals: ')
    reducted_pd_data_categoricals = pd.DataFrame(reducted_pd_data_categoricals)
    #print(reducted_pd_data_categoricals.head())

    self.pd_data = pd.concat( [self.pd_data[ self.pd_data.columns[:14] ] , reducted_pd_data_categoricals],axis=1 )

    #print('self.pd_data: ')
    #print(self.pd_data.info())
    # 결과적으로, 1781 * 33 이 인풋

    def split_data(self):
        y = self.pd_data['Type']
        X = self.pd_data.drop('Type', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, stratify=y)

        #X_train = pd.DataFrame( X_train.to_numpy() )
        #X_test = pd.DataFrame(X_test.to_numpy())

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print('self.X_test: ')
        print(self.X_test.head())
        print()

    def prepare_models(self):
        training_data = pd.concat([self.X_train, self.y_train], axis=1)

        s = setup(training_data, target='Type', train_size=0.7,
                  fold_strategy='stratifiedkfold', n_jobs=2, fix_imbalance=True,
                  feature_selection=True, feature_interaction=False,
                  feature_selection_threshold=0.5, interaction_threshold=0.01)


    def compare_models(self):
        tree_based_models = compare_models(include=['dt', 'rf', 'gbc', 'xgboost'] , sort='F1', n_select=4) #xgboost 가 가장 잘나오긴하는군
        print('tree_based_models: ')
        print()
        print(tree_based_models)
        print()

        self.tree_based_models = tree_based_models

    def predict_and_evaluate(self):
        print('- - - - - - - - - - - - - - - - ')
        print('4개 모델에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')
        print()

        for each_model in self.tree_based_models:
            print('each model: ')
            print(each_model)
            print()
            final_model = finalize_model(each_model)
            prediction_result = predict_model(final_model, data=self.X_test)

            eval_f1 = check_metric( self.y_test, prediction_result['Label'] ,metric='F1')
            print('F1: ', eval_f1)
            eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
            print('Prec: ', eval_prec)
            eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
            print('Recall: ', eval_re)
            print()
            print()

        print()
        print()

    def plot_models(self):
        plot_model(self.tree_based_models)

    def create_predict_evaluate_XGBoost(self):
        xgboost = create_model(estimator='xgboost')
        print()
        print('기본 xgboost: ')
        print(xgboost)
        print()
        final_model = finalize_model(xgboost)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

        tuned_xgboost = tune_model(xgboost, n_iter=15, optimize='F1', search_library='optuna', search_algorithm='tpe',
                                   choose_better=True)
        optimized_tuned_xgboost = optimize_threshold(tuned_xgboost, optimize='F1')
        print()
        print('튜닝된 xgboost: ')
        print(optimized_tuned_xgboost)
        print()
        final_model = finalize_model(optimized_tuned_xgboost)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

        ensemble_xgboost_bagging =  ensemble_model(optimized_tuned_xgboost, method='Bagging', optimize='F1', choose_better=False)
        ensemble_xgboost_bagging = optimize_threshold(ensemble_xgboost_bagging, optimize='F1')
        print()
        print('배깅 버전 xgboost: ')
        print(ensemble_xgboost_bagging)
        print()
        final_model = finalize_model(ensemble_xgboost_bagging)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

        blender = blend_models([tuned_xgboost, optimized_tuned_xgboost, ensemble_xgboost_bagging],
                               method='auto', choose_better=True, optimize='F1',
                               probability_threshold=0.5)  # auto면 soft voting이 기본, 안될때 hard voting
        best_blender = optimize_threshold(blender, optimize='F1')
        print()
        print('블랜딩 버전 xgboost: ')
        print(best_blender)
        print()
        final_model = finalize_model(best_blender)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

        stacker = stack_models([tuned_xgboost, optimized_tuned_xgboost, ensemble_xgboost_bagging],
                               meta_model=None, method='auto', choose_better=True, optimize='F1',
                               probability_threshold=0.5)
        best_stacker = optimize_threshold(stacker, optimize='F1')
        print()
        print('스태킹 버전 xgboost: ')
        print(best_stacker)
        print()
        final_model = finalize_model(best_stacker)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

    def predict_and_evaluate_four_models(self):
        print('- - - - - - - - - - - - - - - - ')
        print('4개 모델에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')
        print()

        for each_model in self.tree_based_models:
            print('each model: ')
            print(each_model)
            print()
            final_model = finalize_model(each_model)
            prediction_result = predict_model(final_model, data=self.X_test)

            eval_f1 = check_metric( self.y_test, prediction_result['Label'] ,metric='F1')
            print('F1: ', eval_f1)
            eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
            print('Prec: ', eval_prec)
            eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
            print('Recall: ', eval_re)
            print()
            print()

        print()
        print()

    def construct_improved_model(self):
        xgboost = self.tree_based_models[0]
        gbc = self.tree_based_models[1]

        print('1등 모델: ')
        print(xgboost)
        print()

        print('2등 모델: ')
        print(gbc)
        print()

        tuned_xgboost = tune_model(xgboost, n_iter=10, optimize='F1', search_library='optuna', search_algorithm='tpe',
                              choose_better=True)
        best_tuned_xgboost = optimize_threshold(tuned_xgboost, optimize='F1')

        tuned_gbc = tune_model(gbc, n_iter=10, optimize='F1', search_library='optuna', search_algorithm='tpe',
                                   choose_better=True)
        best_tuned_gbc = optimize_threshold(tuned_gbc, optimize='F1')

        stacker = stack_models([xgboost,tuned_xgboost, best_tuned_xgboost],
                               meta_model=None, method='auto', choose_better=True, optimize='F1',
                               probability_threshold=0.5)

        # prob_thres 다시 최적화
        best_stacker = optimize_threshold(stacker, optimize='F1')

        print('1등 모델을 이용해서 Stacker 구현: ')
        print(best_stacker)
        print()
        final_model_stacker = finalize_model(best_stacker)
        self.final_model_stacker = final_model_stacker

        blender = blend_models([xgboost, tuned_xgboost, best_tuned_xgboost],
                               method='auto', choose_better=True, optimize='F1',
                               probability_threshold=0.5) # auto면 soft voting이 기본, 안될때 hard voting
        best_blender = optimize_threshold(blender, optimize='F1')

        print('1등 모델을 이용해서 Blender 구현: ')
        print(best_blender)
        print()
        final_model_blender = finalize_model(best_blender)
        self.final_model_blender = final_model_blender



    def predict_and_evaluate_final_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('최종 모델(Stacker 앙상블)에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')
        print()

        final_model = finalize_model(self.final_model_stacker)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()

        print('- - - - - - - - - - - - - - - - ')
        print('최종 모델(Blender 앙상블)에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')
        print()

        final_model = finalize_model(self.final_model_blender)
        prediction_result = predict_model(final_model, data=self.X_test)

        eval_f1 = check_metric(self.y_test, prediction_result['Label'], metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)
        print()
        print()
