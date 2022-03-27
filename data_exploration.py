import pandas as pd
#import pandas_profiling
import numpy as np

from scipy.stats import kstest

import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self):
        self.raw_data = None
        self.load_raw_data()

        print('[raw data 칼럼 정보]')
        print(self.raw_data.info())
        print()

        self.meaningful_raw_data = None # 의미없는 칼럼 제거

    def load_raw_data(self):
        self.raw_data = pd.read_csv('C:/Users/Dae-Young Park/Desktop/AI스터디/연습용데이터셋/dataset_website.csv')


    def show_statistics(self):
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        print('[raw data 통계 정보 (연속형 변수만)]')
        print( self.raw_data.describe() )
        print()

    def show_label_info(self):
        print('[레이블 정보]')
        print(self.raw_data['Type'].value_counts())
        print()

    def show_correlation(self):
        #self.meaningful_raw_data = self.raw_data.drop(['Name', 'Malware'], axis = 1)

        print('[컬럼별 상관관계 정보]')
        print(self.raw_data.corr())
        print()

    def show_correlation_label_attribute(self):
        print('[레이블과 다른 속성들 사이의 상관관계 (연속형 변수만)]')
        corr_matrix = self.raw_data.corr()
        print(corr_matrix["Malware"].sort_values(ascending=False))
        print()
        # 결과 확인: 생각보다 특정 feature에 의해 영향받는건 없네..?

    def check_normality_test(self):
        # Kolmogorove-Smirnov test: 표본이 2000개 이상인 데이터셋에 적용하는 정규성 검정
        # 귀무가설: 데이터셋이 정규분포를 따름

        print('[feature 별 정규성 검정]')
        for col, item in self.meaningful_raw_data.iteritems():
            print( kstest(self.meaningful_raw_data[col],'norm') )

        print()
        # 만약, 특정 모델이 정규분포를 따라야 성능 좋다고 한다면, 그 feature에 대해선, 분포 transform 해야만 함 (선형 회귀모델이 대표적인 예)


