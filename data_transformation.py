import pandas as pd
import numpy as np
#import pandas_profiling

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

class DataPreprocessor:
    def __init__(self):
        self.raw_data = None
        self.imputed_raw_data = None
        self.pd_scaled_data = None
        self.pd_scaled_data_list = []
        self.load_raw_data()

    def load_raw_data(self):
        self.raw_data = pd.read_csv('C:/Users/Dae-Young Park/Desktop/AI스터디/연습용데이터셋/dataset_website.csv')

    def remove_unnecessary_features(self):
        self.raw_data = self.raw_data.drop(['URL'], axis = 1)

    def remove_incorrect_data(self):
        # 레이블이 NaN을 포함한 row 에 대해선 제거
        pass
    
    def change_missing_value(self):
        # None, NAN -> -1 로 바꾸어두기
        #self.raw_data = self.raw_data.replace( {'CONTENT_LENGTH': np.nan} , {'CONTENT_LENGTH': -1} )
        self.raw_data = self.raw_data.replace({'WHOIS_REGDATE': 'None'}, {'WHOIS_REGDATE': -1})
        self.raw_data = self.raw_data.replace({'WHOIS_UPDATED_DATE': 'None'}, {'WHOIS_UPDATED_DATE': -1})

    def encode_object_data(self):
        """col_CHARSET_ENCODED = LabelEncoder.fit_transform( self.raw_data['CHARSET'] )
        col_SERVER_ENCODED = LabelEncoder.fit_transform(self.raw_data['SERVER'])
        col_WHOIS_REGDATE_ENCODED = LabelEncoder.fit_transform(self.raw_data['WHOIS_REGDATE'])
        col_WHOIS_UPDATED_DATE_ENCODED = LabelEncoder.fit_transform(self.raw_data['WHOIS_UPDATED_DATE'])"""

        data_string_cols = self.raw_data[['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']]
        #print(data_string_cols.info())

        data_string_cols_with_dummies = pd.get_dummies(data_string_cols, prefix_sep='--') # sklearn 으로 one-hot하는 것보다 더미 변수로 만드는게 더 편하네
        #print(data_string_cols_with_dummies)

        data_without_string_cols = self.raw_data.drop(['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis = 1)

        self.raw_data = pd.concat([data_without_string_cols, data_string_cols_with_dummies], axis=1)
        self.raw_data.to_csv('final_raw_data_before_impute.csv')
        print(self.raw_data)


    def impute_missing_value(self, imputation_method = 'IterativeImputer'): # CONTENT_LENGTH의 -1들
        if imputation_method == 'simple':
            imputer = SimpleImputer(strategy = "mean")
        elif imputation_method == 'IterativeImputer': # 이걸로 돌리면 성능 높아질 수 있음
            imputer = IterativeImputer(missing_values = np.nan) # 파라미터 나중에 자세히 보기

        self.imputed_raw_data = imputer.fit_transform( self.raw_data )
        self.imputed_raw_data = pd.DataFrame(self.imputed_raw_data)
        print(self.imputed_raw_data)


    def save_final_raw_data(self , imputation_method = 'IterativeImputer'):
        if imputation_method == 'simple':
            self.imputed_raw_data.to_csv('final_raw_data_SimpleImputer.csv') # 메뉴얼하게 칼럼명 수정함
        elif imputation_method == 'IterativeImputer':
            self.imputed_raw_data.to_csv('final_raw_data_SimpleImputer.csv')

    def show_data_profile(self):
        #profile = ProfileReport(self.pd_data, title="data profile")
        #print(profile)
        #profile.to_file("data_profile_report.html")
        #print(self.pd_data.dtypes)
        pr = self.pd_data.profile_report()
        print(pr)

    def make_scaled_data(self):
        standard = StandardScaler()
        standard.fit(self.pd_data)

        self.pd_scaled_data = standard.transform(self.pd_data)

    def put_cleaned_data(self):
        return self.pd_scaled_data


class FeatureEngineer:
    def __init__(self):
        self.pd_data = None # 정제중인 데이터
        self.load_raw_data()

    def load_raw_data(self):
        self.raw_data = pd.read_csv('C:/Users/Dae-Young Park/Desktop/AI스터디/연습용데이터셋/dataset_PEmalwares.csv')

    def check_anova(self): # feature selection 을 위한 ANOVA 검정
        None