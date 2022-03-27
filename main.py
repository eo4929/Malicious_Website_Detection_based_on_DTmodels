
from data_exploration import DataAnalyzer
from data_transformation import DataPreprocessor, FeatureEngineer
from model_construction import ModelMaker


if __name__ == '__main__':
    """data_analyzer = DataAnalyzer()
    data_analyzer.show_statistics()
    data_analyzer.show_label_info()
    data_analyzer.show_correlation()"""

    """data_preprocessor = DataPreprocessor()
    data_preprocessor.remove_unnecessary_features()
    data_preprocessor.change_missing_value()
    data_preprocessor.encode_object_data()
    data_preprocessor.impute_missing_value(imputation_method='simple')

    data_preprocessor.save_final_raw_data(imputation_method='simple')"""

    models = ModelMaker()
    models.load_data()
    models.split_data()
    models.prepare_models()
    models.compare_models()
    models.predict_and_evaluate_four_models()

    models.construct_improved_model()
    models.predict_and_evaluate_final_model()
    #models.plot_models()
    #models.predict_and_evaluate()