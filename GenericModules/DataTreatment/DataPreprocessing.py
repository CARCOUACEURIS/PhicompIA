import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, coo_matrix

from dataclasses import dataclass, field, InitVar
from LibraryModelsAndObjects.PreprocessingObjects import LoadPreprocessorObjects


@dataclass
class DataPreprocessing:
    # INIT ##############################################################################################

    scaler_name: InitVar[str] = 'robust'
    x_encoder_name: InitVar[str] = 'onehot'
    impute_num_name: InitVar[str] = 'knn'
    impute_cat_name: InitVar[str] = 'simple'

    y_encoder_name: InitVar[str] = 'label'

    imputer_missing_num: object = field(init=False)
    imputer_missing_cat: object = field(init=False)

    _num_features = make_column_selector(dtype_include=np.number)
    _cat_features = make_column_selector(dtype_exclude=np.number)

    def __post_init__(self, scaler_name, x_encoder_name, impute_num_name, impute_cat_name, y_encoder_name):

        loader = LoadPreprocessorObjects()
        self.imputer_missing_num = loader.load_imputer(impute_num_name)
        self.imputer_missing_cat = loader.load_imputer(impute_cat_name)

        self.scaler = loader.load_scaler(scaler_name)
        self.x_encoder = loader.load_encoder(x_encoder_name)

        self.y_encoder = loader.load_encoder(y_encoder_name)

    def _create_numerical_preprocessor(self):
        numerical_pipeline = make_pipeline(self.imputer_missing_num, self.scaler)
        return make_column_transformer((numerical_pipeline, self._num_features))

    def _create_categorical_preprocessor(self):
        categorical_pipeline = make_pipeline(self.imputer_missing_cat, self.x_encoder)
        return make_column_transformer((categorical_pipeline, self._cat_features))

    def preprocessing_data_into_train_test_set(self, x_data, y_data,test_size, rs, for_classification, format_outfile):

        x_train, y_train, x_test, y_test = self.train_test_split_data(x_data, y_data, test_size, rs, for_classification)

        x_train, y_train, num_prep, cat_prep, encoder_y = self.preprocessing_dataset(x_train, y_train, format_outfile,
                                                                                     already_fitted=False)

        x_test, y_test, _, _, _ = self.preprocessing_dataset(x_test, y_test, format_outfile, num_prep, cat_prep,
                                                             encoder_y, True)

        return (x_train, y_train), (x_test, y_test), num_prep, cat_prep, encoder_y

    # TRAIN TEST SPLIT ##############################################################################################

    @staticmethod
    def train_test_split_data(x_data, y_data, test_size, random_state=42, for_classification=True):
        stratify = None
        if for_classification:
            stratify = y_data

        return train_test_split(x_data, y_data, test_size=test_size, random_state=random_state, stratify=stratify)

    # PREPROCESSING ##############################################################################################

    def preprocessing_dataset(self, x_data, y_data=None,
                              format_outfile='array', numerical_preprocessor=None,
                              categorical_preprocessor=None, encoder_y_data=None,
                              already_fitted=False):

        # chose preprocessor
        if numerical_preprocessor is None:
            numerical_preprocessor = self._create_numerical_preprocessor()

        if categorical_preprocessor is None:
            categorical_preprocessor = self._create_categorical_preprocessor()

        # chose transforming method => then transform data
        if already_fitted:
            x_num = numerical_preprocessor.fit_transform(x_data)
            x_cat = categorical_preprocessor.fit_transform(x_data)
        else:
            x_num = numerical_preprocessor.transform(x_data)
            x_cat = categorical_preprocessor.transform(x_data)

        # y_data
        if y_data is not None:
            y_data = self.preprocessing_y_data(y_data, encoder_y_data, already_fitted, format_outfile)

        # out file format
        if format_outfile is 'dataframe':
            x_data = pd.DataFrame(hstack([coo_matrix(x_num), coo_matrix(x_cat)],format='array'))

        else:
            x_data = hstack([coo_matrix(x_num), coo_matrix(x_cat)], format=format_outfile)

        return x_data, y_data, numerical_preprocessor, categorical_preprocessor, encoder_y_data

    def preprocessing_y_data(self, y_data, encoder_y_data=None, already_fitted=True, format_outfile=None):

        # modify data
        y_data = np.ravel(y_data)

        # chose encoder
        if encoder_y_data is not None:
            encoder = encoder_y_data
        else:
            encoder = self.y_encoder

        # encode data
        if encoder is not None:
            if already_fitted:
                y_data = encoder_y_data.transform(y_data)
            else:
                y_data = encoder_y_data.fit_transform(y_data)

        # return data
        if format_outfile.lower() == 'dataframe':
            return pd.DataFrame(y_data)
        if format_outfile.lower() == 'series':
            return pd.Series(y_data)
        return y_data
