import pandas as pd
from dataclasses import dataclass


@dataclass
class CleanData:

    def __post_init__(self):
        self.report = dict()
        self.toolbox = CleanDataToolBox()

    def generic_cleaning_dataframe(self, data, list_columns_to_save, list_columns_to_delete, list_columns_to_keep_aside,
                                   drop_duplicates_rows=True, threshold_na_in_column=0.5, threshold_na_in_row=0.2,
                                   threshold_correlation=0.95, reset_index=True):

        if reset_index:
            data = data.reset_index(drop=True)

        # drop duplicates rows
        if drop_duplicates_rows:
            data = self.toolbox.drop_duplicates_row(data)

        # drop rows by checking Nan
        data = self.toolbox.drop_row_with_na(data, threshold_na_in_row)

        # save data from columns_to_save_out && delete data from columns_to_delete
        data, saved_data = self.save_and_drop_columns(data, list_columns_to_save, list_columns_to_delete,
                                                      list_columns_to_keep_aside)

        # drop columns by checking Nan & constants
        data = self.toolbox.drop_na_and_constants_columns(data, threshold_na_in_column)

        # drop columns by checking correlation
        data = self.toolbox.drop_correlated_data(data, threshold=threshold_correlation)

        return data, saved_data

    def save_and_drop_columns(self, data, list_columns_to_save, list_columns_to_delete, list_columns_to_keep_aside):

        # drop data to save
        data, kept_aside_data = self.toolbox.split_data_from_list(data, list_columns_to_keep_aside)

        # drop data to save
        if list_columns_to_save not in [None, [None], []]:
            __, data = self.toolbox.split_data_from_list(data, list_columns_to_save)

        # delete data
        if list_columns_to_save not in [None, [None], []]:
            data, __ = self.toolbox.split_data_from_list(data, list_columns_to_delete)

        return data, kept_aside_data


@dataclass
class CleanDataToolBox:

    # COLUMNS TREATMENT ############################################################################################

    @staticmethod
    def split_data_from_list(data, list_data_to_drop):

        if list_data_to_drop is not None:
            if type(list_data_to_drop) is str:
                list_data_to_drop = [list_data_to_drop]

            # split
            try:
                data_dropped = data[list_data_to_drop]
                data = data.drop(list_data_to_drop, axis=1)

                if data_dropped.shape[1] == 0:  # check data dropped
                    return data, pd.DataFrame()
                return data, data_dropped
            except KeyError:
                return data, pd.DataFrame()

        return data, pd.DataFrame()

    @staticmethod
    def drop_correlated_data(data, threshold=0.95):

        if threshold is None:
            return data

        column_to_drop = set()
        corr_matrix = abs(data.corr())

        # browse the lower part of the correlated matrix
        for index_col in range(len(corr_matrix.columns)):
            for index_row in range(index_col):
                if corr_matrix.iloc[index_col, index_row] >= threshold:
                    column_to_drop.add(corr_matrix.columns[index_col])
        return data.drop(column_to_drop, axis=1)

    def drop_na_and_constants_columns(self, data, threshold_na=0.5):

        set_drop = set()

        for column_name in data.columns:
            if self._check_na_rate_in_column(data[column_name], threshold_na):
                set_drop.add(column_name)
            if self._check_column_is_constant(data[column_name]):
                set_drop.add(column_name)

        return data.drop(set_drop, axis=1)

    @staticmethod
    def _check_na_rate_in_column(column, threshold):
        na_rate = column.isna().sum() / len(column)
        if na_rate >= threshold:
            return True
        return False

    @staticmethod
    def _check_column_is_constant(column):
        if len(list(column.value_counts())) <= 1:
            return True
        return False

    @staticmethod
    def get_modality_of_a_column(column):
        return list(pd.unique(column))

    # ROWS TREATMENT ############################################################################################

    @staticmethod
    def drop_duplicates_row(data):
        return data.drop_duplicates()

    #  TO TEST
    def drop_row_if_value_unwanted_in_a_column(self, data, column_name, set_value_unwanted):
        for value in self.get_modality_of_a_column(data[column_name]):
            if value in set_value_unwanted:
                data = data.drop([data[column_name] == value], axis=0)
        return data

    # TO TEST
    @staticmethod
    def drop_row_with_na(data, threshold=0.2):
        list_drop = []
        for index in range(data.shape[0]):
            list_null = list(data.iloc[index].isnull())
            if (list_null.count(True) / len(list_null)) >= threshold:
                list_drop.append(index)

        data = data.drop(list_drop, axis=0)
        return data.reset_index(drop=True)
