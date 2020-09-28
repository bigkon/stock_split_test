import pickle
from logging import getLogger

from numpy import uint8
from xgboost import XGBClassifier
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..settings import STOCKS_CSV, TRANSFORMER_LOCATION, MODEL_LOCATION, TRAIN_ITERATIONS
from ._base import BaseModeling


L = getLogger(__name__)


class ModelCreator(BaseModeling):
    """
    Uses stocks.csv and splits.csv data to create model, which will be used for predictions
    """

    def get_raw_data(self) -> DataFrame:
        """
        Read stocks data from csv file, join it with splits data and normalize
        """
        data = read_csv(STOCKS_CSV, parse_dates=['date'])
        data = data.loc[data['date'] < '2020-01-01']
        data['split'] = (data['splitFactor'] != 1).astype(uint8)
        data.drop(columns=['splitFactor'], inplace=True)
        return data

    def train_model(self):
        """Use normalized data to create and train prediction model"""
        L.info('Reading and preparing data')
        data = self.prepare_data(self.get_raw_data())
        L.info('Read data finished')
        y = data['split'].values
        data.drop(columns=['split', 'date', 'symbol', 'open', 'close', 'low', 'high', 'volume'], inplace=True)
        transformer = ColumnTransformer([
            ('weekday', OneHotEncoder(categories=[list(range(7))]), ['weekday']),
            ('scale', StandardScaler(), make_column_selector(pattern='_diff'))
        ], remainder='passthrough')
        L.info('Normalizing data')
        weight = sum(y == 0) / y.sum()
        data = transformer.fit_transform(data)
        L.info('Normalize finished')
        with open(TRANSFORMER_LOCATION, 'wb') as f:
            pickle.dump(transformer, f)
        L.info('Training model, weight %s', weight)
        results = []
        for attempt in range(1, TRAIN_ITERATIONS + 1):
            train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.33, random_state=attempt, stratify=y)
            model = XGBClassifier(random_state=0, scale_pos_weight=weight)
            model.fit(train_x, train_y)
            predicted = model.predict(test_x)
            accuracy = accuracy_score(test_y, predicted)
            precision = precision_score(test_y, predicted)
            recall = recall_score(test_y, predicted)
            f1 = f1_score(test_y, predicted)
            L.debug('Train attempt #%s: accuracy %s, precision %s, recall %s, f1 score %s',
                    attempt, accuracy, precision, recall, f1)
            results.append((model, accuracy, precision, recall, f1))
        best_model, accuracy, precision, recall, f1 = max(results, key=lambda x: x[4])
        L.info('Select best model fit with stats: accuracy %s, precision %s, recall %s, f1 score %s',
               accuracy, precision, recall, f1)
        with open(MODEL_LOCATION, 'wb') as f:
            pickle.dump(best_model, f)
        L.info('Model trained and saved')
