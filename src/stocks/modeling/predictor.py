import pickle
import warnings
from logging import getLogger

from numpy import array
from pandas import DataFrame
from pandas.core.common import SettingWithCopyWarning

from ..settings import MODEL_LOCATION, TRANSFORMER_LOCATION
from ._base import BaseModeling


warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
L = getLogger(__name__)


class Predictor(BaseModeling):

    @property
    def model(self):
        try:
            with open(MODEL_LOCATION, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise Exception('Model does not exists. Make sure it was prepared.') from e
        except Exception as e:
            raise Exception('Unknown error while loading model') from e

    @property
    def transformer(self):
        try:
            with open(TRANSFORMER_LOCATION, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise Exception('Transformer does not exists. Make sure model was prepared.') from e
        except Exception as e:
            raise Exception('Unknown error while loading transformer') from e

    def predict(self, data: DataFrame) -> array:
        cols = ['date', 'open', 'close', 'low', 'high', 'volume']
        df = data[cols]
        model = self.model
        transformer = self.transformer
        try:
            df = self.prepare_data(df, train=False)
        except Exception as e:
            raise Exception('Failed to normalize data') from e
        df = df.drop(columns=cols)
        data = transformer.transform(df)
        results = model.predict(data)
        L.info('Processed %s rows, %s splits predicted', len(data), results.sum())
        return results
