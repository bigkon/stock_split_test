from numpy import nan, inf, uint8
from pandas import DataFrame

from ..settings import MODEL_DAYS_PERIOD


class BaseModeling(object):

    def prepare_data(self, df: DataFrame, train: bool = True) -> DataFrame:
        """
        Preprocess raw data to normalize it
        """
        df['swing'] = ((df['high'] - df['open']) + (df['open'] - df['low'])) / df['open']
        df['diff'] = (df['close'] - df['open']) / df['open']
        for offset in range(1, MODEL_DAYS_PERIOD + 1):
            df_prev = df.shift(periods=offset)
            df_before_prev = df.shift(periods=offset + 1)
            if offset == 1:  # previous day data
                df['is_significant'] = (((df_prev['close'] - df['close']) / df_prev['close']).abs() > 0.3).astype(uint8)
                df['price_change_between_days'] = (df_prev['close'] - df['open']) / df_prev['close']
            df['swing_diff_{}'.format(-offset)] = df_prev['swing'] - df_before_prev['swing']
            df['diff_diff_{}'.format(-offset)] = df_prev['diff'] - df_before_prev['diff']
            df['volume_diff_{}'.format(-offset)] = (df_prev['volume'] - df_before_prev['volume']) / df_before_prev['volume']
            df['current_swing_diff_{}'.format(-offset)] = df['swing'] - df_prev['swing']
            df['current_diff_diff_{}'.format(-offset)] = df['diff'] - df_prev['diff']
            df['current_volume_diff_{}'.format(-offset)] = (df['volume'] - df_prev['volume']) / df_prev['volume']
            if train:
                mask = df_prev['symbol'] != df_before_prev['symbol']
                df.loc[mask, 'swing_diff_{}'.format(-offset)] = nan
                df.loc[mask, 'diff_diff_{}'.format(-offset)] = nan
                df.loc[mask, 'volume_diff_{}'.format(-offset)] = nan
                del df_prev, df_before_prev

        df.fillna(0, inplace=True)
        df['weekday'] = df['date'].dt.dayofweek
        df.replace([-inf, inf], 0, inplace=True)
        return df
