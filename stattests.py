import pandas as pd
import numpy as np
import abc

from scipy.stats import ttest_ind_from_stats, ttest_ind, mannwhitneyu, norm

import config as cfg


class EstimatorCriteriaValues:
    def __init__(self, pvalue: float, statistic: float):
        self.pvalue = pvalue
        self.statistic = statistic


class Statistics:
    def __init__(self, mean_0: float, mean_1: float, var_0: float, var_1: float, n_0: int, n_1: int, x=None, y=None):
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.var_0 = var_0
        self.var_1 = var_1
        self.n_0 = n_0
        self.n_1 = n_1
        self.x = x
        self.y = y


class MetricStats(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df) -> Statistics:
        pass


class Estimator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, Statistics) -> EstimatorCriteriaValues:
        pass


class BaseStatsRatio(MetricStats):

    def __call__(self, df) -> Statistics:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        n_0 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        n_1 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        mean_0 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        mean_1 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        var_0 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[0]].var()
        var_1 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[1]].var()

        return Statistics(mean_0, mean_1, var_0, var_1, n_0, n_1)




class MannStatsRatio(MetricStats):

    def __call__(self, df) -> Statistics:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        n_0 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        n_1 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        mean_0 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        mean_1 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        var_0 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[0]].var()
        var_1 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[1]].var()
        x = df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]] / df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]]
        y = df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]] / df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]]

        return Statistics(mean_0, mean_1, var_0, var_1, n_0, n_1, x, y)



class PropStatsRatio(MetricStats):

    def __call__(self, df) -> Statistics:
        _unique_variants = df[cfg.VARIANT_COL].unique()
        n_0 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        n_1 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        mean_0 = (sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])) / n_0
        mean_1 = (sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])) / n_1
        var_0 = mean_0 * (1- mean_0) / n_0
        var_1 = mean_1 * (1- mean_1) / n_1

        return Statistics(mean_0, mean_1, var_0, var_1, n_0, n_1)


class Linearization():

    def __call__(self, num_0, den_0, num_1, den_1):
        k = np.sum(num_0) / np.sum(den_0)
        l_0 = num_0 - k * den_0
        l_1 = num_1 - k * den_1
        return l_0, l_1


class TTestFromStats(Estimator):

    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = ttest_ind_from_stats(
                mean1=stat.mean_0,
                std1=np.sqrt(stat.var_0),
                nobs1=stat.n_0,
                mean2=stat.mean_1,
                std2=np.sqrt(stat.var_1),
                nobs2=stat.n_1
            )
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)


class MannWhitneyFromStats(Estimator):
    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = mannwhitneyu(
                stat.x,
                stat.y,
                alternative='two-sided'
            )
        except Exception as e:
            cfg.logger.error(e)
            pvalue, statistic = None, None

        return EstimatorCriteriaValues(pvalue, statistic)


class ZtestFromStats(Estimator):
    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            statistic = (stat.mean_0 - stat.mean_1)/np.sqrt(stat.var_0 + stat.var_1)
            pvalue = (1 - norm.cdf(abs(statistic)))

        except Exception as e:
            cfg.logger.error(e)
            pvalue, statistic = None, None

        return EstimatorCriteriaValues(pvalue, statistic)


def calculate_statistics(df, estimator):
    mappings = {
        "t_test": BaseStatsRatio(),
        "mann_whitney": MannStatsRatio(),
        "prop_test": PropStatsRatio()
    }

    calculate_method = mappings[estimator]

    return calculate_method(df)

def apply_condition(self, df, condition, field):
    comparison_field = condition.get('condition_field', '')
    comparison_value = condition.get('comparison_value', '')
    comparison_sign = condition.get('comparison_sign', '')

    if comparison_field and comparison_value and comparison_sign:
        if comparison_sign == 'equal':
            df = df[df[field + '_conditions'][comparison_field] == comparison_value]
        elif comparison_sign == 'not_equal':
            df = df[df[field + '_conditions'][comparison_field] != comparison_value]

    return df


def calculate_linearization(df):
    _variants = df[cfg.VARIANT_COL].unique()
    linearization = Linearization()

    df['l_ratio'] = 0
    if (df['den'] == df['n']).all():
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[0], 'num']
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[1], 'num']
    else:
        l_0, l_1 = linearization(
            df['num'][df[cfg.VARIANT_COL] == _variants[0]],
            df['den'][df[cfg.VARIANT_COL] == _variants[0]],
            df['num'][df[cfg.VARIANT_COL] == _variants[1]],
            df['den'][df[cfg.VARIANT_COL] == _variants[1]]
        )
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = l_0
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = l_1

    return df

