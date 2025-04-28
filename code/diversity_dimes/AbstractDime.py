import pandas as pd
from multiprocessing import Pool
import numpy as np
import sys


class AbstractDime:

    def __init__(self, *args, **kwargs):
        self.d = None
        self.workers = None
        if "workers" in kwargs and type(kwargs["workers"]) == int and kwargs["workers"] > 1:
            self.workers = kwargs["workers"]

    def compute_importance(self, queries: pd.DataFrame) -> pd.DataFrame:
        if self.workers is not None:
            return self._parallel_compute_importance(queries)
        else:
            return self._compute_importance(queries)

    def _compute_importance(self, queries: pd.DataFrame) -> pd.DataFrame:
        """
        :param queries: it must be a pandas dataframe with (at least) two fields, <query_id, representation>. query_representation must
                        be a np.array
        :return: a pandas dataframe with three columns, query_id, dimension, importance
        """

        self.d = len(queries.representation.values[0])
        out = list(queries.apply(self.querywise_compute_importance, axis=1))
        importance = pd.concat(out)

        return importance

    def _parallel_compute_importance(self, queries: pd.DataFrame) -> pd.DataFrame:
        """
        Parallel version of compute_importance
        :param queries: it must be a pandas dataframe with (at least) two fields, <query_id, representation>. query_representation must
                        be a np.array
        :return:  a pandas dataframe with three columns, query_id, dimension, importance
        """
        with Pool() as pool:
            importance = pd.concat(pool.map(self._compute_importance, np.array_split(queries, self.workers)))

        return importance

    def querywise_compute_importance(self, query: pd.Series) -> pd.DataFrame:
        """
        :param query: series with two fields, query_id and representation
        :return: a pandas dataframe with three columns, query_id, dimension, importance
        """
        raise NotImplementedError
