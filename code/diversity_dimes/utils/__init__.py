import numpy as np
import pandas as pd
from .crossvalidation import crossvalidation

def get_masked_encoding(queries: pd.DataFrame, importance: pd.DataFrame, alpha: float) -> np.array:
    """
    This function takes the queries and constructs a masked representation based on the importance with a cutoff alpha
    :param queries:
    :param importance:
    :param alpha:
    :return:
    """
    qembs = np.array(queries.representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])

    rep_size = len(queries.representation.values[0])
    n_dims = int(np.round(alpha * rep_size))

    importance["drank"] = importance.groupby("query_id")["importance"].rank(method="first", ascending=False).astype(int)
    selected_dims = importance.loc[importance["drank"] <= n_dims][["query_id", "dimension"]]

    tmp = selected_dims.merge(q2r)
    rows = np.array(tmp["row"])
    cols = np.array(tmp["dimension"])

    mask = np.zeros_like(qembs)
    mask[rows, cols] = 1
    enc_queries = np.where(mask, qembs, 0)

    return enc_queries, q2r


def get_soft_encoding(queries, importance):
    qembs = np.array(queries.representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])

    rep_size = len(queries.representation.values[0])
    importance["drank"] = importance.groupby("query_id")["importance"].rank(method="first", ascending=False).astype(int)

    soft_importance = np.zeros((len(q2r), rep_size))

    imp_values = np.linspace(0, 1, rep_size)[::-1]
    for row, v in q2r.iterrows():
        i, q = v
        # soft_importance[i, importance.loc[importance.query_id==q, "dimension"].to_list()] = importance.loc[importance.query_id==q, "drank"].to_list()
        dims = importance.loc[importance.query_id == q, "dimension"].to_list()
        imp_order = (importance.loc[importance.query_id == q, "drank"]-1).to_list()
        soft_importance[i, dims] = imp_values[imp_order]

    return np.multiply(qembs, soft_importance), q2r


def get_masked_encoding_v2(queries: pd.DataFrame, importance: pd.DataFrame) -> np.array:
    """
    This function takes the queries and constructs a masked representation based on the importance with a cutoff alpha
    :param queries:
    :param importance:
    :param alpha:
    :return:
    """
    qembs = np.array(queries.representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])

    selected_dims = importance.loc[importance["keep"]][["query_id", "dimension"]]

    tmp = selected_dims.merge(q2r)
    rows = np.array(tmp["row"])
    cols = np.array(tmp["dimension"])

    mask = np.zeros_like(qembs)
    mask[rows, cols] = 1
    enc_queries = np.where(mask, qembs, 0)

    return enc_queries, q2r
