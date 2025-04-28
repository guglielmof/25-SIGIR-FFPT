import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd
from scipy.stats import norm


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


class NegDime(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.qrels = kwargs["qrels"].copy()
        self.run = kwargs["run"]

        self.name = "NegDime"

    def querywise_compute_importance(self, query, *args, **kwargs):
        qemb = query.representation

        local_run = self.run.loc[self.run.query_id == query.query_id]

        annotated_run = local_run.merge(self.qrels, how="left").fillna(0)
        nrdoc_id = annotated_run.loc[annotated_run[annotated_run.relevance == 0]["score"].idxmax()].doc_id

        nrdoc_emb = self.docs_encoder.get_encoding(nrdoc_id)
        local_run = local_run[local_run.doc_id != nrdoc_id].iloc[:5]
        prdoc_emb = self.docs_encoder.get_encoding(local_run.doc_id.to_list())

        pr_itxm = np.multiply(qemb[np.newaxis, :], prdoc_emb)
        nr_itxv = np.multiply(qemb, nrdoc_emb)

        means = np.mean(pr_itxm, axis=0)
        stds = np.std(pr_itxm, axis=0)

        importance = [1 - norm(loc=means[i], scale=stds[i]).cdf(nr_itxv[i]) for i in range(len(means))]

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": importance})
