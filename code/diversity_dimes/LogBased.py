from .AbstractDime import AbstractDime

from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


class AbstractLogBased(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log = kwargs["log"]


class UnbiasedLogBased(AbstractLogBased):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def querywise_compute_importance(self, query, *args, **kwargs):
        # We have 1000 simulations of clicks, each with its clicked documents. The user is not biased by the position
        local_log = self.log[self.log.query_id == query.query_id]
        local_log["weight"] = local_log["click"]

        docs = local_log[["doc_id", "representation", "weight"]].groupby("doc_id").mean()
        local_relevance = docs["weight"].to_list()

        dembs = np.array(docs["representation"].to_list())
        qemb = query.representation
        itx_mat = np.multiply(qemb[np.newaxis, :], dembs)
        # prel_emb = self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0]
        if not np.all(np.array(local_relevance) == local_relevance[0]):
            importance = corr2_coeff(itx_mat.T, np.array([local_relevance])).ravel()
        else:
            importance = np.mean(itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class BiasedLogBased(AbstractLogBased):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def querywise_compute_importance(self, query, *args, **kwargs):
        # We have 1000 simulations of clicks, each with its clicked documents. The user is not biased by the position
        local_log = self.log[self.log.query_id == query.query_id]
        local_log["weight"] = local_log["click"] / (1 / local_log["rank"])

        docs = local_log[["doc_id", "representation", "weight"]].groupby("doc_id").mean()
        local_relevance = docs["weight"].to_list()

        dembs = np.array(docs["representation"].to_list())
        qemb = query.representation
        itx_mat = np.multiply(qemb[np.newaxis, :], dembs)
        # prel_emb = self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0]
        if not np.all(np.array(local_relevance) == local_relevance[0]):
            importance = corr2_coeff(itx_mat.T, np.array([local_relevance])).ravel()
        else:
            importance = np.mean(itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class SimulatedPerfectUnbiasedLog(UnbiasedLogBased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedPerfectUnbiasedLog"


class SimulatedPerfectBiasedLog(BiasedLogBased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedPerfectBiasedLog"


class SimulatedNoisyUnbiasedLog(UnbiasedLogBased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedNoisyUnbiasedLog"


class SimulatedNoisyBiasedLog(BiasedLogBased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedNoisyBiasedLog"


class AbstractLogBasedOthers(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log = kwargs["log"]
        self.subtopics = kwargs["subtopics"]


class UnbiasedLogBasedOthers(AbstractLogBasedOthers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def querywise_compute_importance(self, query, *args, **kwargs):

        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]

        # We have 1000 simulations of clicks, each with its clicked documents. The user is not biased by the position
        local_log = self.log[self.log.query_id.isin(local_subtopics.query_id)]
        local_log["weight"] = local_log["click"]

        docs = local_log[["doc_id", "representation", "weight"]].groupby("doc_id").mean()
        local_relevance = docs["weight"].to_list()

        dembs = np.array(docs["representation"].to_list())
        qemb = query.representation
        itx_mat = np.multiply(qemb[np.newaxis, :], dembs)
        # prel_emb = self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0]
        if not np.all(np.array(local_relevance) == local_relevance[0]):
            importance = corr2_coeff(itx_mat.T, np.array([local_relevance])).ravel()
        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class BiasedLogBasedOthers(AbstractLogBasedOthers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def querywise_compute_importance(self, query, *args, **kwargs):

        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]

        # We have 1000 simulations of clicks, each with its clicked documents. The user is not biased by the position
        local_log = self.log[self.log.query_id.isin(local_subtopics.query_id)]
        local_log["weight"] = local_log["click"] / (1 / local_log["rank"])

        docs = local_log[["doc_id", "representation", "weight"]].groupby("doc_id").mean()
        local_relevance = docs["weight"].to_list()

        dembs = np.array(docs["representation"].to_list())
        qemb = query.representation
        itx_mat = np.multiply(qemb[np.newaxis, :], dembs)

        # prel_emb = self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0]
        if not np.all(np.array(local_relevance) == local_relevance[0]):
            importance = corr2_coeff(itx_mat.T, np.array([local_relevance])).ravel()
        else:
            importance = np.mean(itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class SimulatedPerfectUnbiasedLogOthers(UnbiasedLogBasedOthers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedPerfectUnbiasedLog"


class SimulatedPerfectBiasedLogOthers(BiasedLogBasedOthers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedPerfectBiasedLog"


class SimulatedNoisyUnbiasedLogOthers(UnbiasedLogBasedOthers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedNoisyUnbiasedLog"


class SimulatedNoisyBiasedLogOthers(BiasedLogBasedOthers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SimulatedNoisyBiasedLog"
