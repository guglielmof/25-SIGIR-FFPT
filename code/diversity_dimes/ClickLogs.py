from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.stats import spearmanr
import scipy.stats
from sklearn.preprocessing import normalize


def corr2_coeff(A, B, axis=1):
    if axis == 1:
        A = A.T
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return (np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))).ravel()


def mmc(A, B):
    u = B
    V = A

    xorder = np.argsort(u)
    usort = u[xorder]
    udist = usort[1:] - usort[:-1]
    Vsort = V[:, xorder]
    Vdist = Vsort[:, 1:] - Vsort[:, :-1]
    auc = np.sum(np.multiply(udist[np.newaxis, :], Vdist), axis=1)
    print(len(auc))
    return auc


class ClickLogs(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.name = "ClickLogs"

        def no_propensity(docs_ranks):
            docs_ranks["propensity"] = np.ones(len(docs_ranks))
            return docs_ranks

        self.propensity = no_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation
        '''
        prel_emb = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].to_list()[0])#.values[0]
        #pnr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())

        pre_itx_vec = np.multiply(qemb, prel_emb)
        #pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pnr_embs)
        #importance = np.multiply(pre_itx_vec[np.newaxis, :], 1 / pnr_itx_mat)
        importance = pre_itx_vec
        '''
        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        weighted_doc_rep = np.multiply(doc_repr, np.array(doc_weight.weight)[:, np.newaxis])
        itx_mat = np.multiply(qemb[np.newaxis, :], weighted_doc_rep)
        importance = np.mean(itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsPropensityCorrection(ClickLogs):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.name = "ClickLogsPropensityCorrection"
        self.eta = 1

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity


class CorrelationClickLogs(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.name = "ClickLogs"

        def no_propensity(docs_ranks):
            docs_ranks["propensity"] = np.ones(len(docs_ranks))
            return docs_ranks

        self.propensity = no_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]):
            importance = corr2_coeff(itx_mat, np.array([doc_weight.weight.to_list()]))
        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class CorrelationClickLogsPropensityCorrection(CorrelationClickLogs):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.name = "ClickLogsPropensityCorrection"
        self.eta = 1

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity


class DCClickLogs(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]):

            # pearson
            importance = corr2_coeff(itx_mat, np.array([doc_weight.weight.to_list()]))

            '''
            # mmc: it should allow to go beyond linear correlation
            # u = np.array(doc_weight.weight.to_list())
            # V = itx_mat
            # importance = mmc(V.T, u)
            '''

            '''
            #spearman
            res = spearmanr(itx_mat, np.array(doc_weight.weight.to_list()))
            importance = res.statistic[-1, :-1]
            '''

            '''
            means = np.mean(itx_mat, axis=0)
            stds = np.std(itx_mat, axis=0)
            p = np.array(doc_weight.weight.to_list())
            q = np.array([scipy.stats.norm(loc=means[i], scale=stds[i]).cdf(itx_mat[:, i]) for i in range(len(means))])[:, np.argwhere(p != 0)[0]]
            kldiv = np.multiply(np.log(p)[np.newaxis, :], np.log(np.multiply(p[np.newaxis, :], 1 / q)))
            importance = np.sum(kldiv, axis=1)
            '''

            '''
            witx_mat = np.multiply(itx_mat, (2*np.array(doc_weight.weight.to_list())-1)[:,np.newaxis])
            importance = np.max(witx_mat, axis=0)
            '''

            # importance = np.dot(normalize(itx_mat).T, normalize((2 * np.array(doc_weight.weight.to_list()) - 1).reshape(1, -1)).T)
            # importance = corr2_coeff(itx_mat+np.random.multivariate_normal(np.mean(axis=0), np.cov(itx_mat), size=itx_mat.shape), np.array([doc_weight.weight.to_list()]))

            '''
            disc = np.maximum(1, np.log2(np.arange(1, len(doc_weight) + 1)))

            idcg = sorted(doc_weight.weight.to_list(), reverse=True)
            idcg = np.sum(np.divide(idcg, disc))

            as_itx = np.argsort(-itx_mat, axis=0)
            as_erel = np.array(doc_weight.weight.to_list())[as_itx]  # (2*np.array(doc_weight.weight.to_list())-1)[as_itx]
            importance = np.sum(np.divide(as_erel, disc[:, np.newaxis]), axis=0)
            '''

            '''
            X = itx_mat
            X = normalize(X)
            X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
            y = np.array(doc_weight.weight.to_list())
            beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
            importance = beta[1:]
            '''
            '''
            X = itx_mat.T
            y = np.array(doc_weight.weight.to_list())

            X = np.stack([X, np.ones(X.shape)], axis=2)
            K = np.einsum('rdm,d->rm', np.einsum('rnm,ndm->rdm', np.linalg.inv(np.einsum('rnm,mnd->rdm', X.T, X).T).T, X.T), y)
            importance = K[0, :]
            
            '''
        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsCorrelation(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)

        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()

        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]) and np.any(doc_repr!=doc_repr[0, :]):

            # pearson
            importance = corr2_coeff(itx_mat, np.array([doc_weight.weight.to_list()]))

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsLinearModel(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]) and np.any(doc_repr!=doc_repr[0, :]):

            X = itx_mat.T
            y = np.array(doc_weight.weight.to_list())

            X = np.stack([X, np.ones(X.shape)], axis=2)
            K = np.einsum('rdm,d->rm', np.einsum('rnm,ndm->rdm', np.linalg.inv(np.einsum('rnm,mnd->rdm', X.T, X).T).T, X.T), y)
            importance = K[0, :]

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsWeightedMax(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]):

            witx_mat = np.multiply(itx_mat, (2 * np.array(doc_weight.weight.to_list()) - 1)[:, np.newaxis])
            importance = np.max(witx_mat, axis=0)

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsWeightedMean(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]):

            witx_mat = np.multiply(itx_mat, (2 * np.array(doc_weight.weight.to_list()) - 1)[:, np.newaxis])
            importance = np.mean(witx_mat, axis=0)

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsMean(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsLinearModelSignItx(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation
        qembsign = np.sign(qemb)
        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qembsign[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]):

            X = itx_mat.T
            y = np.array(doc_weight.weight.to_list())

            X = np.stack([X, np.ones(X.shape)], axis=2)
            K = np.einsum('rdm,d->rm', np.einsum('rnm,ndm->rdm', np.linalg.inv(np.einsum('rnm,mnd->rdm', X.T, X).T).T, X.T), y)
            importance = K[0, :]

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class ClickLogsLinearModelNoItx(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation
        qembsign = np.sign(qemb)
        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = doc_repr

        if np.any(doc_weight.weight != doc_weight.weight[0]) and np.any(doc_repr!=doc_repr[0, :]):

            X = itx_mat.T
            y = np.array(doc_weight.weight.to_list())

            X = np.stack([X, np.ones(X.shape)], axis=2)
            K = np.einsum('rdm,d->rm', np.einsum('rnm,ndm->rdm', np.linalg.inv(np.einsum('rnm,mnd->rdm', X.T, X).T).T, X.T), y)
            importance = K[0, :]

        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})



class ClickLogsSlope2(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicklog = kwargs["clicklog"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.eta = 1
        self.name = "ClickLogs"

        def known_propensity(docs_ranks):
            docs_ranks["propensity"] = (1 / docs_ranks["rank"]) ** self.eta
            return docs_ranks

        self.propensity = known_propensity

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_log = self.clicklog.loc[self.clicklog.query_id == query.query_id].reset_index(drop=True)
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        doc_weight = local_log[["doc_id", "rank"]].drop_duplicates()
        clickproba = local_log.groupby("doc_id")["click"].mean().reset_index().rename({"click": "p_click"}, axis=1)

        doc_weight = self.propensity(doc_weight).merge(clickproba)
        doc_weight["weight"] = doc_weight.p_click / doc_weight.propensity

        doc_repr = self.docs_encoder.get_encoding(doc_weight.doc_id.to_list())
        itx_mat = np.multiply(qemb[np.newaxis, :], doc_repr)

        if np.any(doc_weight.weight != doc_weight.weight[0]) and np.any(doc_repr!=doc_repr[0, :]):

            X = itx_mat.T
            y = np.array(doc_weight.weight.to_list())

            X = np.stack([X, np.ones(X.shape)], axis=2)
            K = np.einsum('rdm,d->rm', np.einsum('rnm,ndm->rdm', np.linalg.inv(np.einsum('rnm,mnd->rdm', X.T, X).T).T, X.T), y)
            slope = K[0, :]
            intercept = K[1, :]

            minX = np.min(itx_mat, axis=0)
            maxX = np.max(itx_mat, axis=0)
            importance = np.multiply(maxX, slope) - np.multiply(minX, slope)
        else:
            importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})

