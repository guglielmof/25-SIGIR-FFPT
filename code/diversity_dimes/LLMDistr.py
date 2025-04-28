from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd
from scipy.stats import norm


class LLMDistr(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pseudorelevants = kwargs["pseudorelevants"]
        self.run = kwargs["run"]
        self.docs_encoder = kwargs["docs_encoder"]
        self.name = "LLMDistr"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        # local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation
        prel_emb = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0])
        # pnr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())

        local_run = self.run.loc[self.run.query_id == query.query_id].iloc[:100]

        pre_itx_vec = np.multiply(qemb, prel_emb)
        prdoc_emb = self.docs_encoder.get_encoding(local_run.doc_id.to_list())
        # pre_itx_vec = prel_emb

        pr_itxm = np.multiply(qemb[np.newaxis, :], prdoc_emb)

        means = np.mean(pr_itxm, axis=0)
        stds = np.std(pr_itxm, axis=0)

        importance = [norm(loc=means[i], scale=stds[i]).cdf(pre_itx_vec[i]) for i in range(len(means))]
        #importance = [np.exp(pre_itx_vec[i]) for i in range(len(means))]

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": importance})
