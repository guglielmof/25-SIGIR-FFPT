from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd


class LLM(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.pseudorelevants = kwargs["pseudorelevants"]

        self.name = "OpposingDocuments"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        #local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation
        prel_emb = self.pseudorelevants.loc[self.pseudorelevants.query_id == query.query_id, "representation"].values[0]
        #pnr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())

        pre_itx_vec = np.multiply(qemb, prel_emb)
        #pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pnr_embs)
        #importance = np.multiply(pre_itx_vec[np.newaxis, :], 1 / pnr_itx_mat)
        importance = pre_itx_vec
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
