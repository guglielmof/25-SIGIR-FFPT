from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd


class EntityFeedback(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.entities = kwargs["entities"]
        self.top_perc = 0.2

        self.name = "EntityFeedback"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        #local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        qemb = query.representation

        local_entities = self.entities.loc[self.entities.query_id == query.query_id]
        local_entities = local_entities.sort_values("expected_relevance", ascending=False)
        print(len(local_entities))
        local_entities = local_entities.iloc[:max(2, int(len(local_entities)*self.top_perc))]
        print(len(local_entities))
        prel_emb = np.array(local_entities.representation.to_list())
        #pnr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())

        pre_itx_vec = np.multiply(qemb[np.newaxis,:], prel_emb)
        #pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pnr_embs)
        #importance = np.multiply(pre_itx_vec[np.newaxis, :], 1 / pnr_itx_mat)
        importance = np.mean(pre_itx_vec, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
