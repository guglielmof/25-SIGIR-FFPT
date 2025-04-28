from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd


class LLMOthers(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.pseudorelevants = kwargs["pseudorelevants"]

        self.name = "LLMOthers"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        qemb = query.representation
        pr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())
        pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pr_embs)

        importance = np.mean(pnr_itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class LLMOthersFirst(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.pseudorelevants = kwargs["pseudorelevants"]

        self.name = "LLMOthersFirst"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        local_subtopics = local_subtopics[local_subtopics.subtopic_id == local_subtopics.subtopic_id.min()]

        qemb = query.representation
        pr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())
        pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pr_embs)

        importance = np.mean(pnr_itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})



class LLMOthersPrevious(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.pseudorelevants = kwargs["pseudorelevants"]

        self.name = "LLMOthersPrevious"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        local_subtopics = local_subtopics[local_subtopics.subtopic_id == local_subtopics.subtopic_id.max()]

        qemb = query.representation
        pr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())
        pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pr_embs)

        importance = np.mean(pnr_itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})



class LLMOthersAllPrevious(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]
        self.pseudorelevants = kwargs["pseudorelevants"]

        self.name = "LLMOthersAllPrevious"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # we want to remove those dimensions where the query interacts more with pseudo-relevants
        # for other intents
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        qemb = query.representation
        pr_embs = np.array(self.pseudorelevants.loc[self.pseudorelevants.query_id.isin(local_subtopics.query_id), "representation"].to_list())
        pnr_itx_mat = np.multiply(qemb[np.newaxis, :], pr_embs)

        importance = np.mean(pnr_itx_mat, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
