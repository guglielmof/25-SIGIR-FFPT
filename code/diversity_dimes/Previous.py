import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd


class Previous(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ref_importance = kwargs["ref_importance"]
        self.subtopics = kwargs["subtopics"]
        self.name = "Previous"

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id != query.query_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        local_subtopics = local_subtopics[local_subtopics.subtopic_id == local_subtopics.subtopic_id.max()]

        importance_matrix = np.zeros((len(local_subtopics), self.ref_importance.dimension.max() + 1))
        small_importance = self.ref_importance[self.ref_importance.query_id.isin(local_subtopics.query_id)]
        for i in range(len(local_subtopics)):
            stid = local_subtopics.iloc[i]["query_id"]
            local_importance = small_importance.loc[small_importance.query_id == stid]
            dims_idx = local_importance.dimension.to_list()
            dims_imp = local_importance.importance.to_list()
            importance_matrix[i, dims_idx] = dims_imp

        importance = np.mean(importance_matrix, axis=0)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})