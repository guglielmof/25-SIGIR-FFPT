from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd


class StringBasedFirst(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]

        self.name = "StringBasedFirst"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # step 1 take the representation of all the subtopics for the same topic
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        local_subtopics = local_subtopics[local_subtopics.subtopic_id == local_subtopics.subtopic_id.min()]

        centroid = np.array(local_subtopics.representation.to_list())
        centroid = np.mean(centroid, axis=0)
        importance = np.multiply(query.representation, centroid)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class StringBasedPrevious(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]

        self.name = "StringBasedPrevious"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # step 1 take the representation of all the subtopics for the same topic
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]
        local_subtopics = local_subtopics[local_subtopics.subtopic_id == local_subtopics.subtopic_id.max()]

        centroid = np.array(local_subtopics.representation.to_list())
        centroid = np.mean(centroid, axis=0)
        importance = np.multiply(query.representation, centroid)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})


class StringBasedAllPrevious(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]

        self.name = "StringBasedAllPrevious"

    def querywise_compute_importance(self, query, *args, **kwargs):
        # step 1 take the representation of all the subtopics for the same topic
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id)]
        local_subtopics.subtopic_id = local_subtopics.subtopic_id.astype(int)
        local_subtopics = local_subtopics[local_subtopics.subtopic_id < int(query.subtopic_id)]

        centroid = np.array(local_subtopics.representation.to_list())
        centroid = np.mean(centroid, axis=0)
        importance = np.multiply(query.representation, centroid)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
