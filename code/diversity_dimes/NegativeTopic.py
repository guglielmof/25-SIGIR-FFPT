from .AbstractDime import AbstractDime
import numpy as np
import pandas as pd
class NegativeTopic(AbstractDime):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subtopics = kwargs["subtopics"]

        self.name = "NegativeTopic"

    def querywise_compute_importance(self, query, *args, **kwargs):
        #step 1 take the representation of all the subtopics for the same topic
        local_subtopics = self.subtopics[(self.subtopics.topic_id == query.topic_id) & (self.subtopics.query_id!=query.query_id)]

        centroid = np.array(local_subtopics.representation.to_list())
        centroid = np.mean(centroid, axis=0)
        #importance = np.multiply(query.representation, centroid)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(-centroid)})
