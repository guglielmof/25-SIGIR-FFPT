import sys

import pandas as pd
import numpy as np

sys.path += ["..", "/mnt"]


def retrieve_faiss(qembs, q2r, indexWrapper, k=1000):
    ip, idx = indexWrapper.index.search(qembs, k)

    run = pd.DataFrame({"row": np.arange(len(q2r)), "doc_id": list(idx), "score": list(ip)}).merge(q2r).drop("row", axis=1).explode(["doc_id", "score"])
    run.doc_id = run.doc_id.map(lambda x: indexWrapper.mapper[x])
    run.score = run.score.astype(float)
    return run[["query_id", "doc_id", "score"]]
