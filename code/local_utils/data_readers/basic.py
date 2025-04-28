import pandas as pd
import ir_datasets

def _read_irdataset_queries(dataset_id, postprocessing=None):
    if postprocessing is None:
        postprocessing = lambda x: x
    # print(postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).queries_iter())))
    return postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).queries_iter()))


def _read_irdataset_qrels(dataset_id, postprocessing=None):
    if postprocessing is None:
        postprocessing = lambda x: x

    return postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).qrels_iter()))
