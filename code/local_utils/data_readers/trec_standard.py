import configparser
from .basic import _read_irdataset_queries, _read_irdataset_qrels
import pandas as pd


def read_trecrobust2004_queries(*args):
    pp = lambda x: x.drop(["narrative", "description"], axis=1).rename({"title": "text"}, axis=1)
    return _read_irdataset_queries(args[0]["Collections"]["trec-robust-2004.datasetid"], postprocessing=pp)


def read_trecrobust2004_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-robust-2004.datasetid"])


def read_trecdl2019_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["trec-dl-2019.datasetid"])


def read_trecdl2019_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2019.datasetid"])


def read_trecdl2020_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["trec-dl-2020.datasetid"])


def read_trecdl2020_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2020.datasetid"])


def read_vaswani_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["vaswani.datasetid"])


def read_vaswani_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["vaswani.datasetid"])


def read_treccastv12019_queries(*args):

    rwu = pd.read_csv(args[0]["Collections"]["trec-cast-v1-2019.rewritten_queries_path"], sep="\t", header=None, names=["query_id", "text"])

    def pp(ds):
        ds = ds.merge(rwu)[["query_id", "text"]]
        ds[["topic_id", "subtopic_id"]] = ds.query_id.str.split("_", expand=True)
        return ds
    return _read_irdataset_queries(args[0]["Collections"]["trec-cast-v1-2019.datasetid"], postprocessing=pp)

def read_treccastv12019_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-cast-v1-2019.datasetid"])


def read_treccastv12020_queries(*args):
    def pp(ds):
        ds = ds.rename({"manual_rewritten_utterance": "text"}, axis=1)[["query_id", "text"]]
        ds[["topic_id", "subtopic_id"]] = ds.query_id.str.split("_", expand=True)
        return ds
    return _read_irdataset_queries(args[0]["Collections"]["trec-cast-v1-2020.datasetid"], postprocessing=pp)


def read_treccastv12020_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-cast-v1-2020.datasetid"])


def read_trec8_queries(*args):
    postprocessing = lambda x: x.rename({"title": "text"}, axis=1)[["query_id", "text"]]
    print(args[0]["Collections"]["trec8.datasetid"])
    return _read_irdataset_queries(args[0]["Collections"]["trec8.datasetid"], postprocessing=postprocessing)


def read_trec8_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec8.datasetid"])





