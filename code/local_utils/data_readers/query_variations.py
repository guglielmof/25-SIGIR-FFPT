import configparser
from .basic import _read_irdataset_queries, _read_irdataset_qrels
import pandas as pd


def read_uqv100_queries(*args):
    if type(args[0]) == str:
        path = args[0]
    elif type(args[0]) == configparser.ConfigParser:
        path = args[0]["Collections"]["uqv100.queries.path"]
    else:
        raise ValueError("unrecognized input type")
    queries = pd.read_csv(path, header=None, names=["full"])
    queries[["query_id", "text"]] = queries.full.str.split(" ", n=1, expand=True)
    queries.drop("full", axis=1, inplace=True)
    queries[["topic_id", "subtopic_id"]] = queries.query_id.str.split("-", n=1, expand=True)
    return queries


def read_uqv100_qrels(*args):
    if type(args[0]) == str:
        path = args[0]
    elif type(args[0]) == configparser.ConfigParser:
        path = args[0]["Collections"]["uqv100.qrels.path"]
    else:
        raise ValueError("unrecognized input type")
    qrels = pd.read_csv(path)
    qrels[["topic_id", "subtopic_id"]] = qrels.query_id.str.split("-", n=1, expand=True)

    return qrels


def read_trecdl2019qv_queries(*args):
    path = args[0]["Collections"]["trec-dl-2019-qv.query_path"]
    queries = pd.read_csv(path, dtype={"query_id": str, "vid": str})
    queries["topic_id"] = queries.query_id
    queries = queries.rename({"vid": "subtopic_id"}, axis=1)
    queries["query_id"] = queries.topic_id + "_" + queries.subtopic_id
    orig_queries = _read_irdataset_queries(args[0]["Collections"]["trec-dl-2019-qv.datasetid"])
    orig_queries["subtopic_id"] = "orig"
    orig_queries["topic_id"] = orig_queries.query_id
    orig_queries["query_id"] = orig_queries.topic_id + "_" + orig_queries.subtopic_id
    queries = pd.concat([queries, orig_queries])
    return queries


def read_trecdl2019qv_qrels(*args):
    queries = read_trecdl2019qv_queries(*args)
    qrels = _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2019-qv.datasetid"])
    tmp_queries = queries[["query_id", "topic_id"]].drop_duplicates()
    qrels = qrels.rename({"query_id": "topic_id"}, axis=1)
    qrels = tmp_queries.merge(qrels).drop("topic_id", axis=1)
    return qrels


def read_trecdl2020qv_queries(*args):
    path = args[0]["Collections"]["trec-dl-2020-qv.query_path"]
    queries = pd.read_csv(path, dtype={"query_id": str, "vid": str})
    queries["topic_id"] = queries.query_id
    queries = queries.rename({"vid": "subtopic_id"}, axis=1)
    queries["query_id"] = queries.topic_id + "_" + queries.subtopic_id
    orig_queries = _read_irdataset_queries(args[0]["Collections"]["trec-dl-2020-qv.datasetid"])
    orig_queries["subtopic_id"] = "orig"
    orig_queries["topic_id"] = orig_queries.query_id
    orig_queries["query_id"] = orig_queries.topic_id + "_" + orig_queries.subtopic_id
    queries = pd.concat([queries, orig_queries])
    return queries


def read_trecdl2020qv_qrels(*args):
    queries = read_trecdl2020qv_queries(*args)

    qrels = _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2020-qv.datasetid"])
    tmp_queries = queries[["query_id", "topic_id"]].drop_duplicates()
    qrels = qrels.rename({"query_id": "topic_id"}, axis=1)
    qrels = tmp_queries.merge(qrels).drop("topic_id", axis=1)
    return qrels
