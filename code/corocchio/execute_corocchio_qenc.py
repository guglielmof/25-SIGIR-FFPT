import faiss
import os
import pandas as pd

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

sys.path += ["code", "..", "/mnt"]
import argparse
import configparser
import importlib
from memmap_interface import MemmapCorpusEncoding
import local_utils.retrieve
from MYRETRIEVE.code.evaluating.evaluate import compute_measure
from MYRETRIEVE.code.indexes.FaissIndex import FaissIndex
from time import time
import numpy as np
import diversity_dimes.utils
from multiprocessing import Pool


def corocchio(queries, docs_encoder, clicklog):
    nsims = clicklog.rep_idx.max() + 1

    clicklog["weight"] = clicklog["click"] / (1 / clicklog["rank"])

    clicklog["representation"] = list(docs_encoder.get_encoding(clicklog.doc_id.to_list()))
    clicklog["representation"] = clicklog["representation"] * clicklog["weight"]

    rcontrib = clicklog.groupby("query_id")["representation"].apply(lambda x: np.sum(x, axis=0) / nsims).reset_index().rename(
        {"representation": "prf_representation"}, axis=1)

    local_queries = pd.merge(queries, rcontrib)

    alpha, beta = 0.4, 0.6
    local_queries["representation"] = alpha * local_queries["representation"] + beta * local_queries["prf_representation"]

    return local_queries[["query_id", "text", "representation"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("-e", "--encoder")
    parser.add_argument("-p", "--propensity")
    parser.add_argument("-l", "--length")

    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    queries = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.query_reader"])(config)
    qrels = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.qrels_reader"])(config)
    encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()

    queries = queries.loc[queries.query_id.isin(qrels.query_id)].reset_index(drop=True)
    queries["representation"] = list(encoder.encode_queries(queries.text))

    corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")
    import ir_datasets
    docs = pd.DataFrame(ir_datasets.load(config["Collections"][f"{args.collection}.datasetid"]).docs_iter())
    if args.collection == "trec-robust-2004":
        docs["text"] = docs["title"] + " " + docs["body"]
        docs = docs.drop(["title", "body", "marked_up_doc"], axis=1)

    print("loading faiss index ... ", end="", flush=True)
    start_time = time()
    indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
    print(f"done in {time() - start_time:.2f}s", flush=True)


    output = None
    for um in ["perfect", "binarized", "nearrandom"]:

        simulations = pd.read_csv(f"/mnt/25-CLICKMODELS/data/simulations/{args.collection}_{args.encoder}_{um}_{args.propensity}_{args.length}.csv", dtype={"query_id": str, "doc_id": str})
        nsims = simulations.rep_idx.max() + 1

        simulations["weight"] = simulations["click"] / (1 / simulations["rank"])

        sim_docs = simulations[["doc_id"]].drop_duplicates()
        sim_docs = sim_docs.merge(docs)
        sim_docs["representation"] = list(encoder.encode_queries(sim_docs.text))
        simulations = simulations.merge(sim_docs)
        simulations["representation"] = simulations["representation"] * simulations["weight"]

        rcontrib = simulations.groupby("query_id")["representation"].apply(lambda x: np.sum(x, axis=0) / nsims).reset_index().rename(
            {"representation": "prf_representation"}, axis=1)
        local_queries = pd.merge(queries, rcontrib)

        alpha, beta = 0.4, 0.6
        local_queries["prf_representation"] = alpha * local_queries["representation"] + beta * local_queries["prf_representation"]

        print("computing masked retrieval ... ", end="", flush=True)
        start_time = time()

        qembs = np.array(local_queries.prf_representation.to_list())
        q2r = pd.DataFrame(enumerate(local_queries.query_id.to_list()), columns=["row", "query_id"])
        run = local_utils.retrieve.retrieve_faiss(qembs, q2r, indexWrapper)
        run["rank"] = run.groupby("query_id")["score"].rank(ascending=False)
        perf = compute_measure(run, qrels, ["nDCG@10", "nDCG@20", "nDCG@50", "nDCG@100", "AP", "R@1000", "nDCG@3", "RR"]).rename({"value": um}, axis=1)
        if output is None:
            output = perf
        else:
            output = output.merge(perf)
        print(f"done in {time() - start_time:.2f}s", flush=True)

    output.to_csv(f"data/performance/corocchio_{args.collection}_{args.encoder}_{args.propensity}_{args.length}.csv", index=False)
