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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("-e", "--encoder")
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

    simulations = pd.read_csv(f"data/simulations/{args.collection}_{args.encoder}_{args.simulation}.csv", dtype={"query_id": str, "doc_id": str})
    nsims = simulations.simid.max() + 1

    if "_unbiased" in args.simulation:
        simulations["weight"] = simulations["click"]
    else:
        simulations["weight"] = simulations["click"] / (1 / simulations["rank"])

    simulations["representation"] = list(docs_encoder.get_encoding(simulations.doc_id.to_list()))
    simulations["representation"] = simulations["representation"] * simulations["weight"]

    rcontrib = simulations.groupby("query_id")["representation"].apply(lambda x: np.sum(x, axis=0) / nsims).reset_index().rename(
        {"representation": "prf_representation"}, axis=1)
    queries = pd.merge(queries, rcontrib)
    alpha, beta = 0.4, 0.6
    queries["prf_representation"] = alpha * queries["representation"] + beta * queries["prf_representation"]

    print("loading faiss index ... ", end="", flush=True)
    start_time = time()
    indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
    print(f"done in {time() - start_time:.2f}s", flush=True)

    print("computing masked retrieval ... ", end="", flush=True)
    start_time = time()

    qembs = np.array(queries.representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])
    run = local_utils.retrieve.retrieve_faiss(qembs, q2r, indexWrapper)
    run["rank"] = run.groupby("query_id")["score"].rank(ascending=False)
    perf = compute_measure(run, qrels, ["nDCG@10", "nDCG@100", "AP", "R@1000"])
    print(perf.groupby("measure")["value"].mean().reset_index())

    qembs = np.array(queries.prf_representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])
    run = local_utils.retrieve.retrieve_faiss(qembs, q2r, indexWrapper)
    run["rank"] = run.groupby("query_id")["score"].rank(ascending=False)
    perf = compute_measure(run, qrels, ["nDCG@10", "nDCG@100", "AP", "R@1000", "nDCG@3"])
    print(perf.groupby("measure")["value"].mean().reset_index())
