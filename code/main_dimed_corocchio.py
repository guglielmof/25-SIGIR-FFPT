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
    parser.add_argument("-d", "--dime")
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    indexWrapper = None

    orig_queries = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.query_reader"])(config)
    qrels = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.qrels_reader"])(config)
    encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()

    orig_queries = orig_queries.loc[orig_queries.query_id.isin(qrels.query_id)].reset_index(drop=True)
    orig_queries["representation"] = list(encoder.encode_queries(orig_queries.text))

    corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")

    if indexWrapper is None:
        print("loading faiss index ... ", end="", flush=True)
        start_time = time()
        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        print(f"done in {time() - start_time:.2f}s", flush=True)

    output = None
    for um in ["perfect", "binarized", "nearrandom"]:
        cl = pd.read_csv(f"../25-CLICKMODELS/data/simulations/{args.collection}_{args.encoder}_{um}_1_10.csv", dtype={"doc_id": str, "query_id": str})

        #compute the corocchio representation of the queries
        queries = corocchio(orig_queries, docs_encoder, cl)

        kwargs = {"clicklog": cl, "docs_encoder": docs_encoder}

        print("initializing importance estimator and computing importance ... ", end="", flush=True)
        start_time = time()
        estimator = getattr(importlib.import_module("diversity_dimes"), args.dime)(**kwargs)
        importance = estimator.compute_importance(queries)
        importance.to_csv(f"{config['DEFAULT']['dime_importance_dir']}/{args.collection}_{args.encoder}_{args.dime}.csv", index=False)
        print(f"done in {time() - start_time:.2f}s", flush=True)

        print("computing masked retrieval ... ", end="", flush=True)
        start_time = time()


        def alpha_retrieve(parallel_args):
            importance, queries, alpha = parallel_args
            masked_qembs, q2r = diversity_dimes.utils.get_masked_encoding(queries, importance, alpha)
            run = local_utils.retrieve.retrieve_faiss(masked_qembs, q2r, indexWrapper)
            run["alpha"] = alpha
            return run


        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        with Pool(processes=len(alphas)) as pool:
            run = pd.concat(pool.map(alpha_retrieve, [[importance, queries, a] for a in alphas]))

        print(f"done in {time() - start_time:.2f}s", flush=True)

        perf = run.groupby("alpha").apply(lambda x: compute_measure(x, qrels, ["AP", "R@1000", "MRR", "nDCG@3", "nDCG@10", "nDCG@100"])) \
            .reset_index().drop("level_1", axis=1)
        orig_pref = perf.loc[perf.alpha == 1.].drop("alpha", axis=1)

        # avg_perf = perf.groupby(["alpha", "measure"])["value"].mean().reset_index().pivot_table(index="measure", columns="alpha", values="value")
        cv_perf = perf.groupby("measure").apply(diversity_dimes.utils.crossvalidation).reset_index() \
            .rename({"value": f"{um}"}, axis=1).drop("level_1", axis=1)

        if output is None:
            output = cv_perf
        else:
            output = pd.merge(output, cv_perf)
    #print(f"{args.dime}_{args.collection}_{args.encoder}")
    #output.to_csv(f"data/performance/{args.dime}_{args.collection}_{args.encoder}.csv", index=False)
