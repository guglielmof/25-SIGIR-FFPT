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
    parser.add_argument("-d", "--dime")
    parser.add_argument("-p", "--propensity")
    parser.add_argument("-l", "--length")

    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    indexWrapper = None

    queries = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.query_reader"])(config)
    qrels = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.qrels_reader"])(config)
    encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()

    queries = queries.loc[queries.query_id.isin(qrels.query_id)].reset_index(drop=True)
    queries["representation"] = list(encoder.encode_queries(queries.text))

    corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")

    if indexWrapper is None:
        print("loading faiss index ... ", end="", flush=True)
        start_time = time()
        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        print(f"done in {time() - start_time:.2f}s", flush=True)

    output = None
    for um in ["perfect", "binarized", "nearrandom"]:
        cl = pd.read_csv(f"/mnt/25-CLICKMODELS/data/simulations/{args.collection}_{args.encoder}_{um}_{args.propensity}_{args.length}.csv", dtype={"doc_id": str, "query_id": str})
        #cl = pd.read_csv(f"/ssd/data/faggioli/25-CLICKMODELS/data/simulations/{args.collection}_{args.encoder}_{um}_{args.propensity}_{args.length}.csv",
        #                 dtype={"doc_id": str, "query_id": str})

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

        perf = run.groupby("alpha").apply(lambda x: compute_measure(x, qrels, ["AP", "R@1000", "MRR", "nDCG@3", "nDCG@10", "nDCG@100", "nDCG@20", "nDCG@50"])) \
            .reset_index().drop("level_1", axis=1)
        orig_pref = perf.loc[perf.alpha == 1.].drop("alpha", axis=1)

        # avg_perf = perf.groupby(["alpha", "measure"])["value"].mean().reset_index().pivot_table(index="measure", columns="alpha", values="value")
        cv_perf = perf.groupby("measure").apply(diversity_dimes.utils.crossvalidation).reset_index() \
            .rename({"value": f"{um}"}, axis=1).drop("level_1", axis=1)

        if output is None:
            output = cv_perf
        else:
            output = pd.merge(output, cv_perf)
    print(f"{args.dime}_{args.collection}_{args.encoder}")
    output.to_csv(f"data/performance/{args.dime}_{args.collection}_{args.encoder}_{args.propensity}_{args.length}.csv", index=False)
