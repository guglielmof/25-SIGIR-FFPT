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
    if args.collection in ["trec-web-diversity-2009-st", "trec-web-diversity-2010-st", "trec-web-diversity-2011-st", "trec-web-diversity-2012-st"]:
        queries[["topic_id", "subtopic_id"]] = queries.query_id.str.split("_", expand=True)
        without_subtopics = (queries.groupby("topic_id")["query_id"].count() <= 1).reset_index()
        queries = queries.loc[~queries.topic_id.isin(without_subtopics.loc[without_subtopics["query_id"], "topic_id"])].reset_index(drop=True)

    queries["representation"] = list(encoder.encode_queries(queries.text))

    corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")

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
    perf = compute_measure(run, qrels, ["AP", "MRR", "nDCG@3", "nDCG@10"])
    # print(perf.groupby("measure")["value"].mean().reset_index())

    cut_run = run.loc[run["rank"] <= 10]
    cut_run = cut_run.merge(qrels[["query_id", "doc_id", "relevance"]], how="left")
    cut_run = cut_run.fillna(0)
    cut_run = cut_run[["query_id", "doc_id", "rank", "relevance"]]


    def simulate_log(input_run, dist="perfect", eta_bias=1, nsim=1000):
        run = input_run.copy()
        distributions = {"perfect": {0: 0, 1: 0, 2: 1, 3: 1}, "noisy": {0: 0.2, 1: 0.4, 2: 0.8, 3: 0.9}}

        run["rclick"] = run["relevance"].map(distributions[dist])
        run["bias"] = (1 / run["rank"]) ** eta_bias
        run["pclick"] = run["rclick"] * run["bias"]

        simulations = []
        for i in range(nsim):
            srun = run.copy()
            srun["click"] = np.random.random(size=len(run))
            srun["click"] = srun["click"] < srun["pclick"]
            srun["simid"] = i
            simulations.append(srun[["query_id", "doc_id", "rank", "simid", "click"]])
        simulations = pd.concat(simulations)
        return simulations


    pb = simulate_log(cut_run, dist="perfect", eta_bias=1)
    pu = simulate_log(cut_run, dist="perfect", eta_bias=0)
    nb = simulate_log(cut_run, dist="noisy", eta_bias=1)
    nu = simulate_log(cut_run, dist="noisy", eta_bias=0)

    pb.to_csv(f"data/simulations/{args.collection}_{args.encoder}_perfect_biased.csv", index=False)
    pu.to_csv(f"data/simulations/{args.collection}_{args.encoder}_perfect_unbiased.csv", index=False)
    nb.to_csv(f"data/simulations/{args.collection}_{args.encoder}_noisy_biased.csv", index=False)
    nu.to_csv(f"data/simulations/{args.collection}_{args.encoder}_noisy_unbiased.csv", index=False)
