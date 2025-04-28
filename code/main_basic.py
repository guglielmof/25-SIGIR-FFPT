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
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    indexWrapper = None

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
    if args.dime in ["NegativeTopic", "PositiveTopic", "StringBased"]:
        kwargs = {"subtopics": queries}

    elif args.dime in ["OpposingDocuments", "LLM", "OpposingDocumentsOnly"]:
        model = "llama-3.1-70b-versatile"
        pseudorelevants = pd.read_csv(f"data/misc/pseudorelevants/{args.collection}_{model}.csv", dtype={"query_id": str})
        pseudorelevants["representation"] = list(encoder.encode_queries(pseudorelevants.pr_text))

        kwargs = {"subtopics": queries, "pseudorelevants": pseudorelevants}
    elif args.dime in ["Oracle"]:
        kwargs = {"qrels": qrels, "docs_encoder": docs_encoder, "add_non_relevant": True, "workers": 60}
    elif args.dime in ["OthersDime"]:
        ref_importance = pd.read_csv(f"{config['DEFAULT']['dime_importance_dir']}/{args.collection}_{args.encoder}_Oracle.csv")
        kwargs = {"ref_importance": ref_importance, "subtopics": queries, "workers": 60}
    elif args.dime in ["AllPrevious", "Previous", "First"]:
        ref_importance = pd.read_csv(f"{config['DEFAULT']['dime_importance_dir']}/{args.collection}_{args.encoder}_Oracle.csv")
        kwargs = {"ref_importance": ref_importance, "subtopics": queries.copy()}
        queries = queries.loc[queries.subtopic_id.astype(int) > 1]
        qrels = qrels.loc[qrels.query_id.isin(queries.query_id)]

    elif args.dime in ["StringBasedFirst", "StringBasedPrevious", "StringBasedAllPrevious"]:
        kwargs = {"subtopics": queries}
        queries = queries.loc[queries.subtopic_id.astype(int) > 1]
        qrels = qrels.loc[qrels.query_id.isin(queries.query_id)]

    elif args.dime in ["LLMOthers"]:
        model = "llama-3.1-70b-versatile"
        pseudorelevants = pd.read_csv(f"data/misc/pseudorelevants/{args.collection}_{model}.csv")
        pseudorelevants["representation"] = list(encoder.encode_queries(pseudorelevants.pr_text))

        kwargs = {"subtopics": queries, "pseudorelevants": pseudorelevants}
    elif args.dime in ["LLMOthersFirst", "LLMOthersPrevious", "LLMOthersAllPrevious"]:
        model = "llama-3.1-70b-versatile"
        pseudorelevants = pd.read_csv(f"data/misc/pseudorelevants/{args.collection}_{model}.csv")
        pseudorelevants["representation"] = list(encoder.encode_queries(pseudorelevants.pr_text))

        kwargs = {"subtopics": queries, "pseudorelevants": pseudorelevants}
        queries = queries.loc[queries.subtopic_id.astype(int) > 1]
        qrels = qrels.loc[qrels.query_id.isin(queries.query_id)]

    elif args.dime in ["EntityFeedback"]:
        model = "llama-3.1-70b-versatile"
        entities = pd.read_csv(f"data/misc/entities_annotation/{args.collection}_{model}.csv")
        entities["representation"] = list(encoder.encode_queries(entities.entities))
        kwargs = {"subtopics": queries, "entities": entities}
        '''
        elif args.dime in ["SimulatedPerfectUnbiasedLog", "SimulatedPerfectBiasedLog", "SimulatedNoisyUnbiasedLog", "SimulatedNoisyBiasedLog"]:
    
            log_type = {"SimulatedPerfectUnbiasedLog": "perfect_unbiased", "SimulatedPerfectBiasedLog": "perfect_biased",
                        "SimulatedNoisyUnbiasedLog": "noisy_unbiased", "SimulatedNoisyBiasedLog": "noisy_biased"}
    
            log = pd.read_csv(f"data/simulations/{args.collection}_{args.encoder}_{log_type[args.dime]}.csv", dtype={"query_id": str, "doc_id": str})
            log["representation"] = list(docs_encoder.get_encoding(log.doc_id.to_list()))
            kwargs = {"log": log}
    
        elif args.dime in ["SimulatedPerfectUnbiasedLogOthers", "SimulatedPerfectBiasedLogOthers", "SimulatedNoisyUnbiasedLogOthers", "SimulatedNoisyBiasedLogOthers"]:
    
            log_type = {"SimulatedPerfectUnbiasedLogOthers": "perfect_unbiased", "SimulatedPerfectBiasedLogOthers": "perfect_biased",
                        "SimulatedNoisyUnbiasedLogOthers": "noisy_unbiased", "SimulatedNoisyBiasedLogOthers": "noisy_biased"}
    
            log = pd.read_csv(f"data/simulations/{args.collection}_{args.encoder}_{log_type[args.dime]}.csv", dtype={"query_id": str, "doc_id": str})
            log["representation"] = list(docs_encoder.get_encoding(log.doc_id.to_list()))
            kwargs = {"log": log, "subtopics": queries}
        '''

    elif args.dime in ["ClickLogs", "ClickLogsPropensityCorrection", "CorrelationClickLogsPropensityCorrection", "DCClickLogs"]:
        cl = pd.read_csv(f"../25-CLICKMODELS/data/simulations/trec-dl-2019_{args.encoder}_perfect_1_10.csv", dtype={"doc_id": str, "query_id": str})
        kwargs = {"clicklog": cl, "docs_encoder": docs_encoder}

    elif args.dime in ["NegDime"]:

        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        qembs = np.array(queries.representation.to_list())
        q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])
        run = local_utils.retrieve.retrieve_faiss(qembs, q2r, indexWrapper)

        kwargs = {"run": run, "qrels": qrels, "docs_encoder": docs_encoder}

    elif args.dime in ["LLMDistr"]:

        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        qembs = np.array(queries.representation.to_list())
        q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])
        run = local_utils.retrieve.retrieve_faiss(qembs, q2r, indexWrapper)

        model = "llama-3.1-70b-versatile"
        pseudorelevants = pd.read_csv(f"data/misc/pseudorelevants/{args.collection}_{model}.csv", dtype={"query_id": str})
        pseudorelevants["representation"] = list(encoder.encode_queries(pseudorelevants.pr_text))

        kwargs = {"run": run, "qrels": qrels, "docs_encoder": docs_encoder, "pseudorelevants": pseudorelevants}
    else:
        raise ValueError("unrecognized dime")

    print("initializing importance estimator and computing importance ... ", end="", flush=True)
    start_time = time()
    estimator = getattr(importlib.import_module("diversity_dimes"), args.dime)(**kwargs)
    importance = estimator.compute_importance(queries)
    importance.to_csv(f"{config['DEFAULT']['dime_importance_dir']}/{args.collection}_{args.encoder}_{args.dime}.csv", index=False)
    print(f"done in {time() - start_time:.2f}s", flush=True)


    if indexWrapper is None:
        print("loading faiss index ... ", end="", flush=True)
        start_time = time()
        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        print(f"done in {time() - start_time:.2f}s", flush=True)

    print("computing masked retrieval ... ", end="", flush=True)
    start_time = time()


    def alpha_retrieve(parallel_args):
        importance, queries, alpha = parallel_args
        masked_qembs, q2r = diversity_dimes.utils.get_masked_encoding(queries, importance, alpha)
        run = local_utils.retrieve.retrieve_faiss(masked_qembs, q2r, indexWrapper)
        run["alpha"] = alpha
        return run


    if args.collection == "uqv100":
        queries = queries[queries.subtopic_id == "50-1"].reset_index(drop=True)
        qrels = qrels[qrels.query_id.isin(queries.query_id)]
    if args.collection in ["trec-dl-2019-qv", "trec-dl-2020-qv"]:
        queries = queries[queries.subtopic_id == "orig"].reset_index(drop=True)
        qrels = qrels[qrels.query_id.isin(queries.query_id)]

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    with Pool(processes=len(alphas)) as pool:
        run = pd.concat(pool.map(alpha_retrieve, [[importance, queries, a] for a in alphas]))

    print(f"done in {time() - start_time:.2f}s", flush=True)

    perf = run.groupby("alpha").apply(lambda x: compute_measure(x, qrels, ["AP", "R@1000", "MRR", "nDCG@3", "nDCG@10", "nDCG@20", "nDCG@50", "nDCG@100"])) \
        .reset_index().drop("level_1", axis=1)
    orig_pref = perf.loc[perf.alpha == 1.].drop("alpha", axis=1)

    # avg_perf = perf.groupby(["alpha", "measure"])["value"].mean().reset_index().pivot_table(index="measure", columns="alpha", values="value")
    cv_perf = perf.groupby("measure").apply(diversity_dimes.utils.crossvalidation).reset_index().drop("level_1", axis=1)

    print(f"{args.dime}_{args.collection}_{args.encoder}")
    cv_perf.to_csv(f"data/performance/{args.dime}_{args.collection}_{args.encoder}.csv", index=False)
    orig_pref.to_csv(f"data/performance/nodime_{args.collection}_{args.encoder}.csv", index=False)
