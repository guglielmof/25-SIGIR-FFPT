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
from glob import glob
from bioinfokit.analys import stat


def tukey_comparison(df, alpha=0.05):
    print(df)
    res = stat()
    means = df.groupby("model")["value"].mean().reset_index()
    best_sys = means.loc[means.value.idxmax(), "model"]
    res.tukey_hsd(df=df[["value", "model", "query_id"]], res_var='value', xfac_var='model', anova_model='value ~ C(query_id) + C(model)')
    tukey_res = res.tukey_summary
    print(tukey_res.to_string())
    tukey_res = tukey_res.loc[((tukey_res["group1"] == best_sys) | (tukey_res["group2"] == best_sys)) & (tukey_res["p-value"] >= alpha)]

    ns_sys = list(set(tukey_res.group1.unique()).union(set(tukey_res.group2.unique())).union({best_sys}))
    return ns_sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    df = []
    for f in glob("data/performance/*"):
        print(f)
        df.append(pd.read_csv(f, dtype={"query_id": str}))
        m, c, e = f.rsplit("/", 1)[-1].rsplit(".")[0].split("_", 2)
        if "_" in e:
            e, p, l = e.split("_")
        else:
            p, l = "NA", "NA"
        df[-1][["model", "collection", "encoder", "propensity", "cl_len"]] = m, c, e, p, l
        if df[-1]["model"].unique()[0] in ["LLM", "nodime", "vprf20", "vprf5"]:
            df[-1]["nearrandom"] = df[-1]["value"]
            df[-1]["binarized"] = df[-1]["value"]
            df[-1] = df[-1].rename({"value": "perfect"}, axis=1)

    df = pd.concat(df) \
        .melt(id_vars=["query_id", "measure", "model", "collection", "encoder", "propensity", "cl_len"], value_vars=["perfect", "binarized", "nearrandom"],
              var_name="user_model")

    df = df.loc[~df.measure.isin(["RR", "R@1000"])].reset_index(drop=True)
    df = df[df.propensity.isin(['1', "NA"]) & df.cl_len.isin(['20', "NA"])]
    df[["query_id", "measure", "model", "collection", "encoder", "user_model", "value"]].to_csv("tmp_perf.csv", index=False)

    '''
    for c in df.collection.unique():
        for m in df.measure.unique():
            tmp_df = df[(df.collection == c) & (df.measure == m)].drop(["collection", "measure"], axis=1)
            tmp_df.to_csv("tmp", index=False)
            avg = tmp_df.groupby(["encoder", "user_model", "model"])["value"].mean().reset_index()
            top_group = tmp_df.groupby(["encoder", "user_model"]).apply(tukey_comparison).reset_index().rename({0: "top_group"}, axis=1)
            best = avg.loc[avg.groupby(["encoder", "user_model"])["value"].idxmax(), ["encoder", "user_model", "model"]]
            print(top_group)
    '''
