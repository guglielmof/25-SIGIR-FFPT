import argparse
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dime")
    parser.add_argument("--cl_len", default="20")

    args = parser.parse_args()
    sns.set(font_scale=1.8, style="whitegrid")

    dsets = []
    for f in glob("data/performance/*"):
        if len(f.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("_")) < 5:
            continue
        dime_id, coll_id, encd_id, prop, cl_len = f.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("_")
        #if (dime_id == args.dime or dime_id == "corocchio") and cl_len == args.cl_len:
        if (dime_id == args.dime) and cl_len == args.cl_len:
            ds = pd.read_csv(f)
            ds = ds.melt(id_vars=["query_id", "measure"], value_vars=["perfect", "binarized", "nearrandom"], var_name="user_model")
            ds[["collection", "encoder", "propensity", "dime"]] = [coll_id, encd_id, int(prop), dime_id]
            dsets.append(ds)

    dsets = pd.concat(dsets)
    # avg_perf = dsets.groupby(["measure", "user_model", "collection", "encoder", "propensity"])["value"].mean().reset_index()

    encoders = ["contriever", "dragon", "tasb"]
    measures = ["nDCG@10", "nDCG@20"]
    user_models = ["perfect", "nearrandom"]

    # dsets = dsets[dsets.encoder.isin(encoders) & dsets.measure.isin(measures) & dsets.user_model.isin(user_models)]
    collections = ["trec-dl-2019", "trec-dl-2020", "trec-robust-2004"]
    '''
    fig, axs = plt.subplot_mosaic([
        [f"{c}_{e}_{m}_{u}" for e in encoders for m in measures for u in user_models] for c in collections
    ], layout='constrained', sharex=True, sharey=True)

    for c in collections:
        for e in encoders:
            for m in measures:
                for u in user_models:
                    utm = {"perfect": "P", "nearrandom": "R"}
                    data = dsets[(dsets.collection == c) & (dsets.encoder == e) & (dsets.measure == m) & (dsets.user_model == u)]
                    g = sns.barplot(data, y="value", x="propensity", hue="encoder", ax=axs[f"{c}_{e}_{m}_{u}"], hue_order=[args.dime, "corocchio"])
                    axs[f"{c}_{e}_{m}_{u}"].get_legend().set_visible(False)
                    axs[f"{c}_{e}_{m}_{u}"].set_title(f"{m}, {utm[u]}")

    plt.savefig("data/figures/propensity_tasb.pdf")
    '''

    ctc = {"trec-dl-2019": "DL '19", "trec-dl-2020": "DL '20", "trec-robust-2004": "RB '04"}
    fig, axs = plt.subplots(nrows=len(collections), ncols=len(measures)*len(user_models),  layout='constrained', sharex='col', sharey='row', figsize=(10, 6))

    for ridx, c in enumerate(collections):
        cidx = 0
        for u in user_models:
            for m in measures:
                ax = axs[ridx][cidx]
                utm = {"perfect": "P", "nearrandom": "R"}
                data = dsets[(dsets.collection == c) & (dsets.encoder.isin(encoders)) & (dsets.measure == m) & (dsets.user_model == u)]
                g = sns.barplot(data, y="value", x="propensity", hue="encoder", ax=ax, hue_order=encoders)
                ax.get_legend().set_visible(False)
                if ridx == 0:
                    ax.set_title(f"{m}, {utm[u]}")
                ax.set_xlabel("k")
                ax.set_ylabel(ctc[c])
                cidx += 1

    plt.savefig("data/figures/propensity.pdf")

    plt.show()
