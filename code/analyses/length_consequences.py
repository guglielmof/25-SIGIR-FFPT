import argparse
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dime")
    parser.add_argument("--prop", default="1")

    args = parser.parse_args()
    sns.set(font_scale=1.8, style="whitegrid")

    dsets = []
    for f in glob("data/performance/*"):
        dime_id, coll_id, encd_id = f.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("_", 2)
        if "_" in encd_id:
            encd_id, prop, cl_len = encd_id.split("_")
        else:
            prop, cl_len = "NA", "NA"
        if dime_id in [args.dime, "corocchio"] and prop == args.prop:
            ds = pd.read_csv(f)
            ds = ds.melt(id_vars=["query_id", "measure"], value_vars=["perfect", "binarized", "nearrandom"], var_name="user_model")
            ds[["dime", "collection", "encoder", "propensity", "cl_len"]] = [dime_id, coll_id, encd_id, int(prop), int(cl_len)]
            dsets.append(ds)

    dsets = pd.concat(dsets)
    # avg_perf = dsets.groupby(["measure", "user_model", "collection", "encoder", "propensity"])["value"].mean().reset_index()

    encoders = ["tasb", "contriever", "dragon"]
    measures = ["nDCG@10", "nDCG@20"]
    user_models = ["perfect", "nearrandom"]

    collections = ["trec-dl-2019", "trec-dl-2020", "trec-robust-2004"]

    dsets = dsets[dsets.encoder.isin(encoders) & dsets.measure.isin(measures) & dsets.user_model.isin(user_models)]
    #dsets = dsets[dsets.cl_len<50]

    '''
    fig, axs = plt.subplot_mosaic([
        [f"{c}_{m}_{u}" for u in user_models for m in measures] for c in collections
    ], layout='constrained', sharex=True, figsize=(14, 8))
    '''

    utm = {"perfect": "P", "nearrandom": "R"}
    ctc = {"trec-dl-2019": "DL '19", "trec-dl-2020": "DL '20", "trec-robust-2004": "RB '04"}
    fig, axs = plt.subplots(nrows=len(collections), ncols=len(measures)*len(user_models),  layout='constrained', sharex='col', sharey='row', figsize=(14, 8))

    for ridx, c in enumerate(collections):
        cidx = 0
        for u in user_models:
            for m in measures:
                ax = axs[ridx][cidx]
                data = dsets[(dsets.collection == c) & (dsets.measure == m) & (dsets.user_model == u)]
                g = sns.lineplot(data, x="cl_len", y="value", ax=ax, errorbar=None,
                                 style="dime", style_order=[args.dime, "corocchio"],
                                 hue="encoder", hue_order=["contriever", "tasb", "dragon"])
                ax.set_xticks([2, 5, 10, 20, 50])

                ax.get_legend().set_visible(False)
                if ridx == 0:
                    ax.set_title(f"{m}, {utm[u]}")
                ax.set_xlabel("k")


                ax.set_ylabel(ctc[c])

                cidx += 1

    plt.savefig("data/figures/selection_effect.pdf")

    plt.show()
