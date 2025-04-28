import pandas as pd

it = pd.read_csv("data/analyses/out.csv")

def stringify(row):
    return f"{'\\bf{' if row['best'] else''}{row['mean_value']:.3f}{'}' if row['best'] else ''}{'$^*$' if row['tpgr'] else ''}"

m2m = {"nodime": "retrieval only", "vprf5": "VPRF-5", "vprf20": "VPRF-20", "LLM": "DIME$_{LLM}$", "ClickLogsMean": "DIME$_{PRF}$", "corocchio": "CoRocchio",
'ClickLogsWeightedMean': "CoDIME$_{wavg}$", 'ClickLogsWeightedMax': "CoDIME$_{wmax}$", 'ClickLogsCorrelation': "CoDIME$_{corr}$", 'ClickLogsLinearModel': "CoDIME$_{slope}$"}
model_order = ["retrieval only", "VPRF-5", "VPRF-20", "DIME$_{LLM}$", "DIME$_{PRF}$", "CoRocchio", "CoDIME$_{wavg}$", "CoDIME$_{wmax}$", "CoDIME$_{corr}$", "CoDIME$_{slope}$"]

utm = {"perfect": "P", "binarized": "B", "nearrandom": "R"}
ctc = {"trec-dl-2019": "DL '19", "trec-dl-2020": "DL '20", "trec-robust-2004": "RB '04"}


it["user_model"] = it.user_model.map(utm)
it["collection"] = it.collection.map(ctc)
it["model"] = it.model.map(m2m)

um_order = ["P", "B", "R"]
it.model = pd.Categorical(it.model, categories=model_order, ordered=True)
it.user_model = pd.Categorical(it.user_model, categories=um_order, ordered=True)
for c in it.collection.unique():
    #for m in it.measure.unique():
    for m in ["nDCG@10", "nDCG@20", "nDCG@50", "nDCG@100"]:
        print(f"{m} {c}")
        tmp_tbl = it[(it.measure == m) & (it.collection == c)].drop(["measure", "collection"], axis=1)
        tmp_tbl["string"] = tmp_tbl.apply(stringify, axis=1)
        tmp_tbl = tmp_tbl.sort_values("model")[["model", "encoder", "user_model", "string"]]
        tmp_tbl = tmp_tbl.pivot_table(index="model", columns=["encoder", "user_model"], values="string", aggfunc=lambda x: x)
        print(tmp_tbl.to_string())

#tmp
codime = it[(it.measure.isin(["nDCG@10", "nDCG@20", "nDCG@50", "nDCG@100"])) & (it.model == r"CoDIME$_{slope}$")].drop(["best", "tpgr"], axis=1)
corocchio =  it[(it.measure.isin(["nDCG@10", "nDCG@20", "nDCG@50", "nDCG@100"])) & (it.model == "CoRocchio")].drop(["best", "tpgr"], axis=1)
merged = codime.merge(corocchio, on=["measure", "collection", "user_model", "encoder"])
merged["diff"] = merged["mean_value_x"] - merged["mean_value_y"]
merged.groupby(["measure"])["diff"].max()