from .basic import _read_irdataset_queries, _read_irdataset_qrels


def read_trecwebdiversity_queries(dataset_id):
    def query_processing(queries):
        subtopics = queries.copy()[["query_id", "subtopics"]]
        info = queries.copy()[["query_id", "description", "type"]]
        queries = queries[["query_id", "query"]].rename({"query": "text"}, axis=1)
        subtopics = subtopics.explode("subtopics")

        def split_subtopics(st):
            return [st.number, st.text.strip(), st.type]

        subtopics["subtopics"] = subtopics.subtopics.apply(split_subtopics)
        subtopics["subtopic_id"] = subtopics.subtopics.apply(lambda x: x[0])
        subtopics["text"] = subtopics.subtopics.apply(lambda x: x[1])
        subtopics["type"] = subtopics.subtopics.apply(lambda x: x[2])

        subtopics = subtopics.drop("subtopics", axis=1)

        return queries, subtopics, info

    return _read_irdataset_queries(dataset_id, query_processing)

def read_trecwebdiversity_qrels(dataset_id):

    def postprocessing(qrels):
        qrels = qrels[["query_id", "doc_id", "relevance"]]
        qrels = qrels.drop_duplicates()
        qrels = qrels.groupby(["query_id", "doc_id"])["relevance"].max().reset_index()
        return qrels
    return _read_irdataset_qrels(dataset_id, postprocessing)

def read_trec18webdiversity_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2009.datasetid"]
    return read_trecwebdiversity_queries(dataset_id)[0]


def read_trec19webdiversity_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2010.datasetid"]
    return read_trecwebdiversity_queries(dataset_id)[0]


def read_trec20webdiversity_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2011.datasetid"]
    return read_trecwebdiversity_queries(dataset_id)[0]


def read_trec21webdiversity_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2012.datasetid"]
    return read_trecwebdiversity_queries(dataset_id)[0]


def read_trec18webdiversity_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2009.datasetid"]
    return read_trecwebdiversity_qrels(dataset_id)


def read_trec19webdiversity_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2010.datasetid"]
    return read_trecwebdiversity_qrels(dataset_id)


def read_trec20webdiversity_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2011.datasetid"]
    return read_trecwebdiversity_qrels(dataset_id)


def read_trec21webdiversity_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2012.datasetid"]
    return read_trecwebdiversity_qrels(dataset_id)


def read_ST_queries(dataset_id):
    subtopics = read_trecwebdiversity_queries(dataset_id)[1]
    subtopics = subtopics.rename({"query_id": "topic_id"}, axis=1)
    subtopics["query_id"] = subtopics["topic_id"] + "_" + subtopics["subtopic_id"]
    return subtopics[["query_id", "text"]]


def read_ST_qrels(dataset_id):
    subtopics_qrels = _read_irdataset_qrels(dataset_id)
    subtopics_qrels = subtopics_qrels.rename({"query_id": "topic_id"}, axis=1)
    subtopics_qrels["query_id"] = subtopics_qrels["topic_id"] + "_" + subtopics_qrels["subtopic_id"]
    assessed_subtopics = set(subtopics_qrels.loc[subtopics_qrels["relevance"] > 0, "query_id"].unique())
    subtopics_qrels = subtopics_qrels.loc[subtopics_qrels.query_id.isin(assessed_subtopics)].reset_index(drop=True)[["doc_id", "query_id", "relevance"]]

    return subtopics_qrels


def read_trec18webdiversityST_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2009.datasetid"]
    return read_ST_queries(dataset_id)


def read_trec19webdiversityST_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2010.datasetid"]
    return read_ST_queries(dataset_id)


def read_trec20webdiversityST_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2011.datasetid"]
    return read_ST_queries(dataset_id)


def read_trec21webdiversityST_queries(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2012.datasetid"]
    return read_ST_queries(dataset_id)


def read_trec18webdiversityST_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2009.datasetid"]
    return read_ST_qrels(dataset_id)


def read_trec19webdiversityST_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2010.datasetid"]
    return read_ST_qrels(dataset_id)



def read_trec20webdiversityST_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2011.datasetid"]
    return read_ST_qrels(dataset_id)



def read_trec21webdiversityST_qrels(*args):
    dataset_id = args[0]["Collections"]["trec-web-diversity-2012.datasetid"]
    return read_ST_qrels(dataset_id)
