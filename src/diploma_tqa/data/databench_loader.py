from databench_eval import utils


def load_qa(name: str = "semeval", split: str = "dev", limit: int | None = None):

    # Loads QA split from Databench

    qa = utils.load_qa(name=name, split=split)
    # can choose limit to run a subset of Databench
    if limit is not None:
        qa = qa.select(range(min(limit, len(qa))))
    return qa


def load_table(dataset_id: str, lite: bool = False):

    # Loads table with a dataset identifier, if lite = False : load the full version

    if lite:
        return utils.load_sample(dataset_id)
    return utils.load_table(dataset_id)