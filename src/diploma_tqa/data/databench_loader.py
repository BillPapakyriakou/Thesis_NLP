from databench_eval import utils


def load_qa(name: str = "semeval", split: str = "dev", limit: int | None = None):
    qa = utils.load_qa(name=name, split=split)
    if limit is not None:
        qa = qa.select(range(min(limit, len(qa))))
    return qa


def load_table(dataset_id: str, lite: bool = False):
    if lite:
        return utils.load_sample(dataset_id)
    return utils.load_table(dataset_id)