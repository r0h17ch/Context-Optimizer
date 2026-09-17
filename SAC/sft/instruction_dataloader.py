from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch


def derive_query_ids(example):
    """The question prefix of `lm_targets`, i.e. exactly the `question_ids` built in
    instruction_prepare_data.get_ids: "\\n### Question:\\n" + question + "\\n### Answer:\\n".

    For train examples `lm_targets` is question_ids + answer_ids and
    `instruction_target` is ([-100] * len(question_ids) + answer_ids)[1:], so the number
    of -100 entries is len(question_ids) - 1 and the prefix length is that count + 1.
    For eval examples `lm_targets` is already the question alone.

    Deriving it here keeps the 1.9 GB tokenised cache valid -- rebuilding it to add one
    field would cost ~30 min of tokenisation for information already present.
    """
    target = example.get("instruction_target")
    if target is None:
        return example["lm_targets"]
    n_query = int((target == -100).sum()) + 1
    return example["lm_targets"][:n_query]


class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size, add_query_ids=False):
        super(CompressDataset).__init__()
        self.examples = examples
        self.batch_size = batch_size
        self.add_query_ids = add_query_ids

    def __iter__(self):
        keys = list(self.examples[0].keys())
        if self.add_query_ids:
            keys.append("query_ids")
        batch = {key: [] for key in keys}
        count = 0  # 初始化计数器

        for example in self.examples:
            for key, value in example.items():
                batch[key].append(value)
            if self.add_query_ids:
                batch["query_ids"].append(derive_query_ids(example))

            count += 1  # 更新计数器

            # 当计数器达到 batch_size 时 yield 批次
            if count == self.batch_size:
                # yield {key: torch.stack(value) for key, value in batch.items()}
                # only for batch_size == 1
                assert self.batch_size==1
                yield {key: torch.stack(value) if value[0] is not None else None for key, value in batch.items()}
                batch = {key: [] for key in batch}  # 重置批次
                count = 0  # 重置计数器


def get_dataset(task_type, examples, batch_size, add_query_ids=False):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size, add_query_ids=add_query_ids)

    raise Exception("Don't exist [{task_type}] task.")
