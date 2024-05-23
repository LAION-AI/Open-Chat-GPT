import json
import os

import numpy as np
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from data.augmentation import MODEL_STORE
from data.helper import PoetryDialogueTask, PoetryRecord


def make_splits(ds, test=0.2, val=0.05):
    """Splits out the specified fraction of test/val samples from the passed dataset.
    Remaining 1 - (test + val) fraction of samples will be in train set."""
    train_test = ds.train_test_split(test_size=test + val)
    test_val = train_test["test"].train_test_split(test_size=val / test)
    return DatasetDict(
        {
            "train": train_test["train"],
            "test": test_val["train"],
            "validation": test_val["test"],
        }
    )


def main(output_dir: str = "data"):
    """Download and prepare the dataset for use."""
    np.random.seed(42)
    merve_poetry = make_splits(load_dataset("merve/poetry")["train"])
    matthh_poetry = make_splits(load_dataset("matthh/gutenberg-poetry-corpus")["train"])
    os.makedirs(output_dir, exist_ok=True)

    MODEL_STORE.load()

    for split in ["train", "test", "validation"]:
        with open(f"{output_dir}/{split}.jsonl", "w", encoding="utf8") as output:
            for i in tqdm(range(len(merve_poetry[split])), desc=split):
                # if i > 3:
                #     break
                record = merve_poetry[split][i]
                record = PoetryRecord(
                    poem=record["content"].replace("\r\n", "\n"),
                    title=record["poem name"],
                    author=record["author"],
                    theme=record["type"],
                    time_period=record["age"],
                )

                dialogue = PoetryDialogueTask.random_task().prepare_dialogue(record)
                # dialogue = list(PoetryDialogueTask)[i].prepare_dialogue(record)
                output.write(f"{json.dumps({'conversation': dialogue})}\n")

            for i in tqdm(range(len(matthh_poetry[split])), desc=split):
                # if i > 3:
                #     break
                record = matthh_poetry[split][i]
                record = PoetryRecord(
                    poem=record["content"],
                    title=record["title"],
                    author=record["author"],
                    theme=None,
                    time_period=record["author_birth"],
                )

                dialogue = PoetryDialogueTask.random_task().prepare_dialogue(record)
                # dialogue = list(PoetryDialogueTask)[i].prepare_dialogue(record)
                output.write(f"{json.dumps({'conversation': dialogue})}\n")


if __name__ == "__main__":
    main()
