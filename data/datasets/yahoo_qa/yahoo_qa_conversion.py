import argparse
import json
from typing import Dict, List

from datasets import Dataset, load_dataset
from detoxify import Detoxify
from tqdm import tqdm


def map_data_to_new_format(data: List[Dict], extend: bool, to_take: int, detoxify: bool) -> List[Dict]:
    new_data = []
    predictor = Detoxify("multilingual") if detoxify else None
    for d in tqdm(data, desc="Mapping data to new format"):
        new_d = {
            "INSTRUCTION": d["question"],
            "RESPONSE": d["answer"],
            "SOURCE": "Yahoo Q&A",
            "METADATA": {
                "topic": d["main_category"],
                "toxicity": float(predictor.predict(d["question"])["toxicity"]) if detoxify else None,
            },
        }
        if extend:
            range_to_take = (
                range(len(d["nbestanswers"]) - 1) if len(d["nbestanswers"]) - 1 < to_take else range(to_take)
            )
            for i in range_to_take:
                augmented_d = {
                    "INSTRUCTION": d["question"],
                    "RESPONSE": d["nbestanswers"][i + 1],
                    "SOURCE": "Yahoo Q&A",
                    "METADATA": {
                        "topic": d["main_category"],
                        "toxicity": float(predictor.predict(d["question"])["toxicity"]) if detoxify else None,
                    },
                }
            new_data.append(augmented_d)

        new_data.append(new_d)
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map data from the Yahoo Answers Dataset to the Open Assistant Dataset Forma and optionally extend it with the others best answer."
    )
    parser.add_argument("--extend", action="store_true", help="extend dataset by taking more answer by question")
    parser.add_argument("--export-json", action="store_true", help="export the datasets to Json")
    parser.add_argument(
        "--detoxify",
        action="store_true",
        help="generate a toxicity score between 0-1 using detoxify multilingual model",
    )
    parser.add_argument("--extension-number", type=int, default=1, help="number of answer to take in addition")

    args = parser.parse_args()
    dataset: Dataset = load_dataset("yahoo_answers_qa", split="train")

    new_data: List[Dict] = map_data_to_new_format(dataset, args.extend, args.extension_number, args.detoxify)

    if args.export_json:
        with open("converted.json", "w") as f:
            json.dump(new_data, f, indent=4)
