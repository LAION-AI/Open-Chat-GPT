import json
import math
import os
import random
import re
import sys
from string import punctuation

import kaggle
import pandas as pd

CLINICAL_NOTE_GENERATION_TEMPLATE = """User: Write a clinical note about a patient with the following {section}: {section_information}.
Rosey: {note}"""


def preprocess(mt_dataset):
    def filter_for_notes(row):
        normalized_transcript = row["transcription"].lower()
        if "chief complaint:" in normalized_transcript:
            return True
        return False

    mt_dataset = mt_dataset.dropna(subset=["description", "transcription"])
    mt_note_subset = mt_dataset[mt_dataset.apply(filter_for_notes, axis=1)]
    return mt_note_subset


def is_chief_complaint(section):
    return "chief complaint" in section.lower()


def get_conversations(dataset):
    def normalize_transcript(x):
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)
        x = re.sub(r":\s+", ": ", x)
        x = re.sub(r"\.\s+", ". ", x)
        x = re.sub(r":(\s)*\,+", ": ", x)
        x = re.sub(r"\.\,+", ". ", x)
        return x

    conversations = []
    for idx in range(len(dataset)):
        transcript = normalize_transcript(dataset.iloc[idx]["transcription"])
        sections = re.findall(r"\b[A-Z]+(?: [A-Z]+)*:", transcript)
        if len(sections) >= 2:
            note_prompt = transcript.split(sections[0])[1].split(sections[1])[0]
        else:
            continue
        section_name = sections[0].lower().strip(punctuation)
        if len(note_prompt.split(" ")) > 30 and is_chief_complaint(section_name):
            # There are some chief complaints that seem to be HPI
            section_name = "history of present illness"
        conversations.append(
            CLINICAL_NOTE_GENERATION_TEMPLATE.format(
                section=section_name, section_information=note_prompt, note=transcript
            )
        )
    return conversations


def main(output_dir: str = "data"):
    """Download and prepare the dataset for use."""
    os.makedirs(output_dir, exist_ok=True)
    kaggle.api.dataset_download_files("tboyle10/medicaltranscriptions", "data", unzip=True)
    # Use the `dtype` parameter of `pd.read_csv`.
    mt_samples = preprocess(pd.read_csv("data/mtsamples.csv"))
    conversations = get_conversations(mt_samples)
    random.shuffle(conversations)
    train_limit = math.ceil(len(conversations) * 0.6)
    dev_limit = math.ceil(len(conversations) * 0.8)
    train, validation, test = (
        conversations[:train_limit],
        conversations[train_limit:dev_limit],
        conversations[dev_limit:],
    )
    splits = {"train": train, "validation": validation, "test": test}
    for split in ["train", "validation", "test"]:
        with open(f"{output_dir}/mt_note_generation_{split}.jsonl", "w") as f:
            for conversation in splits[split]:
                f.write(f"{json.dumps({'conversation': conversation})}\n")


if __name__ == "__main__":
    sys.exit(main())
