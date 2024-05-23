# Dataset Card for Yahoo Questions and Answers

## Dataset Description

- **Original:** [HuggingFace](https://huggingface.co/datasets/yahoo_answers_qa)

## Dataset Summary

The dataset is a regroupment of Yahoo questions, topics, and best answers.

## Script

### What it is

This script is made to convert the original datasets to the expected Open
Assistant datasets format.

This script can aggregate multiple best answers for one question to create more
unique entries.

It can also generate, if activated, a toxicity number between 0-1 using
[Detoxify](https://github.com/unitaryai/detoxify).

### What it's not

This script does not automatically upload any datasets to Hugging Face. It can
only create a JSON file to export the newly generated datasets.

### Usage

`python yahoo_qa_conversion.py --help` should do the job 😄

## Author

The script to convert the original datasets was created by
[Shadowner](https://github.com/Shadowner).
