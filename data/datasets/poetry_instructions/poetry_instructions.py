# Copyright 2023 The OpenAssistant Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This template serves as a starting point for contributing a dataset to the OpenAssistant repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associates local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/dataset_script

This template is adapted from the one provided by BigScience's BigBIO library:
https://github.com/bigscience-workshop/biomedical/blob/main/templates/template.py

TODO: Before submitting your script, delete this docstring and replace it with a description of your dataset.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

# TODO: import the schema (i.e. features) that fits your dataset:
from hub import OpenAssistantConfig, features

# TODO: Add BibTeX citation where appropriate
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

# TODO: create a module level variable with your dataset name (should match the script name)
#  E.g. The Pile: [dataset_name] --> the_pile
_DATASETNAME = "[poetry_instructions]"
# TODO: create a pretty display name for your dataset
_DISPLAYNAME = "Poetry Instructions"

# TODO: Add a description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset contains prompt-response data for poetry generation.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = ""

# TODO: Add links to the URLs needed to download your dataset files.
# This variable can be a relative path for datasets whose files need to be
# manually downloaded or preprocessed in advance.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {"train": "./data/train.jsonl", "test": "./data/test.jsonl", "validation": "./data/validation.jsonl"}
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [
    "dialogue-modeling"
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: ThePile --> ThePileDataset
class PoetryInstructionsDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version(_VERSION)

    # You will be able to load each dataset with
    # dataset = datasets.load_dataset('my_dataset')

    # TODO: For each dataset, implement a config for each subset;
    #  If a dataset contains more than one subset, implement a config for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. the_pile_[schema_name]
    #   - version: VERSION
    #   - description: one line description for the dataset
    #   - schema: open_assistant_[schema_name]
    #   - subset_id: subset id is the canonical name for the dataset (eg. the_pile)
    #  where [schema_name] = (language_modeling)

    BUILDER_CONFIGS = [
        OpenAssistantConfig(
            name=f"{_DATASETNAME}_dialogue_modeling",
            version=VERSION,
            description=f"OpenAssistant dataset config for {_DATASETNAME}",
            schema_name="dialogue_modeling",
            subset_id=_DATASETNAME,
        )
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME

    def _info(self) -> datasets.DatasetInfo:
        # TODO: Implement the schema for your dataset here.
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access a config choice, that will be in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        if self.config.schema == "dialogue_modeling":
            key = 0
            with open(filepath[split], "r", encoding="utf8") as data:
                while True:
                    line = data.readline()
                    if not line:
                        return
                    yield key, json.loads(line)
                    key += 1


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
