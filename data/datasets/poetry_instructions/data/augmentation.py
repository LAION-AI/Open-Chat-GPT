from typing import List

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy


def extract_keywords(poem: str, num_keywords: int) -> List[str]:
    model_out = MODEL_STORE.keyword_extractor(poem, num_keywords=num_keywords)
    # print(model_out)
    return [e["word"] for e in model_out]


class FixedERPipeline(TokenClassificationPipeline):
    """Pipeline for Entity Recognition, modified to allow specification
    of the number of entities to extract
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs,
        )
        self._num_keywords = 1
        self.non_entity_label = "O"
        self.key_label = "KEY"
        (self.b_key_label_idx,) = [
            idx for idx, lbl in self.model.config.id2label.items() if lbl == "B-KEY"
        ]  # index for the B-KEY entity

    def __call__(self, *args, num_keywords: int = 1, **kwargs):
        self._num_keywords = num_keywords
        return super().__call__(*args, **kwargs)

    """Taken from
    https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L341

    Modified to fix the total number of keyphrases returned
    """

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        aggregation_strategy = AggregationStrategy.SIMPLE
        sorted_entities = sorted(
            [e for e in pre_entities], key=lambda e: e["scores"][self.b_key_label_idx], reverse=True
        )
        extracted_b_key_words = set()
        extracted_b_key_idxs = set()
        # extract top-n tokens with highest B-KEY score, skipping duplicates
        for e in sorted_entities:
            if e["word"] not in extracted_b_key_words:
                extracted_b_key_words.add(e["word"])
                extracted_b_key_idxs.add(e["index"])
                if len(extracted_b_key_idxs) >= self._num_keywords:
                    break

        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                # if entity is one of our extracted B-KEYs, fix prediction to indicate new keyphrase
                if pre_entity["index"] in extracted_b_key_idxs:
                    entity_idx = self.b_key_label_idx
                else:
                    pre_entity["scores"][self.b_key_label_idx] = 0
                    entity_idx = pre_entity["scores"].argmax()

                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    """Taken from
    https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L420

    Modified to prevent lone I-KEYs from being extracted
    """

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.
        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group

        # modify to set as non-entity if no B-KEY exists in group
        entity = self.non_entity_label
        score = np.nanmean([entity["score"] for entity in entities])
        for e in entities:
            bi, tag = self.get_tag(e["entity"])
            if bi == "B" and tag == self.key_label:
                entity = self.key_label
                score = e["score"]
                break

        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": score,
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group


class ModelStore:
    def __init__(self):
        self.keyword_extractor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self):
        self.keyword_extractor = FixedERPipeline(
            model="yanekyuk/bert-uncased-keyword-extractor",
            device=self.device,
        )


MODEL_STORE = ModelStore()
