from dataclasses import dataclass
from enum import Enum

import numpy as np

from .augmentation import extract_keywords
from .template import (
    CONTINUE_POEM_TEMPLATE,
    NEW_POEM_ABOUT_IN_STYLE_OF_TEMPLATE,
    NEW_POEM_ABOUT_TEMPLATE,
    NEW_POEM_IN_STYLE_OF_TEMPLATE,
)


@dataclass
class PoetryRecord:
    poem: str
    title: str
    author: str
    theme: str
    time_period: str


class PoetryDialogueTask(Enum):
    CONTINUE = "make_continue_poem_dialogue"
    IN_STYLE = "make_poem_in_style_dialogue"
    ABOUT_KEYWORDS = "make_poem_about_keywords_dialogue"
    ABOUT_KEYWORDS_IN_STYLE = "make_poem_about_keywords_in_style_dialogue"

    @staticmethod
    def make_continue_poem_dialogue(record: "PoetryRecord") -> str:
        line_splits = record.poem.split("\n")
        if len(line_splits) <= 1:
            return PoetryDialogueTask.random_task_excluding(PoetryDialogueTask.CONTINUE).prepare_dialogue(record)
        line_split_idx = np.random.randint(1, len(line_splits))
        return CONTINUE_POEM_TEMPLATE.format(
            poem_start="\n".join(line_splits[:line_split_idx]),
            poem_end="\n".join(line_splits[line_split_idx:]),
        )

    @staticmethod
    def make_poem_in_style_dialogue(record: "PoetryRecord") -> str:
        return NEW_POEM_IN_STYLE_OF_TEMPLATE.format(
            author=record.author,
            poem=record.poem,
        )

    @staticmethod
    def make_poem_about_keywords_dialogue(record: "PoetryRecord") -> str:
        keywords = extract_keywords(record.poem, num_keywords=np.random.randint(1, 4))
        keyword_string = keywords[-1]
        if len(keywords) > 1:
            keyword_string = ", ".join(keywords[:-1]) + " and " + keyword_string
        return NEW_POEM_ABOUT_TEMPLATE.format(
            about=keyword_string,
            poem=record.poem,
        )

    @staticmethod
    def make_poem_about_keywords_in_style_dialogue(record: "PoetryRecord") -> str:
        keywords = extract_keywords(record.poem, num_keywords=np.random.randint(1, 4))
        keyword_string = keywords[-1]
        if len(keywords) > 1:
            keyword_string = ", ".join(keywords[:-1]) + " and " + keyword_string
        return NEW_POEM_ABOUT_IN_STYLE_OF_TEMPLATE.format(
            about=keyword_string,
            author=record.author,
            poem=record.poem,
        )

    @staticmethod
    def random_task():
        return np.random.choice(PoetryDialogueTask)

    @staticmethod
    def random_task_excluding(*exclude_tasks: "PoetryDialogueTask"):
        return np.random.choice([t for t in PoetryDialogueTask if t not in exclude_tasks])

    def prepare_dialogue(self, record: PoetryRecord) -> str:
        return getattr(PoetryDialogueTask, self.value)(record)
