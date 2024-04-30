import re
from enum import Enum
from itertools import zip_longest
from random import random, shuffle
from typing import Literal, Optional

from pydantic import BaseModel, validator
from pydantic.fields import ModelField

QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}


def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        QA_SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )


def compute_length(s: str) -> int:
    return len(re.findall(r"\w+", s)) // 5 + 1


class Mode(str, Enum):
    sft = "sft"
    rm = "rm"
    rl = "rl"


class Role(str, Enum):
    prompter = "prompter"
    assistant = "assistant"


class Utterance(BaseModel):
    text: str
    role: Role
    lang: str | None = None
    quality: float | None = None
    humor: float | None = None
    creativity: float | None = None
    context: str | None = None

    @validator("quality", "humor", "creativity")
    def between_0_1(cls, v, field: ModelField) -> float:
        if v is not None and not (0 <= v <= 1):
            raise ValueError(f"Field {field.name} must be between 0 and 1. Received: {v}")
        return v

    def system_tag(
        self,
        eos_token: str,
        enabled: bool = True,
        property_dropout: float = 0.0,
        add_length: bool = True,
    ) -> str:
        if not enabled:
            return ""

        properties: list[tuple[float | str]] = []
        for k, v in self.dict().items():
            if v is not None and k in ["lang", "quality", "humor", "creativity"]:
                properties.append((k, v))

        if add_length:
            properties.append(("length", compute_length(self.text)))

        shuffle(properties)

        # ensure that potentially multi-line context field comes last
        if self.context:
            properties.append(("context", self.context))

        fragments: list[str] = []
        for k, v in properties:
            if random() < property_dropout:
                continue

            if isinstance(v, float):
                fragments.append(f"{k}: {v:0.1f}")
            elif isinstance(v, str):
                if not v.isspace():  # ignore whitespace-only values
                    fragments.append(f"{k}: {v}")
            else:
                fragments.append(f"{k}: {v}")

        if len(fragments) == 0:
            return ""

        content = "\n".join(fragments)
        return f"{QA_SPECIAL_TOKENS['System']}{content}\n{eos_token}"


class DatasetEntry(BaseModel):
    pass


class DatasetEntryLm(DatasetEntry):
    """Language modelling dataset entry"""

    text: str | None = None


class DatasetEntrySft(DatasetEntry):
    """Supervised fine-tuning conversation dataset entry"""

    conversation: list[Utterance]
    system_message: Optional[str]

    def get_formatted(
        self,
        eos_token: str,
        use_system_tag: bool = False,
        system_property_dropout: float = 0.5,
        system_add_length: bool = False,
    ) -> list[str]:
        output: list[str] = []

        for i, m in enumerate(self.conversation):
            if m.role == Role.prompter:
                if use_system_tag and i + 1 < len(self.conversation):
                    a = self.conversation[i + 1]
                    assert a.role == Role.assistant
                    system_tag = a.system_tag(
                        eos_token=eos_token,
                        property_dropout=system_property_dropout,
                        add_length=system_add_length,
                    )
                else:
                    system_tag = ""
                if i == 0 and self.system_message:
                    output.append(
                        f"{QA_SPECIAL_TOKENS['System']}{self.system_message}{eos_token}{QA_SPECIAL_TOKENS['Question']}{m.text}{eos_token}{system_tag}"
                    )
                else:
                    output.append(f"{QA_SPECIAL_TOKENS['Question']}{m.text}{eos_token}{system_tag}")
            else:
                output.append(f"{QA_SPECIAL_TOKENS['Answer']}{m.text}{eos_token}")

        return output


class DatasetEntryRm(DatasetEntry):
    """Reward model dataset entry (conversation history + ranked replies)"""

    messages: list[Utterance] | None  # conversation history
    replies: list[Utterance]  # ordered reply variants, best first

    def get_formatted(
        self,
        eos_token: str,
        use_system_tag: bool = False,
        system_property_dropout: float = 0.5,
        system_add_length: bool = False,
        max_replies: int = 5,
    ) -> tuple[str, list[str]]:
        reply_variants = self.replies
        if len(reply_variants) > max_replies:
            reply_variants = reply_variants[:max_replies]

        # special handling for non-dialogue datasets like Hellaswag
        if self.messages is None or len(self.messages) == 1 and self.messages[0] is None:
            prefix = ""
            replies = [r.text + eos_token for r in reply_variants]
            return prefix, replies

        assert len(self.messages) > 0 and self.messages[-1].role == Role.prompter

        # format conversation history (prefix)
        prefix_messages: list[str] = []
        for i, m in enumerate(self.messages):
            if m.role == Role.prompter:
                prefix_messages.append(f"{QA_SPECIAL_TOKENS['Question']}{m.text}{eos_token}")
            else:
                if use_system_tag:
                    assert m.role == Role.assistant
                    system_tag = m.system_tag(
                        eos_token=eos_token,
                        property_dropout=system_property_dropout,
                        add_length=system_add_length,
                    )
                else:
                    system_tag = ""
                prefix_messages.append(f"{system_tag}{QA_SPECIAL_TOKENS['Answer']}{m.text}{eos_token}")
        prefix = "".join(prefix_messages)

        #  format reply variants
        replies: list[str] = []
        for r in reply_variants:
            assert r.role == Role.assistant
            if use_system_tag:
                system_tag = r.system_tag(
                    eos_token=eos_token,
                    property_dropout=system_property_dropout,
                    add_length=system_add_length,
                )
            else:
                system_tag = ""
            replies.append(f"{system_tag}{QA_SPECIAL_TOKENS['Answer']}{r.text}{eos_token}")

        return prefix, replies


def create_dataset_entry_qa(
    mode: Mode | Literal["sft", "rm", "rl"],
    questions: list[str],
    answers: list[str] | list[list[str]],
    context: Optional[str] = None,
    lang: Optional[str] = None,
) -> DatasetEntry:
    """Helper function to create DatasetEntry objects (DatasetEntrySft or DatasetEntryRm) for simple
    Q&A datasets."""
    if mode == Mode.sft:
        messages: list[Utterance] = []

        for q, a in zip_longest(questions, answers):
            messages.append(Utterance(text=q, role=Role.prompter, lang=lang))
            if isinstance(a, list):
                a = a[0]
            messages.append(Utterance(text=a, role=Role.assistant, lang=lang, context=context))

        return DatasetEntrySft(conversation=messages)
    elif mode == Mode.rm:
        if len(questions) != 1:
            raise RuntimeError("QA dataset entry factory does not support multi-turn conversation for the RM case.")

        if len(answers) == 1 and isinstance(answers[0], list):
            answers = answers[0]

        assert isinstance(answers, list) and len(answers) > 1 and isinstance(answers[0], str)
        conversation_history = [Utterance(text=questions[0], role=Role.prompter, lang=lang)]
        reply_variants = [Utterance(text=a, role=Role.assistant, lang=lang, context=context) for a in answers]
        return DatasetEntryRm(messages=conversation_history, replies=reply_variants)
    # elif mode == Mode.rl:
    else:
        raise RuntimeError(f"Unsupported mode ({mode=})")


def format_pairs(
    pairs: list[str],
    eos_token: str,
    add_initial_reply_token: bool = False,
) -> list[str]:
    assert isinstance(pairs, list)
    conversations = [
        "{}{}{}".format(QA_SPECIAL_TOKENS["Question" if i % 2 == 0 else "Answer"], pairs[i], eos_token)
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(QA_SPECIAL_TOKENS["Answer"])
    return conversations


def format_rl_text(pairs: list[str]) -> str:
    # convert question answer pairs to only the prefix prompt for RLHF
    return "{}{}{}".format(QA_SPECIAL_TOKENS["Question"], pairs[0], QA_SPECIAL_TOKENS["Answer"])


def format_reply(text: str, eos_token: str) -> str:
    return "{}{}{}".format(QA_SPECIAL_TOKENS["Answer"], text, eos_token)
