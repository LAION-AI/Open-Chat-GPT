from dataclasses import dataclass

import datasets


@dataclass
class OpenAssistantConfig(datasets.BuilderConfig):
    """BuilderConfig for OpenAssistant datasets."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None


features = datasets.Features(
    {
        "conversation": datasets.Value("string"),
    }
)
