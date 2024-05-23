"""
    Summarize different spectrum of documents
"""
import random
from collections import defaultdict
from typing import List

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

SUMMARIZATION_SPECIAL_TOKENS = {"Text": "", "Summary": ["TL;DR:", "Summarize this", "Give me the summary"]}

SUMMARY_SPECIAL_PROMPT = {
    "multi_news": ["Summarize in bullet points", "Generate summary in list of points"],
    "xsum": ["Give me summary in one sentence", "Short TLDR", "Give me a concise summary"],
    "samsum": ["TLDR;", "Summarize this dialogue", "Summarize dialogue"],
}

summarization_config_mapping = {
    "cnn_dailymail": (
        "cnn_dailymail",
        "3.0.0",
    ),
    "samsum": ("samsum",),
    "xsum": ("xsum",),
    "multi_news": ("multi_news",),
    "scitldr": (
        "scitldr",
        "AIC",
    ),
    "billsum": ("billsum",),
    "reddit": ("reddit",),
    "tldr_news": ("JulesBelveze/tldr_news",),  # need to fix : JulesBelveze/tldr_news
    "debate_sum": ("Hellisotherpeople/DebateSum",),  # Hellisotherpeople/DebateSum
}

summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "samsum": ("dialogue", "summary"),
    "xsum": ("document", "summary"),
    "multi_news": ("document", "summary"),
    "scitldr": ("source", "target"),
    "billsum": ("text", "summary"),
    "reddit": ("content", "summary"),
    "tldr_news": ("content", "headline"),
    "debate_sum": ("Full-Document", "Extract"),
}


def index_summary_default(text, summary):
    return text.replace("\n\n", "\n"), summary


def index_summary_merge(text, summary):
    return " ".join(text), " ".join(summary)


class SummarizationDataset(Dataset):
    def __init__(self, dataset, cache_dir, split, max_words=512):
        self.name = dataset
        if (dataset in ["billsum", "tldr_news"]) and (split == "validation"):
            split = "test"
        self.dataset = load_dataset(*summarization_config_mapping[dataset], cache_dir=cache_dir, split=split)
        self.text_column, self.summary_column = summarization_name_mapping[dataset]
        self.preprocess_fn = index_summary_merge if dataset == "scitldr" else index_summary_default
        self.max_words = max_words

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text, summary = data[self.text_column], data[self.summary_column]
        text, summary = self.preprocess_fn(text, summary)
        if self.name in SUMMARY_SPECIAL_PROMPT:
            prompt = random.choice(SUMMARIZATION_SPECIAL_TOKENS["Summary"])
        else:
            prompt = random.choice(SUMMARIZATION_SPECIAL_TOKENS["Summary"])

        context = "".join([SUMMARIZATION_SPECIAL_TOKENS["Text"], " ".join(text.split(" ")[: self.max_words]), prompt])
        return (context, summary)


SUMMARIZATION_PROMPTS = [
    "Please summarize the following content:\n{}",
    "Write me a summary for the following article:\n{}",
    "Kindly sum up the following information: {}",
    "Please summarize the following text for me:\n{}",
    "Give me a summary of the following text:\n\n{}",
    "Describe the following information in brief: {}",
    "Will you kindly summarize the following paragraph for me?\n{}",
    "Summarize this: {}",
    "TLDR this: {}",
    "{}\nTLDR;",
    "{}\n\nTL;DR",
    "{} tl;dr:",
    "{}\nPlease summarize the content above",
    "{} Please summarize the preceding statements.",
]


class HFSummaryPairs(Dataset):
    """
    Simplified version of the HFSummary class which uses the original examples
    of the OpenAI dataset.
    https://huggingface.co/datasets/openai/summarize_from_feedback
    """

    def __init__(self, split="train", mode="sft", conf_threshold=-1) -> None:
        super().__init__()
        assert split in ("train", "valid1", "valid2", "test")
        assert mode in ("sft", "rm", "rl")
        self.mode = mode

        self.posts = []
        self.summary_pairs = []

        major_split = split if "train" == split else "validation"
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons")[major_split]
        for data in dataset:
            if (
                "extra" in data
                and "confidence" in data["extra"]
                and data["extra"]["confidence"] is not None
                and conf_threshold > data["extra"]["confidence"]
            ):
                print("skipping {}".format(data["info"]["id"]))
                continue

            if split != "train" and split != data["split"]:
                continue

            if "article" in data["info"] and data["info"]["article"] is not None:
                context = data["info"]["article"]
            elif "post" in data["info"]:
                context = data["info"]["post"]

            self.posts.append(context)
            pos, neg = (0, 1) if data["choice"] == 0 else (1, 0)
            self.summary_pairs.append((data["summaries"][pos]["text"].strip(), data["summaries"][neg]["text"].strip()))

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, index: int) -> tuple | list:
        if index < 0 or index >= len(self.posts):
            raise IndexError()

        context = self.posts[index]
        # return pairs of comparison
        good_summary, bad_summary = self.summary_pairs[index]
        prompt = random.choice(SUMMARIZATION_PROMPTS)

        # pair very big
        # we are going to do some sampling
        # not optimal but good for now
        if self.mode == "sft":
            return [prompt.format(context), good_summary]
        elif self.mode == "rl":
            return (prompt.format(context),)
        elif self.mode == "rm":
            return [prompt.format(context)], [good_summary, bad_summary]

        raise RuntimeError(f"Unsupported mode '{self.mode}'")


def head_to_head_votes(pairs):
    # 1. get all possible choices and fix an order to label the axis later
    options = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    # 2. make square preference matrix
    tallies = np.zeros((len(options), len(options)))
    # count up all preferences (might be (j,i) order, not sure of how I ordered winner and loser in my original implementation)
    for i, j in pairs:
        tallies[i, j] += 1
    return tallies, options


def cycle_detect(pairs):
    """Recursively detect cylces by removing condorcet losers until either only one pair is left or condorcet loosers no longer exist
    This method upholds the invariant that in a ranking for all a,b either a>b or b>a for all a,b.


    Returns
    -------
    out : False if the pairs do not contain a cycle, True if the pairs contain a cycle


    """
    # get all condorcet losers (pairs that loose to all other pairs)
    # idea: filter all losers that are never winners
    # print("pairs", pairs)
    if len(pairs) <= 1:
        return False
    losers = [c_lose for c_lose in np.unique(pairs[:, 1]) if c_lose not in pairs[:, 0]]
    if len(losers) == 0:
        # if we recursively removed pairs, and at some point we did not have
        # a condorcet loser, that means everything is both a winner and loser,
        # yielding at least one (winner,loser), (loser,winner) pair
        return True

    new = []
    for p in pairs:
        if p[1] not in losers:
            new.append(p)
    return cycle_detect(np.array(new))


def get_winner(pairs):
    """
    This returns _one_ concordant winner.
    It could be that there are multiple concordant winners, but in our case
    since we are interested in a ranking, we have to choose one at random.
    """
    try:
        losers = np.unique(pairs[:, 1]).astype(int)
        winners = np.unique(pairs[:, 0]).astype(int)
    except IndexError as e:
        print(pairs)
        raise e
    for w in winners:
        if w not in losers:
            return w


def get_ranking(pairs):
    """
    Abuses concordance property to get a (not necessarily unique) ranking.
    The lack of uniqueness is due to the potential existence of multiple
    equally ranked winners. We have to pick one, which is where
    the non-uniqueness comes from
    """
    if len(pairs) == 1:
        return list(pairs[0])
    w = get_winner(pairs)
    # now remove the winner from the list of pairs
    p_new = np.array([(a, b) for a, b in pairs if a != w])
    return [w] + get_ranking(p_new)


def ranked_pairs(ranks: List[List[int]]):
    """
    Expects a list of rankings for an item like:
        [("w","x","z","y") for _ in range(3)]
        + [("w","y","x","z") for _ in range(2)]
        + [("x","y","z","w") for _ in range(4)]
        + [("x","z","w","y") for _ in range(5)]
        + [("y","w","x","z") for _ in range(1)]
    This code is quite brain melting, but the idea is the following:
    1. create a head-to-head matrix that tallies up all win-lose combinations of preferences
    2. take all combinations that win more than they loose and sort those by how often they win
    3. use that to create an (implicit) directed graph
    4. recursively extract nodes from the graph that do not have incoming edges
    5. said recursive list is the ranking
    """
    tallies, names = head_to_head_votes(ranks)
    tallies = tallies - tallies.T
    # note: the resulting tally matrix should be skew-symmetric
    # order by strength of victory (using tideman's original method, don't think it would make a difference for us)
    sorted_majorities = []
    for i in range(len(tallies)):
        for j in range(len(tallies)):
            # you can never prefer yourself over yourself
            # we also have to pick one of the two choices,
            # if the preference is exactly zero...
            if tallies[i, j] >= 0 and i != j:
                sorted_majorities.append((i, j, tallies[i, j]))
    # we don't explicitly deal with tied majorities here
    sorted_majorities = np.array(sorted(sorted_majorities, key=lambda x: x[2], reverse=True))
    # now do lock ins
    lock_ins = []
    for x, y, _ in sorted_majorities:
        # invariant: lock_ins has no cycles here
        lock_ins.append((x, y))
        # print("lock ins are now",np.array(lock_ins))
        if cycle_detect(np.array(lock_ins)):
            # print("backup: cycle detected")
            # if there's a cycle, delete the new addition and continue
            lock_ins = lock_ins[:-1]
    # now simply return all winners in order, and attach the losers
    # to the back. This is because the overall loser might not be unique
    # and (by concordance property) may never exist in any winning set to begin with.
    # (otherwise he would either not be the loser, or cycles exist!)
    # Since there could be multiple overall losers, we just return them in any order
    # as we are unable to find a closer ranking
    if len(lock_ins) == 0:  # no winner found
        return []
    numerical_ranks = np.array(get_ranking(np.array(lock_ins))).astype(int)
    conversion = [names[n] for n in numerical_ranks]
    return conversion


class HFSummary(Dataset):
    """
    Human feedback data from OpenAI
    https://github.com/openai/summarize-from-feedback
    https://huggingface.co/datasets/openai/summarize_from_feedback

    labeling method : pair comparison, 0 or 1

    """

    def __init__(self, split="train", mode="sft", conf_threshold=-1, max_comparison_per_sample=5) -> None:
        super().__init__()
        assert split in ("train", "valid1", "valid2", "test")
        assert mode in ("sft", "rm", "rl")
        self.mode = mode
        summaries = {}
        # using prompt as our index will allows us
        # to add additional generated prompt later
        self.index2summary = {}
        self.max_comparison_per_sample = max_comparison_per_sample
        major_split = split if "train" == split else "validation"
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons")[major_split]
        for data in dataset:
            if (
                "extra" in data
                and "confidence" in data["extra"]
                and data["extra"]["confidence"] is not None
                and conf_threshold > data["extra"]["confidence"]
            ):
                print("skipping {}".format(data["info"]["id"]))
                continue

            if split != "train" and split != data["split"]:
                continue

            if "article" in data["info"] and data["info"]["article"] is not None:
                context = data["info"]["article"]
            elif "post" in data["info"]:
                context = data["info"]["post"]

            if context not in summaries:
                summaries[context] = []

            pos, neg = (0, 1) if data["choice"] == 0 else (1, 0)
            summaries[context].append((data["summaries"][pos]["text"].strip(), data["summaries"][neg]["text"].strip()))

        ranked_summaries = []
        for context, summary_comparison_pairs in summaries.items():
            if len(summary_comparison_pairs) > 30:
                # outlier example
                continue
            # ranks = self.aggregate_pairs(summary_comparison_pairs)
            # if len(ranks) > 1:
            #     ranked_summaries.append((context, ranks))
            comparison_mapping = {}
            ranks = []
            for pair in summary_comparison_pairs:
                if pair[0] not in comparison_mapping:
                    comparison_mapping[pair[0]] = len(comparison_mapping)
                if pair[1] not in comparison_mapping:
                    comparison_mapping[pair[1]] = len(comparison_mapping)
                ranks.append([comparison_mapping[pair[0]], comparison_mapping[pair[1]]])
            reverse_mapping = {idx: context for context, idx in comparison_mapping.items()}
            ranked = ranked_pairs(ranks)
            if len(ranked) > 1 and mode == "rm":
                ranked_summaries.append((context, [reverse_mapping[rank_id] for rank_id in ranked]))
            else:  # we want to include one ranks for sft and rl
                ranked_summaries.append((context, [reverse_mapping[rank_id] for rank_id in ranked]))

        self.summaries = ranked_summaries

    @staticmethod
    def traverse_dag(reverse_linked_dag, key, depth=0):
        # we want to find all the ancestor of the key:
        # ie ancestor of D are [A, B, C]
        # so we create a reverse linked dag
        if depth > (len(reverse_linked_dag) + 1):
            # cicular rank detected, ignore results
            return []
        output = []
        if key in reverse_linked_dag:
            for element in reverse_linked_dag[key]:
                output.append(element)
                if element in reverse_linked_dag:
                    output += HFSummary.traverse_dag(reverse_linked_dag, element, depth=depth + 1)
        return output

    @staticmethod
    def aggregate_pairs(comparison_pairs):
        # perfect case
        # comparison_pairs = [('B', 'D'), ('A', 'B'), ('C', 'D'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D')]
        # comparison_pairs = [('B', 'C'), ('A', 'B'), ('C', 'D')]
        # comparison_pairs = [('B', 'D'), ('A', 'B'), ('C', 'D')]
        # aggregate_pairs(comparison_pairs) = ['A', 'B', 'C', 'D']
        # A > B > C
        # perfect case:
        # (A, B), (A, C), (B, C)
        # worse case:
        # (A, B), (B, C)
        # A > B = C > D
        # (A, B), (B, D), (A, C), (C, D)
        # TODO: how to handle such case?
        reverse_linked_dag = defaultdict(set)
        for pair in comparison_pairs:
            reverse_linked_dag[pair[1]].add(pair[0])
        node_count = defaultdict(int)
        for key, _ in reverse_linked_dag.items():
            node_count[key]  # make it zero
            for e in HFSummary.traverse_dag(reverse_linked_dag, key):
                node_count[e] += 1
        # those who are referenced the most is the best summary
        # if you want to handle tied results, just find the same frequency
        elements_counts = [(element, count) for element, count in node_count.items()]
        # Sort the list of tuples by count in descending order
        elements_counts.sort(key=lambda x: x[1], reverse=True)
        # Create a list of elements in order of their counts
        sorted_elements = [element for element, _ in elements_counts]
        return sorted_elements

    def __len__(self) -> int:
        return len(self.summaries)

    def __getitem__(self, index) -> tuple | list:
        context, rows = self.summaries[index]
        prompt = random.choice(SUMMARIZATION_PROMPTS)

        if self.mode == "sft":
            return [prompt.format(context), rows[0]]
        elif self.mode == "rl":
            return (prompt.format(context),)
        elif self.mode == "rm":
            if len(rows) > self.max_comparison_per_sample:
                valid_idx = np.random.choice(len(rows), self.max_comparison_per_sample)
                return [prompt.format(context)], [r for idx, r in enumerate(rows) if idx in valid_idx]
            return [prompt.format(context)], rows

        raise RuntimeError(f"Unsupported mode '{self.mode}'")
